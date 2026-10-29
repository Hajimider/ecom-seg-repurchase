"""在线零售客户分群与复购预测分析。

清洗交易数据，计算 RFM 指标并分群，按时间窗口评估复购模型。
结果保存到 outputs。
"""

from pathlib import Path
from io import BytesIO
import sqlite3
import zipfile

import matplotlib
matplotlib.use('Agg')  # 仅保存图片，不打开绘图窗口。
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, davies_bouldin_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve, silhouette_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from report import build_report, configure_plots


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "online_retail.zip"
OUT = ROOT / "outputs"
PROCESSED = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
ANALYSIS_DATE = pd.Timestamp("2011-12-10")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
configure_plots()
MODEL_NAMES = {'logistic': 'Logistic回归', 'random_forest': '随机森林'}


def load_transactions():
    """读取原始交易并转换日期、客户编号、数量和单价的数据类型。"""
    if not RAW.exists():
        raise FileNotFoundError('找不到原始数据，请将 online_retail.zip 放入本项目的 data/raw 文件夹。')
    with zipfile.ZipFile(RAW) as zf:
        name = next((n for n in zf.namelist() if n.lower().endswith(".xlsx")), None)
        if not name:
            raise ValueError("The retail archive has no XLSX workbook")
        # XLSX 也是压缩文件；先读入内存，可减少读取时对外层 ZIP 的重复解压。
        tx = pd.read_excel(BytesIO(zf.read(name)))
    tx.columns = [str(c).strip().replace(" ", "") for c in tx.columns]
    required = {"InvoiceNo", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"}
    missing = required.difference(tx.columns)
    if missing:
        raise ValueError(f"Missing retail columns: {sorted(missing)}")
    tx["InvoiceDate"] = pd.to_datetime(tx["InvoiceDate"], errors="coerce")
    tx["CustomerID"] = pd.to_numeric(tx["CustomerID"], errors="coerce")
    tx["Quantity"] = pd.to_numeric(tx["Quantity"], errors="coerce")
    tx["UnitPrice"] = pd.to_numeric(tx["UnitPrice"], errors="coerce")
    # 商品金额 = 数量 × 单价，负数量对应的金额也为负。
    tx["revenue"] = tx["Quantity"] * tx["UnitPrice"]
    if tx[['InvoiceDate', 'Quantity', 'UnitPrice']].isna().any().any():
        raise ValueError('日期、数量或单价无法解析，请检查原始数据。')
    return tx


def clean_transaction(tx):
    """记录每条交易的清洗标记，并生成客户级分析样本。"""
    flags = pd.DataFrame(index=tx.index)
    flags["missing_customer"] = tx.CustomerID.isna()
    flags["cancelled_invoice"] = tx.InvoiceNo.astype(str).str.startswith("C")
    flags["nonpositive_quantity"] = tx.Quantity <= 0
    flags["nonpositive_price"] = tx.UnitPrice <= 0
    # 此标记为 True 的交易纳入客户分析。
    flags["valid_positive_customer_sale"] = ~(flags.any(axis=1))
    rows = [{"step": "raw", "rows": len(tx), "amount": tx.revenue.sum()}]
    # 逐步记录剩余行数和金额；同一行可同时命中 flags 中的多条规则。
    mask = pd.Series(True, index=tx.index)
    for name in ["missing_customer", "cancelled_invoice", "nonpositive_quantity", "nonpositive_price"]:
        mask &= ~flags[name]
        rows.append({"step": f"after_excluding_{name}", "rows": int(mask.sum()), "amount": float(tx.loc[mask, "revenue"].sum())})
    clean = tx.loc[mask].copy()
    return clean, flags, pd.DataFrame(rows)


def customer_features(tx, cutoff):
    """用截止日之前的有效交易计算某一时间点的 R、F、M。"""
    # 特征只使用截止日之前的购买记录，防止引入未来信息。
    hist = tx[(tx.InvoiceDate < cutoff) & tx.CustomerID.notna() & ~tx.InvoiceNo.astype(str).str.startswith("C") & (tx.Quantity > 0) & (tx.UnitPrice > 0)].copy()
    hist["revenue"] = hist.Quantity * hist.UnitPrice
    g = hist.groupby("CustomerID").agg(last_purchase=("InvoiceDate", "max"), frequency=("InvoiceNo", "nunique"), monetary=("revenue", "sum"))
    g["recency"] = (pd.Timestamp(cutoff) - g.pop("last_purchase")).dt.days
    return g[["recency", "frequency", "monetary"]]


def labels(tx, features, cutoff, end):
    """生成窗口标签：窗口内发生过有效购买的客户记为 1。"""
    # 标签窗口左闭右开：包含 cutoff 当天，不包含 end 当天。
    future = tx[(tx.InvoiceDate >= cutoff) & (tx.InvoiceDate < end) & tx.CustomerID.isin(features.index) & ~tx.InvoiceNo.astype(str).str.startswith("C") & (tx.Quantity > 0) & (tx.UnitPrice > 0)]
    return future.groupby("CustomerID").InvoiceNo.nunique().gt(0).reindex(features.index, fill_value=False).astype(int)


def plot_clustering(z, customer):
    """比较 k=2 至 k=8 的聚类质量，并保存评价图。"""
    rows = []
    for k in range(2, 9):
        labels_k = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20).fit_predict(z)
        rows.append({"k": k, "silhouette": silhouette_score(z, labels_k, sample_size=min(1500, len(z)), random_state=RANDOM_STATE), "calinski_harabasz": __import__("sklearn.metrics", fromlist=["calinski_harabasz_score"]).calinski_harabasz_score(z, labels_k), "davies_bouldin": davies_bouldin_score(z, labels_k)})
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "cluster_k_evaluation.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(scores.k, scores.silhouette, marker="o", label="轮廓系数")
    ax.scatter([4], scores.loc[scores.k == 4, 'silhouette'], s=100, color='#c2410c', zorder=3, label='画像使用四簇')
    ax.set_xlabel("聚类数 k")
    ax.set_ylabel("轮廓系数（越高分离越清晰）")
    ax.set_title("RFM 聚类数量评价")
    ax.set_xticks(scores.k)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cluster_k_evaluation.png", dpi=180)
    plt.close(fig)
    return scores


def plot_repurchase(y, predictions):
    """绘制两个复购模型在同一测试窗口上的 ROC、PR 曲线。"""
    plt.figure(figsize=(7.5, 6))
    for name, p in predictions.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{MODEL_NAMES[name]} (AUC={roc_auc_score(y, p):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="#777")
    plt.xlabel("假阳性率")
    plt.ylabel("真阳性率（召回率）")
    plt.title("复购预测 ROC 曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "repurchase_roc.png", dpi=180)
    plt.close()
    plt.figure(figsize=(7.5, 6))
    for name, p in predictions.items():
        precision, recall, _ = precision_recall_curve(y, p)
        plt.plot(recall, precision, label=f"{MODEL_NAMES[name]} (AP={average_precision_score(y, p):.3f})")
    plt.axhline(np.mean(y), linestyle="--", color="#777", label="测试集复购率")
    plt.xlabel("召回率")
    plt.ylabel("查准率")
    plt.title("复购预测 PR 曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "repurchase_pr.png", dpi=180)
    plt.close()


def main():
    """运行完整零售分析并保存清洗、分群和预测结果。"""
    print('案例实务：正在读取交易数据并进行清洗……', flush=True)
    tx = load_transactions()
    clean, flags, cleaning = clean_transaction(tx)
    tx.to_csv(PROCESSED / "online_retail.csv", index=False, encoding="utf-8-sig")
    cleaning.to_csv(OUT / "cleaning_reconciliation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"rule": flags.columns, "rows_flagged": [int(flags[c].sum()) for c in flags.columns], "amount_flagged": [float(tx.loc[flags[c], "revenue"].sum()) for c in flags.columns]}).to_csv(OUT / "cleaning_flags.csv", index=False, encoding="utf-8-sig")

    # 将交易写入 SQLite，核对 SQL 与 Python 的汇总结果。
    db = sqlite3.connect(OUT / "retail.sqlite")
    tx.drop(columns=["revenue"]).to_sql("transactions", db, if_exists="replace", index=False)
    sql_text = (ROOT / "sql" / "retail_analysis.sql").read_text(encoding="utf-8")
    sql_text = "\n".join(line for line in sql_text.splitlines() if not line.lstrip().startswith("--"))
    statements = [s.strip() for s in sql_text.split(";") if s.strip()]
    sql_outputs = {}
    for statement in statements:
        query = pd.read_sql_query(statement, db)
        if "month" in query.columns:
            query.to_csv(OUT / "monthly_sales.csv", index=False, encoding="utf-8-sig")
            sql_outputs["monthly"] = query
        if "CustomerID" in query.columns:
            query.to_csv(OUT / "customer_rfm.csv", index=False, encoding="utf-8-sig")
            sql_outputs["rfm"] = query
    db.close()
    print('SQL汇总完成，正在构建客户画像和聚类……', flush=True)

    # 月度金额采用清洗后的可识别客户正向商品口径。
    month = clean.assign(month=clean.InvoiceDate.dt.to_period("M").astype(str)).groupby("month").agg(positive_merchandise_amount=("revenue", "sum"), invoices=("InvoiceNo", "nunique"), customers=("CustomerID", "nunique")).reset_index()
    month.to_csv(OUT / "monthly_sales.csv", index=False, encoding="utf-8-sig")
    customer = clean.groupby("CustomerID").agg(recency=("InvoiceDate", lambda x: (ANALYSIS_DATE - x.max()).days), frequency=("InvoiceNo", "nunique"), monetary=("revenue", "sum"))
    # 逐项核对 SQL 与 Python 的月度金额及客户 RFM。
    pd.testing.assert_frame_equal(sql_outputs['monthly'].reset_index(drop=True), month.reset_index(drop=True), check_dtype=False, atol=1e-7, rtol=1e-10)
    sql_rfm = sql_outputs['rfm'].set_index('CustomerID').rename(columns={'recency_days': 'recency'})
    pd.testing.assert_frame_equal(sql_rfm[customer.columns].sort_index(), customer.sort_index(), check_dtype=False, atol=1e-7, rtol=1e-10)
    # R 越小越活跃，因此 R 的分数方向与 F、M 相反。
    customer["R_score"] = pd.qcut(customer.recency.rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    customer["F_score"] = pd.qcut(customer.frequency.rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    customer["M_score"] = pd.qcut(customer.monetary.rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    customer["RFM_segment"] = np.select([(customer.R_score >= 3) & (customer.F_score >= 3) & (customer.M_score >= 3), (customer.R_score <= 2) & (customer.F_score >= 3), customer.F_score <= 2], ["高价值活跃", "高频待唤回", "低频客户"], default="发展中客户")
    segments = customer.groupby("RFM_segment").agg(customers=("recency", "size"), recency_mean=("recency", "mean"), frequency_mean=("frequency", "mean"), monetary_mean=("monetary", "mean")).reset_index()
    segments.to_csv(OUT / "rfm_segments.csv", index=False, encoding="utf-8-sig")
    # 聚类前压缩金额和频次的长尾，再把三个指标放到相近尺度。
    vals = np.log1p(customer[["recency", "frequency", "monetary"]])
    z = StandardScaler().fit_transform(vals)
    k_scores = plot_clustering(z, customer)
    # 固定四簇用于画像；各候选 k 的评分见 cluster_k_evaluation.csv。
    km = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=20).fit(z)
    customer["cluster"] = km.labels_
    silhouette = float(silhouette_score(z, km.labels_, sample_size=min(1500, len(z)), random_state=RANDOM_STATE))
    cluster_summary = customer.groupby("cluster").agg(customers=("recency", "size"), recency_mean=("recency", "mean"), frequency_mean=("frequency", "mean"), monetary_mean=("monetary", "mean")).reset_index()
    cluster_summary.to_csv(OUT / "kmeans_segments.csv", index=False, encoding="utf-8-sig")
    pd.crosstab(customer.RFM_segment, customer.cluster).to_csv(OUT / "rfm_cluster_crosstab.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    heat = segments.set_index("RFM_segment")[["recency_mean", "frequency_mean", "monetary_mean"]].copy()
    heat = (heat - heat.mean()) / heat.std()
    im = ax.imshow(heat.values, aspect="auto", cmap="RdYlBu_r")
    ax.set_xticks(range(3), ["最近购买间隔 R（越小越近期）", "购买频次 F", "商品金额 M"])
    ax.set_yticks(range(len(heat)), heat.index)
    ax.set_title("RFM 客户群体画像（标准化均值）")
    fig.colorbar(im, ax=ax, label="群体均值的标准化值")
    fig.tight_layout()
    fig.savefig(OUT / "rfm_profile.png", dpi=180)
    plt.close(fig)

    # 训练和测试按时间推进：特征只看截点以前，标签只看截点之后。
    train_cut, train_end = pd.Timestamp("2011-06-01"), pd.Timestamp("2011-09-01")
    test_cut, test_end = pd.Timestamp("2011-09-01"), pd.Timestamp("2011-12-01")
    xtr, ytr = customer_features(tx, train_cut), None
    ytr = labels(tx, xtr, train_cut, train_end)
    xte = customer_features(tx, test_cut)
    yte = labels(tx, xte, test_cut, test_end)
    feature_cols = ["recency", "frequency", "monetary"]
    # 两个模型使用相同的 RFM 特征；类别权重调整训练中各类样本的贡献。
    models = {"logistic": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)), "random_forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1))}
    metric_rows, prediction_map = [], {}
    for name, model in models.items():
        print(f'正在训练复购模型：{MODEL_NAMES[name]}', flush=True)
        # 用前一窗口训练，用后一窗口测试。
        model.fit(xtr[feature_cols], ytr)
        p = model.predict_proba(xte[feature_cols])[:, 1]
        prediction_map[name] = p
        pred = p >= 0.5
        metric_rows.append({"model": name, "train_n": len(xtr), "train_positive_rate": float(ytr.mean()), "test_n": len(xte), "test_positive_rate": float(yte.mean()), "roc_auc": roc_auc_score(yte, p), "pr_auc": average_precision_score(yte, p), "precision_0_5": precision_score(yte, pred, zero_division=0), "recall_0_5": recall_score(yte, pred, zero_division=0), "f1_0_5": f1_score(yte, pred, zero_division=0)})
    # 保存概率模型的排序指标和 0.5 阈值下的分类指标。
    pd.DataFrame(metric_rows).to_csv(OUT / "repurchase_metrics.csv", index=False, encoding="utf-8-sig")
    plot_repurchase(yte, prediction_map)

    # 比较不同阈值下的名单规模、查准率和召回率，供选择触达范围时参考。
    threshold_rows = []
    for name, p in prediction_map.items():
        for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            pred = p >= threshold
            threshold_rows.append({"model": name, "threshold": threshold, "selected_rate": float(pred.mean()), "precision": precision_score(yte, pred, zero_division=0), "recall": recall_score(yte, pred, zero_division=0), "f1": f1_score(yte, pred, zero_division=0)})
    pd.DataFrame(threshold_rows).to_csv(OUT / "repurchase_thresholds.csv", index=False, encoding="utf-8-sig")

    summary = {"raw_rows": len(tx), "clean_rows": len(clean), "missing_customer_rows": int(flags.missing_customer.sum()), "cancelled_invoice_rows": int(flags.cancelled_invoice.sum()), "negative_quantity_rows": int((tx.Quantity < 0).sum()), "nonpositive_price_rows": int(flags.nonpositive_price.sum()), "customers": int(len(customer)), "analysis_date": ANALYSIS_DATE.strftime("%Y-%m-%d"), "silhouette_k4": silhouette, "best_silhouette_k": int(k_scores.loc[k_scores.silhouette.idxmax(), "k"]), "train_cutoff": train_cut.strftime("%Y-%m-%d"), "train_end": train_end.strftime("%Y-%m-%d"), "test_cutoff": test_cut.strftime("%Y-%m-%d"), "test_end": test_end.strftime("%Y-%m-%d"), "train_n": len(xtr), "test_n": len(xte), "train_repeat_rate": float(ytr.mean()), "test_repeat_rate": float(yte.mean()), "positive_merchandise_amount_clean": float(clean.revenue.sum()), "positive_merchandise_amount_all": float(tx.loc[(tx.Quantity > 0) & (tx.UnitPrice > 0), "revenue"].sum()), "signed_amount_all": float(tx.revenue.sum()), "random_state": RANDOM_STATE}
    summary.update(data_start=tx.InvoiceDate.min().strftime('%Y-%m-%d'), data_end=tx.InvoiceDate.max().strftime('%Y-%m-%d'), sql_python_consistent=True)
    pd.DataFrame([summary]).to_csv(OUT / 'summary.csv', index=False, encoding='utf-8-sig')
    path = build_report()
    print(f'完成：有效交易{len(clean):,}行，客户{len(customer):,}名。')
    print(f'报告：{path}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='电商客户分群、复购预测与结果报告')
    parser.add_argument('--check', action='store_true', help='核验已有结果并重建报告，不重新训练模型')
    args = parser.parse_args()
    if args.check:
        print(f'结果核验通过，可视化报告：{build_report()}')
    else:
        main()
