"""读取零售分析结果，生成离线 HTML 报告。"""
from base64 import b64encode
from datetime import datetime, timezone, timedelta
from html import escape
from pathlib import Path
from textwrap import dedent, indent

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / 'outputs'
ACCENT = '#0f4c5c'
MODEL_NAMES = {'logistic': 'Logistic回归', 'random_forest': '随机森林'}


def configure_plots():
    """设置图表配色和中文字体。"""
    plt.rcParams.update(
        {
            'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans'],
            'axes.unicode_minus': False,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.prop_cycle': cycler(color=['#0f4c5c', '#c2410c', '#64748b', '#8b5e83']),
            'axes.facecolor': '#fffdf8',
            'figure.facecolor': '#fffdf8',
            'axes.labelcolor': '#334155',
            'text.color': '#172033',
            'axes.titlepad': 15,
            'axes.grid': True,
            'grid.alpha': .18,
            'font.size': 10,
            'lines.linewidth': 2,
            'savefig.bbox': 'tight',
        },
    )


def read_table(name, columns):
    """读取结果表，检查文件是否为空、必要列是否齐全。"""
    path = OUT / name
    if not path.is_file():
        raise FileNotFoundError(f'缺少分析结果 {name}，请先运行本项目的 run.py。')
    frame = pd.read_csv(path)
    if frame.empty or not set(columns).issubset(frame.columns):
        raise ValueError(f'{name}为空或缺少必要字段：{columns}')
    return frame


def table_html(frame, names):
    """选择并重命名展示列；转义单元格中的 HTML 字符。"""
    data = frame[list(names)].copy().rename(columns=names)
    table = data.to_html(
        index=False,
        border=0,
        escape=True,
        float_format=lambda value: f'{value:,.4f}',
        na_rep='—',
    )
    return (
        '<div class="table-scroll" tabindex="0" aria-label="结果表，可横向滚动">\n'
        + indent(table, '    ')
        + '\n</div>'
    )


def figure_html(name, title, note):
    """将 PNG 嵌入页面，离线打开时无需另带图片。"""
    encoded = b64encode((OUT / name).read_bytes()).decode('ascii')
    return dedent(f"""\
        <figure>
            <figcaption>{escape(title)}</figcaption>
            <img src="data:image/png;base64,{encoded}" alt="{escape(title)}">
            <p class="note">{escape(note)}</p>
        </figure>
        """)


def figures_html(*figures):
    """把图表放入现有的双列布局，窄屏下由 CSS 切换为单列。"""
    return (
        '<div class="two">\n'
        + indent('\n'.join(figure.strip() for figure in figures), '    ')
        + '\n</div>'
    )

CSS = """
:root {
    --accent: ACCENT; --ink: #172033; --muted: #586577; --line: #dcdedc; --paper: #fffdf8;
    --bg: #f7f3ea;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 16px/1.75 "Microsoft YaHei", "PingFang SC", sans-serif;
}
a { color: var(--accent); text-underline-offset: 4px; }
a:focus-visible, summary:focus-visible, [tabindex]:focus-visible {
    outline: 3px solid #c2410c; outline-offset: 4px;
}
.skip {
    position: absolute; top: -100px; left: 20px; background: white; padding: 8px;
}
.skip:focus { top: 10px; }
.wrap { max-width: 1220px; margin: auto; padding: 0 36px; }
.mast {
    display: flex; justify-content: space-between; gap: 14px; padding: 24px 0;
    border-bottom: 2px solid var(--accent); font-size: 13px; letter-spacing: .08em;
}
.brand { font-weight: 800; color: var(--accent); }
.hero {
    display: grid; grid-template-columns: 1.7fr 1fr; gap: 64px; align-items: center;
    padding: 46px 0 30px;
}
.eyebrow {
    font-size: 13px; color: var(--accent); font-weight: 700; letter-spacing: .1em;
}
h1 { font: 700 clamp(30px, 4vw, 46px)/1.35 "SimSun", "Songti SC", serif; margin: 12px 0 20px; }
h1 span {
    display: block; font-size: .62em; font-weight: 400; margin-top: 8px; color: var(--muted);
}
.lead { color: var(--muted); max-width: 650px; margin: 0; }
.hero-stat { border-left: 4px solid var(--accent); padding: 16px 0 16px 26px; }
.hero-stat strong {
    font: 600 66px/1.15 "Consolas", monospace; letter-spacing: -.07em; color: var(--accent);
}
.hero-stat p { margin: 4px 0; font-size: 14px; }
.hero-stat .note { max-width: 300px; }
.kpis {
    display: grid; grid-template-columns: repeat(4, 1fr); background: var(--paper);
    border: 1px solid var(--line); margin: 8px 0 24px;
}
.kpi { padding: 20px 22px; border-right: 1px solid var(--line); }
.kpi:last-child { border: 0; }
.kpi span { display: block; font-size: 13px; color: var(--muted); }
.kpi b { display: block; font: 600 27px/1.6 "Consolas", monospace; font-variant-numeric: tabular-nums; }
.kpi small { font-size: 12px; color: var(--muted); }
nav {
    display: flex; gap: 26px; flex-wrap: wrap; border-bottom: 1px solid var(--line);
    padding: 6px 0 18px; margin-bottom: 24px; font-size: 14px;
}
nav a { text-decoration: none; font-weight: 700; }
section { padding: 26px 0; border-bottom: 1px solid var(--line); scroll-margin-top: 20px; }
h2 { font-size: 24px; line-height: 1.4; margin: 0 0 8px; }
h3 { font-size: 18px; margin: 8px 0 12px; }
.section-note { color: var(--muted); margin: 0 0 20px; }
.finding {
    background: #e9eef2; border-left: 4px solid var(--accent); padding: 18px 24px; margin: 18px 0;
}
.finding p { margin: 0; }
.two {
    display: grid; grid-template-columns: 1fr 1fr; gap: 22px; align-items: start; min-width: 0;
}
.two>* { min-width: 0; }
figure {
    margin: 0; background: var(--paper); border: 1px solid var(--line); padding: 20px; min-width: 0;
}
figcaption { font-weight: 700; font-size: 17px; margin-bottom: 14px; }
figure img { display: block; width: 100%; height: auto; }
.note {
    font-size: 13px; line-height: 1.65; color: var(--muted); margin: 12px 0 0;
}
.table-scroll {
    overflow-x: auto; max-width: 100%; background: var(--paper); border: 1px solid var(--line);
    margin: 16px 0;
}
table {
    border-collapse: collapse; width: 100%; font-size: 14px; font-variant-numeric: tabular-nums;
}
th, td {
    padding: 12px 16px; text-align: right; white-space: nowrap; border-bottom: 1px solid #e3e5e7;
}
th { background: var(--accent); color: white; font-weight: 600; }
th:first-child, td:first-child { text-align: left; }
tbody tr:nth-child(even) { background: #f0f3f4; }
tbody tr:last-child td { border-bottom: 0; }
.methods { background: var(--paper); padding: 22px; border: 1px solid var(--line); }
.methods dl { margin: 0; }
.methods dt { font-weight: 700; margin-top: 16px; }
.methods dt:first-child { margin-top: 0; }
.methods dd { margin: 3px 0 0; color: var(--muted); }
details {
    margin: 14px 0; padding: 14px 18px; background: var(--paper); border: 1px solid var(--line);
}
summary { cursor: pointer; font-weight: 700; min-height: 28px; }
ul { padding-left: 22px; }
li { margin: 8px 0; }
.downloads {
    display: flex; flex-wrap: wrap; gap: 10px 22px; margin: 16px 0;
}
.downloads a { font-size: 14px; }
footer { padding: 28px 0 38px; font-size: 12px; color: var(--muted); }

@media (max-width: 760px) {
    .wrap { padding: 0 18px; }
    .mast { font-size: 11px; flex-direction: column; gap: 4px; }
    .hero { grid-template-columns: 1fr; gap: 20px; padding: 28px 0; }
    .hero-stat { padding: 10px 0 10px 20px; }
    .hero-stat strong { font-size: 50px; }
    .kpis { grid-template-columns: 1fr 1fr; }
    .kpi { padding: 14px; border-bottom: 1px solid var(--line); }
    .kpi:nth-child(2) { border-right: 0; }
    .kpi b { font-size: 23px; }
    .two { grid-template-columns: 1fr; }
    nav { gap: 14px 20px; }
    section { padding: 24px 0; }
    figure { padding: 14px; }
    h2 { font-size: 22px; }
    th, td { padding: 10px 12px; }
}

@media (prefers-reduced-motion: reduce) {
    html { scroll-behavior: auto; }
}

@media print {
    body { background: white; }
    .wrap { max-width: none; }
    nav, .downloads { display: none; }
    figure, .kpis { break-inside: avoid; }
}
"""


def write_report(title, subject, intro, hero_value, hero_label, hero_note, kpis, nav, body):
    """组装页面并保存报告。"""
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M')
    cards = '\n'.join(
        dedent(f"""\
            <div class="kpi">
                <span>{escape(label)}</span>
                <b>{escape(value)}</b>
                <small>{escape(note)}</small>
            </div>
            """).strip()
        for label, value, note in kpis
    )
    links = '\n'.join(f'<a href="#{key}">{escape(label)}</a>' for key, label in nav)
    html = dedent(f"""\
        <!doctype html>
        <html lang="zh-CN">
            <head>
                <meta charset="utf-8"/>
                <meta name="viewport" content="width=device-width,initial-scale=1">
                <meta
                    http-equiv="Content-Security-Policy"
                    content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"
                >
                <title>{escape(title)}</title>
                <style>
                    {indent(CSS.replace('ACCENT', ACCENT), " " * 20).lstrip()}
                </style>
            </head>
            <body>
                <a class="skip" href="#content">跳到研究结果</a>
                <div class="wrap">
                    <header>
                        <div class="mast">
                            <span class="brand">{escape(subject)} · 研究结果</span>
                            <span>UCI公开数据 / 离线可视化报告</span>
                        </div>
                        <div class="hero">
                            <div>
                                <div class="eyebrow">{escape(title)}</div>
                                <h1>{title.split('：')[0]}<span>{escape(intro)}</span></h1>
                                <p class="lead">查看交易清洗结果、客户分群和复购预测表现。完整结果表见页末。</p>
                            </div>
                            <aside class="hero-stat">
                                <p>{escape(hero_label)}</p>
                                <strong>{escape(hero_value)}</strong>
                                <p class="note">{escape(hero_note)}</p>
                            </aside>
                        </div>
                        <div class="kpis">
                            {indent(cards, " " * 28).lstrip()}
                        </div>
                        <nav aria-label="报告章节">
                            {indent(links, " " * 28).lstrip()}
                        </nav>
                    </header>
                    <main id="content">
                        {indent(body, " " * 24).lstrip()}
                    </main>
                    <footer>数据来源：UCI Machine Learning Repository · 报告生成时间：{stamp}（北京时间）<br/>页面图像已内嵌，可单独离线打开；CSV下载需保留同目录的结果文件。</footer>
                </div>
            </body>
        </html>
        """)
    target = OUT / '结果报告.html'
    target.write_text(html, encoding='utf-8')
    return target


def build_report():
    """核对客户数、金额和模型指标，绘图并生成报告。"""
    configure_plots()
    s = read_table('summary.csv', ['raw_rows', 'clean_rows', 'customers', 'positive_merchandise_amount_clean', 'test_cutoff', 'test_end', 'data_end', 'sql_python_consistent']).iloc[0]
    metrics = read_table(
        'repurchase_metrics.csv',
        ['model', 'roc_auc', 'pr_auc', 'test_n', 'test_positive_rate'],
    )
    segments = read_table(
        'rfm_segments.csv',
        ['RFM_segment', 'customers', 'recency_mean', 'frequency_mean', 'monetary_mean'],
    )
    clusters = read_table(
        'kmeans_segments.csv',
        ['cluster', 'customers', 'recency_mean', 'frequency_mean', 'monetary_mean'],
    )
    month = read_table(
        'monthly_sales.csv',
        ['month', 'positive_merchandise_amount', 'invoices', 'customers'],
    )
    cleaning = read_table('cleaning_reconciliation.csv', ['step', 'rows', 'amount'])
    thresholds = read_table(
        'repurchase_thresholds.csv',
        ['model', 'threshold', 'selected_rate', 'precision', 'recall', 'f1'],
    )
    k_scores = read_table('cluster_k_evaluation.csv', ['k', 'silhouette'])
    if set(metrics.model) != set(MODEL_NAMES) or len(metrics) != 2:
        raise ValueError('复购结果必须包含两个唯一模型。')
    if not (metrics.test_n.eq(s.test_n).all() and metrics.test_positive_rate.sub(s.test_repeat_rate).abs().lt(1e-12).all()):
        raise ValueError('两个模型的测试人数或事件率不一致。')
    for col in ['roc_auc', 'pr_auc', 'precision_0_5', 'recall_0_5', 'f1_0_5']:
        if not np.isfinite(metrics[col]).all() or not metrics[col].between(0, 1).all():
            raise ValueError(f'模型指标{col}存在缺失或超出0—1范围。')
    if not (segments.customers.sum() == clusters.customers.sum() == s.customers):
        raise ValueError('规则分层、聚类与摘要中的客户总数不一致。')
    amount = s.positive_merchandise_amount_clean
    if not (np.isclose(month.positive_merchandise_amount.sum(), amount, atol=.01, rtol=0)
            and np.isclose((segments.customers * segments.monetary_mean).sum(), amount, atol=.01, rtol=0)
            and cleaning.iloc[-1]['rows'] == s.clean_rows):
        raise ValueError('月度、客户金额或清洗行数对账失败。')
    if pd.Timestamp(s.test_end) > pd.Timestamp(s.data_end) or s.sql_python_consistent != True:
        raise ValueError('标签窗口超过观测终点，或SQL与Python汇总未通过核对。')

    # 末月数据不完整，用颜色和日期标注。
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(month))
    amounts = month.positive_merchandise_amount / 10000
    ax.plot(x, amounts, marker='o', color=ACCENT)
    ax.fill_between(x, amounts, alpha=.07, color=ACCENT)
    ax.scatter(x[-1], amounts.iloc[-1], color='#c2410c', s=60, zorder=3)
    ax.annotate(
        f'末月截至{pd.Timestamp(s.data_end).day}日',
        (x[-1], amounts.iloc[-1]),
        xytext=(-10, 24),
        textcoords='offset points',
        ha='right',
        color='#c2410c',
    )
    ax.set_xticks(x, month.month, rotation=45, ha='right')
    ax.set_ylabel('可识别客户正向商品金额（万GBP）')
    ax.set_title('月度商品金额趋势')
    fig.tight_layout()
    fig.savefig(OUT / 'monthly_sales.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ordered = segments.sort_values('customers')
    bars = ax.barh(ordered.RFM_segment, ordered.customers, color=ACCENT, height=.6)
    ax.bar_label(
        bars,
        labels=[f'{v:,}人（{v/s.customers:.1%}）' for v in ordered.customers],
        padding=6,
        fontsize=10,
    )
    ax.set_xlim(0, ordered.customers.max() * 1.45)
    ax.set_xlabel('客户人数')
    ax.set_title('RFM规则分层的客户规模')
    fig.tight_layout()
    fig.savefig(OUT / 'rfm_segments.png', dpi=180)
    plt.close(fig)

    display = metrics.copy()
    display['model'] = display.model.map(MODEL_NAMES)
    comparison = table_html(
        display,
        {
            'model': '复购模型',
            'roc_auc': '测试AUC ↑',
            'pr_auc': '平均精度AP ↑',
            'precision_0_5': '查准率',
            'recall_0_5': '召回率',
            'f1_0_5': 'F1',
        },
    )
    images = figures_html(
        figure_html('monthly_sales.png', '月度商品金额', '统计有客户编号的正向商品交易，末月仅有部分日期的数据。'),
        figure_html('rfm_segments.png', '各客户群的人数', '按历史RFM分位数分层，群体名称描述历史购买行为，未来价值和营销效果尚待验证。'),
    )
    profiles = figures_html(
        figure_html('rfm_profile.png', 'RFM规则分层画像', '颜色表示四个规则群体的均值经标准化后的数值；R越小，最近一次购买越近，F和M越大，历史购买越多。'),
        figure_html('cluster_k_evaluation.png', '不同聚类数的轮廓系数', '橙色点标出画像采用的四簇，轮廓系数最高的聚类数见下表。'),
    )
    curves = figures_html(
        figure_html('repurchase_roc.png', '复购预测ROC曲线', '两个模型在相同的后续时间窗口、同一批客户上测试。'),
        figure_html('repurchase_pr.png', '复购预测的查准率与召回率', '水平线为测试集复购率；AP按平均精度计算，与梯形积分的PR曲线面积有区别。'),
    )
    segment_table = table_html(
        segments,
        {
            'RFM_segment': '客户群',
            'customers': '客户数',
            'recency_mean': 'R均值（天）',
            'frequency_mean': 'F均值（发票）',
            'monetary_mean': 'M均值（GBP）',
        },
    )
    cluster_table = table_html(
        clusters,
        {
            'cluster': '算法簇',
            'customers': '客户数',
            'recency_mean': 'R均值（天）',
            'frequency_mean': 'F均值（发票）',
            'monetary_mean': 'M均值（GBP）',
        },
    )
    clean_display = cleaning.copy()
    clean_display['step'] = clean_display.step.replace(
        {
            'raw': '原始记录',
            'after_excluding_missing_customer': '排除缺失客户编号后',
            'after_excluding_cancelled_invoice': '再排除取消发票后',
            'after_excluding_nonpositive_quantity': '再排除非正数量后',
            'after_excluding_nonpositive_price': '再排除非正单价后',
        },
    )
    clean_table = table_html(clean_display, {'step': '清洗步骤', 'rows': '保留行数', 'amount': '带符号商品金额（GBP）'})
    th = thresholds[thresholds.model.eq('logistic')].copy()
    th['selected_n'] = np.rint(th.selected_rate * s.test_n).astype(int)
    th_table = table_html(
        th,
        {
            'threshold': '预测阈值',
            'selected_n': '选中客户数',
            'selected_rate': '选中比例',
            'precision': '查准率',
            'recall': '召回率',
            'f1': 'F1',
        },
    )
    logistic = metrics.set_index('model').loc['logistic']
    best_k = int(k_scores.loc[k_scores.silhouette.idxmax(), 'k'])
    body = dedent(f"""\
        <section id="transactions">
            <h2>交易清洗与客户覆盖</h2>
            <p class="section-note">{escape(str(s.data_start))} 至 {escape(str(s.data_end))}；金额以GBP计。</p>
            <div class="finding">
                <p>原始 <b>{int(s.raw_rows):,}</b> 行交易中，客户分析保留 <b>{int(s.clean_rows):,}</b> 行（{s.clean_rows/s.raw_rows:.2%}）。可识别客户的正向商品金额为 <b>£{amount:,.2f}</b>；包含匿名客户的原始正向商品金额为 <b>£{s.positive_merchandise_amount_all:,.2f}</b>，两者覆盖范围不同。</p>
            </div>
            {indent(images, " " * 12).lstrip()}
            <details>
                <summary>查看逐步清洗的行数和金额</summary>
                {indent(clean_table, " " * 16).lstrip()}
                <p class="note">表中是每一步筛选后的剩余记录，各规则单独命中的行数可能重叠。最终客户样本仅含正向商品交易，未做退货与原发票的逐笔冲销。</p>
            </details>
        </section>
        <section id="segments">
            <h2>客户分群与画像</h2>
            <p class="section-note">RFM规则按分位数分层，K-Means按交易特征的距离聚类。数据没有已知客户类别，无法将其中一种分群当作另一种的正确答案。</p>
            {indent(profiles, " " * 12).lstrip()}
            {indent(segment_table, " " * 12).lstrip()}
            <details>
                <summary>查看四个聚类的RFM均值</summary>
                {indent(cluster_table, " " * 16).lstrip()}
                <p class="note">在k=2—8中，k={best_k}的轮廓系数最高。此处另用四簇划分客户，四簇轮廓系数为{s.silhouette_k4:.4f}。</p>
            </details>
        </section>
        <section id="prediction">
            <h2>用历史RFM预测未来三个月复购</h2>
            <p class="section-note">下表查准率、召回率和F1均对应0.5阈值。</p>
            {indent(comparison, " " * 12).lstrip()}
            {indent(curves, " " * 12).lstrip()}
            <details>
                <summary>查看Logistic模型在不同阈值下的客户名单规模</summary>
                {indent(th_table, " " * 16).lstrip()}
                <p class="note">降低阈值会扩大名单，通常也会提高召回率。名单中的客户是否会因营销而增加购买，需要随机营销实验验证。</p>
            </details>
        </section>
        <section id="method">
            <h2>时间窗口与分析局限</h2>
            <div class="two">
                <div class="methods">
                    <dl>
                        <dt>训练标签窗口</dt>
                        <dd>{s.train_cutoff} ≤ 交易日期 &lt; {s.train_end}；{int(s.train_n):,}名历史客户。</dd>
                        <dt>测试标签窗口</dt>
                        <dd>{s.test_cutoff} ≤ 交易日期 &lt; {s.test_end}；{int(s.test_n):,}名历史客户。</dd>
                        <dt>特征计算</dt>
                        <dd>RFM只使用截止日前的交易；标准化和缺失填补参数在训练集估计，测试集沿用。测试标签窗口早于数据终点，覆盖完整。</dd>
                    </dl>
                </div>
                <div class="methods">
                    <h3>结果核对与局限</h3>
                    <ul>
                        <li>分析脚本已逐项核对SQL与Python的月度汇总及客户RFM；本页面检查已保存结果的一致性。</li>
                        <li>分群描述历史购买行为，客户终身价值尚未估计。</li>
                        <li>只评估了一次时间划分，未估计模型性能差异的置信区间，也未进行概率校准或营销实验。</li>
                    </ul>
                </div>
            </div>
        </section>
        """)
    files = [
        ('cleaning_reconciliation.csv', '清洗对账'),
        ('monthly_sales.csv', '月度汇总'),
        ('rfm_segments.csv', '规则群体画像'),
        ('kmeans_segments.csv', '算法簇画像'),
        ('cluster_k_evaluation.csv', '簇数评价'),
        ('repurchase_metrics.csv', '复购模型评价'),
        ('repurchase_thresholds.csv', '阈值结果'),
        ('summary.csv', '数据与窗口摘要'),
    ]
    downloads = '\n'.join(f'<a href="{name}" download>{label}</a>' for name, label in files)
    body += dedent(f"""\
        <section id="downloads">
            <h2>结果明细</h2>
            <div class="downloads">
                {indent(downloads, " " * 16).lstrip()}
            </div>
            <a href="https://archive.ics.uci.edu/dataset/352/online+retail">UCI数据来源</a>
        </section>
        """)
    return write_report(
        '电商客户分群与复购：交易数据研究',
        '案例实务',
        'RFM客户分群与未来三个月复购预测',
        f'{logistic.roc_auc:.4f}',
        'Logistic复购模型 · 时间测试AUC',
        f'后续窗口测试{int(s.test_n):,}名客户，复购率{s.test_repeat_rate:.2%}。',
        [
            ('有效交易', f'{int(s.clean_rows):,}', '可识别客户的正向交易行'),
            ('分析客户', f'{int(s.customers):,}', '客户级RFM聚合'),
            ('正向商品金额', f'£{amount/1000000:.3f}M', '客户样本金额，非会计净收入'),
            ('测试复购率', f'{s.test_repeat_rate:.2%}', '后续三个月窗口'),
        ],
        [
            ('transactions', '交易与覆盖'),
            ('segments', '客户画像'),
            ('prediction', '复购预测'),
            ('method', '时间与方法'),
            ('downloads', '结果明细'),
        ],
        body,
    )


if __name__ == '__main__':
    print(f'结果核验通过：{build_report()}')


