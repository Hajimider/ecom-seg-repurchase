-- run.py 将原始交易写入 transactions 表后，执行以下查询。
-- 仅统计有客户编号、非取消、数量与单价均为正且早于分析日的交易。
-- 金额为客户样本的正向商品金额，未做逐笔退货冲销。

-- 月度金额、发票数和客户数。同一客户或发票在当月只计一次。
SELECT strftime('%Y-%m', InvoiceDate) AS month,
       SUM(Quantity * UnitPrice) AS positive_merchandise_amount,
       COUNT(DISTINCT InvoiceNo) AS invoices,
       COUNT(DISTINCT CustomerID) AS customers
FROM transactions
WHERE CustomerID IS NOT NULL
  AND InvoiceNo NOT LIKE 'C%'
  AND Quantity > 0 AND UnitPrice > 0
  AND InvoiceDate < '2011-12-10'
GROUP BY strftime('%Y-%m', InvoiceDate)
ORDER BY month;

-- 客户 RFM：R 为最后购买距分析日的天数，F 为去重发票数，M 为商品金额合计。
SELECT CustomerID,
       CAST(julianday('2011-12-10') - julianday(MAX(InvoiceDate)) AS INTEGER) AS recency_days,
       COUNT(DISTINCT InvoiceNo) AS frequency,
       SUM(Quantity * UnitPrice) AS monetary
FROM transactions
WHERE CustomerID IS NOT NULL
  AND InvoiceNo NOT LIKE 'C%'
  AND Quantity > 0 AND UnitPrice > 0
  AND InvoiceDate < '2011-12-10'
GROUP BY CustomerID;
