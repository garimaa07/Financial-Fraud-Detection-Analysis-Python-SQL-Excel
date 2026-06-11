-- =============================================================================
-- queries.sql
-- Financial Fraud Detection System — SQL Analysis Layer
-- Database: PostgreSQL  |  Table: transactions
-- =============================================================================
-- Schema reference:
--   transactions(
--     id           SERIAL PRIMARY KEY,
--     time_seconds FLOAT,          -- seconds since first transaction
--     v1..v28      FLOAT,          -- PCA-anonymised features
--     amount       FLOAT,          -- transaction amount (USD)
--     class        SMALLINT,       -- 0 = legitimate, 1 = fraud
--     hour_of_day  SMALLINT,       -- 0–23 (derived)
--     amount_scaled FLOAT          -- standardised amount
--   );
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 1. Class distribution — fraud vs. legitimate
-- -----------------------------------------------------------------------------
SELECT
    CASE class WHEN 1 THEN 'Fraud' ELSE 'Legitimate' END AS transaction_type,
    COUNT(*)                                              AS total_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 4)   AS percentage
FROM transactions
GROUP BY class
ORDER BY class DESC;


-- -----------------------------------------------------------------------------
-- 2. Top 20 highest-value fraudulent transactions
-- -----------------------------------------------------------------------------
SELECT
    id,
    ROUND(amount::numeric, 2)   AS amount_usd,
    hour_of_day,
    ROUND(amount_scaled::numeric, 4) AS amount_z_score
FROM transactions
WHERE class = 1
ORDER BY amount DESC
LIMIT 20;


-- -----------------------------------------------------------------------------
-- 3. Average transaction amount — fraud vs. legitimate
-- -----------------------------------------------------------------------------
SELECT
    CASE class WHEN 1 THEN 'Fraud' ELSE 'Legitimate' END AS transaction_type,
    ROUND(AVG(amount)::numeric, 2)    AS avg_amount,
    ROUND(MEDIAN(amount)::numeric, 2) AS median_amount,
    ROUND(MAX(amount)::numeric, 2)    AS max_amount,
    ROUND(MIN(amount)::numeric, 2)    AS min_amount,
    COUNT(*)                          AS count
FROM transactions
GROUP BY class;


-- -----------------------------------------------------------------------------
-- 4. Fraud count by hour of day (24-hour buckets)
-- -----------------------------------------------------------------------------
SELECT
    hour_of_day,
    COUNT(*)                                                    AS fraud_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)         AS pct_of_all_fraud,
    CASE
        WHEN hour_of_day BETWEEN 23 AND 23 OR hour_of_day BETWEEN 0 AND 4
        THEN 'High-Risk (Late Night)'
        WHEN hour_of_day BETWEEN 5 AND 8 OR hour_of_day BETWEEN 20 AND 22
        THEN 'Medium-Risk'
        ELSE 'Low-Risk (Business Hours)'
    END AS risk_band
FROM transactions
WHERE class = 1
GROUP BY hour_of_day
ORDER BY fraud_count DESC;


-- -----------------------------------------------------------------------------
-- 5. Fraud rate by hour bucket (fraud / total transactions in that hour)
-- -----------------------------------------------------------------------------
SELECT
    hour_of_day,
    COUNT(*)                                                AS total_transactions,
    SUM(class)                                              AS fraud_count,
    ROUND(SUM(class) * 100.0 / COUNT(*), 4)                AS fraud_rate_pct
FROM transactions
GROUP BY hour_of_day
ORDER BY fraud_rate_pct DESC;


-- -----------------------------------------------------------------------------
-- 6. High-value fraud transactions (amount > 95th percentile of all fraud)
-- -----------------------------------------------------------------------------
WITH fraud_percentiles AS (
    SELECT PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount) AS p95
    FROM transactions
    WHERE class = 1
)
SELECT
    t.id,
    ROUND(t.amount::numeric, 2)  AS amount_usd,
    t.hour_of_day,
    t.amount_scaled
FROM transactions t
JOIN fraud_percentiles fp ON t.amount > fp.p95
WHERE t.class = 1
ORDER BY t.amount DESC;


-- -----------------------------------------------------------------------------
-- 7. Transaction volume by 6-hour time blocks
-- -----------------------------------------------------------------------------
SELECT
    CASE
        WHEN hour_of_day BETWEEN 0  AND 5  THEN '00:00–05:59 (Night)'
        WHEN hour_of_day BETWEEN 6  AND 11 THEN '06:00–11:59 (Morning)'
        WHEN hour_of_day BETWEEN 12 AND 17 THEN '12:00–17:59 (Afternoon)'
        ELSE                                     '18:00–23:59 (Evening)'
    END                          AS time_block,
    COUNT(*)                     AS total_txns,
    SUM(class)                   AS fraud_count,
    ROUND(SUM(class) * 100.0 / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(AVG(amount)::numeric, 2)           AS avg_amount
FROM transactions
GROUP BY time_block
ORDER BY fraud_rate_pct DESC;


-- -----------------------------------------------------------------------------
-- 8. Cumulative fraud detection coverage at different amount thresholds
--    (Useful for setting risk-based alert thresholds)
-- -----------------------------------------------------------------------------
WITH thresholds AS (
    SELECT unnest(ARRAY[50, 100, 200, 500, 1000, 2000, 5000]) AS amount_threshold
),
fraud_total AS (
    SELECT COUNT(*) AS total_fraud FROM transactions WHERE class = 1
)
SELECT
    t.amount_threshold,
    COUNT(tr.id)                                          AS fraud_above_threshold,
    ft.total_fraud,
    ROUND(COUNT(tr.id) * 100.0 / ft.total_fraud, 2)      AS pct_fraud_captured
FROM thresholds t
CROSS JOIN fraud_total ft
LEFT JOIN transactions tr
    ON tr.class = 1 AND tr.amount >= t.amount_threshold
GROUP BY t.amount_threshold, ft.total_fraud
ORDER BY t.amount_threshold;


-- -----------------------------------------------------------------------------
-- 9. Rolling 10-transaction window — fraud streak detection
--    (Flags consecutive fraud bursts, useful for alert fatigue analysis)
-- -----------------------------------------------------------------------------
WITH ranked AS (
    SELECT
        id,
        time_seconds,
        class,
        SUM(class) OVER (ORDER BY time_seconds ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)
            AS fraud_in_last_10
    FROM transactions
)
SELECT *
FROM ranked
WHERE fraud_in_last_10 >= 3      -- 3+ frauds in last 10 transactions
ORDER BY time_seconds;


-- -----------------------------------------------------------------------------
-- 10. Export: flagged transactions for Excel audit dashboard
--     (Run this and export result to reports/flagged_for_audit.csv)
-- -----------------------------------------------------------------------------
SELECT
    id                                      AS transaction_id,
    ROUND(amount::numeric, 2)              AS amount_usd,
    hour_of_day,
    amount_scaled                           AS z_score,
    CASE
        WHEN amount > 500  AND hour_of_day BETWEEN 0 AND 4 THEN 'CRITICAL'
        WHEN amount > 200                                   THEN 'HIGH'
        WHEN hour_of_day BETWEEN 0 AND 4                    THEN 'MEDIUM'
        ELSE                                                     'LOW'
    END                                     AS risk_level,
    class                                   AS is_fraud
FROM transactions
WHERE class = 1
   OR (amount > 500 AND hour_of_day BETWEEN 0 AND 4)   -- High-value late-night
ORDER BY amount DESC, hour_of_day;
