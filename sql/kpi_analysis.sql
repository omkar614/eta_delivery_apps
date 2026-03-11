-- KPI & Business Insight Queries for Delivery ETA Analysis


-- Overall KPI summary — the five core ETA metrics across all deliveries
SELECT
    COUNT(*)                                                          AS total_orders,
    ROUND(AVG(actual_duration_min)::NUMERIC, 2)                      AS avg_actual_min,
    ROUND(AVG(estimated_total_duration)::NUMERIC, 2)                 AS avg_estimated_min,

    -- Average absolute difference between actual and estimated time
    ROUND(AVG(ABS(eta_error_min))::NUMERIC, 2)                       AS mae_minutes,

    -- Percentage error relative to actual delivery time
    ROUND((AVG(ABS(eta_error_min) / NULLIF(actual_duration_min,0)) * 100)::NUMERIC, 2) AS mape_pct,

    -- Orders delivered within 10 minutes of the ETA
    ROUND((AVG(CASE WHEN is_on_time  THEN 1.0 ELSE 0.0 END) * 100)::NUMERIC, 2) AS on_time_rate_pct,

    -- Orders that arrived more than 15 minutes late
    ROUND((AVG(CASE WHEN is_eta_breach THEN 1.0 ELSE 0.0 END) * 100)::NUMERIC, 2) AS breach_rate_pct,

    -- On-time rate specifically during peak hours
    ROUND((AVG(CASE WHEN is_peak_hour AND is_on_time THEN 1.0
                    WHEN is_peak_hour THEN 0.0
                    ELSE NULL END) * 100)::NUMERIC, 2) AS peak_on_time_pct,

    ROUND((AVG(dasher_util_ratio) * 100)::NUMERIC, 2)               AS avg_dasher_util_pct,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY actual_duration_min)::NUMERIC, 2) AS median_delivery_min
FROM deliveries;


-- Break down the same KPIs by market to spot regional differences
SELECT
    market_id,
    COUNT(*)                                                          AS total_orders,
    ROUND(AVG(actual_duration_min)::NUMERIC, 2)                      AS avg_actual_min,
    ROUND(AVG(ABS(eta_error_min))::NUMERIC, 2)                       AS mae,
    ROUND((AVG(ABS(eta_error_min) / NULLIF(actual_duration_min,0)) * 100)::NUMERIC, 2) AS mape_pct,
    ROUND((AVG(CASE WHEN is_on_time  THEN 1.0 ELSE 0.0 END) * 100)::NUMERIC, 2) AS on_time_pct,
    ROUND((AVG(CASE WHEN is_eta_breach THEN 1.0 ELSE 0.0 END) * 100)::NUMERIC, 2) AS breach_pct,
    ROUND((AVG(dasher_util_ratio) * 100)::NUMERIC, 2)               AS avg_util_pct
FROM deliveries
GROUP BY market_id
ORDER BY total_orders DESC;


-- Compare performance during peak hours vs the rest of the day
SELECT
    CASE WHEN is_peak_hour THEN 'Peak Hour' ELSE 'Off-Peak' END      AS period,
    COUNT(*)                                                          AS total_orders,
    ROUND(AVG(actual_duration_min)::NUMERIC, 2)                      AS avg_actual_min,
    ROUND(AVG(ABS(eta_error_min))::NUMERIC, 2)                       AS mae,
    ROUND((AVG(CASE WHEN is_on_time  THEN 1.0 ELSE 0.0 END) * 100)::NUMERIC, 2) AS on_time_pct,
    ROUND((AVG(CASE WHEN is_eta_breach THEN 1.0 ELSE 0.0 END) * 100)::NUMERIC, 2) AS breach_pct,
    ROUND((AVG(dasher_util_ratio) * 100)::NUMERIC, 2)               AS avg_util_pct
FROM deliveries
GROUP BY is_peak_hour
ORDER BY is_peak_hour DESC;


-- Hourly breakdown to see how performance shifts across the full day
SELECT
    hour_of_day,
    COUNT(*)                                                          AS total_orders,
    ROUND(AVG(actual_duration_min)::NUMERIC, 2)                      AS avg_actual_min,
    ROUND(AVG(ABS(eta_error_min))::NUMERIC, 2)                       AS mae,
    ROUND((AVG(CASE WHEN is_on_time  THEN 1.0 ELSE 0.0 END) * 100)::NUMERIC, 2) AS on_time_pct,
    ROUND((AVG(dasher_util_ratio) * 100)::NUMERIC, 2)               AS avg_util_pct
FROM deliveries
GROUP BY hour_of_day
ORDER BY hour_of_day;


-- Performance by store category — limited to the top 15 by order volume
SELECT
    store_primary_category,
    COUNT(*)                                                          AS total_orders,
    ROUND(AVG(actual_duration_min)::NUMERIC, 2)                      AS avg_delivery_min,
    ROUND(AVG(ABS(eta_error_min))::NUMERIC, 2)                       AS mae,
    ROUND((AVG(CASE WHEN is_on_time  THEN 1.0 ELSE 0.0 END) * 100)::NUMERIC, 2) AS on_time_pct,
    ROUND(AVG(subtotal / 100.0)::NUMERIC, 2)                        AS avg_order_value
FROM deliveries
GROUP BY store_primary_category
ORDER BY total_orders DESC
LIMIT 15;


-- How dasher utilization affects ETA breach rate — identifies congestion thresholds
SELECT
    CASE
        WHEN dasher_util_ratio < 0.25 THEN '0–25%  (Low)'
        WHEN dasher_util_ratio < 0.50 THEN '25–50% (Moderate)'
        WHEN dasher_util_ratio < 0.75 THEN '50–75% (High)'
        WHEN dasher_util_ratio < 1.00 THEN '75–100% (Critical)'
        ELSE                               '100%+  (Overloaded)'
    END AS util_bucket,
    COUNT(*)                                                          AS total_orders,
    ROUND(AVG(ABS(eta_error_min))::NUMERIC, 2)                       AS mae,
    ROUND((AVG(CASE WHEN is_eta_breach THEN 1.0 ELSE 0.0 END) * 100)::NUMERIC, 2) AS breach_pct,
    ROUND(AVG(actual_duration_min)::NUMERIC, 2)                      AS avg_actual_min
FROM deliveries
GROUP BY util_bucket
ORDER BY util_bucket;


-- Day-of-week breakdown to catch any weekly patterns in delivery performance
SELECT
    day_of_week,
    COUNT(*)                                                          AS total_orders,
    ROUND(AVG(actual_duration_min)::NUMERIC, 2)                      AS avg_actual_min,
    ROUND(AVG(ABS(eta_error_min))::NUMERIC, 2)                       AS mae,
    ROUND((AVG(CASE WHEN is_on_time  THEN 1.0 ELSE 0.0 END) * 100)::NUMERIC, 2) AS on_time_pct
FROM deliveries
GROUP BY day_of_week
ORDER BY
    CASE day_of_week
        WHEN 'Monday'    THEN 1 WHEN 'Tuesday'   THEN 2
        WHEN 'Wednesday' THEN 3 WHEN 'Thursday'  THEN 4
        WHEN 'Friday'    THEN 5 WHEN 'Saturday'  THEN 6
        ELSE 7
    END;


-- At what outstanding order volume does ETA accuracy start to degrade?
SELECT
    CASE
        WHEN total_outstanding_orders < 5   THEN '0–4   (Low Load)'
        WHEN total_outstanding_orders < 10  THEN '5–9   (Medium)'
        WHEN total_outstanding_orders < 20  THEN '10–19 (High)'
        WHEN total_outstanding_orders < 30  THEN '20–29 (Very High)'
        ELSE                                     '30+   (Overloaded)'
    END AS load_bucket,
    COUNT(*)                                                          AS total_orders,
    ROUND(AVG(actual_duration_min)::NUMERIC, 2)                      AS avg_actual_min,
    ROUND(AVG(ABS(eta_error_min))::NUMERIC, 2)                       AS mae,
    ROUND((AVG(CASE WHEN is_eta_breach THEN 1.0 ELSE 0.0 END) * 100)::NUMERIC, 2) AS breach_pct
FROM deliveries
GROUP BY load_bucket
ORDER BY load_bucket;


-- Pre vs post comparison — splits the dataset at the midpoint as a simple A/B proxy
WITH ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (ORDER BY created_at) AS rn,
           COUNT(*) OVER ()                         AS total
    FROM deliveries
),
halves AS (
    SELECT *,
           CASE WHEN rn <= total / 2 THEN 'Pre-Optimization'
                ELSE 'Post-Optimization' END AS phase
    FROM ranked
)
SELECT
    phase,
    COUNT(*)                                                          AS total_orders,
    ROUND(AVG(actual_duration_min)::NUMERIC, 2)                      AS avg_actual_min,
    ROUND(AVG(ABS(eta_error_min))::NUMERIC, 2)                       AS mae,
    ROUND((AVG(ABS(eta_error_min) / NULLIF(actual_duration_min,0)) * 100)::NUMERIC, 2) AS mape_pct,
    ROUND((AVG(CASE WHEN is_on_time AND is_peak_hour THEN 1.0
                    WHEN is_peak_hour THEN 0.0
                    ELSE NULL END) * 100)::NUMERIC, 2)               AS peak_on_time_pct
FROM halves
GROUP BY phase
ORDER BY phase;


-- Top 10 busiest stores and how well they're hitting ETA targets
SELECT
    store_id,
    store_primary_category,
    COUNT(*)                                                          AS total_orders,
    ROUND(AVG(actual_duration_min)::NUMERIC, 2)                      AS avg_actual_min,
    ROUND(AVG(ABS(eta_error_min))::NUMERIC, 2)                       AS mae,
    ROUND((AVG(CASE WHEN is_on_time THEN 1.0 ELSE 0.0 END) * 100)::NUMERIC, 2) AS on_time_pct
FROM deliveries
GROUP BY store_id, store_primary_category
ORDER BY total_orders DESC
LIMIT 10;
