-- Data Cleaning Queries for Delivery ETA Analysis
-- Run these after loading raw CSV data into the deliveries table
-- Mirrors the Python cleaning steps for consistency


-- Check how many rows we're starting with
SELECT COUNT(*) AS raw_row_count
FROM deliveries;


-- Count missing values in each column to understand data quality
SELECT
    'market_id'              AS column_name, COUNT(*) - COUNT(market_id)              AS missing_count FROM deliveries
UNION ALL SELECT 'actual_delivery_time',     COUNT(*) - COUNT(actual_delivery_time)     FROM deliveries
UNION ALL SELECT 'store_primary_category',   COUNT(*) - COUNT(NULLIF(store_primary_category,'NA')) FROM deliveries
UNION ALL SELECT 'order_protocol',           COUNT(*) - COUNT(order_protocol)           FROM deliveries
UNION ALL SELECT 'total_onshift_dashers',    COUNT(*) - COUNT(total_onshift_dashers)    FROM deliveries
UNION ALL SELECT 'total_busy_dashers',       COUNT(*) - COUNT(total_busy_dashers)       FROM deliveries
UNION ALL SELECT 'total_outstanding_orders', COUNT(*) - COUNT(total_outstanding_orders) FROM deliveries
UNION ALL SELECT 'estimated_store_to_consumer_driving_duration',
                 COUNT(*) - COUNT(estimated_store_to_consumer_driving_duration)         FROM deliveries;


-- Standardize missing category values to 'Unknown'
UPDATE deliveries
SET store_primary_category = 'Unknown'
WHERE store_primary_category = 'NA'
   OR store_primary_category IS NULL;


-- Fill missing market IDs with the median value
UPDATE deliveries
SET market_id = (
    SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY market_id)
    FROM deliveries
    WHERE market_id IS NOT NULL
)
WHERE market_id IS NULL;


-- Fill missing dasher availability metrics with median values
UPDATE deliveries
SET total_onshift_dashers = (
    SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_onshift_dashers)
    FROM deliveries WHERE total_onshift_dashers IS NOT NULL
)
WHERE total_onshift_dashers IS NULL;

UPDATE deliveries
SET total_busy_dashers = (
    SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_busy_dashers)
    FROM deliveries WHERE total_busy_dashers IS NOT NULL
)
WHERE total_busy_dashers IS NULL;

UPDATE deliveries
SET total_outstanding_orders = (
    SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_outstanding_orders)
    FROM deliveries WHERE total_outstanding_orders IS NOT NULL
)
WHERE total_outstanding_orders IS NULL;


-- Calculate actual delivery duration in minutes
UPDATE deliveries
SET actual_duration_min = EXTRACT(EPOCH FROM (actual_delivery_time - created_at)) / 60
WHERE actual_delivery_time IS NOT NULL
  AND created_at IS NOT NULL;


-- Calculate total estimated duration in minutes (converted from seconds)
UPDATE deliveries
SET estimated_total_duration = (
    COALESCE(estimated_order_place_duration, 0) +
    COALESCE(estimated_store_to_consumer_driving_duration, 0)
) / 60.0;


-- Remove unrealistic delivery times (less than 1 min or more than 5 hours)
DELETE FROM deliveries
WHERE actual_duration_min IS NULL
   OR actual_duration_min <= 0
   OR actual_duration_min > 300;


-- Calculate ETA error, time-based features, and delivery performance flags
UPDATE deliveries
SET
    eta_error_min     = actual_duration_min - estimated_total_duration,
    hour_of_day       = EXTRACT(HOUR FROM created_at),
    day_of_week       = TO_CHAR(created_at, 'Day'),
    is_weekend        = EXTRACT(DOW FROM created_at) IN (0, 6),
    is_peak_hour      = EXTRACT(HOUR FROM created_at) IN (11,12,13,14,18,19,20,21),
    dasher_util_ratio = CASE
                            WHEN total_onshift_dashers > 0
                            THEN total_busy_dashers / total_onshift_dashers
                            ELSE 0
                        END,
    is_on_time        = (actual_duration_min - estimated_total_duration) <= 10,
    is_eta_breach     = (actual_duration_min - estimated_total_duration) > 15;


-- Validate the cleaned data and check for any remaining missing values
SELECT
    COUNT(*)                          AS clean_rows,
    SUM(CASE WHEN eta_error_min IS NULL THEN 1 ELSE 0 END) AS nulls_remaining,
    ROUND(AVG(actual_duration_min)::NUMERIC, 2)    AS avg_actual_min,
    ROUND(AVG(estimated_total_duration)::NUMERIC, 2) AS avg_est_min
FROM deliveries;
