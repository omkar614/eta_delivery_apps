

DROP TABLE IF EXISTS deliveries;


CREATE TABLE deliveries (
    id                                           SERIAL PRIMARY KEY,

   
    market_id                                    INT,
    store_id                                     INT          NOT NULL,
    store_primary_category                       VARCHAR(100) DEFAULT 'Unknown',
    order_protocol                               INT,

   
    created_at                                   TIMESTAMP    NOT NULL,
    actual_delivery_time                         TIMESTAMP,

    total_items                                  INT          DEFAULT 0,
    subtotal                                     INT          DEFAULT 0,
    num_distinct_items                           INT          DEFAULT 0,
    min_item_price                               INT          DEFAULT 0,
    max_item_price                               INT          DEFAULT 0,

   
    total_onshift_dashers                        FLOAT,
    total_busy_dashers                           FLOAT,
    total_outstanding_orders                     FLOAT,

   
    estimated_order_place_duration               INT,
    estimated_store_to_consumer_driving_duration FLOAT,

    
    actual_duration_min      FLOAT,   -- actual delivery minutes
    estimated_total_duration FLOAT,   -- estimated minutes (total)
    eta_error_min            FLOAT,   -- actual - estimated (+ = late)
    hour_of_day              INT,
    day_of_week              VARCHAR(15),
    is_weekend               BOOLEAN,
    is_peak_hour             BOOLEAN,
    dasher_util_ratio        FLOAT,   -- busy / onshift
    is_on_time               BOOLEAN, -- within ETA + 10 min buffer
    is_eta_breach            BOOLEAN  -- > 15 min late
);

CREATE INDEX idx_market_id   ON deliveries(market_id);
CREATE INDEX idx_store_id    ON deliveries(store_id);
CREATE INDEX idx_created_at  ON deliveries(created_at);
CREATE INDEX idx_hour        ON deliveries(hour_of_day);
CREATE INDEX idx_peak        ON deliveries(is_peak_hour);
CREATE INDEX idx_category    ON deliveries(store_primary_category);
