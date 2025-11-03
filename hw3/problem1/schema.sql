DROP TABLE IF EXISTS stop_events;
DROP TABLE IF EXISTS trips;
DROP TABLE IF EXISTS line_stops;
DROP TABLE IF EXISTS stops;
DROP TABLE IF EXISTS lines;

CREATE TABLE lines (
    line_id SERIAL PRIMARY KEY,  -- or use line_name as natural key?
    line_name VARCHAR(50) NOT NULL UNIQUE,
    vehicle_type VARCHAR(10) CHECK (vehicle_type IN ('rail', 'bus'))
);

CREATE TABLE stops (
    stop_id SERIAL PRIMARY KEY,
    stop_name VARCHAR(255) NOT NULL,
    latitude DECIMAL(9, 6) NOT NULL,
    longitude DECIMAL(9, 6) NOT NULL,
    UNIQUE (latitude, longitude)
);

CREATE TABLE line_stops (
    line_id INT NOT NULL,
    stop_id INT NOT NULL,
    sequence_number INT NOT NULL,
    time_offset_from_start INTERVAL,
    PRIMARY KEY (line_id, sequence_number),
    FOREIGN KEY (line_id) REFERENCES lines(line_id),
    FOREIGN KEY (stop_id) REFERENCES stops(stop_id),
    CHECK (sequence_number > 0)
);

CREATE TABLE trips (
    trip_id SERIAL PRIMARY KEY,
    line_id INT NOT NULL,
    departure_time TIME NOT NULL,
    vehicle_id INT NOT NULL,
    FOREIGN KEY (line_id) REFERENCES lines(line_id)
);


CREATE TABLE stop_events (
    event_id SERIAL PRIMARY KEY,
    trip_id INT NOT NULL,
    stop_id INT NOT NULL,
    scheduled_time TIME,
    actual_time TIME,
    passengers_on INT NOT NULL,
    passengers_off INT NOT NULL,
    FOREIGN KEY (trip_id) REFERENCES trips(trip_id),
    FOREIGN KEY (stop_id) REFERENCES stops(stop_id),
    CHECK (passengers_on >= 0),
    CHECK (passengers_off >= 0)
);