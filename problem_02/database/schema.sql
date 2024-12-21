CREATE TABLE ingredients (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    quantity VARCHAR(50),
    unit VARCHAR(50),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE recipes (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    cuisine VARCHAR(50),
    taste VARCHAR(50),
    preparation_time INT,
    ingredients JSONB,
    steps TEXT,
    reviews FLOAT DEFAULT 0.0
);
