# Homework 3 Analysis

## Part E: Analysis Questions

### Schema Decisions: Natural vs. Surrogate Keys

*   **Decision**: Surrogate keys
*   **Why**: They are simple integers that enable fast queries and joins. They are also more stable, as changes to names do not affect the relationships across the entire system.

### Constraints Added

*   **`UNIQUE` Constraints**:
    *   `stops(latitude, longitude)`: To prevent creating duplicate stops at the same physical location.

*   **`CHECK` Constraints**:
    *   `line_stops(sequence_number > 0)`: Preserve positive of the stop order.
    *   `stop_events(passengers_on >= 0, passengers_off >= 0)`: Impossible to be negative for the passenger counts.

### Most Complex Query

*   **Query**: Q10
*   **Why**: Because it performes a nested query within the HAVING clause, the entire algorithm became more complex compared to others.

### Foreign Keys and Data Integrity

The foreign key on `stop_events.trip_id` ensures each event is linked to a valid trip. If an attempt is made to insert an event that does not exist in the table (e.g., `trip_id` = 9999), the database will reject the operation.

### Why a Relational Database is a Good Fit

*   **Clear Relationships**: There is a clear relationship between entities, which can be perfectly modeled using foreign keys.
*   **Data Integrity**: SQL constraints (such as primary keys, foreign keys, and CHECK constraints) ensure data accuracy and consistency, which is critical for real-world transportation systems.
*   **Powerful Queries**: The various questions we need to answer—such as linking schedules to actual events and aggregating passenger data—can all be efficiently addressed using SQL.