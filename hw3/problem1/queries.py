import argparse
import psycopg2
import psycopg2.extras
import json
import sys
import decimal

# --- Query Definitions ---
# A dictionary to hold all our SQL queries and their descriptions.
QUERIES = {
    "Q1": {
        "description": "List all stops on Route 20 in order",
        "sql": """
            SELECT s.stop_name, ls.sequence_number AS sequence, ls.time_offset_from_start AS time_offset
            FROM line_stops ls
            JOIN lines l ON ls.line_id = l.line_id
            JOIN stops s ON ls.stop_id = s.stop_id
            WHERE l.line_name = 'Route 20'
            ORDER BY ls.sequence_number;
        """
    },
    "Q2": {
        "description": "Trips during morning rush (7-9 AM)",
        "sql": """
            SELECT t.trip_id, l.line_name, t.departure_time AS scheduled_departure
            FROM trips t
            JOIN lines l ON t.line_id = l.line_id
            WHERE t.departure_time >= '07:00:00' AND t.departure_time < '09:00:00'
            ORDER BY t.departure_time;
        """
    },
    "Q3": {
        "description": "Transfer stops (stops on 2+ routes)",
        "sql": """
            SELECT s.stop_name, COUNT(DISTINCT ls.line_id) AS line_count
            FROM stops s
            JOIN line_stops ls ON s.stop_id = ls.stop_id
            GROUP BY s.stop_id, s.stop_name
            HAVING COUNT(DISTINCT ls.line_id) >= 2
            ORDER BY line_count DESC, stop_name;
        """
    },
    "Q4": {
        "description": "Complete route for trip T0001",
        "sql": """
            SELECT ls.sequence_number, s.stop_name
            FROM trips t
            JOIN line_stops ls ON t.line_id = ls.line_id
            JOIN stops s ON ls.stop_id = s.stop_id
            WHERE t.trip_id = 1
            ORDER BY ls.sequence_number;
        """
    },
    "Q5": {
        "description": "Routes serving both Wilshire / Veteran and Le Conte / Broxton",
        "sql": """
            SELECT l.line_name
            FROM lines l
            JOIN line_stops ls1 ON l.line_id = ls1.line_id
            JOIN stops s1 ON ls1.stop_id = s1.stop_id
            JOIN line_stops ls2 ON l.line_id = ls2.line_id
            JOIN stops s2 ON ls2.stop_id = s2.stop_id
            WHERE s1.stop_name = 'Wilshire / Veteran' AND s2.stop_name = 'Le Conte / Broxton';
        """
    },
    "Q6": {
        "description": "Average ridership by line",
        "sql": """
            SELECT l.line_name, AVG(se.passengers_on) AS avg_passengers
            FROM lines l
            JOIN trips t ON l.line_id = t.line_id
            JOIN stop_events se ON t.trip_id = se.trip_id
            GROUP BY l.line_id, l.line_name
            ORDER BY avg_passengers DESC;
        """
    },
    "Q7": {
        "description": "Top 10 busiest stops",
        "sql": """
            SELECT s.stop_name, SUM(se.passengers_on + se.passengers_off) AS total_activity
            FROM stops s
            JOIN stop_events se ON s.stop_id = se.stop_id
            GROUP BY s.stop_id, s.stop_name
            ORDER BY total_activity DESC
            LIMIT 10;
        """
    },
    "Q8": {
        "description": "Count delays by line (>2 min late)",
        "sql": """
            SELECT l.line_name, COUNT(*) AS delay_count
            FROM stop_events se
            JOIN trips t ON se.trip_id = t.trip_id
            JOIN lines l ON t.line_id = l.line_id
            WHERE se.actual_time > se.scheduled_time + INTERVAL '2 minutes'
            GROUP BY l.line_id, l.line_name
            ORDER BY delay_count DESC;
        """
    },
    "Q9": {
        "description": "Trips with 3+ delayed stops",
        "sql": """
            SELECT trip_id, COUNT(*) AS delayed_stop_count
            FROM stop_events
            WHERE actual_time > scheduled_time + INTERVAL '2 minutes'
            GROUP BY trip_id
            HAVING COUNT(*) >= 3
            ORDER BY delayed_stop_count DESC;
        """
    },
    "Q10": {
        "description": "Stops with above-average ridership",
        "sql": """
            SELECT s.stop_name, SUM(se.passengers_on) AS total_boardings
            FROM stops s
            JOIN stop_events se ON s.stop_id = se.stop_id
            GROUP BY s.stop_id, s.stop_name
            HAVING SUM(se.passengers_on) > (
                SELECT AVG(total_boardings)
                FROM (
                    SELECT SUM(passengers_on) AS total_boardings
                    FROM stop_events
                    GROUP BY stop_id
                ) AS subquery
            )
            ORDER BY total_boardings DESC;
        """
    }
}

def execute_query(conn, query_name, format_type):
    """
    Executes a single SQL query and prints the result in the specified format.
    """
    query_info = QUERIES.get(query_name)
    if not query_info:
        print(f"Error: Query '{query_name}' not found.", file=sys.stderr)
        return

    try:
        # Use DictCursor to get results as dictionaries
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(query_info["sql"])
            results = cur.fetchall()
            
            # Convert each DictRow to a standard dict
            results_list = [dict(row) for row in results]

            # Convert special data types (time, interval) to strings for JSON serialization
            for row in results_list:
                for key, value in row.items():
                    if isinstance(value, decimal.Decimal):
                        row[key] = float(value)
                    if hasattr(value, 'isoformat'): # Checks for time, date, datetime objects
                        row[key] = value.isoformat()
                    elif hasattr(value, 'total_seconds'): # Check if it's a timedelta/interval
                        row[key] = str(value)

            output = {
                "query": query_name,
                "description": query_info["description"],
                "results": results_list,
                "count": len(results_list)
            }

            if format_type == 'json':
                print(json.dumps(output, indent=2))
            else:
                # Simple text format as a fallback
                print(f"--- {query_name}: {query_info['description']} ---")
                for row in results_list:
                    print(row)
                print(f"Count: {len(results_list)}\n")

    except psycopg2.Error as e:
        print(f"Database error while running {query_name}: {e}", file=sys.stderr)
        conn.rollback()

def main():
    """
    Main function to parse arguments and run queries.
    """
    parser = argparse.ArgumentParser(description="Run SQL queries against the transit database.")
    parser.add_argument('--host', default='db', help='Database host')
    parser.add_argument('--dbname', required=True, help='Database name')
    parser.add_argument('--user', default='transit', help='Database user')
    parser.add_argument('--password', default='transit123', help='Database password')
    parser.add_argument('--format', default='json', choices=['json', 'text'], help='Output format')
    
    # Group for choosing which queries to run
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument('--query', choices=list(QUERIES.keys()), help='Run a single query by its ID (e.g., Q1)')
    query_group.add_argument('--all', action='store_true', help='Run all available queries')

    args = parser.parse_args()

    conn = None
    try:
        conn = psycopg2.connect(
            host=args.host,
            dbname=args.dbname,
            user=args.user,
            password=args.password
        )

        if args.all:
            for query_name in QUERIES:
                execute_query(conn, query_name, args.format)
        else:
            execute_query(conn, args.query, args.format)

    except psycopg2.Error as e:
        print(f"Connection error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main()