import os, sys, csv, argparse
from pathlib import Path
from typing import Dict, Tuple
import psycopg2
from io import StringIO

def load_data(host, dbname, user, password, datadir):
    """
    Connects to PostgreSQL, creates schema, and loads data from CSV files.
    """
    conn = None
    total_rows_loaded = 0
    try:
        # --- 1. Connect to the database ---
        conn = psycopg2.connect(
            host=host,
            dbname=dbname,
            user=user,
            password=password
        )
        print(f"Connected to {dbname}@{host}")
        cur = conn.cursor()

        # --- 2. Create schema ---
        print("Creating schema...")
        schema_path = 'schema.sql'
        with open(schema_path, 'r') as f:
            cur.execute(f.read())
        
        # Get created table names for the report
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = [row[0] for row in cur.fetchall()]
        print(f"Tables created: {', '.join(tables)}\n")

        # --- 3. Load data in the correct order ---
        
        # Load lines.csv
        lines_path = os.path.join(datadir, 'lines.csv')
        print(f"Loading {lines_path}...")
        with open(lines_path, 'r') as f:
            # Skip header row
            next(f)
            # Use copy_from for efficient bulk loading
            cur.copy_from(f, 'lines', sep=',', columns=('line_name', 'vehicle_type'))
            print(f"{cur.rowcount} rows")
            total_rows_loaded += cur.rowcount

        # Load stops.csv
        stops_path = os.path.join(datadir, 'stops.csv')
        print(f"Loading {stops_path}...")
        with open(stops_path, 'r') as f:
            next(f)
            cur.copy_from(f, 'stops', sep=',', columns=('stop_name', 'latitude', 'longitude'))
            print(f"{cur.rowcount} rows")
            total_rows_loaded += cur.rowcount

        # --- Pre-fetch name-to-ID mappings for foreign keys ---
        cur.execute("SELECT line_name, line_id FROM lines")
        line_name_to_id = {name: line_id for name, line_id in cur.fetchall()}
        
        cur.execute("SELECT stop_name, stop_id FROM stops")
        stop_name_to_id = {name: stop_id for name, stop_id in cur.fetchall()}

        # Load line_stops.csv (requires mapping names to IDs)
        line_stops_path = os.path.join(datadir, 'line_stops.csv')
        print(f"Loading {line_stops_path}...")
        output = StringIO()
        with open(line_stops_path, 'r') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            for row in reader:
                line_name, stop_name, sequence, time_offset = row
                line_id = line_name_to_id.get(line_name)
                stop_id = stop_name_to_id.get(stop_name)
                if line_id is not None and stop_id is not None:
                    # Format time_offset as a PostgreSQL interval string
                    interval_str = f"{time_offset} minutes"
                    output.write(f"{line_id}\t{stop_id}\t{sequence}\t{interval_str}\n")
        output.seek(0)
        cur.copy_expert(
            "COPY line_stops(line_id, stop_id, sequence_number, time_offset_from_start) FROM STDIN",
            output
        )
        print(f"{cur.rowcount} rows")
        total_rows_loaded += cur.rowcount

        # Load trips.csv (requires mapping and data transformation)
        trips_path = os.path.join(datadir, 'trips.csv')
        print(f"Loading {trips_path}...")
        output = StringIO()
        with open(trips_path, 'r') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            for row in reader:
                trip_id_str, line_name, scheduled_departure, vehicle_id_str = row
                trip_id = int(trip_id_str[1:]) # 'T0001' -> 1
                line_id = line_name_to_id.get(line_name)
                # Extract only the time part from the timestamp
                departure_time = scheduled_departure.split(' ')[1]
                vehicle_id = int(vehicle_id_str[1:]) # 'V101' -> 101
                if line_id is not None:
                    output.write(f"{trip_id}\t{line_id}\t{departure_time}\t{vehicle_id}\n")
        output.seek(0)
        cur.copy_expert(
            "COPY trips(trip_id, line_id, departure_time, vehicle_id) FROM STDIN",
            output
        )
        print(f"{cur.rowcount} rows")
        total_rows_loaded += cur.rowcount
        
        # Load stop_events.csv (requires mapping and data transformation)
        stop_events_path = os.path.join(datadir, 'stop_events.csv')
        print(f"Loading {stop_events_path}...")
        output = StringIO()
        with open(stop_events_path, 'r') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            for row in reader:
                trip_id_str, stop_name, scheduled, actual, passengers_on, passengers_off = row
                trip_id = int(trip_id_str[1:])
                stop_id = stop_name_to_id.get(stop_name)
                # Extract time part, handling potential empty strings
                scheduled_time = scheduled.split(' ')[1] if scheduled else None
                actual_time = actual.split(' ')[1] if actual else None
                if stop_id is not None:
                    # Use '\\N' for NULL values in COPY command
                    scheduled_val = scheduled_time if scheduled_time else '\\N'
                    actual_val = actual_time if actual_time else '\\N'
                    output.write(f"{trip_id}\t{stop_id}\t{scheduled_val}\t{actual_val}\t{passengers_on}\t{passengers_off}\n")
        output.seek(0)
        cur.copy_expert(
            "COPY stop_events(trip_id, stop_id, scheduled_time, actual_time, passengers_on, passengers_off) FROM STDIN",
            output
        )
        print(f"{cur.rowcount} rows")
        total_rows_loaded += cur.rowcount

        # --- 4. Commit changes and report total ---
        conn.commit()
        print(f"\nTotal: {total_rows_loaded} rows loaded")

    except psycopg2.Error as e:
        print(f"Database error: {e}", file=sys.stderr)
        if conn:
            conn.rollback()
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"File not found: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description="Load transit data into PostgreSQL")
    parser.add_argument('--host', required=True, help='Database host')
    parser.add_argument('--dbname', required=True, help='Database name')
    parser.add_argument('--user', required=True, help='Database user')
    parser.add_argument('--password', required=True, help='Database password')
    parser.add_argument('--datadir', required=True, help='Directory containing CSV data files')
    args = parser.parse_args()

    load_data(
        host=args.host,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
        datadir=args.datadir
    )

if __name__ == '__main__':
    main()