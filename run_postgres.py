import os
import sys
import subprocess
import time
import pandas as pd
import sqlalchemy
from sqlalchemy import text

# Arguments
# -start: start PostgreSQL container
# -collect: run data collection scripts
# -start-collect: start container and run data collection scripts
# -push-to-render: push data from local database to Render
# -full-process: start container, collect data, and push to Render

def start_container(container_name):
    # First check if Docker is running
    check_docker_cmd = 'docker ps > nul 2>&1'
    if os.name != 'nt':  # For non-Windows systems
        check_docker_cmd = 'docker ps > /dev/null 2>&1'
    
    docker_running = os.system(check_docker_cmd) == 0
    
    if not docker_running:
        print("Error: Docker doesn't appear to be running. Please start Docker Desktop first.")
        return False
        
    # Check if container exists
    check_container_cmd = f'docker container inspect {container_name} > nul 2>&1'
    if os.name != 'nt':  # For non-Windows systems
        check_container_cmd = f'docker container inspect {container_name} > /dev/null 2>&1'
    
    container_exists = os.system(check_container_cmd) == 0
    
    if not container_exists:
        print(f"Container '{container_name}' does not exist. Creating it...")
        # Create the postgres container if it doesn't exist
        create_cmd = f'docker run --name {container_name} -e POSTGRES_PASSWORD=mysecretpassword -p 5431:5432 -d postgres'
        if os.system(create_cmd) != 0:
            print(f"Failed to create {container_name}")
            return False
        print(f"Created {container_name}")
        # Wait for PostgreSQL to be ready after creation
        print("Waiting for PostgreSQL to be ready...")
        time.sleep(10)  # Give more time for initial setup
        return True
    
    # Container exists, try to start it
    cmd = f'docker start {container_name}'
    result = os.system(cmd)
    if result == 0:
        print(f'Started {container_name}')
        # Wait for PostgreSQL to be ready
        print("Waiting for PostgreSQL to be ready...")
        time.sleep(5)
        return True
    else:
        print(f'Failed to start {container_name}')
        return False

def run_data_collectors():
    # Set environment variables for the collectors
    env = os.environ.copy()
    env['DB_HOST'] = 'localhost'
    env['DB_PORT'] = '5431'
    env['DB_USER'] = 'postgres'
    env['DB_PASSWORD'] = 'mysecretpassword'
    env['DB_NAME'] = 'postgres'
    
    # Store original directory
    original_dir = os.getcwd()
    
    # List of collector scripts to run
    collectors = [
        'tariff_news_collector.py',
        'historical_data_collector.py'
    ]
    
    # Run each collector
    for collector in collectors:
        print(f"Running {collector}...")
        # Change directory only once
        collector_dir = os.path.join(original_dir, 'src', 'collectors')
        os.chdir(collector_dir)
        
        try:
            result = subprocess.run(['python', collector], env=env, check=True)
            print(f"Successfully ran {collector}")
        except subprocess.CalledProcessError as e:
            print(f"Error running {collector}, return code: {e.returncode}")
        except Exception as e:
            print(f"Error executing {collector}: {str(e)}")
        
        # Return to original directory after each collector
        os.chdir(original_dir)

def push_data_to_render():
    """Push data from local PostgreSQL to Render database using SQLAlchemy"""
    try:
        print("Pushing data to Render PostgreSQL database using SQLAlchemy...")
        
        # Local DB connection with improved connection parameters
        local_conn_str = "postgresql://postgres:mysecretpassword@localhost:5431/postgres"
        local_engine = sqlalchemy.create_engine(
            local_conn_str,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            connect_args={"connect_timeout": 10}
        )
        
        # Render DB connection with improved connection parameters - UPDATED PORT TO 5432
        render_conn_str = "postgresql://tariff_db_mwke_user:OgyMpx6azihJIqRknrWlOmLkEbgiuj13@dpg-cv9l4g3tq21c73blj9ug-a.oregon-postgres.render.com:5432/tariff_db_mwke"
        render_engine = sqlalchemy.create_engine(
            render_conn_str,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            connect_args={"connect_timeout": 10}
        )
        
        # Check both connections with proper resource cleanup
        try:
            with local_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                # Connection automatically closed when exiting the with block
            print("Successfully connected to local database")
        except Exception as e:
            print(f"Error connecting to local database: {str(e)}")
            return False
            
        try:
            with render_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                # Connection automatically closed when exiting the with block
            print("Successfully connected to Render database")
        except Exception as e:
            print(f"Error connecting to Render database: {str(e)}")
            return False
        
        # Tables to transfer
        tables = ["tariff_events", "market_data"]  # Add all your tables here
        
        for table in tables:
            print(f"Transferring table: {table}")
            
            # First check if table exists in local database
            table_exists = False
            try:
                with local_engine.connect() as conn:
                    check_table = text(f"SELECT to_regclass('{table}')")
                    result = conn.execute(check_table).scalar()
                    table_exists = result is not None
                    
                if not table_exists:
                    print(f"Table {table} does not exist in local database. Skipping.")
                    continue
            except Exception as e:
                print(f"Error checking table {table} in local database: {str(e)}")
                continue
            
            # Special handling for tariff_events table - create sequence first
            if table == "tariff_events":
                try:
                    with render_engine.connect() as conn:
                        # Check if sequence exists
                        seq_check = text("SELECT to_regclass('tariff_events_id_seq')")
                        seq_exists = conn.execute(seq_check).scalar() is not None
                        
                        if not seq_exists:
                            # Create sequence
                            create_seq = text("CREATE SEQUENCE tariff_events_id_seq")
                            conn.execute(create_seq)
                            conn.commit()
                            print("Created sequence tariff_events_id_seq")
                except Exception as e:
                    print(f"Error managing sequence: {str(e)}")
            
            # Get schema from local database - using with statement for connection cleanup
            schema = None
            try:
                with local_engine.connect() as conn:
                    schema_query = text(f"""
                    SELECT column_name, data_type, 
                           character_maximum_length, column_default, is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = '{table}'
                    ORDER BY ordinal_position
                    """)
                    schema = pd.read_sql(schema_query, conn)
            except Exception as e:
                print(f"Error getting schema for {table}: {str(e)}")
                continue
                
            if schema is None or schema.empty:
                print(f"Could not retrieve schema for {table}. Skipping.")
                continue
            
            # Create table in Render if it doesn't exist
            columns = []
            for _, row in schema.iterrows():
                col_name = row['column_name']
                col_type = row['data_type']
                
                # Handle specific data types
                if col_type == 'character varying' and row['character_maximum_length']:
                    # Convert float length to integer for varchar
                    length = int(row['character_maximum_length'])
                    col_type = f"varchar({length})"
                elif col_type == 'ARRAY':
                    col_type = "text[]"  # Simplified array handling
                
                # Add nullability
                nullable = "NULL" if row['is_nullable'] == 'YES' else "NOT NULL"
                
                # Add default if exists - but handle sequences specially
                if row['column_default'] and 'nextval' in str(row['column_default']):
                    if table == "tariff_events" and col_name == "id":
                        default = "DEFAULT nextval('tariff_events_id_seq'::regclass)"
                    else:
                        default = f"DEFAULT {row['column_default']}"
                else:
                    default = f"DEFAULT {row['column_default']}" if row['column_default'] else ""
                
                columns.append(f"{col_name} {col_type} {nullable} {default}".strip())
            
            create_table_sql = f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(columns)})"
            
            table_created = False
            try:
                with render_engine.connect() as conn:
                    conn.execute(text(create_table_sql))
                    conn.commit()
                print(f"Created or verified table {table} in Render database")
                table_created = True
            except Exception as e:
                print(f"Error creating table {table} in Render database: {str(e)}")
                print(f"SQL was: {create_table_sql}")
                
                # Alternative approach for tariff_events if standard approach fails
                if table == "tariff_events":
                    try:
                        print("Trying alternative approach with SERIAL type...")
                        alt_create_sql = text("""
                        CREATE TABLE IF NOT EXISTS tariff_events (
                            id SERIAL PRIMARY KEY,
                            title text NOT NULL, 
                            description text NULL, 
                            date date NOT NULL, 
                            source varchar(100) NULL, 
                            url text NULL, 
                            event_type varchar(50) NULL, 
                            affected_sectors text[] NULL, 
                            collected_at timestamp without time zone NOT NULL DEFAULT now()
                        )
                        """)
                        with render_engine.connect() as conn:
                            conn.execute(alt_create_sql)
                            conn.commit()
                        print("Created tariff_events table with alternative approach")
                        table_created = True
                    except Exception as e2:
                        print(f"Alternative approach also failed: {str(e2)}")
                        continue
                else:
                    continue
            
            if not table_created:
                print(f"Could not create table {table}. Skipping data transfer.")
                continue
                
            # Transfer data in chunks to avoid memory issues
            try:
                # Get total rows for progress reporting
                total_rows = 0
                with local_engine.connect() as conn:
                    count_query = text(f"SELECT COUNT(*) FROM {table}")
                    total_rows = conn.execute(count_query).scalar()
                print(f"Found {total_rows} rows to transfer in {table}")
                
                # Use smaller chunks and explicitly dispose of DataFrames to manage memory
                if total_rows > 0:
                    # Smaller chunk size to reduce memory usage
                    chunk_size = 1000
                    
                    # Clear existing data if we're replacing
                    with render_engine.connect() as conn:
                        conn.execute(text(f"DELETE FROM {table}"))
                        conn.commit()
                    
                    # Process in chunks with explicit resource cleanup
                    offset = 0
                    while offset < total_rows:
                        # Use SQL directly to limit memory usage
                        with local_engine.connect() as conn:
                            chunk_query = text(f"SELECT * FROM {table} OFFSET {offset} LIMIT {chunk_size}")
                            chunk_df = pd.read_sql(chunk_query, conn)
                        
                        print(f"Transferring rows {offset} to {min(offset + chunk_size, total_rows)} of {total_rows}...")
                        
                        # Write to Render
                        with render_engine.connect() as conn:
                            # Convert DataFrame to list of tuples for more efficient insert
                            columns = chunk_df.columns.tolist()
                            tuples = [tuple(x) for x in chunk_df.to_numpy()]
                            
                            # Explicitly set statement timeout
                            conn.execute(text("SET statement_timeout = 30000"))  # 30 seconds
                            
                            # Create placeholders for values
                            placeholders = ', '.join(['%s'] * len(columns))
                            column_names = ', '.join(columns)
                            
                            # Insert data in batches
                            batch_size = 100
                            for i in range(0, len(tuples), batch_size):
                                batch = tuples[i:i+batch_size]
                                values = ', '.join([str(conn.connection.cursor().mogrify(f"({placeholders})", x).decode('utf-8')) for x in batch])
                                insert_query = f"INSERT INTO {table} ({column_names}) VALUES {values}"
                                conn.execute(text(insert_query))
                            
                            conn.commit()
                        
                        # Increment offset and explicitly clean up DataFrame
                        offset += chunk_size
                        del chunk_df
                        
                    print(f"Successfully transferred all {total_rows} rows for {table}")
                else:
                    print(f"No data found in {table}")
                    
            except Exception as e:
                print(f"Error transferring data for {table}: {str(e)}")
                continue
            
        print("Data transfer completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during data transfer: {str(e)}")
        return False
    finally:
        # Ensure engines are properly disposed to release all connections
        try:
            if 'local_engine' in locals():
                local_engine.dispose()
            if 'render_engine' in locals():
                render_engine.dispose()
        except Exception as e:
            print(f"Error disposing database engines: {str(e)}")

# Main function
def main():
    # Read input argument
    argument = sys.argv[1] if len(sys.argv) > 1 else None

    # Process argument
    if argument == '-start':
        start_container('postgres-container')
        
    elif argument == '-collect':
        run_data_collectors()
        
    elif argument == '-start-collect':
        if start_container('postgres-container'):
            print("Container started, now running data collectors...")
            run_data_collectors()
    
    elif argument == '-push-to-render':
        push_data_to_render()
        
    elif argument == '-full-process':
        if start_container('postgres-container'):
            print("Container started, now running data collectors...")
            run_data_collectors()
            print("Data collection completed, now pushing to Render...")
            push_data_to_render()
        
    else:
        print("Usage: python script.py [-start|-collect|-start-collect|-push-to-render|-full-process]")
        print("  -start          Start PostgreSQL container")
        print("  -collect        Run data collection scripts")
        print("  -start-collect  Start container and run data collection scripts")
        print("  -push-to-render Push data from local database to Render")
        print("  -full-process   Start container, collect data, and push to Render")

if __name__ == "__main__":
    main()