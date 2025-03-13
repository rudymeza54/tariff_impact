import os
import sys
import subprocess
import time

# Arguments
# -start: start PostgreSQL container
# -collect: run data collection scripts
# -start-collect: start container and run data collection scripts

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
        
    else:
        print("Usage: python script.py [-start|-collect|-start-collect]")
        print("  -start         Start PostgreSQL container")
        print("  -collect       Run data collection scripts")
        print("  -start-collect Start container and run data collection scripts")

if __name__ == "__main__":
    main()