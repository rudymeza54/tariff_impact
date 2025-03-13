# test_db_connection.py
import os
import sqlalchemy
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection function
def get_db_connection():
    """Create a database connection using environment variables or defaults"""
    db_host = os.environ.get('DB_HOST', 'localhost')
    db_user = os.environ.get('DB_USER', 'postgres')
    db_pass = os.environ.get('DB_PASSWORD', 'mysecretpassword')
    db_name = os.environ.get('DB_NAME', 'postgres')
    db_port = os.environ.get('DB_PORT', '5431')  # Host port is 5431 in your new container
    
    logger.info(f"Connecting to database: {db_name} on {db_host}:{db_port} as {db_user}")
    
    connection_string = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    engine = sqlalchemy.create_engine(connection_string)
    return engine

# Test the connection
try:
    logger.info("Testing database connection...")
    engine = get_db_connection()
    
    # Try a simple query
    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text("SELECT 1"))
        value = result.scalar()
        
    logger.info(f"Connection successful! Test query returned: {value}")
    
    # Check if we can create a simple test table
    logger.info("Testing table creation...")
    with engine.connect() as conn:
        conn.execute(sqlalchemy.text("""
            CREATE TABLE IF NOT EXISTS connection_test (
                id SERIAL PRIMARY KEY,
                test_column VARCHAR(50)
            )
        """))
        conn.commit()
        
    logger.info("✅ Database connection and table creation test PASSED")
    
except Exception as e:
    logger.error(f"❌ Database connection test FAILED with error: {str(e)}")
    # Print the full error for debugging
    import traceback
    traceback.print_exc()