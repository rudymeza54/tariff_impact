# test_historical_collection.py
import os
import sys
import logging
import sqlalchemy

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the correct path to find the modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

# Set database connection environment variables
os.environ['DB_HOST'] = 'localhost'
os.environ['DB_USER'] = 'postgres'
os.environ['DB_PASSWORD'] = 'mysecretpassword'
os.environ['DB_NAME'] = 'postgres'
os.environ['DB_PORT'] = '5431'

try:
    # Import the historical data collection function
    from collectors.market_data_collector import collect_historical_data, get_db_connection
    
    # Use a larger set of tickers for historical collection
    import collectors.market_data_collector as mdc
    
    # Set tickers to include both agricultural and lumber sectors
    mdc.TICKERS = {
        'Agriculture_Companies': ['ADM', 'BG', 'CTVA', 'DE'],
        'Agriculture_ETFs': ['CORN', 'SOYB', 'WEAT'],
        'Lumber_Companies': ['WY', 'RYN', 'PCH', 'WFG'],
        'Lumber_ETFs': ['WOOD', 'CUT']
    }
    
    # Collect data from March 1, 2025
    records_added = collect_historical_data(start_date="2025-03-01")
    
    # Check data in database
    engine = get_db_connection()
    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text("SELECT COUNT(*) FROM market_data"))
        total_count = result.scalar()
        
        # Get date range
        result = conn.execute(sqlalchemy.text("SELECT MIN(date), MAX(date) FROM market_data"))
        date_range = result.fetchone()
        
    logger.info(f"Total records in database: {total_count}")
    logger.info(f"Date range: {date_range[0]} to {date_range[1]}")
    
    logger.info("✅ Historical data collection test PASSED")
    
except Exception as e:
    logger.error(f"❌ Historical data collection test FAILED with error: {str(e)}")
    import traceback
    traceback.print_exc()