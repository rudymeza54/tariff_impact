# test_tariff_collector.py
import os
import sys
import logging
import sqlalchemy
from sqlalchemy import text

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
    # Import the tariff news collector module
    from collectors.tariff_news_collector import collect_tariff_news, get_db_connection
    
    # Run the collector
    total_saved = collect_tariff_news()
    
    # Check what we collected
    engine = get_db_connection()
    with engine.connect() as conn:
        # Count total events
        result = conn.execute(text("SELECT COUNT(*) FROM tariff_events"))
        total_count = result.scalar()
        
        # Count events by sector
        result = conn.execute(text("""
            SELECT unnest(affected_sectors) as sector, COUNT(*) 
            FROM tariff_events 
            GROUP BY sector
        """))
        sectors = result.fetchall()
        
        # Count events by type
        result = conn.execute(text("""
            SELECT event_type, COUNT(*) 
            FROM tariff_events 
            GROUP BY event_type
            ORDER BY COUNT(*) DESC
        """))
        event_types = result.fetchall()
        
        # Get most recent event
        result = conn.execute(text("""
            SELECT title, date, source 
            FROM tariff_events 
            ORDER BY date DESC 
            LIMIT 1
        """))
        recent_event = result.fetchone()
    
    # Report findings
    logger.info(f"Total tariff events in database: {total_count}")
    
    if sectors:
        logger.info("Events by sector:")
        for sector in sectors:
            logger.info(f"  {sector[0]}: {sector[1]} events")
    
    if event_types:
        logger.info("Events by type:")
        for event_type in event_types:
            logger.info(f"  {event_type[0]}: {event_type[1]} events")
    
    if recent_event:
        logger.info(f"Most recent event: {recent_event[0]} ({recent_event[1]}) from {recent_event[2]}")
    
    logger.info("✅ Tariff news collection test PASSED")
    
except Exception as e:
    logger.error(f"❌ Tariff news collection test FAILED with error: {str(e)}")
    import traceback
    traceback.print_exc()