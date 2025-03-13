# setup_schema.py
import os
import sqlalchemy
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Create a database connection using environment variables or defaults"""
    db_host = os.environ.get('DB_HOST', 'localhost')
    db_user = os.environ.get('DB_USER', 'postgres')
    db_pass = os.environ.get('DB_PASSWORD', 'mysecretpassword')
    db_name = os.environ.get('DB_NAME', 'postgres')
    db_port = os.environ.get('DB_PORT', '5431')  # Host port is 5431 in your new container
    
    connection_string = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    engine = sqlalchemy.create_engine(connection_string)
    return engine

def create_tables():
    """Create necessary tables for the tariff analysis project"""
    engine = get_db_connection()
    
    # Market data table
    market_data_table = """
    CREATE TABLE IF NOT EXISTS market_data (
        id SERIAL PRIMARY KEY,
        date DATE NOT NULL,
        open FLOAT,
        high FLOAT,
        low FLOAT, 
        close FLOAT NOT NULL,
        volume BIGINT,
        symbol VARCHAR(20) NOT NULL,
        sector VARCHAR(100) NOT NULL,
        data_type VARCHAR(20) NOT NULL,
        collected_at TIMESTAMP NOT NULL DEFAULT NOW(),
        UNIQUE(date, symbol)
    );
    """
    
    # Tariff events table
    tariff_events_table = """
    CREATE TABLE IF NOT EXISTS tariff_events (
        id SERIAL PRIMARY KEY,
        title TEXT NOT NULL,
        description TEXT,
        date DATE NOT NULL,
        source VARCHAR(100),
        url TEXT,
        event_type VARCHAR(50),
        affected_sectors TEXT[],
        collected_at TIMESTAMP NOT NULL DEFAULT NOW()
    );
    """
    
    # Impact analysis table
    impact_analysis_table = """
    CREATE TABLE IF NOT EXISTS tariff_impact_analysis (
        id SERIAL PRIMARY KEY,
        event_id INTEGER REFERENCES tariff_events(id),
        sector VARCHAR(100) NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        pre_event_avg FLOAT,
        post_event_5d FLOAT,
        post_event_10d FLOAT,
        post_event_30d FLOAT,
        impact_5d_pct FLOAT,
        impact_10d_pct FLOAT,
        impact_30d_pct FLOAT,
        volume_change_pct FLOAT,
        analyzed_at TIMESTAMP NOT NULL DEFAULT NOW(),
        UNIQUE(event_id, symbol)
    );
    """
    
    with engine.connect() as conn:
        logger.info("Creating market_data table...")
        conn.execute(sqlalchemy.text(market_data_table))
        
        logger.info("Creating tariff_events table...")
        conn.execute(sqlalchemy.text(tariff_events_table))
        
        logger.info("Creating tariff_impact_analysis table...")
        conn.execute(sqlalchemy.text(impact_analysis_table))
        
        conn.commit()
        
    logger.info("✅ Database schema created successfully")

if __name__ == "__main__":
    create_tables()