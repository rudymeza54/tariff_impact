# src/collectors/historical_data_collector.py

import os
import pandas as pd
import sqlalchemy
from sqlalchemy import text
import logging
import yfinance as yf
from datetime import datetime, timedelta
import time
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dates for Market Pull
start = '2024-03-01'
end = datetime.now().strftime('%Y-%m-%d')  # Use current date

def get_db_connection():
    """Create a database connection using environment variables"""
    db_host = os.environ.get('DB_HOST', 'localhost')
    db_user = os.environ.get('DB_USER', 'postgres')
    db_pass = os.environ.get('DB_PASSWORD', 'mysecretpassword')
    db_name = os.environ.get('DB_NAME', 'postgres')
    db_port = os.environ.get('DB_PORT', '5431')
    
    connection_string = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    engine = sqlalchemy.create_engine(connection_string)
    return engine

def get_existing_symbols():
    """Get a list of symbols already in the database"""
    try:
        engine = get_db_connection()
        
        with engine.connect() as conn:
            query = text("SELECT DISTINCT symbol, sector FROM market_data")
            result = conn.execute(query)
            symbols = [(row.symbol, row.sector) for row in result]
        
        return symbols
    except Exception as e:
        logger.error(f"Error getting existing symbols: {str(e)}")
        # Fall back to predefined symbols
        symbols = [
            ('ADM', 'Agriculture_Companies'),
            ('BG', 'Agriculture_Companies'),
            ('CTVA', 'Agriculture_Companies'),
            ('DE', 'Agriculture_Companies'),
            ('WY', 'Lumber_Companies'),
            ('PCH', 'Lumber_Companies'),
            ('RYN', 'Lumber_Companies'),
            ('WFG', 'Lumber_Companies'),
            ('CORN', 'Agriculture_ETFs'),
            ('WEAT', 'Agriculture_ETFs'),
            ('SOYB', 'Agriculture_ETFs'),
            ('WOOD', 'Lumber_ETFs'),
            ('CUT', 'Lumber_ETFs')
        ]
        logger.info("Using predefined symbol list as fallback")
        return symbols

def download_historical_data(symbol, sector, start_date, end_date):
    """Download historical data for a single symbol"""
    symbol_str = str(symbol).strip()
    logger.info(f"Downloading historical data for {symbol_str} in sector {sector}")
    
    try:
        # Add a delay to avoid rate limiting
        time.sleep(random.uniform(1, 2))
        
        # Download historical data
        ticker_data = yf.download(
            symbol_str,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        if ticker_data.empty:
            logger.warning(f"No historical data returned for {symbol_str}")
            return None
        
        # Create a new DataFrame with all the data we need
        result_data = []
        
        # Process each row
        for idx, row in ticker_data.iterrows():
            data_dict = {
                'date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
                'open': float(row['Open']) if 'Open' in row else 0.0,
                'high': float(row['High']) if 'High' in row else 0.0,
                'low': float(row['Low']) if 'Low' in row else 0.0,
                'close': float(row['Close']) if 'Close' in row else 0.0,
                'volume': int(row['Volume']) if 'Volume' in row else 0,
                'symbol': symbol_str,
                'sector': str(sector),
                'data_type': 'etf' if 'ETF' in str(sector) else 'stock'
            }
            result_data.append(data_dict)
        
        # Convert to DataFrame
        if not result_data:
            logger.warning(f"No processable historical data for {symbol_str}")
            return None
            
        result_df = pd.DataFrame(result_data)
        logger.info(f"Successfully downloaded {len(result_df)} historical records for {symbol_str}")
        return result_df
        
    except Exception as e:
        logger.error(f"Error downloading historical data for {symbol_str}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def get_todays_data(symbol, sector):
    """Get today's data for a specific symbol"""
    symbol_str = str(symbol).strip()
    today = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Getting today's data for {symbol_str} in sector {sector}")
    
    try:
        # Fetch live data for the symbol
        ticker = yf.Ticker(symbol_str)
        live_data = ticker.history(period="1d")  # Get today's data

        if live_data.empty:
            logger.warning(f"No live data returned for {symbol_str}")
            return None

        # Get the latest price
        latest_price = float(live_data['Close'].iloc[-1])
        
        # Create a dictionary for today's data
        todays_record = {
            'date': today,
            'open': float(live_data['Open'].iloc[-1]) if 'Open' in live_data.columns else latest_price,
            'high': float(live_data['High'].iloc[-1]) if 'High' in live_data.columns else latest_price,
            'low': float(live_data['Low'].iloc[-1]) if 'Low' in live_data.columns else latest_price,
            'close': latest_price,
            'volume': int(live_data['Volume'].iloc[-1]) if 'Volume' in live_data.columns else 0,
            'symbol': symbol_str,
            'sector': str(sector),
            'data_type': 'etf' if 'ETF' in str(sector) else 'stock'
        }
        
        logger.info(f"Successfully got today's data for {symbol_str}")
        return pd.DataFrame([todays_record])
        
    except Exception as e:
        logger.error(f"Error getting today's data for {symbol_str}: {str(e)}")
        return None

def collect_market_data(start_date=start, end_date=end):
    """Collect both historical and today's market data for all symbols"""
    logger.info(f"Collecting market data from {start_date} to {end_date}")
    
    # Get symbols
    symbols = get_existing_symbols()
    
    if not symbols:
        logger.error("No symbols found")
        return None
    
    # Today's date
    today = datetime.now().strftime('%Y-%m-%d')
    logger.info(f"Today's date: {today}")
    
    # Adjust end date for historical data to yesterday
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Create empty list to hold all data
    all_data_frames = []
    
    # Process each symbol
    for symbol_info in symbols:
        if isinstance(symbol_info, tuple) and len(symbol_info) >= 2:
            symbol = symbol_info[0]
            sector = symbol_info[1]
        else:
            symbol = str(symbol_info)
            sector = "Unknown"
        
        # Get historical data (up to yesterday)
        historical_df = download_historical_data(symbol, sector, start_date, yesterday)
        if historical_df is not None and not historical_df.empty:
            all_data_frames.append(historical_df)
        
        # Get today's data separately
        today_df = get_todays_data(symbol, sector)
        if today_df is not None and not today_df.empty:
            all_data_frames.append(today_df)
        
        # Small delay between symbols
        time.sleep(random.uniform(0.5, 1.0))
    
    # Combine all data
    if all_data_frames:
        combined_data = pd.concat(all_data_frames, ignore_index=True)
        logger.info(f"Total collected records: {len(combined_data)}")
        
        # Log some statistics
        dates = combined_data['date'].unique()
        logger.info(f"Date range in data: {min(dates)} to {max(dates)}")
        today_count = len(combined_data[combined_data['date'] == today])
        logger.info(f"Records for today ({today}): {today_count}")
        
        return combined_data
    else:
        logger.error("No data collected")
        return None

def save_market_data(data):
    """Save market data to the database"""
    if data is None or data.empty:
        logger.error("No data to save")
        return 0
    
    # Required columns
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'sector', 'data_type']
    
    # Check if all required columns exist
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}")
        return 0
    
    # Select only needed columns
    df = data[required_cols].copy()
    
    # Fix data types for numeric columns
    for col in ['open', 'high', 'low', 'close']:
        # Convert placeholder strings to numeric values
        mask = df[col] == "markets still open"
        if mask.any():
            logger.info(f"Converting placeholder values for {col}")
            # For columns with placeholder values, use previous day's closing price
            today = datetime.now().strftime('%Y-%m-%d')
            for symbol in df.loc[mask, 'symbol'].unique():
                # Get yesterday's close for this symbol
                yesterday_data = df[(df['symbol'] == symbol) & (df['date'] < today)].sort_values('date', ascending=False)
                if not yesterday_data.empty:
                    last_close = yesterday_data.iloc[0]['close']
                    if isinstance(last_close, (int, float)):
                        df.loc[(df['symbol'] == symbol) & mask, col] = last_close
                    else:
                        df.loc[(df['symbol'] == symbol) & mask, col] = 0.0
    
    # Fix volume placeholder
    mask = df['volume'] == "markets still open"
    if mask.any():
        logger.info("Converting volume placeholder values")
        df.loc[mask, 'volume'] = 0
    
    # Ensure numeric columns are numeric
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
    
    # Log the first few rows for today
    today = datetime.now().strftime('%Y-%m-%d')
    today_data = df[df['date'] == today]
    if not today_data.empty:
        logger.info(f"Sample of today's data: {today_data.iloc[0].to_dict()}")
    
    # Handle existing data for today
    try:
        engine = get_db_connection()
        
        # Delete today's data that might already exist
        with engine.connect() as conn:
            today_symbols = df[df['date'] == today]['symbol'].unique()
            for symbol in today_symbols:
                delete_query = text(f"DELETE FROM market_data WHERE date = '{today}' AND symbol = '{symbol}'")
                conn.execute(delete_query)
            conn.commit()
            
        # Save the data
        df.to_sql('market_data', engine, if_exists='replace', index=False)
        logger.info(f"Successfully saved {len(df)} records")
        return len(df)
        
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        
        # Try row by row if bulk insert fails
        success_count = 0
        for _, row in df.iterrows():
            try:
                row_dict = row.to_dict()
                # Make sure all values are of correct types
                for col in ['open', 'high', 'low', 'close']:
                    if not isinstance(row_dict[col], (int, float)):
                        row_dict[col] = 0.0
                
                if not isinstance(row_dict['volume'], (int, float)):
                    row_dict[col] = 0
                else:
                    row_dict['volume'] = int(row_dict['volume'])
                
                single_row = pd.DataFrame([row_dict])
                single_row.to_sql('market_data', engine, if_exists='replace', index=False)
                success_count += 1
            except Exception as inner_e:
                logger.error(f"Error saving row: {str(inner_e)}")
        
        logger.info(f"Saved {success_count} records individually")
        return success_count

if __name__ == "__main__":
    try:
        # Collect data
        data = collect_market_data()
        
        # Save data
        if data is not None:
            rows_saved = save_market_data(data)
            logger.info(f"Total rows saved: {rows_saved}")
        else:
            logger.error("No data to save")
    except Exception as e:
        logger.error(f"Critical error in main process: {str(e)}")