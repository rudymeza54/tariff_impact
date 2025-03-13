# api.py
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import text
import os
import logging
import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["https://tariffimpact.netlify.app"]) # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection
def get_db_connection():
    db_host = os.environ.get('DB_HOST', 'localhost')
    db_user = os.environ.get('DB_USER', 'postgres')
    db_pass = os.environ.get('DB_PASSWORD', 'mysecretpassword')
    db_name = os.environ.get('DB_NAME', 'postgres')
    db_port = os.environ.get('DB_PORT', '5431')
    
    connection_string = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    logger.info(f"Attempting to connect with: {connection_string.replace(db_pass, '******')}")
    engine = sqlalchemy.create_engine(connection_string)
    return engine

# Helper class for creating mock data when database is not available
class TariffImpactAnalyzer:
    def __init__(self):
        self.event_cache = {}
        
    def get_normalized_prices(self, event_id, sector, days):
        """Get normalized price data for visualization."""
        try:
            # Try to get data from database
            engine = get_db_connection()
            
            # Get event date
            with engine.connect() as conn:
                event_query = text("SELECT date FROM tariff_events WHERE id = :event_id")
                event_date = conn.execute(event_query, {"event_id": event_id}).scalar()
                
            if not event_date:
                # Use fallback date if event not found
                event_date = datetime.datetime(2025, 3, 5)
            
            # Calculate date range
            event_date = pd.to_datetime(event_date)
            start_date = (event_date - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
            end_date = (event_date + pd.Timedelta(days=days)).strftime('%Y-%m-%d')
            
            # Updated query to match database schema
            query = """
            SELECT date, symbol, sector, close, volume
            FROM market_data
            WHERE date BETWEEN :start_date AND :end_date
            """
            
            params = {"start_date": start_date, "end_date": end_date}
            
            if sector and sector != 'all':
                query += f" AND sector = '{sector}'"
            
            query += " ORDER BY symbol, date"
            
            # Execute query
            with engine.connect() as conn:
                data = pd.read_sql(text(query), conn, params=params)
                
            # Add missing name column
            data['name'] = data['symbol']
                
            logger.info(f"Successfully retrieved {len(data)} price data records")
            # Normalize prices and calculate days from event
            result = self._normalize_prices(data, event_date)
            return result
            
        except Exception as e:
            logger.error(f"Error getting normalized prices: {str(e)}")
            # Return empty result when database access fails
            return {'eventDate': '', 'dates': [], 'companies': [], 'series': []}
    
    def _normalize_prices(self, data, event_date):
        """Normalize price data for visualization."""
        try:
            # Convert date to datetime
            data['date'] = pd.to_datetime(data['date'])
            
            # Calculate days from event
            data['day'] = (data['date'] - event_date).dt.days
            
            # Calculate baseline (average pre-event price)
            pre_event = data[data['day'] < 0]
            if pre_event.empty:
                # Handle case with no pre-event data
                baseline_prices = {symbol: 1 for symbol in data['symbol'].unique()}
            else:
                baseline_prices = pre_event.groupby('symbol')['close'].mean().to_dict()
            
            # Normalize prices
            data['normalized_price'] = data.apply(
                lambda row: (row['close'] / baseline_prices.get(row['symbol'], 1)) * 100, 
                axis=1
            )
            
            # Format for visualization
            companies = []
            for symbol in data['symbol'].unique():
                symbol_data = data[data['symbol'] == symbol].iloc[0]
                companies.append({
                    'symbol': symbol,
                    'name': symbol_data['name'],
                    'sector': symbol_data['sector'],
                    'color': self._get_color(symbol, symbol_data['sector'])
                })
            
            # Get all dates
            dates = sorted(data['date'].unique())
            date_strs = [d.strftime('%Y-%m-%d') for d in dates]
            
            # Create series data
            series = []
            for symbol in data['symbol'].unique():
                symbol_data = data[data['symbol'] == symbol].copy()
                series_data = []
                
                for _, row in symbol_data.iterrows():
                    series_data.append({
                        'date': row['date'].strftime('%Y-%m-%d'),
                        'price': row['close'],
                        'normalized_price': row['normalized_price'],
                        'day': int(row['day'])
                    })
                
                series.append({
                    'symbol': symbol,
                    'data': sorted(series_data, key=lambda x: x['date'])
                })
            
            return {
                'eventDate': event_date.strftime('%Y-%m-%d'),
                'dates': date_strs,
                'companies': companies,
                'series': series
            }
        except Exception as e:
            logger.error(f"Error normalizing prices: {str(e)}")
            return {
                'eventDate': event_date.strftime('%Y-%m-%d'),
                'dates': [],
                'companies': [],
                'series': []
            }
            
    def get_event_impact(self, event_id, sector):
        """Get impact analysis data"""
        try:
            # Try to get from database
            engine = get_db_connection()
            
            # Get event date
            with engine.connect() as conn:
                event_query = text("SELECT date FROM tariff_events WHERE id = :event_id")
                event_date = conn.execute(event_query, {"event_id": event_id}).scalar()
                
            if not event_date:
                # Use fallback date if event not found
                event_date = datetime.datetime(2025, 3, 5)
                
            # Calculate date range (expand to ensure we have enough data)
            event_date = pd.to_datetime(event_date)
            pre_start = (event_date - pd.Timedelta(days=30)).strftime('%Y-%m-%d')  # Increased from 10 to 30
            pre_end = (event_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            post_start = event_date.strftime('%Y-%m-%d')
            post_end = (event_date + pd.Timedelta(days=30)).strftime('%Y-%m-%d')  # Increased from 10 to 30
            
            # Get all market data for the time period
            query = """
            SELECT date, symbol, sector, close
            FROM market_data
            WHERE date BETWEEN :pre_start AND :post_end
            ORDER BY symbol, date
            """
            
            # Execute query
            with engine.connect() as conn:
                data = pd.read_sql(text(query), conn, params={
                    "pre_start": pre_start, 
                    "post_end": post_end
                })
            
            if data.empty:
                logger.warning(f"No market data found for event {event_id}")
                return []
                
            # Add date as datetime    
            data['date'] = pd.to_datetime(data['date'])
            
            # Calculate impact for each symbol
            impact_results = []
            
            for symbol in data['symbol'].unique():
                symbol_data = data[data['symbol'] == symbol]
                
                # Get pre-event and post-event data
                pre_data = symbol_data[symbol_data['date'] < event_date]
                post_data = symbol_data[symbol_data['date'] >= event_date]
                
                # Skip if we have no data for either period
                if pre_data.empty or post_data.empty:
                    logger.warning(f"No data for {symbol} on one side of the event")
                    continue
                    
                # Get average prices instead of requiring multiple points
                pre_price = pre_data['close'].mean()
                post_price = post_data['close'].mean()
                sector_name = symbol_data['sector'].iloc[0]
                
                # Calculate percentage change
                pct_change = ((post_price / pre_price) - 1) * 100
                abs_change = post_price - pre_price
                
                impact_results.append({
                    'symbol': symbol,
                    'name': symbol,
                    'sector': sector_name,
                    'pre_price': float(pre_price),
                    'post_price': float(post_price),
                    'pct_change': float(pct_change),
                    'absolute_change': float(abs_change),
                    'impact': float(pct_change)  # For D3 compatibility
                })
                
            logger.info(f"Calculated impact for {len(impact_results)} symbols")
            return impact_results
                
        except Exception as e:
            logger.error(f"Error getting impact data: {str(e)}")
            # Return empty result when database access fails
            return []
            
    def get_trading_volumes(self, event_id, sector, days):
        """Get trading volume data"""
        try:
            # Try to get from database
            engine = get_db_connection()
            
            # Get event date
            with engine.connect() as conn:
                event_query = text("SELECT date FROM tariff_events WHERE id = :event_id")
                event_date = conn.execute(event_query, {"event_id": event_id}).scalar()
                
            if not event_date:
                # Use fallback date if event not found
                event_date = datetime.datetime(2025, 3, 5)
                
            # Calculate date range
            event_date = pd.to_datetime(event_date)
            start_date = (event_date - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
            end_date = (event_date + pd.Timedelta(days=days)).strftime('%Y-%m-%d')
            
            # Updated query to match database schema
            query = """
            SELECT date, symbol, sector, volume
            FROM market_data
            WHERE date BETWEEN :start_date AND :end_date
            """
            
            params = {"start_date": start_date, "end_date": end_date}
            
            if sector and sector != 'all':
                query += f" AND sector = '{sector}'"
            
            query += " ORDER BY symbol, date"
            
            # Execute query
            with engine.connect() as conn:
                data = pd.read_sql(text(query), conn, params=params)
                
            # Add missing name column
            data['name'] = data['symbol']
                
            # Normalize volumes and calculate days from event
            result = self._normalize_volumes(data, event_date)
            return result
            
        except Exception as e:
            logger.error(f"Error getting trading volumes: {str(e)}")
            # Return empty result when database access fails
            return {'eventDate': '', 'dates': [], 'companies': [], 'series': []}
            
    def get_sector_heatmap(self, event_id, sector):
        """Get sector heatmap data"""
        try:
            # Try to get from database
            engine = get_db_connection()
            
            # Get event date
            with engine.connect() as conn:
                event_query = text("SELECT date FROM tariff_events WHERE id = :event_id")
                event_date = conn.execute(event_query, {"event_id": event_id}).scalar()
                    
            if not event_date:
                # Use fallback date if event not found
                event_date = datetime.datetime(2025, 3, 5)
                    
            # Calculate date range
            event_date = pd.to_datetime(event_date)
            start_date = (event_date - pd.Timedelta(days=15)).strftime('%Y-%m-%d')
            end_date = (event_date + pd.Timedelta(days=15)).strftime('%Y-%m-%d')
                
            # Convert event_date to string format for the query
            event_date_str = event_date.strftime('%Y-%m-%d')
                
            # First, check if today's data exists
            today = datetime.datetime.now().strftime('%Y-%m-%d')
                
            with engine.connect() as conn:
                check_query = text("""
                    SELECT COUNT(*) FROM market_data WHERE date = :today
                """)
                today_count = conn.execute(check_query, {"today": today}).scalar()
                    
            logger.info(f"Found {today_count} records for today ({today})")
                
            # Use a simple query that handles string values correctly without updating the database
            query = """
            WITH filtered_data AS (
                SELECT 
                    sector, 
                    date, 
                    CASE 
                        WHEN close::text = 'markets still open' THEN open::numeric
                        ELSE 
                            CASE 
                                WHEN close IS NULL THEN open::numeric
                                ELSE COALESCE(close::numeric, open::numeric)
                            END
                    END as adjusted_close
                FROM market_data
                WHERE date BETWEEN :start_date AND :end_date
            ),
            base_prices AS (
                SELECT sector, date, AVG(adjusted_close) as sector_price
                FROM filtered_data
                GROUP BY sector, date
            ),
            event_prices AS (
                SELECT sector, AVG(adjusted_close) as event_price
                FROM filtered_data
                WHERE date = :event_date
                GROUP BY sector
            )
            SELECT b.sector, b.date, b.sector_price, 
                (b.sector_price / e.event_price - 1) * 100 as pct_change
            FROM base_prices b
            JOIN event_prices e ON b.sector = e.sector
            ORDER BY b.sector, b.date
            """
                
            params = {"start_date": start_date, "end_date": end_date, "event_date": event_date_str}
                
            # Log the query parameters for debugging
            logger.info(f"Heatmap query params: start_date={start_date}, end_date={end_date}, event_date={event_date_str}")
                
            # Execute query
            with engine.connect() as conn:
                data = pd.read_sql(text(query), conn, params=params)
                    
            # Log the returned data dates
            if not data.empty:
                unique_dates = sorted(data['date'].unique())
                logger.info(f"Data returned for dates: {unique_dates}")
                if today not in unique_dates:
                    logger.warning(f"Today's date ({today}) not found in results")
                    
            # Calculate days from event
            data['date'] = pd.to_datetime(data['date'])
            data['day'] = (data['date'] - event_date).dt.days
                    
            # Convert to format for heatmap
            result = self._format_heatmap_data(data, event_date)
            return result
                
        except Exception as e:
            logger.error(f"Error getting heatmap data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())  # Print full traceback for debugging
            # Use empty data
            return {}
        

            
    def get_event_summary(self, event_id):
        """Get summary metrics for an event"""
        try:
            # Try to get impact data (will use cached version if available)
            impact_data = self.get_event_impact(event_id, 'all')
            
            if not impact_data:
                raise ValueError("No impact data available")
                
            # Calculate overall effect (average of all impacts)
            lumber_impacts = [d['pct_change'] for d in impact_data if d['sector'].lower() == 'lumber_companies']
            if lumber_impacts:
                overall_effect = sum(lumber_impacts) / len(lumber_impacts)
            else:
                overall_effect = sum(d['pct_change'] for d in impact_data) / len(impact_data)
            
            # Find most affected (most negative)
            most_affected_idx = min(range(len(impact_data)), key=lambda i: impact_data[i]['pct_change'])
            most_affected = f"{impact_data[most_affected_idx]['symbol']}: {impact_data[most_affected_idx]['pct_change']:.2f}%"
            
            # Find least affected (least negative)
            least_affected_idx = max(range(len(impact_data)), key=lambda i: impact_data[i]['pct_change'])
            least_affected = f"{impact_data[least_affected_idx]['symbol']}: {impact_data[least_affected_idx]['pct_change']:.2f}%"
            
            # Get event details
            event_details = self._get_event_details(event_id)
            
            return {
                "overallEffect": overall_effect,
                "mostAffected": most_affected,
                "leastAffected": least_affected,
                "eventDate": event_details.get('date', '2025-03-05'),
                "eventDescription": event_details.get('description', '')
            }
            
        except Exception as e:
            logger.error(f"Error getting event summary: {str(e)}")
            # Return empty summary when database access fails
            return {
                "overallEffect": 0,
                "mostAffected": "N/A",
                "leastAffected": "N/A",
                "eventDate": "",
                "eventDescription": "Unable to retrieve event data"
            }
    
    # Helper methods for data formatting        
    def _normalize_volumes(self, data, event_date):
        """Normalize trading volumes"""
        # Convert date to datetime
        data['date'] = pd.to_datetime(data['date'])
        
        # Calculate days from event
        data['day'] = (data['date'] - event_date).dt.days
        
        # Calculate baseline (average pre-event volume)
        pre_event = data[data['day'] < 0]
        if pre_event.empty:
            # Handle case with no pre-event data
            baseline_volumes = {symbol: 1 for symbol in data['symbol'].unique()}
        else:
            baseline_volumes = pre_event.groupby('symbol')['volume'].mean().to_dict()
        
        # Normalize volumes
        data['normalized_volume'] = data.apply(
            lambda row: (row['volume'] / baseline_volumes.get(row['symbol'], 1)) * 100, 
            axis=1
        )
        
        # Format for visualization (similar to normalize_prices)
        companies = []
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].iloc[0]
            companies.append({
                'symbol': symbol,
                'name': symbol_data['name'],
                'sector': symbol_data['sector'],
                'color': self._get_color(symbol, symbol_data['sector'])
            })
        
        # Get all dates
        dates = sorted(data['date'].unique())
        date_strs = [d.strftime('%Y-%m-%d') for d in dates]
        
        # Create series data
        series = []
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            series_data = []
            
            for _, row in symbol_data.iterrows():
                series_data.append({
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'volume': row['volume'],
                    'normalized_volume': row['normalized_volume'],
                    'day': int(row['day'])
                })
            
            series.append({
                'symbol': symbol,
                'data': sorted(series_data, key=lambda x: x['date'])
            })
        
        return {
            'eventDate': event_date.strftime('%Y-%m-%d'),
            'dates': date_strs,
            'companies': companies,
            'series': series
        }
        
    def _format_heatmap_data(self, data, event_date):
        """Format data for heatmap visualization"""
        # Get unique sectors and dates
        sectors = sorted(data['sector'].unique())
        dates = sorted(data['date'].unique())
        date_strs = [d.strftime('%Y-%m-%d') for d in dates]
        
        # Format data points
        data_points = []
        for _, row in data.iterrows():
            data_points.append({
                'sector': row['sector'],
                'date': row['date'].strftime('%Y-%m-%d'),
                'value': row['pct_change'],
                'day': row['day']
            })
        
        return {
            'sectors': sectors,
            'dates': date_strs,
            'eventDate': event_date.strftime('%Y-%m-%d'),
            'data': data_points
        }
        
    def _get_event_details(self, event_id):
        """Get event details from database or cache"""
        # Special case for 'latest' event_id
        if event_id == 'latest':
            try:
                engine = get_db_connection()
                
                with engine.connect() as conn:
                    query = text("""
                        SELECT * FROM tariff_events 
                        ORDER BY date DESC, id DESC
                        LIMIT 1
                    """)
                    result = conn.execute(query).mappings().first()
                    
                    if result:
                        event = dict(result)
                        # Handle numpy types
                        for key, value in event.items():
                            if isinstance(value, np.integer):
                                event[key] = int(value)
                            elif isinstance(value, np.floating):
                                event[key] = float(value)
                            elif isinstance(value, pd.Timestamp):
                                event[key] = value.strftime('%Y-%m-%d')
                                
                        self.event_cache['latest'] = event
                        return event
            except Exception as e:
                logger.error(f"Error getting latest event: {str(e)}")
                
                # Return empty data if event not found
                empty_data = {
                    'id': 0,
                    'date': '',
                    'title': "No event data available",
                    'description': "Unable to retrieve latest event data",
                    'sectors': []
                }
                
                self.event_cache['latest'] = empty_data
                return empty_data
        
        # Regular event_id handling
        if event_id in self.event_cache:
            return self.event_cache[event_id]
            
        try:
            engine = get_db_connection()
            
            with engine.connect() as conn:
                query = text("SELECT * FROM tariff_events WHERE id = :event_id")
                result = conn.execute(query, {"event_id": event_id}).mappings().first()
                
                if result:
                    event = dict(result)
                    # Handle numpy types
                    for key, value in event.items():
                        if isinstance(value, np.integer):
                            event[key] = int(value)
                        elif isinstance(value, np.floating):
                            event[key] = float(value)
                        elif isinstance(value, pd.Timestamp):
                            event[key] = value.strftime('%Y-%m-%d')
                            
                    self.event_cache[event_id] = event
                    return event
        except Exception as e:
            logger.error(f"Error getting event details: {str(e)}")
            
        # Return empty event data when database access fails
        empty_event = {
            'id': 0,
            'date': '',
            'title': 'No event data available',
            'description': 'Unable to retrieve event details',
            'sectors': []
        }
        
        self.event_cache[event_id] = empty_event
        return empty_event
        
    def _get_color(self, symbol, sector):
        """Get color for a symbol based on sector"""
        color_map = {
            "Lumber_Companies": {
                "WY": "#FF5733",
                "RYN": "#C70039",
                "PCH": "#FFC300",
                "WFG": "#FF5733",
                "WOOD": "#900C3F",
                "CUT": "#581845"
            },
            "Agriculture_Companies": {
                "ADM": "#2E86C1",
                "BG": "#3498DB",
                "CORN": "#85C1E9",
                "SOYB": "#AED6F1",
                "WEAT": "#D6EAF8",
                "DE": "#1ABC9C",
                "CTVA": "#48C9B0"
            }
        }
        
        # Return color if found, otherwise use default colors
        if sector in color_map and symbol in color_map[sector]:
            return color_map[sector][symbol]
        
        # Default colors by sector
        if "lumber" in sector.lower():
            return "#8B4513"  # Brown
        elif "agriculture" in sector.lower():
            return "#228B22"  # Green
        else:
            return "#777777"  # Gray

# Helper class for visualization formatting
class EnhancedVisualizations:
    """Formatting utilities for visualization data"""
    
    def __init__(self):
        # Initialize color map
        self.color_mapping = {
            "Lumber_Companies": {
                "WY": "#FF5733",
                "RYN": "#C70039",
                "PCH": "#FFC300",
                "WFG": "#FF5733",
                "WOOD": "#900C3F",
                "CUT": "#581845"
            },
            "Agriculture_Companies": {
                "ADM": "#2E86C1",
                "BG": "#3498DB",
                "CORN": "#85C1E9",
                "SOYB": "#AED6F1",
                "WEAT": "#D6EAF8",
                "DE": "#1ABC9C",
                "CTVA": "#48C9B0"
            }
        }
    
    def format_price_data(self, price_data):
        """Pass through for price data (already formatted by analyzer)"""
        return price_data
    
    def format_impact_data(self, impact_data):
        """Pass through for impact data (already formatted by analyzer)"""
        return impact_data
    
    def format_volume_data(self, volume_data):
        """Pass through for volume data (already formatted by analyzer)"""
        return volume_data
    
    def format_heatmap_data(self, heatmap_data):
        """Pass through for heatmap data (already formatted by analyzer)"""
        return heatmap_data
        
    def _get_color(self, symbol, sector):
        """Get color for a symbol"""
        if sector in self.color_mapping and symbol in self.color_mapping[sector]:
            return self.color_mapping[sector][symbol]
        
        # Default colors by sector
        if "lumber" in sector.lower():
            return "#8B4513"  # Brown
        elif "agriculture" in sector.lower():
            return "#228B22"  # Green
        else:
            return "#777777"  # Gray

# Initialize analyzer and visualizer
analyzer = TariffImpactAnalyzer()
visualizer = EnhancedVisualizations()

# API Routes
@app.route('/api/events')
def get_all_events():
    """Get all tariff events"""
    try:
        engine = get_db_connection()
        
        with engine.connect() as conn:
            query = text("SELECT * FROM tariff_events ORDER BY date DESC")
            result = conn.execute(query)
            events = [dict(row._mapping) for row in result]
            
            # Handle numpy types
            for event in events:
                for key, value in event.items():
                    if isinstance(value, np.integer):
                        event[key] = int(value)
                    elif isinstance(value, np.floating):
                        event[key] = float(value)
                    elif isinstance(value, pd.Timestamp):
                        event[key] = value.strftime('%Y-%m-%d')
            
            # Mark the latest event
            if events:
                events[0]['is_latest'] = True
            
            logger.info(f"Successfully retrieved {len(events)} tariff events")
            return jsonify(events)
    except Exception as e:
        logger.error(f"Error getting events: {str(e)}")
        
        # Return empty list if database not available
        return jsonify([])

@app.route('/api/filter-options')
def get_filter_options():
    """Get available filter options"""
    try:
        engine = get_db_connection()
        
        with engine.connect() as conn:
            # Get available sectors
            sector_query = text("SELECT DISTINCT sector FROM market_data")
            sectors = [row[0] for row in conn.execute(sector_query)]
            
            # Include 'all' option
            sectors = ['all'] + sectors
            
            # Get available companies
            company_query = text("""
            SELECT DISTINCT symbol, sector FROM market_data
            ORDER BY sector, symbol
            """)
            
            companies = []
            for row in conn.execute(company_query):
                companies.append({
                    "symbol": row[0],
                    "name": row[0],  # Use symbol as name
                    "sector": row[1],
                    "color": visualizer._get_color(row[0], row[1])
                })
            
            # Define time windows
            time_windows = [10, 20, 30, 60, 90]
            
        return jsonify({
            "sectors": sectors,
            "companies": companies,
            "timeWindows": time_windows
        })
    except Exception as e:
        logger.error(f"Error getting filter options: {str(e)}")
        
        # Return default options when database is not available
        default_options = {
            "sectors": ["all"],
            "companies": [],
            "timeWindows": [10, 20, 30, 60, 90]
        }
        
        return jsonify(default_options)

@app.route('/api/price-data', methods=['GET'])
def get_price_data():
    """Get normalized price data for the price trend chart"""
    try:
        event_id = request.args.get('event_id', '1')
        sector = request.args.get('sector', 'all')
        days = int(request.args.get('days', 30))
        
        # Use analyzer to get normalized price data
        price_data = analyzer.get_normalized_prices(event_id, sector, days)
        
        # Format for visualization
        result = visualizer.format_price_data(price_data)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting price data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/impact-data', methods=['GET'])
def get_impact_data():
    """Get impact data for the differential impact chart"""
    try:
        event_id = request.args.get('event_id', '1')
        sector = request.args.get('sector', 'all')
        
        # Get impact analysis from the analyzer
        impact_data = analyzer.get_event_impact(event_id, sector)
        
        # Format for visualization
        result = visualizer.format_impact_data(impact_data)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting impact data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/volume-data', methods=['GET'])
def get_volume_data():
    """Get trading volume data"""
    try:
        event_id = request.args.get('event_id', '1')
        sector = request.args.get('sector', 'all')
        days = int(request.args.get('days', 30))
        
        # Get volume data
        volume_data = analyzer.get_trading_volumes(event_id, sector, days)
        
        # Format for visualization
        result = visualizer.format_volume_data(volume_data)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting volume data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/heatmap-data', methods=['GET'])
def get_heatmap_data():
    """Get sector impact heatmap data"""
    try:
        event_id = request.args.get('event_id', '1')
        sector = request.args.get('sector', 'all')
        
        # Get heatmap data from analyzer
        heatmap_data = analyzer.get_sector_heatmap(event_id, sector)
        
        # If a specific sector is requested, filter the sectors list
        if sector != 'all':
            # Filter sectors list to only include the selected sector
            heatmap_data['sectors'] = [s for s in heatmap_data['sectors'] if s == sector]
            # Filter data points to only include the selected sector
            heatmap_data['data'] = [d for d in heatmap_data['data'] if d['sector'] == sector]
        
        # Format for visualization
        result = visualizer.format_heatmap_data(heatmap_data)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting heatmap data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/event-summary', methods=['GET'])
def get_event_summary():
    """Get summary metrics for an event"""
    try:
        event_id = request.args.get('event_id', '1')
        
        # Get summary from analyzer
        summary = analyzer.get_event_summary(event_id)
        
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error getting event summary: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/impact')
def get_impact():
    """Get impact analysis results (old endpoint)"""
    try:
        event_id = request.args.get('event_id')
        sector = request.args.get('sector', 'all')  # Add sector parameter with default 'all'
        
        engine = get_db_connection()
        
        # Get event date for reference
        with engine.connect() as conn:
            event_query = text("SELECT date FROM tariff_events WHERE id = :event_id")
            event_date = conn.execute(event_query, {"event_id": event_id}).scalar()
            
        if not event_date:
            return jsonify({"error": "Event not found"}), 404
            
        # Calculate impact for each symbol
        event_date = pd.to_datetime(event_date)
        start_date = (event_date - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
        end_date = (event_date + pd.Timedelta(days=10)).strftime('%Y-%m-%d')
        
        # Build the query with potential sector filter
        query = """
        SELECT date, symbol, sector, close, volume
        FROM market_data
        WHERE date BETWEEN :start_date AND :end_date
        """
        
        # Add sector filter if specified
        if sector and sector != 'all':
            query += f" AND sector = '{sector}'"
            
        query += " ORDER BY symbol, date"
        
        # Get market data - updated to match database schema
        with engine.connect() as conn:
            data_query = text(query)
            
            data = pd.read_sql(data_query, conn, params={
                "start_date": start_date,
                "end_date": end_date
            })
            
        data['date'] = pd.to_datetime(data['date'])
        
        # Calculate impacts manually
        impacts = []
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            
            # Pre and post event
            pre_event = symbol_data[symbol_data['date'] < event_date]
            post_event = symbol_data[symbol_data['date'] >= event_date]
            
            if len(pre_event) < 3 or len(post_event) < 3:
                continue
                
            # Get sector
            sector = symbol_data['sector'].iloc[0]
            
            # Calculate average price change
            pre_price = pre_event['close'].iloc[0]
            post_price = post_event['close'].iloc[-1]
            price_change = ((post_price / pre_price) - 1) * 100
            
            # Calculate average volume change
            pre_volume = pre_event['volume'].mean()
            post_volume = post_event['volume'].mean()
            volume_change = ((post_volume / pre_volume) - 1) * 100
            
            impacts.append({
                'symbol': symbol,
                'name': symbol,  # Add name field using symbol
                'sector': sector,
                'price_change': float(price_change),
                'volume_change': float(volume_change)
            })
            
        return jsonify(impacts)
    except Exception as e:
        logger.error(f"Error calculating impact: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
# Route to get latest tariff event
@app.route('/api/latest-tariff-event', methods=['GET'])
def get_latest_tariff_event():
    """Get the most recent tariff event for automatic dashboard updates"""
    try:
        engine = get_db_connection()
        
        with engine.connect() as conn:
            # Query to get the most recent tariff event
            query = text("""
                SELECT id, title, date, description, affected_sectors, source
                FROM tariff_events
                ORDER BY date DESC, id DESC
                LIMIT 1
            """)
            
            result = conn.execute(query).mappings().first()
            
            if result:
                event = dict(result)
                # Handle numpy types
                for key, value in event.items():
                    if isinstance(value, np.integer):
                        event[key] = int(value)
                    elif isinstance(value, np.floating):
                        event[key] = float(value)
                    elif isinstance(value, pd.Timestamp):
                        event[key] = value.strftime('%Y-%m-%d')
                
                # Format the event response
                formatted_event = {
                    'id': event['id'],
                    'title': event['title'],
                    'date': event['date'],
                    'description': event.get('description', ''),
                    'affected_sectors': event.get('affected_sectors', []),
                    'source': event.get('source', 'Unknown'),
                    'is_latest': True
                }
                
                return jsonify(formatted_event)
            else:
                return jsonify({"error": "No tariff events found"}), 404
    except Exception as e:
        logger.error(f"Error getting latest tariff event: {str(e)}")
        
        # Return empty response if database access fails
        return jsonify({"error": "Unable to retrieve tariff event data"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)