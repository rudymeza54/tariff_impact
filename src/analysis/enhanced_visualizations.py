# enhanced_visualizations.py
import os
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import text
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set styling for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
sns.set_context("paper", font_scale=1.2)

def get_db_connection():
    """Create a database connection using environment variables"""
    db_host = os.environ.get('DB_HOST', 'localhost')
    db_user = os.environ.get('DB_USER', 'postgres')
    db_pass = os.environ.get('DB_PASSWORD', 'postgres')
    db_name = os.environ.get('DB_NAME', 'postgres')
    db_port = os.environ.get('DB_PORT', '5431')
    
    connection_string = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    engine = sqlalchemy.create_engine(connection_string)
    return engine

def get_tariff_event(event_id):
    """Get a specific tariff event from the database"""
    engine = get_db_connection()
    
    with engine.connect() as conn:
        query = text("SELECT * FROM tariff_events WHERE id = :event_id")
        result = conn.execute(query, {"event_id": event_id})
        event = result.fetchone()
        
    if event:
        return dict(event._mapping)
    return None

def get_market_data(start_date=None, end_date=None, symbols=None):
    """Get market data for visualization"""
    engine = get_db_connection()
    
    query = """
    SELECT date, symbol, sector, close, volume
    FROM market_data
    WHERE 1=1
    """
    
    params = {}
    
    if start_date:
        query += " AND date >= :start_date"
        params['start_date'] = start_date
        
    if end_date:
        query += " AND date <= :end_date"
        params['end_date'] = end_date
        
    if symbols:
        placeholders = [f":symbol_{i}" for i in range(len(symbols))]
        query += f" AND symbol IN ({','.join(placeholders)})"
        for i, symbol in enumerate(symbols):
            params[f'symbol_{i}'] = symbol
            
    query += " ORDER BY symbol, date"
    
    # Execute query
    with engine.connect() as conn:
        data = pd.read_sql(text(query), conn, params=params)
    
    return data

def plot_price_trends(event_id, window_days=30, output_dir='visualizations'):
    """
    Create plot showing price trends before and after a tariff event
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get event details
    event = get_tariff_event(event_id)
    if not event:
        logger.error(f"Event with ID {event_id} not found")
        return
    
    event_date = pd.to_datetime(event['date'])
    event_title = event['title']
    
    # Get price data around the event
    start_date = (event_date - pd.Timedelta(days=window_days)).strftime('%Y-%m-%d')
    end_date = (event_date + pd.Timedelta(days=window_days)).strftime('%Y-%m-%d')
    
    # Look for lumber companies and ETFs
    data = get_market_data(start_date, end_date)
    
    # Get unique symbols and filter for relevant sectors
    lumber_sectors = ['Lumber_Companies', 'Lumber_ETFs']
    lumber_symbols = data[data['sector'].isin(lumber_sectors)]['symbol'].unique().tolist()
    
    # Get agriculture symbols for comparison
    ag_sectors = ['Agriculture_Companies', 'Agriculture_ETFs']
    ag_symbols = data[data['sector'].isin(ag_sectors)]['symbol'].unique().tolist()
    
    # Normalize prices for each symbol
    companies_to_plot = lumber_symbols[:4]  # Limit to 4 lumber companies for readability
    
    # Create DataFrame for plotting
    plot_data = data[data['symbol'].isin(companies_to_plot + [ag_symbols[0]])].copy()
    plot_data['date'] = pd.to_datetime(plot_data['date'])
    
    # Create normalized prices
    normalized_data = []
    
    for symbol in companies_to_plot + [ag_symbols[0]]:
        symbol_data = plot_data[plot_data['symbol'] == symbol].copy()
        if len(symbol_data) == 0:
            continue
            
        # Get the base price (day before event)
        base_date = event_date - pd.Timedelta(days=1)
        base_price = symbol_data[symbol_data['date'] <= base_date]['close'].iloc[-1]
        
        # Normalize
        symbol_data['normalized_price'] = (symbol_data['close'] / base_price) * 100
        symbol_data['days_from_event'] = (symbol_data['date'] - event_date).dt.days
        
        normalized_data.append(symbol_data)
    
    if not normalized_data:
        logger.error("No data available for plotting")
        return
        
    normalized_df = pd.concat(normalized_data)
    
    # Plot normalized prices
    plt.figure(figsize=(12, 7))
    
    # Plot each lumber company
    for symbol in companies_to_plot:
        symbol_data = normalized_df[normalized_df['symbol'] == symbol]
        if len(symbol_data) > 0:
            plt.plot('days_from_event', 'normalized_price', 
                    data=symbol_data, label=symbol, linewidth=2)
    
    # Plot one ag company for comparison
    ag_symbol = ag_symbols[0]
    ag_data = normalized_df[normalized_df['symbol'] == ag_symbol]
    if len(ag_data) > 0:
        plt.plot('days_from_event', 'normalized_price', 
                data=ag_data, label=f"{ag_symbol} (Control)", 
                linestyle='--', color='gray', linewidth=2)
    
    # Add event line
    plt.axvline(x=0, color='red', linestyle='-', alpha=0.7, label='Tariff Event')
    
    # Add shading for after event
    plt.axvspan(0, normalized_df['days_from_event'].max(), alpha=0.1, color='red')
    
    # Add labels and title
    plt.title(f"Price Impact of {event_title}")
    plt.xlabel("Days from Event")
    plt.ylabel("Normalized Price (Base = 100)")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"event_{event_id}_price_trends.png"))
    plt.close()
    
    logger.info(f"Price trends plot saved to {output_dir}/event_{event_id}_price_trends.png")

def plot_impact_heatmap(event_id, output_dir='visualizations'):
    """
    Create a heatmap showing impact of tariff across different companies
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get event details
    event = get_tariff_event(event_id)
    if not event:
        logger.error(f"Event with ID {event_id} not found")
        return
    
    event_date = pd.to_datetime(event['date'])
    event_title = event['title']
    
    # Get price data around the event
    start_date = (event_date - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
    end_date = (event_date + pd.Timedelta(days=10)).strftime('%Y-%m-%d')
    
    # Look for lumber companies and ETFs
    data = get_market_data(start_date, end_date)
    data['date'] = pd.to_datetime(data['date'])
    
    # Manually calculate impacts for each symbol (5 days before vs 5 days after)
    impacts = []
    
    for sector in ['Lumber_Companies', 'Lumber_ETFs', 'Agriculture_Companies', 'Agriculture_ETFs']:
        sector_data = data[data['sector'] == sector]
        
        for symbol in sector_data['symbol'].unique():
            symbol_data = sector_data[sector_data['symbol'] == symbol]
            
            # Pre and post event data
            pre_event = symbol_data[symbol_data['date'] < event_date]
            post_event = symbol_data[symbol_data['date'] >= event_date]
            
            if len(pre_event) < 3 or len(post_event) < 3:
                continue
                
            # Calculate returns
            pre_event['return'] = pre_event['close'].pct_change() * 100
            post_event['return'] = post_event['close'].pct_change() * 100
            
            # Average returns
            pre_return = pre_event['return'].mean()
            post_return = post_event['return'].mean()
            
            # Calculate impact
            impact = post_return - pre_return
            
            # Calculate cumulative price change
            pre_price = pre_event['close'].iloc[0]
            post_price = post_event['close'].iloc[-1]
            price_change = ((post_price / pre_price) - 1) * 100
            
            # Calculate volume changes
            pre_volume = pre_event['volume'].mean()
            post_volume = post_event['volume'].mean()
            volume_change = ((post_volume / pre_volume) - 1) * 100
            
            impacts.append({
                'symbol': symbol,
                'sector': sector,
                'impact': impact,
                'price_change': price_change,
                'volume_change': volume_change
            })
    
    if not impacts:
        logger.error("No impact data available for heatmap")
        return
        
    impact_df = pd.DataFrame(impacts)
    
    # Create a pivot table for the heatmap
    pivot_impact = impact_df.pivot(index='symbol', columns='sector', values='impact')
    pivot_impact = pivot_impact.fillna(0)  # Replace NaNs with 0
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    
    # Custom colormap (red for negative, green for positive)
    cmap = sns.diverging_palette(10, 120, as_cmap=True)
    
    # Create heatmap
    ax = sns.heatmap(pivot_impact, cmap=cmap, center=0, 
                   linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    # Add labels and title
    plt.title(f"Tariff Impact by Company and Sector\n{event_title}")
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"event_{event_id}_impact_heatmap.png"))
    plt.close()
    
    logger.info(f"Impact heatmap saved to {output_dir}/event_{event_id}_impact_heatmap.png")



def plot_year_over_year_comparison(event_id, output_dir='visualizations'):
    """
    Create plot comparing March 2024 vs March 2025 data to show long-term impact
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get event details
    event = get_tariff_event(event_id)
    if not event:
        logger.error(f"Event with ID {event_id} not found")
        return
    
    event_date = pd.to_datetime(event['date'])
    event_title = event['title']
    
    # Calculate date ranges for 2024 and 2025
    start_date_2024 = "2024-03-01"
    end_date_2024 = "2024-03-31"
    start_date_2025 = "2025-03-01"
    end_date_2025 = "2025-03-31"
    
    # Get market data for both periods
    data_2024 = get_market_data(start_date=start_date_2024, end_date=end_date_2024)
    data_2025 = get_market_data(start_date=start_date_2025, end_date=end_date_2025)
    
    # Check if we have data for both periods
    if data_2024.empty or data_2025.empty:
        logger.error("Insufficient data for year-over-year comparison")
        return
    
    # Convert dates to datetime
    data_2024['date'] = pd.to_datetime(data_2024['date'])
    data_2025['date'] = pd.to_datetime(data_2025['date'])
    
    # Get lumber symbols
    lumber_sectors = ['Lumber_Companies', 'Lumber_ETFs']
    lumber_symbols = list(set(data_2024[data_2024['sector'].isin(lumber_sectors)]['symbol']).intersection(
                        set(data_2025[data_2025['sector'].isin(lumber_sectors)]['symbol'])))
    
    # Get agriculture symbols for comparison
    ag_sectors = ['Agriculture_Companies', 'Agriculture_ETFs']
    ag_symbols = list(set(data_2024[data_2024['sector'].isin(ag_sectors)]['symbol']).intersection(
                    set(data_2025[data_2025['sector'].isin(ag_sectors)]['symbol'])))
    
    # Calculate average prices by sector and date
    def calculate_sector_averages(df, sectors):
        sector_data = df[df['sector'].isin(sectors)].copy()
        # Normalize each symbol's price series to 100 at the first date
        symbols = sector_data['symbol'].unique()
        normalized_data = []
        
        for symbol in symbols:
            symbol_data = sector_data[sector_data['symbol'] == symbol].copy()
            if len(symbol_data) > 0:
                base_price = symbol_data.iloc[0]['close']
                symbol_data['normalized_price'] = (symbol_data['close'] / base_price) * 100
                normalized_data.append(symbol_data)
        
        if not normalized_data:
            return pd.DataFrame()
            
        # Combine all normalized prices
        combined = pd.concat(normalized_data)
        
        # Calculate average by date
        avg_by_date = combined.groupby('date')['normalized_price'].mean().reset_index()
        return avg_by_date
    
    # Calculate sector averages
    lumber_avg_2024 = calculate_sector_averages(data_2024, lumber_sectors)
    lumber_avg_2025 = calculate_sector_averages(data_2025, lumber_sectors)
    ag_avg_2024 = calculate_sector_averages(data_2024, ag_sectors)
    ag_avg_2025 = calculate_sector_averages(data_2025, ag_sectors)
    
    # Convert dates to days in March for comparison
    lumber_avg_2024['day'] = lumber_avg_2024['date'].dt.day
    lumber_avg_2025['day'] = lumber_avg_2025['date'].dt.day
    ag_avg_2024['day'] = ag_avg_2024['date'].dt.day
    ag_avg_2025['day'] = ag_avg_2025['date'].dt.day
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    # Plot each sector-year combination
    if not lumber_avg_2024.empty:
        plt.plot(lumber_avg_2024['day'], lumber_avg_2024['normalized_price'], 
                label='Lumber 2024', color='blue', linewidth=2)
    
    if not lumber_avg_2025.empty:
        plt.plot(lumber_avg_2025['day'], lumber_avg_2025['normalized_price'], 
                label='Lumber 2025', color='blue', linewidth=2, linestyle='--')
    
    if not ag_avg_2024.empty:
        plt.plot(ag_avg_2024['day'], ag_avg_2024['normalized_price'], 
                label='Agriculture 2024', color='green', linewidth=2)
    
    if not ag_avg_2025.empty:
        plt.plot(ag_avg_2025['day'], ag_avg_2025['normalized_price'], 
                label='Agriculture 2025', color='green', linewidth=2, linestyle='--')
    
    # Add event line
    event_day = event_date.day
    plt.axvline(x=event_day, color='red', linestyle='-', 
               label=f'Tariff Event (March {event_day}, 2025)', alpha=0.7)
    
    # Add labels and title
    plt.title(f"Year-over-Year Comparison: March 2024 vs 2025\nImpact of {event_title}")
    plt.xlabel("Day of March")
    plt.ylabel("Normalized Price (Base = 100)")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks to show days of March
    plt.xticks(range(1, 32, 2))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"event_{event_id}_year_over_year.png"))
    plt.close()
    
    logger.info(f"Year-over-year comparison saved to {output_dir}/event_{event_id}_year_over_year.png")

def plot_volume_analysis(event_id, window_days=10, output_dir='visualizations'):
    """
    Create plot showing trading volume changes around the tariff event
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get event details
    event = get_tariff_event(event_id)
    if not event:
        logger.error(f"Event with ID {event_id} not found")
        return
    
    event_date = pd.to_datetime(event['date'])
    event_title = event['title']
    
    # Get price data around the event
    start_date = (event_date - pd.Timedelta(days=window_days)).strftime('%Y-%m-%d')
    end_date = (event_date + pd.Timedelta(days=window_days)).strftime('%Y-%m-%d')
    
    # Look for lumber companies and ETFs
    data = get_market_data(start_date, end_date)
    data['date'] = pd.to_datetime(data['date'])
    
    # Get lumber sectors
    lumber_sectors = ['Lumber_Companies', 'Lumber_ETFs']
    lumber_data = data[data['sector'].isin(lumber_sectors)].copy()
    
    # Calculate average daily volume by sector
    lumber_data['days_from_event'] = (lumber_data['date'] - event_date).dt.days
    
    # Group by sector and day
    volume_by_day = lumber_data.groupby(['sector', 'days_from_event'])['volume'].mean().reset_index()
    
    # Create the volume plot
    plt.figure(figsize=(12, 6))
    
    # Plot each sector
    for sector in volume_by_day['sector'].unique():
        sector_data = volume_by_day[volume_by_day['sector'] == sector]
        plt.plot('days_from_event', 'volume', data=sector_data, label=sector, marker='o')
    
    # Add event line
    plt.axvline(x=0, color='red', linestyle='-', alpha=0.7, label='Tariff Event')
    
    # Add labels and title
    plt.title(f"Trading Volume Around {event_title}")
    plt.xlabel("Days from Event")
    plt.ylabel("Average Trading Volume")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Format y-axis with thousands separators
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"event_{event_id}_volume_analysis.png"))
    plt.close()
    
    logger.info(f"Volume analysis plot saved to {output_dir}/event_{event_id}_volume_analysis.png")

def plot_comparative_impact(event_id, output_dir='visualizations'):
    """
    Create bar chart comparing impact across different lumber companies
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get event details
    event = get_tariff_event(event_id)
    if not event:
        logger.error(f"Event with ID {event_id} not found")
        return
    
    event_date = pd.to_datetime(event['date'])
    event_title = event['title']
    
    # Get price data around the event
    start_date = (event_date - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
    end_date = (event_date + pd.Timedelta(days=10)).strftime('%Y-%m-%d')
    
    # Get market data
    data = get_market_data(start_date, end_date)
    data['date'] = pd.to_datetime(data['date'])
    
    # Get lumber and control companies
    lumber_sectors = ['Lumber_Companies', 'Lumber_ETFs']
    control_sectors = ['Agriculture_Companies', 'Agriculture_ETFs']
    
    lumber_data = data[data['sector'].isin(lumber_sectors)].copy()
    control_data = data[data['sector'].isin(control_sectors)].copy()
    
    # Calculate 5-day price changes post event
    def calculate_price_change(df, event_date):
        results = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            
            # Get pre-event price (day before event or closest)
            pre_data = symbol_data[symbol_data['date'] < event_date].sort_values('date')
            if len(pre_data) == 0:
                continue
            pre_price = pre_data.iloc[-1]['close']
            
            # Get post-event prices (up to 5 days after)
            post_data = symbol_data[
                (symbol_data['date'] >= event_date) & 
                (symbol_data['date'] <= event_date + pd.Timedelta(days=5))
            ].sort_values('date')
            
            if len(post_data) == 0:
                continue
                
            # Get last available price in the window
            post_price = post_data.iloc[-1]['close']
            
            # Calculate price change
            price_change = ((post_price / pre_price) - 1) * 100
            
            # Get the sector
            sector = symbol_data['sector'].iloc[0]
            
            results.append({
                'symbol': symbol,
                'sector': sector,
                'price_change': price_change
            })
            
        return pd.DataFrame(results)
    
    # Calculate price changes
    lumber_changes = calculate_price_change(lumber_data, event_date)
    control_changes = calculate_price_change(control_data, event_date)
    
    # Calculate average control change
    avg_control_change = control_changes['price_change'].mean()
    
    # Calculate differential impact (vs control average)
    lumber_changes['differential_impact'] = lumber_changes['price_change'] - avg_control_change
    
    # Sort by differential impact
    lumber_changes = lumber_changes.sort_values('differential_impact')
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    
    # Plot bars
    bars = plt.barh(lumber_changes['symbol'], lumber_changes['differential_impact'])
    
    # Color bars based on positive/negative impact
    for i, bar in enumerate(bars):
        if lumber_changes['differential_impact'].iloc[i] < 0:
            bar.set_color('red')
        else:
            bar.set_color('green')
    
    # Add vertical line at 0
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add labels and title
    plt.title(f"Impact of {event_title} Relative to Agricultural Sector")
    plt.xlabel("Differential Price Change (%)")
    plt.ylabel("Lumber Companies and ETFs")
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add average control line text
    plt.text(max(lumber_changes['differential_impact'])*0.8, 
            0.1, 
            f"Control Sector Change: {avg_control_change:.2f}%", 
            transform=plt.gca().transAxes, 
            ha='center', 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"event_{event_id}_comparative_impact.png"))
    plt.close()
    
    logger.info(f"Comparative impact plot saved to {output_dir}/event_{event_id}_comparative_impact.png")

def create_all_visualizations(event_id=10):
    """Create all visualizations for a given event"""
    logger.info(f"Creating enhanced visualizations for event ID {event_id}")
    
    # Create price trends plot
    plot_price_trends(event_id)
    
    # Create impact heatmap
    plot_impact_heatmap(event_id)
    
    # Create volume analysis
    plot_volume_analysis(event_id)
    
    # Create comparative impact chart
    plot_comparative_impact(event_id)
    
    # Create year-over-year comparison
    plot_year_over_year_comparison(event_id)
    
    logger.info("All visualizations created successfully")

if __name__ == "__main__":
    # Default to the March 5, 2025 lumber tariff (event ID 10)
    create_all_visualizations(10)