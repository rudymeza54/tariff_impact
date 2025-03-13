#!/usr/bin/env python3
"""
Tariff Impact Analyzer
Uses causal inference methods to analyze the impact of tariff events on agriculture and lumber markets
"""

import os
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import text
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def get_tariff_events(sector: Optional[str] = None, start_date: Optional[str] = None, 
                      end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch tariff events from the database with optional filtering
    
    Args:
        sector: Filter by affected sector (Agriculture or Lumber)
        start_date: Filter events after this date (YYYY-MM-DD)
        end_date: Filter events before this date (YYYY-MM-DD)
    
    Returns:
        DataFrame of tariff events
    """
    engine = get_db_connection()
    
    # Build query with potential filters
    query = """
    SELECT id, title, date, source, event_type, affected_sectors
    FROM tariff_events
    WHERE 1=1
    """
    
    params = {}
    
    if sector:
        query += " AND :sector = ANY(affected_sectors)"
        params['sector'] = sector
        
    if start_date:
        query += " AND date >= :start_date"
        params['start_date'] = start_date
        
    if end_date:
        query += " AND date <= :end_date"
        params['end_date'] = end_date
    
    query += " ORDER BY date"
    
    # Execute query
    with engine.connect() as conn:
        events = pd.read_sql(text(query), conn, params=params)
    
    return events

def get_market_data(symbols: Optional[List[str]] = None, sectors: Optional[List[str]] = None,
                   start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch market data from the database with optional filtering
    
    Args:
        symbols: List of ticker symbols to include
        sectors: List of sectors to include
        start_date: Filter data after this date (YYYY-MM-DD)
        end_date: Filter data before this date (YYYY-MM-DD)
    
    Returns:
        DataFrame of market data
    """
    engine = get_db_connection()
    
    # Build query with potential filters
    query = """
    SELECT date, symbol, sector, open, high, low, close, volume, data_type
    FROM market_data
    WHERE 1=1
    """
    
    params = {}
    
    if symbols:
        placeholders = [f":symbol_{i}" for i in range(len(symbols))]
        query += f" AND symbol IN ({','.join(placeholders)})"
        for i, symbol in enumerate(symbols):
            params[f'symbol_{i}'] = symbol
            
    if sectors:
        placeholders = [f":sector_{i}" for i in range(len(sectors))]
        query += f" AND sector IN ({','.join(placeholders)})"
        for i, sector in enumerate(sectors):
            params[f'sector_{i}'] = sector
        
    if start_date:
        query += " AND date >= :start_date"
        params['start_date'] = start_date
        
    if end_date:
        query += " AND date <= :end_date"
        params['end_date'] = end_date
    
    query += " ORDER BY symbol, date"
    
    # Execute query
    with engine.connect() as conn:
        data = pd.read_sql(text(query), conn, params=params)
    
    return data

def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns for market data
    
    Args:
        data: DataFrame of market data with 'date', 'symbol', and 'close' columns
    
    Returns:
        DataFrame with daily returns added
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Calculate returns by symbol
    df['return'] = df.groupby('symbol')['close'].pct_change() * 100
    
    return df

def difference_in_differences(data: pd.DataFrame, event_date: str, 
                           treated_group: List[str], control_group: List[str],
                           pre_days: int = 5, post_days: int = 5) -> Dict:
    """
    Perform difference-in-differences analysis to measure tariff impact
    
    Args:
        data: DataFrame with market data including returns
        event_date: The date of the tariff event (YYYY-MM-DD)
        treated_group: List of symbols directly affected by the tariff
        control_group: List of symbols not directly affected (comparison group)
        pre_days: Number of days before event for pre-period
        post_days: Number of days after event for post-period
    
    Returns:
        Dictionary with DiD results
    """
    # Convert event_date to pandas datetime if it's not already
    if not isinstance(event_date, pd.Timestamp):
        event_date = pd.to_datetime(event_date)
    
    # Create period and treatment indicators
    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Define pre and post periods
    pre_start = event_date - pd.Timedelta(days=pre_days)
    pre_end = event_date - pd.Timedelta(days=1)
    post_start = event_date
    post_end = event_date + pd.Timedelta(days=post_days)
    
    # Filter to relevant time period and groups
    mask = (
        ((df['date'] >= pre_start) & (df['date'] <= pre_end)) | 
        ((df['date'] >= post_start) & (df['date'] <= post_end))
    )
    mask = mask & (df['symbol'].isin(treated_group + control_group))
    df = df[mask].copy()
    
    # Create treatment and post indicators
    df['treated'] = df['symbol'].isin(treated_group)
    df['post'] = df['date'] >= event_date
    
    # Prepare for DiD regression
    df['treated_post'] = df['treated'] * df['post']  # Interaction term
    
    # Run DiD regression - Fixed formula syntax
    try:
        # Use a direct model creation instead of the formula API
        y = df['return']
        X = pd.DataFrame({
            'intercept': 1,
            'treated': df['treated'],
            'post': df['post'],
            'treated_post': df['treated_post']
        })
        model = sm.OLS(y, X).fit(cov_type='HC3')
        
        # Extract key statistics
        did_effect = model.params['treated_post']
        did_pvalue = model.pvalues['treated_post']
    except Exception as e:
        logger.error(f"Error in DiD regression: {str(e)}")
        # Handle regression failure with fallback to manual calculation
        grouped = df.groupby(['treated', 'post'])['return'].mean().reset_index()
        result_table = grouped.pivot(index='treated', columns='post', values='return').reset_index()
        result_table.columns = ['treated', 'pre', 'post']
        result_table['diff'] = result_table['post'] - result_table['pre']
        
        treated_diff = result_table.loc[result_table['treated'] == True, 'diff'].values[0]
        control_diff = result_table.loc[result_table['treated'] == False, 'diff'].values[0]
        did_effect = treated_diff - control_diff
        did_pvalue = 0.5  # Default p-value when regression fails
        
        # Create a simple model object for consistent return structure
        model = None
    
    # Calculate group means for verification
    grouped = df.groupby(['treated', 'post'])['return'].mean().reset_index()
    
    # Convert to easily interpretable format
    result_table = grouped.pivot(index='treated', columns='post', values='return').reset_index()
    result_table.columns = ['treated', 'pre', 'post']
    result_table['diff'] = result_table['post'] - result_table['pre']
    
    # Calculate the difference-in-differences manually to verify
    treated_diff = result_table.loc[result_table['treated'] == True, 'diff'].values[0]
    control_diff = result_table.loc[result_table['treated'] == False, 'diff'].values[0]
    manual_did = treated_diff - control_diff
    
    # Prepare results
    results = {
        'model': model,
        'summary': model.summary() if model is not None else None,
        'did_effect': did_effect,
        'did_pvalue': did_pvalue,
        'treated_pre': result_table.loc[result_table['treated'] == True, 'pre'].values[0],
        'treated_post': result_table.loc[result_table['treated'] == True, 'post'].values[0],
        'control_pre': result_table.loc[result_table['treated'] == False, 'pre'].values[0],
        'control_post': result_table.loc[result_table['treated'] == False, 'post'].values[0],
        'treated_diff': treated_diff,
        'control_diff': control_diff,
        'manual_did': manual_did,
        'event_date': event_date,
        'treated_group': treated_group,
        'control_group': control_group
    }
    
    return results

def event_study(data: pd.DataFrame, event_date: str, symbols: List[str], 
               estimation_window: int = 20, event_window: int = 5) -> Dict:
    """
    Perform an event study to measure abnormal returns around a tariff event
    
    Args:
        data: DataFrame with market data including returns
        event_date: The date of the tariff event (YYYY-MM-DD)
        symbols: List of symbols to analyze
        estimation_window: Number of days before event for market model estimation
        event_window: Number of days after event for measuring impact
    
    Returns:
        Dictionary with event study results
    """
    # Convert event_date to pandas datetime if it's not already
    if not isinstance(event_date, pd.Timestamp):
        event_date = pd.to_datetime(event_date)
    
    # Create a copy and ensure date is datetime
    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Define windows
    estimation_start = event_date - pd.Timedelta(days=estimation_window)
    estimation_end = event_date - pd.Timedelta(days=1)
    event_end = event_date + pd.Timedelta(days=event_window)
    
    # Rest of your code...
    
    # Filter to relevant symbols
    df = df[df['symbol'].isin(symbols)].copy()
    
    # Create market return (simple average of all symbols as proxy)
    market_returns = df.groupby('date')['return'].mean().reset_index()
    market_returns.columns = ['date', 'market_return']
    
    # Merge market returns back
    df = pd.merge(df, market_returns, on='date', how='left')
    
    # Results container
    results = {
        'event_date': event_date,
        'symbols': symbols,
        'abnormal_returns': {},
        'cumulative_abnormal_returns': {},
        'statistical_tests': {}
    }
    
    # Analyze each symbol
    for symbol in symbols:
        symbol_data = df[df['symbol'] == symbol].copy()
        
        # Split into estimation and event periods
        estimation_data = symbol_data[(symbol_data['date'] >= estimation_start) & 
                                    (symbol_data['date'] <= estimation_end)]
        event_data = symbol_data[(symbol_data['date'] >= event_date) & 
                               (symbol_data['date'] <= event_end)]
        
        if len(estimation_data) < 10:  # Need enough data for regression
            logger.warning(f"Insufficient data for {symbol} estimation window")
            continue
            
        # Estimate market model: R_i = alpha + beta * R_m + epsilon
        model = sm.OLS(estimation_data['return'], 
                      sm.add_constant(estimation_data['market_return'])).fit()
        
        # Predict expected returns during event window
        event_data['expected_return'] = model.predict(sm.add_constant(event_data['market_return']))
        
        # Calculate abnormal returns
        event_data['abnormal_return'] = event_data['return'] - event_data['expected_return']
        
        # Calculate cumulative abnormal return
        event_data['car'] = event_data['abnormal_return'].cumsum()
        
        # Store results
        results['abnormal_returns'][symbol] = event_data[['date', 'abnormal_return']].set_index('date')
        results['cumulative_abnormal_returns'][symbol] = event_data[['date', 'car']].set_index('date')
        
        # Statistical test for significance
        ar_mean = event_data['abnormal_return'].mean()
        ar_std = event_data['abnormal_return'].std()
        ar_t = ar_mean / (ar_std / np.sqrt(len(event_data)))
        ar_p = 2 * (1 - stats.t.cdf(abs(ar_t), len(event_data) - 1))
        
        results['statistical_tests'][symbol] = {
            'mean_abnormal_return': ar_mean,
            'cumulative_abnormal_return': event_data['car'].iloc[-1],
            't_statistic': ar_t,
            'p_value': ar_p,
            'significant': ar_p < 0.05
        }
    
    return results

def synthetic_control(data: pd.DataFrame, event_date: str, 
                     treated_unit: str, control_pool: List[str],
                     pre_days: int = 20, post_days: int = 10) -> Dict:
    """
    Apply synthetic control method to estimate causal effect of tariff event
    
    Args:
        data: DataFrame with market data including returns
        event_date: The date of the tariff event (YYYY-MM-DD)
        treated_unit: Symbol directly affected by the tariff
        control_pool: List of symbols to use as potential controls
        pre_days: Number of days before event for training
        post_days: Number of days after event for measuring impact
    
    Returns:
        Dictionary with synthetic control results
    """
    # Convert event_date to pandas datetime if it's not already
    if not isinstance(event_date, pd.Timestamp):
        event_date = pd.to_datetime(event_date)
    
    # Create a copy and ensure date is datetime
    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Define periods
    pre_start = event_date - pd.Timedelta(days=pre_days)
    pre_end = event_date - pd.Timedelta(days=1)
    post_start = event_date
    post_end = event_date + pd.Timedelta(days=post_days)
    
    # Filter to relevant time period and units
    all_units = [treated_unit] + control_pool
    mask = (
        ((df['date'] >= pre_start) & (df['date'] <= pre_end)) | 
        ((df['date'] >= post_start) & (df['date'] <= post_end))
    )
    mask = mask & (df['symbol'].isin(all_units))
    df = df[mask].copy()
    
    # Create a period indicator
    df['period'] = 'pre'
    df.loc[df['date'] >= event_date, 'period'] = 'post'
    
    # Rest of your function remains the same...
    
    # Extract pre-treatment data
    pre_data = df[df['period'] == 'pre'].pivot(index='date', columns='symbol', values='close')
    
    # Check if we have enough data for all units
    valid_controls = []
    for control in control_pool:
        if control in pre_data.columns and pre_data[control].isnull().sum() == 0:
            valid_controls.append(control)
    
    if not valid_controls:
        logger.error(f"No valid control units with complete data")
        return None
    
    # Standardize the data to make units comparable
    pre_data_std = pre_data.copy()
    for col in pre_data_std.columns:
        pre_data_std[col] = (pre_data_std[col] - pre_data_std[col].mean()) / pre_data_std[col].std()
    
    # Extract treated and control data
    Y_treated = pre_data_std[treated_unit]
    X_controls = pre_data_std[valid_controls]
    
    # Find optimal weights by solving constrained optimization
    # This is a simplified approach - actual synthetic control uses more sophisticated methods
    from scipy.optimize import minimize
    
    def loss_function(weights):
        # Calculate synthetic control
        synthetic = X_controls.dot(weights)
        # Return mean squared error
        return ((Y_treated - synthetic) ** 2).mean()
    
    # Constraints: weights sum to 1 and are non-negative
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(len(valid_controls))]
    
    # Initial weights are equal
    initial_weights = np.ones(len(valid_controls)) / len(valid_controls)
    
    # Solve the optimization
    result = minimize(loss_function, initial_weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    optimal_weights = result['x']
    
    # Create synthetic control for all periods
    all_data = df.pivot(index='date', columns='symbol', values='close')
    
    # Ensure we have all necessary controls
    synthetic_controls = []
    for i, control in enumerate(valid_controls):
        if control in all_data.columns:
            synthetic_controls.append(all_data[control] * optimal_weights[i])
    
    # Create synthetic unit
    synthetic_unit = pd.DataFrame({
        'date': all_data.index,
        'synthetic': sum(synthetic_controls)
    }).reset_index(drop=True)
    
    # Get actual treated data
    treated_data = df[df['symbol'] == treated_unit][['date', 'close']].rename(
        columns={'close': 'actual'})
    
    # Merge actual and synthetic
    comparison = pd.merge(treated_data, synthetic_unit, on='date')
    comparison['diff'] = comparison['actual'] - comparison['synthetic']
    comparison['period'] = 'pre'
    comparison.loc[comparison['date'] >= event_date, 'period'] = 'post'
    
    # Calculate impact
    pre_diff_mean = comparison[comparison['period'] == 'pre']['diff'].mean()
    post_diff_mean = comparison[comparison['period'] == 'post']['diff'].mean()
    
    # Calculate normalized price series for plotting
    base_value = comparison.loc[comparison['date'] < event_date, 'actual'].iloc[-1]
    comparison['actual_normalized'] = comparison['actual'] / base_value * 100
    comparison['synthetic_normalized'] = comparison['synthetic'] / base_value * 100
    
    # Calculate effect size
    effect = post_diff_mean - pre_diff_mean
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(
        comparison[comparison['period'] == 'pre']['diff'],
        comparison[comparison['period'] == 'post']['diff'],
        equal_var=False
    )
    
    # Prepare final results
    results = {
        'treated_unit': treated_unit,
        'control_units': valid_controls,
        'weights': dict(zip(valid_controls, optimal_weights)),
        'comparison_data': comparison,
        'pre_difference': pre_diff_mean,
        'post_difference': post_diff_mean,
        'effect': effect,
        'effect_percentage': effect / comparison[comparison['period'] == 'pre']['actual'].mean() * 100,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'event_date': event_date
    }
    
    return results

def analyze_tariff_impact(event_id: int, method: str = 'all') -> Dict:
    """
    Analyze the impact of a specific tariff event using the specified method(s)
    
    Args:
        event_id: ID of the tariff event to analyze
        method: Analysis method to use ('did', 'event_study', 'synthetic_control', or 'all')
    
    Returns:
        Dictionary with analysis results
    """
    engine = get_db_connection()
    
    # Convert numpy.int64 to standard Python int if needed
    if hasattr(event_id, 'item'):  # Check if it's a numpy type
        event_id = int(event_id)  # Convert to standard Python int
    
    # Get tariff event details
    with engine.connect() as conn:
        query = text("SELECT * FROM tariff_events WHERE id = :event_id")
        result = conn.execute(query, {"event_id": event_id})
        event = result.fetchone()
    
    if not event:
        logger.error(f"Tariff event with ID {event_id} not found")
        return None
    
    # Extract event details
    event_date = event.date
    event_title = event.title
    affected_sectors = event.affected_sectors
    
    # Define analysis periods
    pre_start = (event_date - timedelta(days=30)).strftime('%Y-%m-%d')
    post_end = (event_date + timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Get relevant market data
    market_data = get_market_data(start_date=pre_start, end_date=post_end)
    
    # Calculate returns
    market_data = calculate_returns(market_data)
    
    # Identify treated and control groups
    treated_symbols = []
    control_symbols = []
    
    # Group symbols by sector
    sectors_dict = market_data.groupby('sector')['symbol'].unique().apply(list).to_dict()
    
    # Determine treated sectors based on event
    for sector in sectors_dict:
        # For lumber tariffs, lumber companies are treated
        if 'Lumber' in affected_sectors and 'Lumber' in sector:
            treated_symbols.extend(sectors_dict[sector])
        # For agriculture tariffs, agriculture companies are treated
        elif 'Agriculture' in affected_sectors and 'Agriculture' in sector:
            treated_symbols.extend(sectors_dict[sector])
        else:
            control_symbols.extend(sectors_dict[sector])
    
    # Make lists unique
    treated_symbols = list(set(treated_symbols))
    control_symbols = list(set(control_symbols))
    
    # Ensure we have data for our symbols
    valid_treated = market_data[market_data['symbol'].isin(treated_symbols)]['symbol'].unique()
    valid_control = market_data[market_data['symbol'].isin(control_symbols)]['symbol'].unique()
    
    logger.info(f"Analyzing impact of event: {event_title} on {event_date}")
    logger.info(f"Treated symbols: {valid_treated.tolist()}")
    logger.info(f"Control symbols: {valid_control.tolist()}")
    
    # Run requested analyses
    results = {
        'event_id': event_id,
        'event_title': event_title,
        'event_date': event_date,
        'affected_sectors': affected_sectors,
        'treated_symbols': valid_treated.tolist(),
        'control_symbols': valid_control.tolist()
    }
    
    if method == 'all' or method == 'did':
        logger.info("Running difference-in-differences analysis")
        did_results = difference_in_differences(
            market_data, 
            event_date, 
            valid_treated.tolist(), 
            valid_control.tolist()
        )
        results['difference_in_differences'] = did_results
    
    if method == 'all' or method == 'event_study':
        logger.info("Running event study analysis")
        es_results = event_study(
            market_data,
            event_date,
            valid_treated.tolist()
        )
        results['event_study'] = es_results
    
    if method == 'all' or method == 'synthetic_control':
        logger.info("Running synthetic control analysis")
        sc_results = {}
        
        # Run synthetic control for each treated unit
        for treated in valid_treated:
            logger.info(f"Creating synthetic control for {treated}")
            # Use all control symbols as the donor pool
            sc_result = synthetic_control(
                market_data,
                event_date,
                treated,
                valid_control.tolist()
            )
            if sc_result:
                sc_results[treated] = sc_result
        
        results['synthetic_control'] = sc_results
    
    # Save results to database
    save_analysis_results(results)
    
    return results

def save_analysis_results(results: Dict) -> None:
    """
    Save analysis results to the database
    
    Args:
        results: Dictionary with analysis results
    """
    engine = get_db_connection()
    
    # Create table if it doesn't exist
    with engine.connect() as conn:
        create_table_query = text("""
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
            p_value FLOAT,
            significant BOOLEAN,
            analyzed_at TIMESTAMP NOT NULL DEFAULT NOW(),
            UNIQUE(event_id, symbol)
        )
        """)
        conn.execute(create_table_query)
        conn.commit()
    
    # Extract key metrics
    event_id = results['event_id']
    event_date = results['event_date']
    
    # Process symbols from different analysis methods
    
    # From DiD
    if 'difference_in_differences' in results:
        did = results['difference_in_differences']
        # DiD provides a group-level effect, so apply to all treated symbols
        for symbol in results['treated_symbols']:
            try:
                # Figure out which sector this symbol belongs to
                with engine.connect() as conn:
                    query = text("""
                        SELECT sector FROM market_data 
                        WHERE symbol = :symbol 
                        LIMIT 1
                    """)
                    sector = conn.execute(query, {"symbol": symbol}).scalar()
                
                # Check if entry already exists
                with engine.connect() as conn:
                    check_query = text("""
                        SELECT id FROM tariff_impact_analysis
                        WHERE event_id = :event_id AND symbol = :symbol
                    """)
                    existing = conn.execute(check_query, {
                        "event_id": event_id,
                        "symbol": symbol
                    }).fetchone()
                
                # Insert or update
                if existing:
                    update_query = text("""
                        UPDATE tariff_impact_analysis
                        SET 
                            pre_event_avg = :pre_avg,
                            post_event_5d = :post_5d,
                            impact_5d_pct = :impact_5d,
                            p_value = :p_value,
                            significant = :significant,
                            analyzed_at = NOW()
                        WHERE event_id = :event_id AND symbol = :symbol
                    """)
                    
                    with engine.connect() as conn:
                        conn.execute(update_query, {
                            "event_id": event_id,
                            "symbol": symbol,
                            "pre_avg": did['treated_pre'],
                            "post_5d": did['treated_post'],
                            "impact_5d": did['did_effect'],
                            "p_value": did['did_pvalue'],
                            "significant": did['did_pvalue'] < 0.05
                        })
                        conn.commit()
                else:
                    insert_query = text("""
                        INSERT INTO tariff_impact_analysis
                        (event_id, sector, symbol, pre_event_avg, post_event_5d, impact_5d_pct,
                         p_value, significant)
                        VALUES 
                        (:event_id, :sector, :symbol, :pre_avg, :post_5d, :impact_5d,
                         :p_value, :significant)
                    """)
                    
                    with engine.connect() as conn:
                        conn.execute(insert_query, {
                            "event_id": event_id,
                            "sector": sector,
                            "symbol": symbol,
                            "pre_avg": did['treated_pre'],
                            "post_5d": did['treated_post'],
                            "impact_5d": did['did_effect'],
                            "p_value": did['did_pvalue'],
                            "significant": did['did_pvalue'] < 0.05
                        })
                        conn.commit()
            except Exception as e:
                logger.error(f"Error saving DiD results for {symbol}: {str(e)}")
    
    # From Event Study
    if 'event_study' in results:
        es = results['event_study']
        
        for symbol, stats in es['statistical_tests'].items():
            try:
                # Figure out which sector this symbol belongs to
                with engine.connect() as conn:
                    query = text("""
                        SELECT sector FROM market_data 
                        WHERE symbol = :symbol 
                        LIMIT 1
                    """)
                    sector = conn.execute(query, {"symbol": symbol}).scalar()
                
                # Check if entry already exists
                with engine.connect() as conn:
                    check_query = text("""
                        SELECT id FROM tariff_impact_analysis
                        WHERE event_id = :event_id AND symbol = :symbol
                    """)
                    existing = conn.execute(check_query, {
                        "event_id": event_id,
                        "symbol": symbol
                    }).fetchone()
                
                # Update post_event_5d with CAR if needed
                if existing:
                    update_query = text("""
                        UPDATE tariff_impact_analysis
                        SET 
                            post_event_5d = :car,
                            p_value = :p_value,
                            significant = :significant,
                            analyzed_at = NOW()
                        WHERE event_id = :event_id AND symbol = :symbol
                    """)
                    
                    with engine.connect() as conn:
                        conn.execute(update_query, {
                            "event_id": event_id,
                            "symbol": symbol,
                            "car": stats['cumulative_abnormal_return'],
                            "p_value": stats['p_value'],
                            "significant": stats['significant']
                        })
                        conn.commit()
            except Exception as e:
                logger.error(f"Error saving Event Study results for {symbol}: {str(e)}")
    
    # From Synthetic Control
    if 'synthetic_control' in results:
        sc = results['synthetic_control']
        
        for symbol, result in sc.items():
            try:
                # Figure out which sector this symbol belongs to
                with engine.connect() as conn:
                    query = text("""
                        SELECT sector FROM market_data 
                        WHERE symbol = :symbol 
                        LIMIT 1
                    """)
                    sector = conn.execute(query, {"symbol": symbol}).scalar()
                
                # Extract impact metrics
                effect_pct = result['effect_percentage']
                p_value = result['p_value']
                
                # Check if entry already exists
                with engine.connect() as conn:
                    check_query = text("""
                        SELECT id FROM tariff_impact_analysis
                        WHERE event_id = :event_id AND symbol = :symbol
                    """)
                    existing = conn.execute(check_query, {
                        "event_id": event_id,
                        "symbol": symbol
                    }).fetchone()
                
                # Insert or update
                if existing:
                    update_query = text("""
                        UPDATE tariff_impact_analysis
                        SET 
                            impact_5d_pct = :impact,
                            p_value = :p_value,
                            significant = :significant,
                            analyzed_at = NOW()
                        WHERE event_id = :event_id AND symbol = :symbol
                    """)
                    
                    with engine.connect() as conn:
                        conn.execute(update_query, {
                            "event_id": event_id,
                            "symbol": symbol,
                            "impact": effect_pct,
                            "p_value": p_value,
                            "significant": p_value < 0.05
                        })
                        conn.commit()
                else:
                    insert_query = text("""
                        INSERT INTO tariff_impact_analysis
                        (event_id, sector, symbol, impact_5d_pct, p_value, significant)
                        VALUES 
                        (:event_id, :sector, :symbol, :impact, :p_value, :significant)
                    """)
                    
                    with engine.connect() as conn:
                        conn.execute(insert_query, {
                            "event_id": event_id,
                            "sector": sector,
                            "symbol": symbol,
                            "impact": effect_pct,
                            "p_value": p_value,
                            "significant": p_value < 0.05
                        })
                        conn.commit()
            except Exception as e:
                logger.error(f"Error saving Synthetic Control results for {symbol}: {str(e)}")

def create_impact_visualizations(results: Dict, output_dir: str = 'visualizations'):
    """
    Create visualizations of tariff impact analysis results
    
    Args:
        results: Dictionary with analysis results
        output_dir: Directory to save visualizations
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("colorblind")
    
    # Extract event details
    event_id = results['event_id']
    event_title = results['event_title']
    event_date = results['event_date']
    
    # 1. Difference-in-Differences visualization
    if 'difference_in_differences' in results:
        did = results['difference_in_differences']
        
        # Create DID plot
        plt.figure(figsize=(12, 6))
        
        # Prepare data for plotting
        pre_values = [did['control_pre'], did['treated_pre']]
        post_values = [did['control_post'], did['treated_post']]
        
        x = ['Control', 'Treated']
        width = 0.35
        
        # Plot bars
        plt.bar([i - width/2 for i in range(len(x))], pre_values, width, label='Pre-Event')
        plt.bar([i + width/2 for i in range(len(x))], post_values, width, label='Post-Event')
        
        # Add lines to visualize the differences
        plt.plot([0 - width/2, 0 + width/2], [pre_values[0], post_values[0]], 'k--', alpha=0.5)
        plt.plot([1 - width/2, 1 + width/2], [pre_values[1], post_values[1]], 'k--', alpha=0.5)
        
        # Add text showing the differences
        plt.text(0, max(pre_values[0], post_values[0]) + 0.5, 
                f"Δ = {post_values[0] - pre_values[0]:.2f}", ha='center')
        plt.text(1, max(pre_values[1], post_values[1]) + 0.5, 
                f"Δ = {post_values[1] - pre_values[1]:.2f}", ha='center')
        
        # Add DiD effect
        plt.text(0.5, max(post_values) + 1.5, 
                f"DiD Effect = {did['did_effect']:.2f} (p = {did['did_pvalue']:.4f})", 
                ha='center', fontweight='bold')
        
        # Customize plot
        plt.xlabel('Group')
        plt.ylabel('Return (%)')
        plt.title(f"Difference-in-Differences Analysis for Event: {event_title}")
        plt.xticks(range(len(x)), x)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add significance indicator
        if did['did_pvalue'] < 0.05:
            plt.text(0.5, max(post_values) + 2.5, "Statistically Significant", 
                    ha='center', color='green', fontweight='bold')
        else:
            plt.text(0.5, max(post_values) + 2.5, "Not Statistically Significant", 
                    ha='center', color='red', fontweight='bold')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"event_{event_id}_did.png"))
        plt.close()
    
    # 2. Event Study visualization
    if 'event_study' in results:
        es = results['event_study']
        
        # Create Event Study CAR plot
        plt.figure(figsize=(12, 6))
        
        # Extract CAR data for each symbol
        cars = {}
        for symbol, car_data in es['cumulative_abnormal_returns'].items():
            cars[symbol] = car_data['car']
        
        # Plot CAR for each symbol
        for symbol, car in cars.items():
            plt.plot(car.index, car, label=symbol)
        
        # Add event line
        plt.axvline(x=event_date, color='r', linestyle='--', label='Event Date')
        
        # Customize plot
        plt.xlabel('Date')
        plt.ylabel('Cumulative Abnormal Return (%)')
        plt.title(f"Cumulative Abnormal Returns After Event: {event_title}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"event_{event_id}_car.png"))
        plt.close()
        
        # Create bar chart of abnormal returns
        plt.figure(figsize=(10, 6))
        
        # Extract significant results
        significant_symbols = []
        significant_cars = []
        nonsig_symbols = []
        nonsig_cars = []
        
        for symbol, stats in es['statistical_tests'].items():
            if stats['significant']:
                significant_symbols.append(symbol)
                significant_cars.append(stats['cumulative_abnormal_return'])
            else:
                nonsig_symbols.append(symbol)
                nonsig_cars.append(stats['cumulative_abnormal_return'])
        
        # Plot bars with different colors for significant/non-significant
        if significant_symbols:
            plt.bar(significant_symbols, significant_cars, color='green', label='Significant (p<0.05)')
        if nonsig_symbols:
            plt.bar(nonsig_symbols, nonsig_cars, color='gray', label='Not Significant')
            
        # Add horizontal line at zero
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Customize plot
        plt.xlabel('Symbol')
        plt.ylabel('Cumulative Abnormal Return (%)')
        plt.title(f"Abnormal Returns by Symbol for Event: {event_title}")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"event_{event_id}_ar_bar.png"))
        plt.close()
    
    # 3. Synthetic Control visualizations
    if 'synthetic_control' in results:
        sc = results['synthetic_control']
        
        # Create a separate plot for each treated unit
        for symbol, result in sc.items():
            plt.figure(figsize=(12, 6))
            
            # Extract comparison data
            comp_data = result['comparison_data']
            
            # Plot actual vs synthetic
            plt.plot(comp_data['date'], comp_data['actual_normalized'], 
                   label=f'Actual {symbol}', linewidth=2)
            plt.plot(comp_data['date'], comp_data['synthetic_normalized'], 
                   label='Synthetic Control', linewidth=2, linestyle='--')
            
            # Add event line
            plt.axvline(x=event_date, color='r', linestyle='--', label='Event Date')
            
            # Shade pre and post periods
            pre_mask = comp_data['period'] == 'pre'
            post_mask = comp_data['period'] == 'post'
            
            plt.fill_between(comp_data['date'], 
                           comp_data['actual_normalized'] - comp_data['synthetic_normalized'], 
                           0, where=pre_mask, color='gray', alpha=0.3, label='Pre-Event Gap')
            
            plt.fill_between(comp_data['date'], 
                           comp_data['actual_normalized'] - comp_data['synthetic_normalized'], 
                           0, where=post_mask, color='red' if result['effect'] < 0 else 'green', 
                           alpha=0.3, label='Treatment Effect')
            
            # Add annotation with effect size
            effect_pct = result['effect_percentage']
            last_date = comp_data['date'].iloc[-1]
            y_pos = max(comp_data['actual_normalized'].max(), comp_data['synthetic_normalized'].max()) + 5
            
            plt.text(event_date, y_pos, 
                   f"Effect: {effect_pct:.2f}% (p = {result['p_value']:.4f})", 
                   ha='center', fontweight='bold')
            
            # Add significance indicator
            if result['significant']:
                plt.text(event_date, y_pos + 5, "Statistically Significant", 
                       ha='center', color='green', fontweight='bold')
            else:
                plt.text(event_date, y_pos + 5, "Not Statistically Significant", 
                       ha='center', color='red', fontweight='bold')
            
            # Customize plot
            plt.xlabel('Date')
            plt.ylabel('Normalized Price (Base=100)')
            plt.title(f"Synthetic Control Analysis for {symbol}: {event_title}")
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"event_{event_id}_{symbol}_synthetic.png"))
            plt.close()
            
        # Create summary bar chart of all effects
        plt.figure(figsize=(10, 6))
        
        # Extract effects
        symbols = []
        effects = []
        colors = []
        
        for symbol, result in sc.items():
            symbols.append(symbol)
            effects.append(result['effect_percentage'])
            colors.append('green' if result['significant'] else 'gray')
        
        # Plot bars
        plt.bar(symbols, effects, color=colors)
        
        # Add horizontal line at zero
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Customize plot
        plt.xlabel('Symbol')
        plt.ylabel('Effect Size (%)')
        plt.title(f"Tariff Impact by Symbol for Event: {event_title}")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add a legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Significant (p<0.05)'),
            Patch(facecolor='gray', label='Not Significant')
        ]
        plt.legend(handles=legend_elements)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"event_{event_id}_effects.png"))
        plt.close()

def run_lumber_tariff_analysis():
    """Run analysis for the most recent lumber tariff event"""
    # Get the recent lumber tariff event
    events = get_tariff_events(sector='Lumber', start_date='2025-01-01')
    
    if events.empty:
        logger.error("No recent lumber tariff events found")
        return None
    
    # Get the most recent event
    recent_event = events.iloc[-1]
    
    logger.info(f"Analyzing impact of event: {recent_event['title']} on {recent_event['date']}")
    
    # Run full analysis
    results = analyze_tariff_impact(recent_event['id'], method='all')
    
    # Create visualizations
    create_impact_visualizations(results)
    
    return results

if __name__ == "__main__":
    # Run analysis for the recent lumber tariff
    run_lumber_tariff_analysis()