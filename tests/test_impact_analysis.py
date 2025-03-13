# test_impact_analysis.py
import os
import sys
import logging

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
    # Import the tariff impact analyzer
    from analysis.tariff_impact_analyzer import run_lumber_tariff_analysis
    
    # Run analysis for the recent lumber tariff
    results = run_lumber_tariff_analysis()
    
    if results:
        logger.info(f"Analysis completed successfully")
        
        # Print main results
        event_info = f"{results['event_title']} on {results['event_date']}"
        logger.info(f"Event: {event_info}")
        
        # Print DiD results if available
        if 'difference_in_differences' in results:
            did = results['difference_in_differences']
            logger.info(f"Difference-in-Differences effect: {did['did_effect']:.2f}%")
            logger.info(f"p-value: {did['did_pvalue']:.4f}")
            logger.info(f"Significant: {did['did_pvalue'] < 0.05}")
        
        # Print synthetic control results if available
        if 'synthetic_control' in results:
            sc = results['synthetic_control']
            logger.info("Synthetic Control results:")
            for symbol, result in sc.items():
                logger.info(f"  {symbol}: Effect = {result['effect_percentage']:.2f}%, p-value = {result['p_value']:.4f}")
        
        logger.info("✅ Impact analysis test PASSED")
    else:
        logger.error("❌ Impact analysis test FAILED - No results returned")
    
except Exception as e:
    logger.error(f"❌ Impact analysis test FAILED with error: {str(e)}")
    import traceback
    traceback.print_exc()