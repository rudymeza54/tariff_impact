# test_visualizations.py - Updated version with fixed imports
import os
import sys
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get tests directory
project_root = os.path.dirname(current_dir)               # Go up to project root
sys.path.append(project_root)                             # Add project root to path

# Set database connection environment variables
os.environ['DB_HOST'] = 'localhost'
os.environ['DB_USER'] = 'postgres'
os.environ['DB_PASSWORD'] = 'mysecretpassword'
os.environ['DB_NAME'] = 'postgres'
os.environ['DB_PORT'] = '5431'

try:
    # Use the full import path from the project root
    from src.analysis.enhanced_visualizations import create_all_visualizations
    
    # Create visualizations for the lumber tariff event (ID 10)
    create_all_visualizations(10)
    
    logger.info("✅ Enhanced visualizations test PASSED")
    
except Exception as e:
    logger.error(f"❌ Enhanced visualizations test FAILED with error: {str(e)}")
    import traceback
    traceback.print_exc()