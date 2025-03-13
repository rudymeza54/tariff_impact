#!/usr/bin/env python3
"""
Enhanced Tariff News Collector
Collects tariff and trade policy news from additional sources with improved detection
"""

import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlalchemy
from sqlalchemy import text
from datetime import datetime, timedelta
import logging
import time
import random
import re
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection
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

# Enhanced keywords to identify relevant news
AGRICULTURE_KEYWORDS = [
    'agriculture', 'farm', 'farmer', 'crop', 'soybean', 'corn', 'wheat', 
    'cotton', 'dairy', 'livestock', 'cattle', 'grain', 'fertilizer',
    'agricultural', 'food', 'produce'
]

LUMBER_KEYWORDS = [
    'lumber', 'timber', 'wood', 'forest', 'logging', 'sawmill', 'plywood', 
    'pulp', 'paper', 'softwood', 'hardwood', 'forestry', 'construction materials'
]

TARIFF_KEYWORDS = [
    'tariff', 'trade war', 'trade dispute', 'trade policy', 'trade agreement',
    'import duty', 'export tax', 'customs duty', 'trade restriction',
    'protectionism', 'trade barrier', 'embargo', 'sanction', 'trade tension',
    'import tax', 'export duty', 'trade conflict', 'escalate', 'retaliate',
    'retaliatory', 'duty', 'Section 232', 'Section 301'
]

COUNTRY_KEYWORDS = [
    'Canada', 'Canadian', 'China', 'Chinese', 'Mexico', 'Mexican', 
    'EU', 'European Union', 'Japan', 'Japanese', 'Brazil', 'Brazilian'
]

# Guardian API key
GUARDIAN_API_KEY = "241131cc-e0f1-4167-a5b8-9ce8bd14701c"
GUARDIAN_API_URL = "https://content.guardianapis.com/search"

def create_tariff_events_table():
    """Create the tariff_events table if it doesn't exist"""
    engine = get_db_connection()
    
    with engine.connect() as conn:
        # Check if table exists
        check_query = text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'tariff_events'
            )
        """)
        result = conn.execute(check_query)
        table_exists = result.scalar()
        
        if not table_exists:
            logger.info("Creating tariff_events table")
            
            # Create table
            create_table_query = text("""
                CREATE TABLE tariff_events (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    date DATE NOT NULL,
                    source VARCHAR(100),
                    url TEXT,
                    event_type VARCHAR(50),
                    affected_sectors TEXT[],
                    collected_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
            """)
            conn.execute(create_table_query)
            conn.commit()
            logger.info("tariff_events table created successfully")
        else:
            logger.info("tariff_events table already exists")

def is_relevant_article(title: str, description: str = '') -> bool:
    """Enhanced check if an article is relevant to tariffs or trade"""
    text = (title + ' ' + description).lower()
    
    # Check for tariff-related terms
    has_tariff = any(keyword.lower() in text for keyword in TARIFF_KEYWORDS)
    
    # Check for country mentions
    has_country = any(country.lower() in text for country in COUNTRY_KEYWORDS)
    
    # Check for agriculture or lumber terms
    has_agriculture = any(keyword.lower() in text for keyword in AGRICULTURE_KEYWORDS)
    has_lumber = any(keyword.lower() in text for keyword in LUMBER_KEYWORDS)
    
    # Check for specific terms related to metal tariffs that might impact construction/lumber
    metal_terms = ['steel', 'aluminum', 'metal', 'construction']
    has_metal = any(term in text for term in metal_terms)
    
    # More lenient relevance checking
    is_tariff_news = has_tariff and has_country
    is_sector_specific = has_tariff and (has_agriculture or has_lumber or has_metal)
    
    # Extra check for Trump tariff news which is often relevant
    has_trump_tariff = 'trump' in text and has_tariff
    
    return is_tariff_news or is_sector_specific or has_trump_tariff

def extract_affected_sectors(text: str) -> list:
    """Extract affected sectors from text with improved detection"""
    sectors = []
    text_lower = text.lower()
    
    if any(keyword in text_lower for keyword in AGRICULTURE_KEYWORDS):
        sectors.append('Agriculture')
        
    if any(keyword in text_lower for keyword in LUMBER_KEYWORDS):
        sectors.append('Lumber')
    
    # Add Metal sector for steel/aluminum tariffs that affect construction
    metal_construction_terms = ['steel', 'aluminum', 'metal', 'construction']
    if any(term in text_lower for term in metal_construction_terms) and not sectors:
        sectors.append('Metal/Construction')
    
    # If we couldn't determine specific sectors but it's a tariff article, mark as General
    if not sectors and any(keyword in text_lower for keyword in TARIFF_KEYWORDS):
        sectors.append('General Trade')
        
    return sectors

def collect_from_guardian(start_date: datetime) -> List[Dict]:
    """Collect tariff-related news from The Guardian API"""
    logger.info("Collecting news from The Guardian API")
    
    # Define the query parameters
    params = {
        "api-key": GUARDIAN_API_KEY,
        "q": "tariff OR trade war OR trade dispute OR agriculture OR lumber",
        "from-date": start_date.strftime("%Y-%m-%d"),
        "to-date": datetime.now().strftime("%Y-%m-%d"),
        "page-size": 50,  # Maximum allowed per request
        "show-fields": "headline,trailText,publication,shortUrl"
    }
    
    news_items = []
    
    try:
        response = requests.get(GUARDIAN_API_URL, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        articles = data.get("response", {}).get("results", [])
        
        for article in articles:
            title = article.get("webTitle", "")
            description = article.get("fields", {}).get("trailText", "")
            url = article.get("webUrl", "")
            date_str = article.get("webPublicationDate", "")
            source = "The Guardian"
            
            # Parse the publication date
            try:
                date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                logger.warning(f"Could not parse date for article: {title}")
                continue
            
            # Check if the article is relevant
            if is_relevant_article(title, description):
                affected_sectors = extract_affected_sectors(title + " " + description)
                
                # Determine event type
                event_type = "News"
                if "impose" in title.lower() or "announce" in title.lower() or "new tariff" in title.lower():
                    event_type = "Tariff Implementation"
                elif "increase" in title.lower() or "raise" in title.lower() or "escalate" in title.lower():
                    event_type = "Tariff Increase"
                elif "reduce" in title.lower() or "cut" in title.lower() or "lower" in title.lower():
                    event_type = "Tariff Reduction"
                
                # Add to news items
                news_items.append({
                    'title': title,
                    'description': description,
                    'date': date,
                    'source': source,
                    'url': url,
                    'event_type': event_type,
                    'affected_sectors': affected_sectors
                })
                
                logger.info(f"Found relevant Guardian article: {title}")
        
    except Exception as e:
        logger.error(f"Error fetching news from The Guardian API: {str(e)}")
    
    return news_items

def save_to_db(news_items: List[Dict]) -> int:
    """Save collected news items to the database"""
    if not news_items:
        logger.info("No news items to save")
        return 0
        
    engine = get_db_connection()
    saved_count = 0
    
    # Insert each item, handling affected_sectors as an array
    with engine.connect() as conn:
        for item in news_items:
            try:
                # Check if this article already exists
                check_query = text("""
                    SELECT id FROM tariff_events 
                    WHERE title = :title AND date = :date AND source = :source
                """)
                
                result = conn.execute(
                    check_query,
                    {"title": item['title'], "date": item['date'], "source": item['source']}
                )
                existing = result.fetchone()
                
                if existing:
                    logger.info(f"Article already exists: {item['title']}")
                    continue
                    
                # Insert new article
                insert_query = text("""
                    INSERT INTO tariff_events 
                    (title, description, date, source, url, event_type, affected_sectors)
                    VALUES (:title, :description, :date, :source, :url, :event_type, :affected_sectors)
                """)
                
                conn.execute(
                    insert_query,
                    {
                        "title": item['title'],
                        "description": item['description'],
                        "date": item['date'],
                        "source": item['source'],
                        "url": item['url'],
                        "event_type": item['event_type'],
                        "affected_sectors": item['affected_sectors']
                    }
                )
                
                saved_count += 1
                logger.info(f"Saved article: {item['title']}")
                
            except Exception as e:
                logger.error(f"Error saving article {item['title']}: {str(e)}")
        
        conn.commit()
    
    return saved_count

def collect_tariff_news(start_date: datetime) -> int:
    """Main function to collect tariff news from all sources"""
    total_saved = 0
    
    # Create table if needed
    create_tariff_events_table()
    
    # Collect from The Guardian API
    guardian_news = collect_from_guardian(start_date)
    saved_count = save_to_db(guardian_news)
    total_saved += saved_count
    
    logger.info(f"Collected {len(guardian_news)} news items from The Guardian and saved {saved_count} new items")
    
    return total_saved

if __name__ == "__main__":
    # Define the start date for news collection
    start_date = datetime(2025, 2, 1)
    
    # Collect and save tariff news
    collect_tariff_news(start_date)