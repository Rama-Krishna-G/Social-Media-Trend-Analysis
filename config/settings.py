""
Configuration settings for the Social Media Trend Analyzer.

This file contains all the configuration settings for the application.
Sensitive information should be loaded from environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = BASE_DIR / 'models'

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Twitter API credentials
TWITTER_CONFIG = {
    'consumer_key': os.getenv('TWITTER_API_KEY'),
    'consumer_secret': os.getenv('TWITTER_API_SECRET'),
    'access_token': os.getenv('TWITTER_ACCESS_TOKEN'),
    'access_token_secret': os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
    'bearer_token': os.getenv('TWITTER_BEARER_TOKEN')
}

# Reddit API credentials
REDDIT_CONFIG = {
    'client_id': os.getenv('REDDIT_CLIENT_ID'),
    'client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
    'user_agent': os.getenv('REDDIT_USER_AGENT', 'social-media-analyzer/1.0'),
    'username': os.getenv('REDDIT_USERNAME'),
    'password': os.getenv('REDDIT_PASSWORD')
}

# NLP and ML settings
NLP_SETTINGS = {
    'spacy_model': 'en_core_web_sm',
    'sentence_transformer': 'all-MiniLM-L6-v2',
    'default_num_topics': 5,
    'max_features': 1000,
    'ngram_range': (1, 2)
}

# Data processing settings
PROCESSING_SETTINGS = {
    'min_word_length': 3,
    'max_doc_freq': 0.95,
    'min_doc_freq': 2,
    'stop_words': 'english',
    'random_state': 42
}

# Visualization settings
VISUALIZATION_SETTINGS = {
    'theme': 'plotly_white',
    'color_scale': 'Viridis',
    'default_height': 600,
    'default_width': 1000,
    'font_family': 'Arial, sans-serif',
    'font_size': 12
}

# Streamlit app settings
STREAMLIT_CONFIG = {
    'page_title': 'Social Media Trend Analyzer',
    'page_icon': '📊',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.FileHandler',
            'formatter': 'standard',
            'level': 'DEBUG',
            'filename': BASE_DIR / 'logs' / 'app.log',
            'mode': 'a',
            'encoding': 'utf-8'
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True
        },
        'app': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}
