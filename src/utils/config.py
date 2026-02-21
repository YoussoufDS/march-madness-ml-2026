"""Configuration module."""

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv('AIzaSyB3iDgg0DdVHQRBxhJgCQaKY5g8ZcFdnjU')
KAGGLE_USERNAME = os.getenv('youssoufabdouramane')
KAGGLE_KEY = os.getenv('KGAT_4b11164385cb45cdd507fa2469784434')

# Model parameters
ELO_K = int(os.getenv('ELO_K', '20'))
ELO_INIT = int(os.getenv('ELO_INIT', '1500'))
ELO_HCA = int(os.getenv('ELO_HCA', '100'))
CURRENT_SEASON = int(os.getenv('CURRENT_SEASON', '2026'))

# Data drift
DRIFT_THRESHOLD = float(os.getenv('DRIFT_THRESHOLD', '0.3'))
RETRAIN_THRESHOLD = float(os.getenv('RETRAIN_THRESHOLD', '0.5'))

# Paths
DATA_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
DRIFT_REPORTS_DIR = 'data/drift_reports'
MODEL_DIR = 'src/models'

# Create directories if they don't exist
for dir_path in [DATA_DIR, PROCESSED_DIR, DRIFT_REPORTS_DIR, MODEL_DIR]:
    os.makedirs(dir_path, exist_ok=True)