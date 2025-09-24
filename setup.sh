#!/bin/bash
# Install system dependencies
apt-get update
apt-get install -y python3-dev python3-pip

# Install Python packages
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
