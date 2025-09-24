#!/bin/bash
# Exit on error
set -e

# Ensure we're using Python 3.10
export PYENV_VERSION=3.10.13

# Install system dependencies
echo "Updating package lists..."
apt-get update
echo "Installing system dependencies..."
apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    build-essential \
    libpython3.10-dev

# Create and activate virtual environment
echo "Creating Python virtual environment..."
python3.10 -m venv /app/venv
source /app/venv/bin/activate

# Upgrade pip and setuptools
echo "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Install Python packages
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download spaCy model
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo "Setup completed successfully!"
