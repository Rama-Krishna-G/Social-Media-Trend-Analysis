#!/bin/bash
set -e

# Add deadsnakes PPA for Python 3.10
echo "Adding deadsnakes PPA for Python 3.10..."
apt-get update
apt-get install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update

# Install Python 3.10 and pip
echo "Installing Python 3.10 and pip..."
apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip

# Set Python 3.10 as the default
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
update-alternatives --set python3 /usr/bin/python3.10

# Upgrade pip and setuptools
echo "Upgrading pip and setuptools..."
python3.10 -m pip install --upgrade pip setuptools wheel

# Install build dependencies
echo "Installing build dependencies..."
apt-get install -y build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev

# Create and activate virtual environment
echo "Creating Python virtual environment..."
python3.10 -m venv /app/venv
source /app/venv/bin/activate

# Install Python packages
echo "Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

# Download spaCy model
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo "Python version: $(python --version)"
echo "Setup completed successfully!"
