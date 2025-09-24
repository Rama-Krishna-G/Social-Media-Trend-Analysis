# Social Media Topic Trend Analyzer

A powerful tool for detecting emerging topics and analyzing sentiment trends across social media platforms.

## Features

- **Data Collection**: Fetch data from Twitter and Reddit APIs
- **Topic Detection**: Identify trending topics using advanced NLP techniques
- **Sentiment Analysis**: Track sentiment trends over time
- **Visualization**: Interactive dashboards and visualizations
- **Time Series Analysis**: Detect patterns and trends in social media discussions

## Project Structure

```
social_media_analyzer/
├── data/                    # Raw and processed data
│   ├── raw/                 # Raw data from APIs
│   └── processed/           # Processed and cleaned data
├── notebooks/               # Jupyter notebooks for exploration
├── src/                     # Source code
│   ├── data/                # Data collection and processing
│   ├── analysis/            # Analysis and modeling
│   ├── visualization/       # Visualization utilities
│   └── utils/               # Utility functions
├── config/                  # Configuration files
├── tests/                   # Test files
├── requirements.txt         # Python dependencies
└── main.py                  # Main application entry point
```

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your API credentials in `config/api_keys.json`

## Usage

Run the main application:
```bash
python main.py
```

## License

MIT
