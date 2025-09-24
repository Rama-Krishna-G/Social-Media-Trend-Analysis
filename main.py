import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Social Media Trend Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import collectors with graceful fallback
COLLECTORS_AVAILABLE = False
TwitterCollector = None
RedditCollector = None

try:
    from src.data.collectors import TwitterCollector, RedditCollector
    COLLECTORS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some collectors are not available: {e}")
    logger.info("The application will run in sample data mode.")

# Import analysis modules
from src.analysis.topic_modeling import TopicModeler
from src.visualization.dashboard import Dashboard, create_dashboard

# Import dummy data generator
from src.data.generators.dummy_data import DummyDataGenerator
import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path

def load_sample_data():
    """Load sample data from the CSV file."""
    sample_file = Path('data/raw/social_media_data_2years.csv')
    
    if not sample_file.exists():
        st.warning("Sample data not found. Generating sample data...")
        try:
            from create_sample_data import generate_sample_data
            generate_sample_data()
        except Exception as e:
            st.error(f"Error generating sample data: {e}")
            return pd.DataFrame()
    
    try:
        # Read the CSV file with proper date parsing
        df = pd.read_csv(
            sample_file,
            parse_dates=['created_at'],
            date_parser=pd.to_datetime
        )
        
        # Ensure required columns exist
        required_columns = ['created_at', 'topic', 'sentiment', 'text']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Required column '{col}' not found in the data")
                return pd.DataFrame()
                
        # Sort by date
        df = df.sort_values('created_at')
        
        return df
        
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return pd.DataFrame()

def main():
    st.title("📊 Social Media Topic Trend Analyzer")
    st.write("Analyze emerging topics and sentiment trends across social media platforms.")
    
    # Initialize data collectors (will be None if API keys are not available)
    twitter_collector = None
    reddit_collector = None
    
    # Sidebar for data source selection
    st.sidebar.header("Data Source")
    if COLLECTORS_AVAILABLE:
        data_source = st.sidebar.radio(
            "Choose data source:",
            ["Sample Data", "Live Data"],
            index=default_source
        )
    else:
        st.sidebar.info("Running in sample data mode (API keys not configured)")
        data_source = "Sample Data"
    
    # Initialize empty dataframe
    df = pd.DataFrame()
    
    # Collect data based on selection
    if data_source == "Live Data" and COLLECTORS_AVAILABLE:
        st.header("Live Data Collection")
        platform = st.radio("Select platform:", ["Twitter", "Reddit"])
        
        if platform == "Twitter":
            query = st.text_input("Search query:", "#python")
            if st.button("Collect Tweets"):
                with st.spinner("Collecting tweets..."):
                    df = collect_twitter_data(query)
        else:  # Reddit
            subreddit = st.text_input("Subreddit:", "python")
            if st.button("Collect Posts"):
                with st.spinner("Collecting Reddit posts..."):
                    df = collect_reddit_data(subreddit)
    else:  # Sample Data
        st.header("Sample Data Analysis")
        st.info("Using pre-generated sample data. Click the button below to load.")
        if st.button("Load Sample Data"):
            if 'data' not in st.session_state:
                with st.spinner("Loading sample data..."):
                    sample_data = load_sample_data()
                    if not sample_data.empty:
                        st.session_state['data'] = sample_data
                        st.sidebar.success(f"Loaded {len(sample_data)} sample posts!")
                    else:
                        st.error("Failed to load sample data. Please try again.")
    
    # Display and analyze data if available
    if 'data' in st.session_state and not st.session_state['data'].empty:
        data = st.session_state['data']
        
        # Display data summary
        st.subheader("📋 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Posts", len(data))
        with col2:
            platforms = data['platform'].value_counts()
            st.metric("Platforms", ", ".join([f"{k} ({v})" for k, v in platforms.items()]))
        with col3:
            st.metric("Unique Topics", data['topic'].nunique() if 'topic' in data.columns else 'N/A')
        with col4:
            if 'sentiment' in data.columns:
                avg_sentiment = data['sentiment'].mean()
                sentiment_label = "😊 Positive" if avg_sentiment > 0.1 else "😐 Neutral" if avg_sentiment > -0.1 else "😞 Negative"
                st.metric("Avg. Sentiment", sentiment_label, delta=f"{avg_sentiment:.2f}")
            else:
                st.metric("Avg. Sentiment", "N/A")
        
        # Display raw data in an expander
        with st.expander("📄 View Raw Data"):
            st.dataframe(data.head(100))  # Show first 100 rows
        
        # Analyze topics
        st.subheader("🔍 Topic Analysis")
        
        # Check if we need to analyze topics (not already done in sample data)
        if 'topic' not in data.columns or data['topic'].isna().all():
            if st.button("🔍 Analyze Topics"):
                with st.spinner("Analyzing topics and sentiment..."):
                    try:
                        topic_modeler = TopicModeler()
                        
                        # Combine title and text for Reddit posts
                        if 'title' in data.columns and 'text' in data.columns:
                            text_data = data['title'] + ' ' + data['text']
                        else:
                            text_data = data['text']
                        
                        # Fit topic model
                        topics = topic_modeler.fit_transform(text_data)
                        data['topic'] = topics
                        
                        # Analyze sentiment if not already present
                        if 'sentiment_score' not in data.columns:
                            data['sentiment_score'] = [topic_modeler.analyze_sentiment(text) for text in text_data]
                            data['sentiment'] = pd.cut(
                                data['sentiment_score'],
                                bins=[-1, -0.1, 0.1, 1],
                                labels=['negative', 'neutral', 'positive']
                            )
                        
                        st.session_state['data'] = data
                        st.success("Analysis complete!")
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
        
        # Display visualizations if we have topic data
        if 'topic' in data.columns and not data['topic'].isna().all():
            st.subheader("📊 Visualizations")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs([
                "Topic Trends", "Sentiment Analysis", 
                "Topic-Sentiment", "Word Cloud"
            ])
            
            # Initialize dashboard
            dashboard = Dashboard()
            
            with tab1:
                st.plotly_chart(
                    dashboard.create_topic_trends_chart(data),
                    use_container_width=True
                )
                
            with tab2:
                st.plotly_chart(
                    dashboard.create_sentiment_timeline(data),
                    use_container_width=True
                )
                
            with tab3:
                st.plotly_chart(
                    dashboard.create_topic_sentiment_chart(data),
                    use_container_width=True
                )
                
            with tab4:
                if 'text' in data.columns and not data['text'].empty:
                    st.write("### Most Common Words in Posts")
                    dashboard.create_word_cloud(
                        data=data,
                        text_col='text',
                        title='Word Cloud of Social Media Posts',
                        max_words=100
                    )
                else:
                    st.warning("No text data available for word cloud generation.")
                fig = dashboard.create_sentiment_timeline(data)
                st.plotly_chart(fig, use_container_width=True)
                
            with tab3:
                # Topic-sentiment chart
                fig = dashboard.create_topic_sentiment_chart(data)
                st.plotly_chart(fig, use_container_width=True)
                
            with tab4:
                # Create word cloud for the most recent 100 posts
                recent_texts = data.sort_values('created_at', ascending=False)\
                                .head(100)['text'].fillna('').astype(str).tolist()
                if any(text.strip() for text in recent_texts):
                    wordcloud_fig = dashboard.create_word_cloud(recent_texts)
                    st.plotly_chart(wordcloud_fig, use_container_width=True)
                else:
                    st.warning("No text data available for word cloud.")
        
        # Add a button to regenerate sample data
        if data_source == "Sample Data":
            if st.sidebar.button("🔄 Regenerate Sample Data"):
                with st.spinner("Generating new sample data..."):
                    from generate_sample_data import generate_sample_data
                    generate_sample_data()
                    st.session_state['data'] = load_sample_data()
                    st.sidebar.success("Sample data regenerated!")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Add custom CSS for better styling
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
        .stAlert {
            border-radius: 5px;
            padding: 1rem;
        }
        .stDataFrame {
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Run the main app
    main()
