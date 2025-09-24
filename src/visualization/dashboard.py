import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

class Dashboard:
    """
    A class for creating interactive dashboards and visualizations for social media analysis.
    """
    
    def __init__(self, theme: str = 'plotly_white'):
        """
        Initialize the Dashboard with a specified theme.
        
        Args:
            theme: Plotly theme to use (default: 'plotly_white')
        """
        self.theme = theme
        self.template = self._get_theme_template(theme)
    
    def _get_theme_template(self, theme: str) -> str:
        """
        Get the Plotly template for the specified theme.
        
        Args:
            theme: Name of the theme
            
        Returns:
            Corresponding Plotly template name
        """
        theme = theme.lower()
        if theme in ['plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn', 'simple_white', 'none']:
            return theme
        return 'plotly_white'
    
    def create_topic_trends_chart(
        self,
        data: pd.DataFrame,
        time_col: str = 'created_at',
        topic_col: str = 'topic',
        title: str = 'Topic Trends Over Time',
        height: int = 500
    ) -> go.Figure:
        """
        Create a line chart showing topic trends over time.
        
        Args:
            data: DataFrame containing the data
            time_col: Name of the datetime column
            topic_col: Name of the topic column
            title: Chart title
            height: Chart height in pixels
            
        Returns:
            Plotly figure object
        """
        if data.empty or time_col not in data.columns or topic_col not in data.columns:
            logging.warning("Invalid data or missing required columns for topic trends chart")
            return go.Figure()
        
        # Resample data by time period (e.g., daily)
        data[time_col] = pd.to_datetime(data[time_col])
        data['date'] = data[time_col].dt.date
        
        # Count topics by date
        topic_counts = data.groupby(['date', topic_col]).size().unstack(fill_value=0)
        topic_percentages = topic_counts.div(topic_counts.sum(axis=1), axis=0) * 100
        
        # Create line chart
        fig = px.line(
            topic_percentages,
            x=topic_percentages.index,
            y=topic_percentages.columns,
            title=title,
            labels={'value': 'Percentage of Posts', 'date': 'Date', 'variable': 'Topic'},
            template=self.template,
            height=height
        )
        
        # Customize layout
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Percentage of Posts",
            legend_title="Topics",
            hovermode="x unified",
            showlegend=True
        )
        
        return fig
    
    def create_sentiment_timeline(
        self,
        data: pd.DataFrame,
        time_col: str = 'created_at',
        sentiment_col: str = 'sentiment',
        title: str = 'Sentiment Over Time',
        height: int = 400
    ) -> go.Figure:
        """
        Create a line chart showing sentiment trends over time.
        
        Args:
            data: DataFrame containing the data
            time_col: Name of the datetime column
            sentiment_col: Name of the sentiment column
            title: Chart title
            height: Chart height in pixels
            
        Returns:
            Plotly figure object
        """
        if data.empty or time_col not in data.columns or sentiment_col not in data.columns:
            logging.warning("Invalid data or missing required columns for sentiment timeline")
            return go.Figure()
        
        # Prepare data
        data = data.copy()
        data[time_col] = pd.to_datetime(data[time_col])
        data['date'] = data[time_col].dt.date
        
        # Convert sentiment to numeric, coercing errors to NaN
        data[sentiment_col] = pd.to_numeric(data[sentiment_col], errors='coerce')
        
        # Drop rows with NaN sentiment values
        valid_data = data.dropna(subset=[sentiment_col])
        
        if valid_data.empty:
            logging.warning("No valid sentiment data available for timeline")
            return go.Figure()
            
        # Calculate daily average sentiment
        daily_sentiment = valid_data.groupby('date', as_index=False)[sentiment_col].mean()
        
        # Create line chart
        fig = px.line(
            daily_sentiment,
            x='date',
            y=sentiment_col,
            title=title,
            labels={'date': 'Date', sentiment_col: 'Sentiment Score'},
            template=self.template,
            height=height
        )
        
        # Add horizontal line at y=0
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="red",
            opacity=0.5,
            annotation_text="Neutral Sentiment",
            annotation_position="bottom right"
        )
        
        # Customize layout
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            hovermode="x",
            showlegend=False,
            yaxis_range=[-1, 1]  # Sentiment scores typically range from -1 to 1
        )
        
        return fig
    
    def create_topic_sentiment_chart(
        self,
        data: pd.DataFrame,
        topic_col: str = 'topic',
        sentiment_col: str = 'sentiment',
        title: str = 'Average Sentiment by Topic',
        height: int = 500
    ) -> go.Figure:
        """
        Create a bar chart showing average sentiment by topic.
        
        Args:
            data: DataFrame containing the data
            topic_col: Name of the topic column
            sentiment_col: Name of the sentiment column
            title: Chart title
            height: Chart height in pixels
            
        Returns:
            Plotly figure object
        """
        if data.empty or topic_col not in data.columns or sentiment_col not in data.columns:
            logging.warning("Invalid data or missing required columns for topic sentiment chart")
            return go.Figure()
        
        # Make a copy of the data to avoid modifying the original
        data = data.copy()
        
        # Ensure sentiment is numeric
        data[sentiment_col] = pd.to_numeric(data[sentiment_col], errors='coerce')
        
        # Drop rows with NaN sentiment or topic values
        valid_data = data.dropna(subset=[sentiment_col, topic_col])
        
        if valid_data.empty:
            logging.warning("No valid data available for topic sentiment chart")
            return go.Figure()
        
        # Calculate average sentiment by topic
        topic_sentiment = valid_data.groupby(topic_col, as_index=False)[sentiment_col].mean()
        
        if topic_sentiment.empty:
            logging.warning("No data available after grouping by topic")
            return go.Figure()
            
        # Sort by sentiment score
        topic_sentiment = topic_sentiment.sort_values(sentiment_col, ascending=False)
        
        # Create bar chart
        fig = px.bar(
            topic_sentiment,
            x=sentiment_col,
            y=topic_col,
            orientation='h',
            title=title,
            labels={topic_col: 'Topic', sentiment_col: 'Average Sentiment'},
            template=self.template,
            height=height
        )
        
        # Add a vertical line at x=0
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="red",
            opacity=0.5
        )
        
        # Customize layout
        fig.update_layout(
            xaxis_range=[-1, 1],  # Assuming sentiment is between -1 and 1
            showlegend=False
        )
        
        return fig
    
    def create_word_cloud(
        self,
        texts: List[str],
        max_words: int = 100,
        width: int = 800,
        height: int = 400,
        background_color: str = 'white'
    ) -> go.Figure:
        """
        Create a word cloud visualization.
        
        Args:
            texts: List of text documents
            max_words: Maximum number of words to include
            width: Width of the figure in pixels
            height: Height of the figure in pixels
            background_color: Background color of the word cloud
            
        Returns:
            Plotly figure object with the word cloud
        """
        if not texts:
            logging.warning("No text provided for word cloud")
            return go.Figure()
        
        # Combine all texts
        text = ' '.join(str(t) for t in texts if t)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            max_words=max_words,
            colormap='viridis'
        ).generate(text)
        
        # Convert to plotly figure
        fig = go.Figure()
        
        # Add word cloud as an image
        fig.add_trace(
            go.Image(
                z=wordcloud.to_array(),
                hoverinfo='none',
                hovertemplate=None
            )
        )
        
        # Hide axes and layout
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0),
            width=width,
            height=height
        )
        
        return fig
    
    def create_hashtag_network(
        self,
        hashtags_list: List[List[str]],
        min_count: int = 2,
        title: str = 'Hashtag Co-occurrence Network',
        height: int = 600
    ) -> go.Figure:
        """
        Create a network graph of co-occurring hashtags.
        
        Args:
            hashtags_list: List of lists, where each sublist contains hashtags from a single post
            min_count: Minimum number of co-occurrences to show an edge
            title: Chart title
            height: Figure height in pixels
            
        Returns:
            Plotly figure object with the network graph
        """
        if not hashtags_list:
            logging.warning("No hashtag data provided")
            return go.Figure()
        
        # Count co-occurrences
        co_occurrences = defaultdict(int)
        nodes = set()
        
        for hashtags in hashtags_list:
            # Convert to lowercase and remove '#' for consistency
            hashtags = [h.lower().lstrip('#') for h in hashtags if h]
            
            # Update nodes
            nodes.update(hashtags)
            
            # Update co-occurrences
            for i in range(len(hashtags)):
                for j in range(i + 1, len(hashtags)):
                    pair = tuple(sorted([hashtags[i], hashtags[j]]))
                    co_occurrences[pair] += 1
        
        # Filter by minimum count
        edges = [(pair[0], pair[1], count) 
                for pair, count in co_occurrences.items() 
                if count >= min_count]
        
        if not edges:
            logging.warning("No co-occurring hashtags found")
            return go.Figure()
        
        # Create nodes and edges data
        nodes_list = list(nodes)
        node_indices = {node: i for i, node in enumerate(nodes_list)}
        
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for src, tgt, weight in edges:
            edge_x.extend([node_indices[src], node_indices[tgt], None])
            edge_y.extend([node_indices[tgt], node_indices[src], None])
            edge_weights.extend([weight, weight, 0])
        
        # Create network graph
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
        )
        
        # Add nodes
        fig.add_trace(
            go.Scatter(
                x=[0] * len(nodes_list),  # Will be updated by layout
                y=[0] * len(nodes_list),  # Will be updated by layout
                mode='markers+text',
                text=nodes_list,
                textposition="bottom center",
                marker=dict(
                    size=20,
                    color='lightblue',
                    line_width=2
                ),
                hoverinfo='text',
                hovertext=[f"Hashtag: {node}" for node in nodes_list]
            )
        )
        
        # Update layout for better visualization
        fig.update_layout(
            title=title,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=30),
            height=height,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        
        return fig
    
def create_hashtag_network(
    self,
    hashtags_list: List[List[str]],
    min_count: int = 2,
    title: str = 'Hashtag Co-occurrence Network',
    height: int = 600
) -> go.Figure:
    """
    Create a network graph of co-occurring hashtags.
    
    Args:
        hashtags_list: List of lists, where each sublist contains hashtags from a single post
        min_count: Minimum number of co-occurrences to show an edge
        title: Chart title
        height: Figure height in pixels
        
    Returns:
        Plotly figure object with the network graph
    """
    if not hashtags_list:
        logging.warning("No hashtag data provided")
        return go.Figure()
        
    # Count co-occurrences
    co_occurrences = defaultdict(int)
    nodes = set()
    
    for hashtags in hashtags_list:
        # Convert to lowercase and remove '#' for consistency
        hashtags = [h.lower().lstrip('#') for h in hashtags if h]
        
        # Update nodes
        nodes.update(hashtags)
        # Define layout
        fig = make_subplots(
            rows=3, cols=2,
            specs=[
                [{"type": "scatter", "rowspan": 2}, {"type": "scatter"}],
                [None, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ],
            subplot_titles=(
                "Topic Trends Over Time",
                "Sentiment Over Time",
                "Average Sentiment by Topic",
                "Word Cloud",
                "Hashtag Network"
            )
        )
        
        # Add visualizations
        topic_trends = self.create_topic_trends(data)
        sentiment_timeline = self.create_sentiment_timeline(data)
        topic_sentiment = self.create_topic_sentiment_chart(data)
        
        # Add traces from figures to subplots
        for trace in topic_trends.data:
            fig.add_trace(trace, row=1, col=1)
            
        for trace in sentiment_timeline.data:
            fig.add_trace(trace, row=1, col=2)
            
        for trace in topic_sentiment.data:
            fig.add_trace(trace, row=2, col=2)
        
        # Add word cloud (if text data is available)
        if 'text' in data.columns:
            wordcloud = self.create_word_cloud(data['text'].dropna().tolist())
            for trace in wordcloud.data:
                fig.add_trace(trace, row=3, col=1)
        
        # Add hashtag network (if hashtag data is available)
        if 'hashtags' in data.columns:
            hashtag_network = self.create_hashtag_network(data['hashtags'].dropna().tolist())
            for trace in hashtag_network.data:
                fig.add_trace(trace, row=3, col=2)
        
        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            template=self.template,
            title_text="Social Media Analysis Dashboard",
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        return fig

# Helper function for backward compatibility
def create_dashboard(data: pd.DataFrame) -> go.Figure:
    """
    Helper function to create a dashboard from a DataFrame.
    
    Args:
        data: DataFrame containing the analysis results
        
    Returns:
        Plotly figure object with the dashboard
    """
    dashboard = Dashboard()
    return dashboard.create_dashboard(data)
