import tweepy
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

# Try to import tweepy, but make it optional
try:
    import tweepy
    TWEETPY_AVAILABLE = True
except ImportError:
    TWEETPY_AVAILABLE = False
    logging.warning("tweepy not installed. Twitter data collection will not be available.")

class TwitterCollector:
    """
    A class to collect tweets using Twitter API v2.
    Requires Twitter API credentials to be set in environment variables.
    """
    
    def __init__(self):
        """Initialize Twitter API client with environment variables."""
        if not TWEETPY_AVAILABLE:
            raise ImportError("tweepy is required for Twitter data collection. "
                           "Install it with: pip install tweepy")
            
        self.client = self._authenticate()
        
    def _authenticate(self):
        """Authenticate with Twitter API using environment variables."""
        try:
            client = tweepy.Client(
                bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
                consumer_key=os.getenv('TWITTER_API_KEY'),
                consumer_secret=os.getenv('TWITTER_API_SECRET'),
                access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
                access_token_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
                wait_on_rate_limit=True
            )
            return client
        except Exception as e:
            logging.error(f"Failed to authenticate with Twitter API: {e}")
            raise
    
    def search(
        self, 
        query: str, 
        max_results: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Search for tweets matching the given query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return (10-100)
            start_time: Start time for search (default: 7 days ago)
            end_time: End time for search (default: now)
            
        Returns:
            DataFrame containing tweets and metadata
        """
        if start_time is None:
            start_time = datetime.utcnow() - timedelta(days=7)
        if end_time is None:
            end_time = datetime.utcnow()
            
        try:
            # Format times as ISO 8601 strings
            start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            end_time_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Define the fields to return
            tweet_fields = [
                'created_at', 'public_metrics', 'text', 'author_id',
                'context_annotations', 'entities'
            ]
            
            # Make the API request
            response = self.client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),  # API limit is 100 per request
                start_time=start_time_str,
                end_time=end_time_str,
                tweet_fields=tweet_fields,
                expansions=['author_id', 'referenced_tweets.id'],
                user_fields=['username', 'name', 'profile_image_url'],
                place_fields=['country_code', 'geo']
            )
            
            # Process the response
            tweets = []
            if response.data:
                for tweet in response.data:
                    tweet_data = {
                        'id': tweet.id,
                        'created_at': tweet.created_at,
                        'text': tweet.text,
                        'author_id': tweet.author_id,
                        'retweet_count': tweet.public_metrics['retweet_count'],
                        'reply_count': tweet.public_metrics['reply_count'],
                        'like_count': tweet.public_metrics['like_count'],
                        'quote_count': tweet.public_metrics['quote_count'],
                        'source': 'twitter'
                    }
                    
                    # Add entities if available
                    if hasattr(tweet, 'entities'):
                        tweet_data['hashtags'] = [tag['tag'] for tag in tweet.entities.get('hashtags', [])]
                        tweet_data['mentions'] = [mention['username'] for mention in tweet.entities.get('mentions', [])]
                        tweet_data['urls'] = [url['expanded_url'] for url in tweet.entities.get('urls', [])]
                    
                    tweets.append(tweet_data)
            
            # Convert to DataFrame
            df = pd.DataFrame(tweets)
            
            # Add user information if available
            if hasattr(response, 'includes') and 'users' in response.includes:
                users = {user.id: user for user in response.includes['users']}
                df['username'] = df['author_id'].map(lambda x: users[x].username if x in users else None)
                df['user_name'] = df['author_id'].map(lambda x: users[x].name if x in users else None)
            
            return df
            
        except Exception as e:
            logging.error(f"Error searching tweets: {e}")
            return pd.DataFrame()
    
    def get_user_timeline(
        self, 
        username: str, 
        max_results: int = 100
    ) -> pd.DataFrame:
        """
        Get tweets from a specific user's timeline.
        
        Args:
            username: Twitter username (without @)
            max_results: Maximum number of tweets to return (10-100)
            
        Returns:
            DataFrame containing user's tweets
        """
        try:
            # Get user ID from username
            user = self.client.get_user(username=username)
            if not user.data:
                return pd.DataFrame()
                
            # Get user's tweets
            tweets = self.client.get_users_tweets(
                id=user.data.id,
                max_results=min(max_results, 100),
                tweet_fields=['created_at', 'public_metrics', 'context_annotations', 'entities'],
                expansions=['author_id', 'referenced_tweets.id']
            )
            
            # Process tweets
            tweet_list = []
            if tweets.data:
                for tweet in tweets.data:
                    tweet_list.append({
                        'id': tweet.id,
                        'created_at': tweet.created_at,
                        'text': tweet.text,
                        'author_id': tweet.author_id,
                        'retweet_count': tweet.public_metrics['retweet_count'],
                        'reply_count': tweet.public_metrics['reply_count'],
                        'like_count': tweet.public_metrics['like_count'],
                        'quote_count': tweet.public_metrics['quote_count'],
                        'source': 'twitter'
                    })
            
            return pd.DataFrame(tweet_list)
            
        except Exception as e:
            logging.error(f"Error getting user timeline for @{username}: {e}")
            return pd.DataFrame()
