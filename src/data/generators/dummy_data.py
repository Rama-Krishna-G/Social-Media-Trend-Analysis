import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import logging

class DummyDataGenerator:
    """
    A class to generate realistic dummy social media data for testing and development.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the DummyDataGenerator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Common words and phrases for generating realistic text
        self.common_words = [
            'the', 'and', 'have', 'that', 'for', 'you', 'with', 'say', 'this', 'they',
            'get', 'like', 'just', 'know', 'what', 'good', 'time', 'people', 'year', 'think',
            'make', 'when', 'which', 'your', 'come', 'could', 'work', 'use', 'than', 'then'
        ]
        
        self.hashtags = [
            '#AI', '#MachineLearning', '#DataScience', '#Python', '#Tech', 
            '#ArtificialIntelligence', '#BigData', '#DeepLearning', '#NeuralNetworks',
            '#Programming', '#Coding', '#DataAnalytics', '#CloudComputing', '#IoT',
            '#Blockchain', '#CyberSecurity', '#DevOps', '#100DaysOfCode', '#WomenInTech'
        ]
        
        self.topics = [
            'Technology', 'Science', 'Politics', 'Sports', 'Entertainment',
            'Business', 'Health', 'Education', 'Environment', 'Fashion'
        ]
        
        self.sentiments = ['positive', 'neutral', 'negative']
        self.platforms = ['twitter', 'reddit']
        
        # User data for generating realistic usernames and handles
        self.first_names = ['Alex', 'Jordan', 'Taylor', 'Morgan', 'Casey', 'Riley', 'Jamie', 'Quinn', 'Avery', 'Peyton']
        self.last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Miller', 'Davis', 'Garcia', 'Rodriguez', 'Wilson']
        
    def generate_text(self, min_words: int = 5, max_words: int = 30) -> str:
        """
        Generate a random text with the specified number of words.
        
        Args:
            min_words: Minimum number of words
            max_words: Maximum number of words
            
        Returns:
            Generated text
        """
        num_words = random.randint(min_words, max_words)
        words = random.choices(self.common_words, k=num_words)
        
        # Capitalize first word and add period at the end
        if words:
            words[0] = words[0].capitalize()
            words[-1] += '.'
            
        return ' '.join(words)
    
    def generate_hashtags(self, min_hashtags: int = 0, max_hashtags: int = 5) -> List[str]:
        """
        Generate a list of random hashtags.
        
        Args:
            min_hashtags: Minimum number of hashtags
            max_hashtags: Maximum number of hashtags
            
        Returns:
            List of hashtags
        """
        num_hashtags = random.randint(min_hashtags, max_hashtags)
        return random.sample(self.hashtags, min(num_hashtags, len(self.hashtags)))
    
    def generate_username(self) -> str:
        """
        Generate a random username.
        
        Returns:
            Random username
        """
        first = random.choice(self.first_names)
        last = random.choice(self.last_names)
        num = random.randint(1, 99)
        return f"{first}{last}{num}"
    
    def generate_date(self, start_date: str = '2023-01-01', end_date: str = '2023-12-31') -> str:
        """
        Generate a random date between start_date and end_date.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            ISO formatted date string
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        delta = end - start
        random_days = random.randint(0, delta.days)
        random_date = start + timedelta(days=random_days)
        return random_date.isoformat()
    
    def generate_sentiment_score(self, sentiment: Optional[str] = None) -> float:
        """
        Generate a sentiment score between -1 and 1.
        
        Args:
            sentiment: If provided, generate score biased towards this sentiment
                       ('positive', 'neutral', 'negative')
                       
        Returns:
            Sentiment score between -1 and 1
        """
        if sentiment == 'positive':
            return random.uniform(0.2, 1.0)
        elif sentiment == 'negative':
            return random.uniform(-1.0, -0.2)
        else:  # neutral
            return random.uniform(-0.2, 0.2)
    
    def generate_twitter_post(self) -> Dict:
        """
        Generate a dummy Twitter post.
        
        Returns:
            Dictionary containing tweet data
        """
        created_at = self.generate_date()
        text = self.generate_text()
        hashtags = self.generate_hashtags()
        
        # Add hashtags to text
        if hashtags:
            text += ' ' + ' '.join(hashtags)
        
        # Generate random engagement metrics
        retweet_count = random.randint(0, 1000)
        like_count = random.randint(0, 5000)
        reply_count = random.randint(0, 200)
        
        # Generate user info
        user_id = random.randint(100000, 999999)
        username = self.generate_username()
        
        # Randomly assign a topic
        topic = random.choice(self.topics)
        
        # Generate sentiment
        sentiment = random.choice(self.sentiments)
        sentiment_score = self.generate_sentiment_score(sentiment)
        
        return {
            'id': f'tw_{random.randint(1000000000000000000, 9999999999999999999)}',
            'created_at': created_at,
            'text': text,
            'author_id': user_id,
            'username': username,
            'retweet_count': retweet_count,
            'reply_count': reply_count,
            'like_count': like_count,
            'quote_count': random.randint(0, 100),
            'hashtags': hashtags,
            'mentions': [f'@{self.generate_username()}' for _ in range(random.randint(0, 3))],
            'urls': [f'https://example.com/{random.randint(1000,9999)}' for _ in range(random.randint(0, 2))],
            'topic': topic,
            'sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'source': 'twitter'
        }
    
    def generate_reddit_post(self) -> Dict:
        """
        Generate a dummy Reddit post.
        
        Returns:
            Dictionary containing Reddit post data
        """
        created_utc = self.generate_date()
        title = self.generate_text(min_words=3, max_words=10)
        text = self.generate_text(min_words=10, max_words=200)
        
        # Generate subreddit name
        subreddit = random.choice([
            'technology', 'science', 'politics', 'worldnews', 'todayilearned',
            'explainlikeimfive', 'askscience', 'dataisbeautiful', 'programming', 'machinelearning'
        ])
        
        # Generate user info
        author = self.generate_username()
        
        # Randomly assign a topic
        topic = random.choice(self.topics)
        
        # Generate sentiment
        sentiment = random.choice(self.sentiments)
        sentiment_score = self.generate_sentiment_score(sentiment)
        
        return {
            'id': f't3_{random.randint(10000000, 99999999)}',
            'created_utc': created_utc,
            'title': title,
            'text': text,
            'author': author,
            'subreddit': subreddit,
            'score': random.randint(0, 10000),
            'upvote_ratio': round(random.uniform(0.5, 1.0), 2),
            'num_comments': random.randint(0, 500),
            'url': f'https://www.reddit.com/r/{subreddit}/comments/{random.randint(100000, 999999)}',
            'permalink': f'/r/{subreddit}/comments/{random.randint(100000, 999999)}',
            'is_self': random.choice([True, False]),
            'over_18': random.choice([True, False]),
            'topic': topic,
            'sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'source': 'reddit'
        }
    
    def generate_dataset(
        self, 
        num_posts: int = 100,
        platforms: Optional[List[str]] = None,
        start_date: str = '2023-01-01',
        end_date: str = '2023-12-31'
    ) -> pd.DataFrame:
        """
        Generate a dataset of dummy social media posts.
        
        Args:
            num_posts: Number of posts to generate
            platforms: List of platforms to include ('twitter', 'reddit')
            start_date: Start date for post timestamps
            end_date: End date for post timestamps
            
        Returns:
            DataFrame containing the generated posts
        """
        if platforms is None:
            platforms = self.platforms
            
        posts = []
        for _ in range(num_posts):
            platform = random.choice(platforms)
            
            if platform == 'twitter':
                post = self.generate_twitter_post()
            elif platform == 'reddit':
                post = self.generate_reddit_post()
            else:
                continue
                
            # Ensure the date is within the specified range
            post['created_at'] = self.generate_date(start_date, end_date)
            if 'created_utc' in post:
                post['created_utc'] = post['created_at']
                
            posts.append(post)
        
        return pd.DataFrame(posts)

# Example usage
if __name__ == "__main__":
    # Initialize the data generator
    generator = DummyDataGenerator()
    
    # Generate a dataset with 100 posts (50% Twitter, 50% Reddit)
    df = generator.generate_dataset(
        num_posts=100,
        platforms=['twitter', 'reddit'],
        start_date='2023-06-01',
        end_date='2023-06-30'
    )
    
    # Save to CSV
    output_file = 'dummy_social_media_data.csv'
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} dummy posts and saved to {output_file}")
    
    # Display basic info
    print("\nSample data:")
    print(df[['created_at', 'source', 'topic', 'sentiment']].head())
