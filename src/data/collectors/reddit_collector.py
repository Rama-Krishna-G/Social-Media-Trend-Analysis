import praw
import pandas as pd
import os
from datetime import datetime
from typing import List, Dict, Optional
import logging

class RedditCollector:
    """
    A class to collect posts and comments from Reddit using PRAW.
    Requires Reddit API credentials to be set in environment variables.
    """
    
    def __init__(self):
        """Initialize Reddit API client with environment variables."""
        self.reddit = self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Reddit API using environment variables."""
        try:
            reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=os.getenv('REDDIT_USER_AGENT', 'social-media-analyzer/1.0')
            )
            return reddit
        except Exception as e:
            logging.error(f"Failed to authenticate with Reddit API: {e}")
            raise
    
    def search(
        self,
        query: str,
        subreddit: str = 'all',
        limit: int = 100,
        sort: str = 'relevance',
        time_filter: str = 'month'
    ) -> pd.DataFrame:
        """
        Search for posts on Reddit matching the given query.
        
        Args:
            query: Search query string
            subreddit: Subreddit to search in (default: 'all')
            limit: Maximum number of posts to return (1-1000)
            sort: Sort method ('relevance', 'hot', 'top', 'new', 'comments')
            time_filter: Time filter for search ('all', 'day', 'hour', 'month', 'week', 'year')
            
        Returns:
            DataFrame containing posts and metadata
        """
        try:
            subreddit = self.reddit.subreddit(subreddit)
            
            # Perform search based on sort method
            if sort == 'hot':
                submissions = subreddit.hot(limit=limit)
            elif sort == 'new':
                submissions = subreddit.new(limit=limit)
            elif sort == 'top':
                submissions = subreddit.top(limit=limit, time_filter=time_filter)
            elif sort == 'rising':
                submissions = subreddit.rising(limit=limit)
            else:  # relevance
                submissions = subreddit.search(
                    query=query,
                    limit=limit,
                    sort=sort,
                    time_filter=time_filter
                )
            
            # Process submissions
            posts = []
            for submission in submissions:
                try:
                    post_data = {
                        'id': submission.id,
                        'created_utc': datetime.utcfromtimestamp(submission.created_utc),
                        'title': submission.title,
                        'text': submission.selftext,
                        'author': str(submission.author) if submission.author else '[deleted]',
                        'score': submission.score,
                        'upvote_ratio': submission.upvote_ratio,
                        'num_comments': submission.num_comments,
                        'subreddit': str(submission.subreddit),
                        'url': submission.url,
                        'permalink': f"https://reddit.com{submission.permalink}",
                        'is_self': submission.is_self,
                        'over_18': submission.over_18,
                        'source': 'reddit'
                    }
                    
                    # Add additional metadata if available
                    if hasattr(submission, 'link_flair_text'):
                        post_data['flair'] = submission.link_flair_text
                    
                    posts.append(post_data)
                except Exception as e:
                    logging.warning(f"Error processing submission {submission.id}: {e}")
                    continue
            
            return pd.DataFrame(posts)
            
        except Exception as e:
            logging.error(f"Error searching Reddit: {e}")
            return pd.DataFrame()
    
    def get_subreddit_posts(
        self,
        subreddit: str,
        limit: int = 100,
        sort: str = 'hot',
        time_filter: str = 'month'
    ) -> pd.DataFrame:
        """
        Get posts from a specific subreddit.
        
        Args:
            subreddit: Name of the subreddit
            limit: Maximum number of posts to return (1-1000)
            sort: Sort method ('hot', 'new', 'top', 'rising')
            time_filter: Time filter for 'top' sort ('all', 'day', 'hour', 'month', 'week', 'year')
            
        Returns:
            DataFrame containing posts from the subreddit
        """
        return self.search(
            query='',
            subreddit=subreddit,
            limit=limit,
            sort=sort,
            time_filter=time_filter
        )
    
    def get_comments(
        self,
        submission_id: str,
        limit: int = 100,
        sort: str = 'top'
    ) -> pd.DataFrame:
        """
        Get comments from a specific submission.
        
        Args:
            submission_id: ID of the submission
            limit: Maximum number of comments to return (1-1000)
            sort: Sort method ('confidence', 'top', 'new', 'controversial', 'old', 'random')
            
        Returns:
            DataFrame containing comments from the submission
        """
        try:
            submission = self.reddit.submission(id=submission_id)
            submission.comment_sort = sort
            submission.comments.replace_more(limit=0)  # Remove MoreComments
            
            comments = []
            for comment in submission.comments.list():
                try:
                    if not comment.author:  # Skip deleted comments
                        continue
                        
                    comment_data = {
                        'id': comment.id,
                        'created_utc': datetime.utcfromtimestamp(comment.created_utc),
                        'author': str(comment.author),
                        'score': comment.score,
                        'text': comment.body,
                        'parent_id': comment.parent_id,
                        'is_submitter': comment.is_submitter,
                        'submission_id': submission_id,
                        'source': 'reddit_comment'
                    }
                    comments.append(comment_data)
                except Exception as e:
                    logging.warning(f"Error processing comment {comment.id}: {e}")
                    continue
            
            return pd.DataFrame(comments)
            
        except Exception as e:
            logging.error(f"Error getting comments for submission {submission_id}: {e}")
            return pd.DataFrame()
