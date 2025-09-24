import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_sample_data():
    """Generate sample social media data for the last 2 years."""
    # Create output directory if it doesn't exist
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Date range: last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)
    
    # Generate dates
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Sample topics and their weights
    topics = [
        'technology', 'politics', 'sports', 'entertainment', 'health',
        'business', 'science', 'education', 'travel', 'food'
    ]
    topic_weights = [0.2, 0.15, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05, 0.03, 0.02]
    
    # Generate sample posts
    posts = []
    post_id = 1
    
    for date in date_range:
        # Generate between 10-20 posts per day
        num_posts = random.randint(10, 20)
        
        for _ in range(num_posts):
            # Select a random topic
            topic = random.choices(topics, weights=topic_weights, k=1)[0]
            
            # Generate random sentiment (-1 to 1)
            sentiment = round(random.uniform(-1, 1), 2)
            
            # Generate engagement metrics
            likes = random.randint(0, 1000)
            shares = random.randint(0, 500)
            comments = random.randint(0, 200)
            
            # Generate realistic post text based on topic
            post_texts = {
                'technology': [
                    f"Just got the new {random.choice(['iPhone', 'Samsung', 'Google Pixel'])} and it's amazing! #tech #gadgets",
                    f"The future of {random.choice(['AI', 'machine learning', 'blockchain'])} looks promising. What do you think?",
                    f"{random.choice(['How to', '5 reasons why', 'The complete guide to'])} {random.choice(['Python', 'JavaScript', 'React', 'Data Science'])} - check it out!"
                ],
                'politics': [
                    f"{random.choice(['Interesting take on', 'Breaking news about', 'My thoughts on'])} the latest {random.choice(['election', 'policy change', 'political debate'])}.",
                    f"{random.choice(['Why', 'How', 'What'])} {random.choice(['the government', 'politicians', 'voters'])} should know about {random.choice(['climate change', 'the economy', 'healthcare'])}",
                    f"{random.choice(['Just watched', 'Read an article about', 'Heard an interesting point about'])} {random.choice(['foreign policy', 'domestic affairs', 'political reform'])}"
                ],
                'sports': [
                    f"What a game! {random.choice(['The team', 'They', 'We'])} {random.choice(['played', 'performed'])} {random.choice(['amazingly', 'terribly', 'okay'])} today.",
                    f"{random.choice(['Prediction', 'Thoughts', 'Analysis'])} for the upcoming {random.choice(['match', 'game', 'tournament'])}?",
                    f"{random.choice(['Incredible', 'Unbelievable', 'Amazing'])} {random.choice(['goal', 'play', 'moment'])} from today's game!"
                ],
                'entertainment': [
                    f"{random.choice(['Just watched', 'Finally saw', 'Checked out'])} {random.choice(['the new movie', 'the latest episode', 'this show'])} - {random.choice(['loved it!', 'hated it.', 'meh.'])}",
                    f"{random.choice(['Can we talk about', 'Thoughts on', 'Just finished'])} {random.choice(['the latest season', 'this series', 'that finale'])}?",
                    f"{random.choice(['Recommendations', 'Looking for suggestions'])} for {random.choice(['movies', 'TV shows', 'music'])} to {random.choice(['watch', 'listen to', 'binge'])} this weekend!"
                ],
                'health': [
                    f"{random.choice(['Tips', 'Advice', 'My experience'])} on {random.choice(['mental health', 'fitness', 'nutrition', 'wellness'])}",
                    f"{random.choice(['Trying out', 'Started', 'Learning about'])} {random.choice(['meditation', 'yoga', 'a new workout routine'])} - {random.choice(['loving it so far!', 'it\'s challenging but rewarding.', 'any tips?'])}",
                    f"{random.choice(['The importance of', 'Why you should care about', 'Breaking down'])} {random.choice(['sleep', 'hydration', 'exercise', 'mental health'])}"
                ]
            }
            
            # Default text if topic not found
            default_text = f"Check out this interesting content about {topic}! #{topic}"
            
            # Create post
            post = {
                'post_id': f'post_{post_id}',
                'created_at': date + timedelta(minutes=random.randint(0, 1439)),
                'platform': random.choice(['Twitter', 'Reddit', 'Facebook', 'Instagram']),
                'topic': topic,
                'sentiment': sentiment,
                'likes': likes,
                'shares': shares,
                'comments': comments,
                'text': random.choice(post_texts.get(topic, [default_text])),
                'author': f'user_{random.randint(1, 1000)}',
                'url': f'https://{random.choice(["twitter.com", "reddit.com", "facebook.com", "instagram.com"])}/user_{random.randint(1, 1000)}/post/{post_id}'
            }
            
            posts.append(post)
            post_id += 1
    
    # Create DataFrame
    df = pd.DataFrame(posts)
    
    # Save to CSV
    output_file = output_dir / 'social_media_data_2years.csv'
    df.to_csv(output_file, index=False)
    print(f"Sample data saved to: {output_file}")
    print(f"Total posts generated: {len(df)}")
    
    return df

if __name__ == "__main__":
    generate_sample_data()
