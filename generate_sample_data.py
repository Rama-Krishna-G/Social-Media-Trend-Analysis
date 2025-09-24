import os
import pandas as pd
from src.data.generators.dummy_data import DummyDataGenerator

def generate_sample_data():
    """
    Generate sample social media data and save it to a CSV file.
    """
    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Initialize the data generator
    generator = DummyDataGenerator(seed=42)
    
    # Generate data for different time periods to show trends
    print("Generating sample data for Q2 2023...")
    
    # Generate data for each month in Q2 2023
    months = [
        ('2023-04-01', '2023-04-30'),
        ('2023-05-01', '2023-05-31'),
        ('2023-06-01', '2023-06-30')
    ]
    
    all_data = []
    
    for start_date, end_date in months:
        # Generate data for this month
        monthly_data = generator.generate_dataset(
            num_posts=200,  # 200 posts per month
            platforms=['twitter', 'reddit'],
            start_date=start_date,
            end_date=end_date
        )
        all_data.append(monthly_data)
    
    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    output_file = 'data/raw/sample_social_media_data.csv'
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} sample posts and saved to {output_file}")
    
    # Print some statistics
    print("\nSample data statistics:")
    print(f"Total posts: {len(df)}")
    print("\nPosts by platform:")
    print(df['source'].value_counts())
    print("\nPosts by topic:")
    print(df['topic'].value_counts())
    print("\nPosts by sentiment:")
    print(df['sentiment'].value_counts())

if __name__ == "__main__":
    generate_sample_data()
