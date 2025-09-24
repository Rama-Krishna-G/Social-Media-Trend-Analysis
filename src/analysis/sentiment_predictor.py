import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os
from pathlib import Path

class SentimentPredictor:
    """
    A class for training and using a sentiment prediction model.
    """
    
    def __init__(self, model_path: str = 'models/sentiment_model.joblib'):
        """
        Initialize the sentiment predictor.
        
        Args:
            model_path: Path to save/load the trained model
        """
        self.model_path = model_path
        self.model = None
        self.vectorizer = None
        
    def train(self, texts: list, labels: list, test_size: float = 0.2, random_state: int = 42):
        """
        Train a sentiment prediction model.
        
        Args:
            texts: List of text documents
            labels: List of sentiment labels (0 for negative, 1 for neutral, 2 for positive)
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Test accuracy score
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state
        )
        
        # Create and train the model pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', LinearSVC())
        ])
        
        self.model.fit(X_train, y_train)
        
        # Save the model
        self._save_model()
        
        # Return test accuracy
        return self.model.score(X_test, y_test)
    
    def predict(self, text: str) -> tuple:
        """
        Predict the sentiment of a given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            tuple: (predicted_label, confidence_score)
        """
        if not self.model:
            self._load_model()
            
        if not self.model:
            raise ValueError("Model not trained or loaded successfully")
            
        # Get prediction and confidence
        prediction = self.model.predict([text])[0]
        decision_scores = self.model.decision_function([text])[0]
        confidence = max(decision_scores) / sum(abs(decision_scores)) if sum(abs(decision_scores)) > 0 else 0.5
        
        # Map numeric prediction to label
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        return sentiment_map.get(prediction, "Neutral"), float(confidence)
    
    def _save_model(self):
        """Save the trained model to disk."""
        if not self.model:
            return
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save the model
        joblib.dump(self.model, self.model_path)
    
    def _load_model(self):
        """Load a trained model from disk."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False
    
    @staticmethod
    def prepare_training_data(data: pd.DataFrame, text_col: str = 'text', 
                            sentiment_col: str = 'sentiment') -> tuple:
        """
        Prepare data for training the sentiment predictor.
        
        Args:
            data: DataFrame containing the data
            text_col: Name of the text column
            sentiment_col: Name of the sentiment column
            
        Returns:
            tuple: (texts, labels)
        """
        # Filter out rows with missing values
        df = data[[text_col, sentiment_col]].dropna()
        
        # Convert sentiment to numeric (assuming -1,0,1 or similar)
        if df[sentiment_col].dtype == 'object':
            sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
            df['label'] = df[sentiment_col].str.lower().map(sentiment_map)
        else:
            # Assume numeric sentiment (-1 to 1)
            df['label'] = pd.cut(
                df[sentiment_col],
                bins=[-float('inf'), -0.1, 0.1, float('inf')],
                labels=[0, 1, 2]  # 0: negative, 1: neutral, 2: positive
            )
        
        return df[text_col].tolist(), df['label'].tolist()
