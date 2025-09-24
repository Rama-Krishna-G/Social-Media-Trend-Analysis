import logging
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set numpy print options to prevent any future warnings
np.set_printoptions(suppress=True)

# Try to import spaCy and other ML dependencies
try:
    import spacy
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.cluster import KMeans
    
    # Import sentence transformers only if available
    try:
        from sentence_transformers import SentenceTransformer
        SENTENCE_TRANSFORMERS_AVAILABLE = True
    except ImportError:
        logger.warning("sentence-transformers not available. BERT-based topic modeling will be disabled.")
        SENTENCE_TRANSFORMERS_AVAILABLE = False
    
    # Import TextBlob only if available
    try:
        from textblob import TextBlob
        TEXTBLOB_AVAILABLE = True
    except ImportError:
        logger.warning("TextBlob not available. Some text processing features may be limited.")
        TEXTBLOB_AVAILABLE = False
    
    # Try to load spaCy model
    try:
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
        SPACY_AVAILABLE = True
    except Exception as e:
        logger.warning(f"Could not load spaCy model: {e}")
        SPACY_AVAILABLE = False
        
    ML_DEPS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some ML dependencies are not available: {e}")
    ML_DEPS_AVAILABLE = False
    SPACY_AVAILABLE = False
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    TEXTBLOB_AVAILABLE = False

class TopicModeler:
    """
    A class for topic modeling and analysis of social media text data.
    Supports multiple topic modeling techniques including LDA, NMF, and BERT-based clustering.
    """
    
    def __init__(self, model_type: str = 'lda', num_topics: int = 5):
        """
        Initialize the TopicModeler.
        
        Args:
            model_type: Type of topic model to use ('lda', 'nmf', or 'bert')
            num_topics: Number of topics to extract
        """
        if not ML_DEPS_AVAILABLE:
            logger.warning("ML dependencies not available. Topic modeling will be limited.")
            
        self.model_type = model_type
        self.num_topics = num_topics
        self.model = None
        self.vectorizer = None
        self.feature_names = []
        
        # Initialize sentence transformer if using BERT
        if model_type == 'bert':
            if not ML_DEPS_AVAILABLE:
                logger.warning("BERT model requires ML dependencies. Falling back to LDA.")
                self.model_type = 'lda'
            else:
                try:
                    self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
                except Exception as e:
                    logger.warning(f"Failed to load BERT model: {e}. Falling back to LDA.")
                    self.model_type = 'lda'
        else:
            self.bert_model = None
        
        # Initialize NLP model if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")
        
        if not ML_DEPS_AVAILABLE:
            logger.warning("Running with limited functionality due to missing ML dependencies")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by removing stopwords, lemmatizing, etc.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Basic cleaning
        text = text.lower().strip()
        
        # Simple preprocessing that works without spaCy
        import re
        from string import punctuation
        
        # Remove URLs, mentions, and special characters
        text = re.sub(r'https?://\S+|@\w+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # If spaCy is available, use it for better preprocessing
        if self.nlp is not None and SPACY_AVAILABLE:
            try:
                doc = self.nlp(text)
                # Lemmatize and remove stopwords and punctuation
                tokens = [token.lemma_ for token in doc 
                         if not token.is_stop and not token.is_punct and not token.is_space]
                return " ".join(tokens)
            except Exception as e:
                logger.warning(f"Error in spaCy processing: {e}")
                # Fall through to simple processing
        
        # Fallback to simple preprocessing
        return ' '.join([word for word in text.split() if word not in punctuation])
    
    def fit_transform(self, texts: List[str]) -> List[int]:
        """
        Fit the topic model and transform the input texts into topic distributions.
        
        Args:
            texts: List of text documents
            
        Returns:
            List of topic assignments for each document
        """
        if not ML_DEPS_AVAILABLE:
            logger.warning("ML dependencies not available. Returning dummy topics.")
            return [0] * len(texts) if texts else []
            
        if not texts or len(texts) == 0:
            return []
            
        try:
            # Preprocess texts
            processed_texts = [self.preprocess_text(str(text)) for text in texts]
            
            if self.model_type == 'bert':
                # Get BERT embeddings
                embeddings = self.bert_model.encode(processed_texts)
                
                # Cluster embeddings
                kmeans = KMeans(n_clusters=self.num_topics, random_state=42)
                topics = kmeans.fit_predict(embeddings)
                
                self.model = kmeans
                return topics.tolist()
                
            else:  # LDA or NMF
                # Create document-term matrix
                if self.model_type == 'nmf':
                    self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                    dtm = self.vectorizer.fit_transform(processed_texts)
                    self.model = NMF(n_components=self.num_topics, random_state=42)
                else:  # LDA (default)
                    self.vectorizer = CountVectorizer(max_features=1000, stop_words='english')
                    dtm = self.vectorizer.fit_transform(processed_texts)
                    self.model = LatentDirichletAllocation(n_components=self.num_topics, random_state=42)
                
                # Get feature names for display
                self.feature_names = self.vectorizer.get_feature_names_out()
                
                # Fit model and get topic assignments
                doc_topics = self.model.fit_transform(dtm)
                return doc_topics.argmax(axis=1).tolist()
                
        except Exception as e:
            logger.error(f"Error in topic modeling: {e}")
            # Return dummy topics in case of error
            return [0] * len(texts)
        
        return []
    
    def _fit_nmf(self, texts: List[str]) -> List[int]:
        """
        Fit and apply NMF topic modeling.
        
        Args:
            texts: List of preprocessed text documents
            
        Returns:
            List of topic assignments
        """
        # Create TF-IDF matrix
        self.vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        tfidf = self.vectorizer.fit_transform(texts)
        
        # Train NMF model
        self.model = NMF(
            n_components=self.num_topics,
            random_state=42,
            alpha=0.1,
            l1_ratio=0.5
        )
        
        # Get topic assignments
        doc_topics = self.model.fit_transform(tfidf)
        topic_assignments = doc_topics.argmax(axis=1)
        
        return topic_assignments.tolist()
    
    def _fit_bert(self, texts: List[str]) -> List[int]:
        """
        Fit and apply BERT-based topic modeling using sentence embeddings and clustering.
        
        Args:
            texts: List of preprocessed text documents
            
        Returns:
            List of topic assignments
        """
        # Get BERT embeddings
        embeddings = self.bert_model.encode(texts, show_progress_bar=False)
        
        # Cluster embeddings
        self.model = KMeans(
            n_clusters=self.num_topics,
            random_state=42,
            n_init=10
        )
        
        # Get topic assignments
        topic_assignments = self.model.fit_predict(embeddings)
        
        return topic_assignments.tolist()
    
    def get_topic_keywords(self, top_n: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get the top keywords for each topic.
        
        Args:
            top_n: Number of keywords to return per topic
            
        Returns:
            Dictionary mapping topic IDs to lists of (keyword, score) tuples
        """
        if self.model is None:
            return {}
            
        if self.model_type in ['lda', 'nmf'] and self.vectorizer is not None:
            feature_names = self.vectorizer.get_feature_names_out()
            
            if self.model_type == 'lda':
                topic_keywords = {}
                for topic_idx, topic in enumerate(self.model.components_):
                    top_keywords = [(feature_names[i], topic[i]) 
                                  for i in topic.argsort()[:-top_n-1:-1]]
                    topic_keywords[topic_idx] = top_keywords
                return topic_keywords
                
            elif self.model_type == 'nmf':
                topic_keywords = {}
                for topic_idx, topic in enumerate(self.model.components_):
                    top_keywords = [(feature_names[i], topic[i]) 
                                  for i in topic.argsort()[:-top_n-1:-1]]
                    topic_keywords[topic_idx] = top_keywords
                return topic_keywords
                
        elif self.model_type == 'bert' and hasattr(self.model, 'cluster_centers_'):
            # For BERT + KMeans, we can't directly get keywords, so we'll return empty for now
            # In a production setting, you might want to use a different approach
            # like finding the most representative documents for each cluster
            # and extracting keywords from them
            return {i: [] for i in range(self.num_topics)}
            
        return {}
    
    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze the sentiment of a text.
        
        Args:
            text: Input text
            
        Returns:
            Sentiment score between -1 (negative) and 1 (positive)
        """
        if not text or not isinstance(text, str):
            return 0.0
            
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    
    def get_topic_sentiment(self, texts: List[str], topics: List[int]) -> Dict[int, float]:
        """
        Calculate the average sentiment for each topic.
        
        Args:
            texts: List of text documents
            topics: List of topic assignments for each document
            
        Returns:
            Dictionary mapping topic IDs to average sentiment scores
        """
        if not texts or not topics or len(texts) != len(topics):
            return {}
            
        topic_sentiments = {i: [] for i in set(topics)}
        
        for text, topic in zip(texts, topics):
            sentiment = self.analyze_sentiment(text)
            topic_sentiments[topic].append(sentiment)
        
        # Calculate average sentiment per topic
        return {
            topic: sum(sentiments) / len(sentiments) 
            for topic, sentiments in topic_sentiments.items()
            if sentiments  # Avoid division by zero
        }
