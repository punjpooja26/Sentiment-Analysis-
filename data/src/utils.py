"""
Utility functions for the sentiment analysis project
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


def clean_text(text):
    """
    Clean and preprocess text data

    Args:
        text (str): Raw text to clean

    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


def remove_stopwords(text):
    """
    Remove stopwords from text

    Args:
        text (str): Text to process

    Returns:
        str: Text with stopwords removed
    """
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)


def lemmatize_text(text):
    """
    Lemmatize text

    Args:
        text (str): Text to lemmatize

    Returns:
        str: Lemmatized text
    """
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)


def preprocess_text(text):
    """
    Complete text preprocessing pipeline

    Args:
        text (str): Raw text

    Returns:
        str: Fully preprocessed text
    """
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text


def rating_to_sentiment(rating):
    """
    Convert rating to sentiment label

    Args:
        rating (int): Rating value (1-5)

    Returns:
        str: Sentiment label (Positive, Negative, Neutral)
    """
    if rating >= 4:
        return 'Positive'
    elif rating <= 2:
        return 'Negative'
    else:
        return 'Neutral'


def sentiment_to_numeric(sentiment):
    """
    Convert sentiment label to numeric value

    Args:
        sentiment (str): Sentiment label

    Returns:
        int: Numeric representation
    """
    sentiment_map = {
        'Negative': 0,
        'Neutral': 1,
        'Positive': 2
    }
    return sentiment_map.get(sentiment, 1)


def numeric_to_sentiment(numeric):
    """
    Convert numeric value to sentiment label

    Args:
        numeric (int): Numeric representation

    Returns:
        str: Sentiment label
    """
    sentiment_map = {
        0: 'Negative',
        1: 'Neutral',
        2: 'Positive'
    }
    return sentiment_map.get(numeric, 'Neutral')


if __name__ == "__main__":
    # Test the functions
    sample_text = "This is a GREAT product! I love it so much!!! http://example.com"
    print("Original:", sample_text)
    print("Cleaned:", clean_text(sample_text))
    print("Preprocessed:", preprocess_text(sample_text))
    print("\nRating to Sentiment:")
    for rating in range(1, 6):
        print(f"Rating {rating}: {rating_to_sentiment(rating)}")
