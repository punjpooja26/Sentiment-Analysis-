"""
Data preprocessing module for sentiment analysis
This module handles data loading, cleaning, and preparation for model training
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from utils import preprocess_text, rating_to_sentiment, sentiment_to_numeric
import warnings
warnings.filterwarnings('ignore')


def create_sample_dataset(output_path='data/raw/sample_reviews.csv', n_samples=1000):
    """
    Create a sample dataset of product reviews for demonstration

    Args:
        output_path (str): Path to save the dataset
        n_samples (int): Number of samples to generate
    """
    print(f"Creating sample dataset with {n_samples} reviews...")

    # Sample positive reviews
    positive_reviews = [
        "This product is amazing! Exceeded all my expectations.",
        "Absolutely love it! Best purchase I've made this year.",
        "Outstanding quality and fast shipping. Highly recommend!",
        "Perfect! Exactly what I was looking for.",
        "Excellent product, great value for money.",
        "I'm very impressed with the quality. Will buy again!",
        "Fantastic! Works perfectly and arrived quickly.",
        "Superior quality, couldn't be happier with this purchase.",
        "This is exactly what I needed. Five stars!",
        "Wonderful product! My family loves it too.",
        "Incredible quality for the price. Very satisfied.",
        "Best product in its category. Highly recommended!",
        "Superb! Exceeded all expectations.",
        "Amazing quality and great customer service.",
        "Perfect condition and fast delivery. Love it!"
    ]

    # Sample negative reviews
    negative_reviews = [
        "Terrible product. Broke after one use.",
        "Very disappointed. Not as described.",
        "Poor quality. Would not recommend.",
        "Waste of money. Don't buy this.",
        "Horrible experience. Product arrived damaged.",
        "Not worth it. Very cheap quality.",
        "Terrible! Stopped working after a week.",
        "Disappointed with the quality. Returning it.",
        "Bad product. Save your money.",
        "Does not work as advertised. Very upset.",
        "Awful quality. Complete waste of money.",
        "Terrible purchase. Regret buying this.",
        "Poor design and cheap materials.",
        "Not what I expected. Very disappointed.",
        "Broke immediately. Don't waste your time."
    ]

    # Sample neutral reviews
    neutral_reviews = [
        "It's okay. Nothing special but does the job.",
        "Average product. Could be better.",
        "Decent quality for the price.",
        "It works fine. Nothing to complain about.",
        "Acceptable product. Meets basic expectations.",
        "It's alright. Not great, not terrible.",
        "Fair quality. Gets the job done.",
        "Mediocre product. Expected more.",
        "It's okay for the price point.",
        "Average quality. Nothing exceptional.",
        "Works as expected. Nothing more, nothing less.",
        "Standard product. No surprises.",
        "It's fine. Does what it's supposed to do.",
        "Neutral opinion. Could go either way.",
        "Adequate for basic needs."
    ]

    # Create balanced dataset
    reviews = []
    ratings = []

    # Generate samples
    n_per_class = n_samples // 3

    for i in range(n_per_class):
        # Positive reviews (4-5 stars)
        reviews.append(np.random.choice(positive_reviews))
        ratings.append(np.random.choice([4, 5]))

    for i in range(n_per_class):
        # Negative reviews (1-2 stars)
        reviews.append(np.random.choice(negative_reviews))
        ratings.append(np.random.choice([1, 2]))

    for i in range(n_samples - 2 * n_per_class):
        # Neutral reviews (3 stars)
        reviews.append(np.random.choice(neutral_reviews))
        ratings.append(3)

    # Create DataFrame
    df = pd.DataFrame({
        'review_text': reviews,
        'rating': ratings
    })

    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Sample dataset created successfully at: {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"\nRating distribution:")
    print(df['rating'].value_counts().sort_index())

    return df


def load_data(file_path='data/raw/sample_reviews.csv'):
    """
    Load data from CSV file

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pandas.DataFrame: Loaded data
    """
    print(f"Loading data from {file_path}...")

    if not os.path.exists(file_path):
        print(f"File not found. Creating sample dataset...")
        return create_sample_dataset(file_path)

    df = pd.read_csv(file_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


def preprocess_data(df):
    """
    Preprocess the dataset

    Args:
        df (pandas.DataFrame): Raw data

    Returns:
        pandas.DataFrame: Preprocessed data
    """
    print("\nPreprocessing data...")

    # Make a copy
    df_processed = df.copy()

    # Handle missing values
    df_processed = df_processed.dropna(subset=['review_text', 'rating'])

    # Convert rating to sentiment
    df_processed['sentiment'] = df_processed['rating'].apply(rating_to_sentiment)

    # Preprocess text
    print("Cleaning and preprocessing text...")
    df_processed['processed_text'] = df_processed['review_text'].apply(preprocess_text)

    # Remove empty texts
    df_processed = df_processed[df_processed['processed_text'].str.len() > 0]

    # Convert sentiment to numeric
    df_processed['sentiment_numeric'] = df_processed['sentiment'].apply(sentiment_to_numeric)

    print(f"Preprocessing complete. Final shape: {df_processed.shape}")
    print(f"\nSentiment distribution:")
    print(df_processed['sentiment'].value_counts())

    return df_processed


def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets

    Args:
        df (pandas.DataFrame): Preprocessed data
        test_size (float): Proportion of test set
        random_state (int): Random seed

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print(f"\nSplitting data into train ({int((1-test_size)*100)}%) and test ({int(test_size*100)}%) sets...")

    X = df['processed_text']
    y = df['sentiment_numeric']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def save_processed_data(X_train, X_test, y_train, y_test, output_dir='data/processed'):
    """
    Save processed data to files

    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        output_dir (str): Output directory
    """
    print(f"\nSaving processed data to {output_dir}...")

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Combine into DataFrames
    train_df = pd.DataFrame({
        'text': X_train,
        'sentiment': y_train
    })

    test_df = pd.DataFrame({
        'text': X_test,
        'sentiment': y_test
    })

    # Save to CSV
    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    test_df.to_csv(f'{output_dir}/test.csv', index=False)

    print("Processed data saved successfully!")


def main():
    """
    Main preprocessing pipeline
    """
    print("="*60)
    print("SENTIMENT ANALYSIS - DATA PREPROCESSING")
    print("="*60)

    # Load or create data
    df = load_data('data/raw/sample_reviews.csv')

    # Display sample data
    print("\nSample raw data:")
    print(df.head())

    # Preprocess data
    df_processed = preprocess_data(df)

    # Display sample processed data
    print("\nSample processed data:")
    print(df_processed[['review_text', 'processed_text', 'sentiment']].head())

    # Split data
    X_train, X_test, y_train, y_test = split_data(df_processed)

    # Save processed data
    save_processed_data(X_train, X_test, y_train, y_test)

    print("\n" + "="*60)
    print("DATA PREPROCESSING COMPLETE!")
    print("="*60)
    print("\nNext step: Run 'python src/model_training.py' to train models")


if __name__ == "__main__":
    main()
