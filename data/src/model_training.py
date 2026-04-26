"""
Model training module for sentiment analysis
This module trains multiple ML models on the preprocessed data
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


def load_processed_data(data_dir='data/processed'):
    """
    Load preprocessed data

    Args:
        data_dir (str): Directory containing processed data

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print("Loading preprocessed data...")

    train_df = pd.read_csv(f'{data_dir}/train.csv')
    test_df = pd.read_csv(f'{data_dir}/test.csv')

    X_train = train_df['text'].values
    y_train = train_df['sentiment'].values
    X_test = test_df['text'].values
    y_test = test_df['sentiment'].values

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def create_features(X_train, X_test, max_features=5000):
    """
    Create TF-IDF features from text

    Args:
        X_train: Training text data
        X_test: Testing text data
        max_features (int): Maximum number of features

    Returns:
        tuple: X_train_tfidf, X_test_tfidf, vectorizer
    """
    print(f"\nCreating TF-IDF features (max_features={max_features})...")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # unigrams and bigrams
        min_df=2,
        max_df=0.8
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"Feature matrix shape: {X_train_tfidf.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    return X_train_tfidf, X_test_tfidf, vectorizer


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Train Logistic Regression model

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels

    Returns:
        tuple: model, train_accuracy, test_accuracy
    """
    print("\n" + "="*60)
    print("Training Logistic Regression Model")
    print("="*60)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        multi_class='multinomial',
        solver='lbfgs'
    )

    print("Fitting model...")
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred,
                                target_names=['Negative', 'Neutral', 'Positive']))

    return model, train_accuracy, test_accuracy


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train Random Forest model

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels

    Returns:
        tuple: model, train_accuracy, test_accuracy
    """
    print("\n" + "="*60)
    print("Training Random Forest Model")
    print("="*60)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    print("Fitting model...")
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred,
                                target_names=['Negative', 'Neutral', 'Positive']))

    return model, train_accuracy, test_accuracy


def train_neural_network(X_train, y_train, X_test, y_test):
    """
    Train Neural Network (MLP) model

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels

    Returns:
        tuple: model, train_accuracy, test_accuracy
    """
    print("\n" + "="*60)
    print("Training Neural Network (MLP) Model")
    print("="*60)

    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )

    print("Fitting model...")
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred,
                                target_names=['Negative', 'Neutral', 'Positive']))

    return model, train_accuracy, test_accuracy


def save_models(models_dict, vectorizer, output_dir='models/saved_models'):
    """
    Save trained models

    Args:
        models_dict (dict): Dictionary of trained models
        vectorizer: Fitted TF-IDF vectorizer
        output_dir (str): Output directory
    """
    print("\n" + "="*60)
    print("Saving Models")
    print("="*60)

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save vectorizer
    vectorizer_path = f'{output_dir}/tfidf_vectorizer.pkl'
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Vectorizer saved: {vectorizer_path}")

    # Save models
    for model_name, model in models_dict.items():
        model_path = f'{output_dir}/{model_name}.pkl'
        joblib.dump(model, model_path)
        print(f"Model saved: {model_path}")

    print("\nAll models saved successfully!")


def main():
    """
    Main training pipeline
    """
    print("\n" + "="*70)
    print(" "*15 + "SENTIMENT ANALYSIS - MODEL TRAINING")
    print("="*70)

    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()

    # Create features
    X_train_tfidf, X_test_tfidf, vectorizer = create_features(X_train, X_test)

    # Store results
    results = {}
    models_dict = {}

    # Train Logistic Regression
    lr_model, lr_train_acc, lr_test_acc = train_logistic_regression(
        X_train_tfidf, y_train, X_test_tfidf, y_test
    )
    results['Logistic Regression'] = {
        'train_accuracy': lr_train_acc,
        'test_accuracy': lr_test_acc
    }
    models_dict['logistic_regression'] = lr_model

    # Train Random Forest
    rf_model, rf_train_acc, rf_test_acc = train_random_forest(
        X_train_tfidf, y_train, X_test_tfidf, y_test
    )
    results['Random Forest'] = {
        'train_accuracy': rf_train_acc,
        'test_accuracy': rf_test_acc
    }
    models_dict['random_forest'] = rf_model

    # Train Neural Network
    nn_model, nn_train_acc, nn_test_acc = train_neural_network(
        X_train_tfidf, y_train, X_test_tfidf, y_test
    )
    results['Neural Network'] = {
        'train_accuracy': nn_train_acc,
        'test_accuracy': nn_test_acc
    }
    models_dict['neural_network'] = nn_model

    # Save models
    save_models(models_dict, vectorizer)

    # Summary
    print("\n" + "="*70)
    print(" "*20 + "MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<25} {'Train Accuracy':<20} {'Test Accuracy':<20}")
    print("-"*70)

    for model_name, scores in results.items():
        print(f"{model_name:<25} {scores['train_accuracy']:<20.4f} {scores['test_accuracy']:<20.4f}")

    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    print("\n" + "="*70)
    print(f"BEST MODEL: {best_model[0]} (Test Accuracy: {best_model[1]['test_accuracy']:.4f})")
    print("="*70)

    print("\n" + "="*70)
    print("MODEL TRAINING COMPLETE!")
    print("="*70)
    print("\nNext step: Run 'python src/model_evaluation.py' for detailed evaluation")
    print("Or run 'streamlit run app.py' to use the web interface")


if __name__ == "__main__":
    main()
