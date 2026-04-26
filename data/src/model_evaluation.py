"""
Model evaluation module for sentiment analysis
This module provides detailed evaluation metrics and visualizations
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


def load_models_and_data():
    """
    Load saved models and test data

    Returns:
        tuple: models_dict, vectorizer, X_test, y_test
    """
    print("Loading models and test data...")

    # Load test data
    test_df = pd.read_csv('data/processed/test.csv')
    X_test = test_df['text'].values
    y_test = test_df['sentiment'].values

    # Load vectorizer
    vectorizer = joblib.load('models/saved_models/tfidf_vectorizer.pkl')

    # Load models
    models_dict = {
        'Logistic Regression': joblib.load('models/saved_models/logistic_regression.pkl'),
        'Random Forest': joblib.load('models/saved_models/random_forest.pkl'),
        'Neural Network': joblib.load('models/saved_models/neural_network.pkl')
    }

    print(f"Loaded {len(models_dict)} models")
    print(f"Test samples: {len(X_test)}")

    return models_dict, vectorizer, X_test, y_test


def evaluate_model(model, X_test_tfidf, y_test, model_name):
    """
    Evaluate a single model

    Args:
        model: Trained model
        X_test_tfidf: Test features
        y_test: True labels
        model_name (str): Name of the model

    Returns:
        dict: Evaluation metrics
    """
    print(f"\nEvaluating {model_name}...")

    # Predictions
    y_pred = model.predict(X_test_tfidf)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': y_pred
    }

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return metrics


def plot_confusion_matrix(cm, model_name, save_path=None):
    """
    Plot confusion matrix

    Args:
        cm: Confusion matrix
        model_name (str): Name of the model
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix: {save_path}")

    plt.close()


def plot_model_comparison(results, save_path=None):
    """
    Plot model comparison

    Args:
        results (dict): Dictionary of model results
        save_path (str): Path to save the plot
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    model_names = list(results.keys())

    data = {metric: [] for metric in metrics}

    for model_name in model_names:
        for metric in metrics:
            data[metric].append(results[model_name][metric])

    # Create DataFrame
    df = pd.DataFrame(data, index=model_names)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot(kind='bar', ax=ax, width=0.8)
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim([0, 1.1])
    plt.legend(title='Metrics', loc='lower right')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot: {save_path}")

    plt.close()


def create_evaluation_report(results, output_path='models/evaluation_report.txt'):
    """
    Create detailed evaluation report

    Args:
        results (dict): Dictionary of model results
        output_path (str): Path to save the report
    """
    print("\nCreating evaluation report...")

    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SENTIMENT ANALYSIS - MODEL EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")

        for model_name, metrics in results.items():
            f.write(f"\n{model_name}\n")
            f.write("-"*80 + "\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n")
            f.write("\n")

        # Best model
        best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
        f.write("\n" + "="*80 + "\n")
        f.write(f"BEST MODEL: {best_model[0]}\n")
        f.write(f"F1-Score: {best_model[1]['f1_score']:.4f}\n")
        f.write("="*80 + "\n")

    print(f"Evaluation report saved: {output_path}")


def main():
    """
    Main evaluation pipeline
    """
    print("\n" + "="*70)
    print(" "*15 + "SENTIMENT ANALYSIS - MODEL EVALUATION")
    print("="*70)

    # Check if models exist
    if not os.path.exists('models/saved_models/logistic_regression.pkl'):
        print("\nERROR: Models not found!")
        print("Please run 'python src/model_training.py' first to train the models.")
        return

    # Load models and data
    models_dict, vectorizer, X_test, y_test = load_models_and_data()

    # Transform test data
    X_test_tfidf = vectorizer.transform(X_test)

    # Evaluate all models
    results = {}

    for model_name, model in models_dict.items():
        metrics = evaluate_model(model, X_test_tfidf, y_test, model_name)
        results[model_name] = metrics

        # Plot confusion matrix
        os.makedirs('models/plots', exist_ok=True)
        plot_path = f"models/plots/{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        plot_confusion_matrix(metrics['confusion_matrix'], model_name, plot_path)

    # Create comparison plot
    plot_model_comparison(results, 'models/plots/model_comparison.png')

    # Create evaluation report
    create_evaluation_report(results)

    # Print summary
    print("\n" + "="*70)
    print(" "*20 + "EVALUATION SUMMARY")
    print("="*70)
    print(f"{'Model':<25} {'Accuracy':<12} {'F1-Score':<12}")
    print("-"*70)

    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['accuracy']:<12.4f} {metrics['f1_score']:<12.4f}")

    # Best model
    best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
    print("\n" + "="*70)
    print(f"BEST MODEL: {best_model[0]} (F1-Score: {best_model[1]['f1_score']:.4f})")
    print("="*70)

    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("- models/evaluation_report.txt")
    print("- models/plots/model_comparison.png")
    print("- models/plots/*_confusion_matrix.png")
    print("\nNext step: Run 'streamlit run app.py' to use the web interface")


if __name__ == "__main__":
    main()
