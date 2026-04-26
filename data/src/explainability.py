"""
Model Explainability Module
Shows which words influenced the prediction
"""

import numpy as np
from collections import Counter
import re


class SentimentExplainer:
    """
    Explains sentiment predictions by highlighting influential words
    """

    def __init__(self):
        # Positive and negative word indicators
        self.positive_indicators = {
            'excellent': 0.9, 'amazing': 0.9, 'perfect': 0.9, 'outstanding': 0.9,
            'fantastic': 0.9, 'wonderful': 0.9, 'great': 0.8, 'good': 0.7,
            'love': 0.8, 'best': 0.9, 'awesome': 0.8, 'brilliant': 0.9,
            'superb': 0.9, 'exceptional': 0.9, 'impressed': 0.8, 'satisfied': 0.7,
            'happy': 0.7, 'pleased': 0.7, 'recommend': 0.7, 'quality': 0.6,
            'fast': 0.6, 'easy': 0.6, 'beautiful': 0.7, 'nice': 0.6
        }

        self.negative_indicators = {
            'terrible': 0.9, 'awful': 0.9, 'horrible': 0.9, 'worst': 0.9,
            'bad': 0.7, 'poor': 0.7, 'disappointed': 0.8, 'disappointing': 0.8,
            'hate': 0.9, 'waste': 0.8, 'useless': 0.8, 'broken': 0.8,
            'defective': 0.8, 'failed': 0.7, 'unhappy': 0.7, 'regret': 0.8,
            'return': 0.6, 'refund': 0.7, 'cheap': 0.6, 'slow': 0.6,
            'difficult': 0.6, 'complicated': 0.6, 'never': 0.5
        }

    def get_word_importance(self, text, model, vectorizer):
        """
        Calculate word importance using feature weights

        Args:
            text (str): Input text
            model: Trained model
            vectorizer: TF-IDF vectorizer

        Returns:
            dict: Word -> importance score
        """
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()

        # Transform text
        text_vector = vectorizer.transform([text.lower()])

        # Get non-zero features
        non_zero_features = text_vector.nonzero()[1]

        # Get word importance
        word_importance = {}

        # Try to get feature importances (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            for idx in non_zero_features:
                word = feature_names[idx]
                importance = model.feature_importances_[idx]
                word_importance[word] = importance

        # For linear models, use coefficients
        elif hasattr(model, 'coef_'):
            for idx in non_zero_features:
                word = feature_names[idx]
                # Get absolute coefficient value
                importance = abs(model.coef_[0][idx]) if len(model.coef_.shape) > 1 else abs(model.coef_[idx])
                word_importance[word] = float(importance)

        return word_importance

    def explain_prediction(self, text, sentiment, confidence):
        """
        Generate explanation for prediction

        Args:
            text (str): Input text
            sentiment (str): Predicted sentiment
            confidence (float): Confidence score

        Returns:
            dict: Explanation with influential words
        """
        words = re.findall(r'\b\w+\b', text.lower())

        # Find influential words
        positive_words = []
        negative_words = []
        neutral_words = []

        for word in set(words):
            if word in self.positive_indicators:
                positive_words.append((word, self.positive_indicators[word]))
            elif word in self.negative_indicators:
                negative_words.append((word, self.negative_indicators[word]))

        # Sort by importance
        positive_words.sort(key=lambda x: x[1], reverse=True)
        negative_words.sort(key=lambda x: x[1], reverse=True)

        # Generate explanation
        explanation = {
            'positive_words': [w[0] for w in positive_words[:5]],
            'negative_words': [w[0] for w in negative_words[:5]],
            'word_count': len(words),
            'unique_words': len(set(words))
        }

        return explanation

    def highlight_text(self, text, sentiment):
        """
        Highlight influential words in text

        Args:
            text (str): Input text
            sentiment (str): Predicted sentiment

        Returns:
            str: HTML with highlighted words
        """
        words = text.split()
        highlighted = []

        for word in words:
            word_lower = word.lower().strip('.,!?;:')

            if word_lower in self.positive_indicators:
                highlighted.append(f'<span style="background-color: #90EE90; padding: 2px 4px; border-radius: 3px;">{word}</span>')
            elif word_lower in self.negative_indicators:
                highlighted.append(f'<span style="background-color: #FFB6C6; padding: 2px 4px; border-radius: 3px;">{word}</span>')
            else:
                highlighted.append(word)

        return ' '.join(highlighted)

    def get_key_phrases(self, text):
        """
        Extract key phrases that influenced the sentiment

        Args:
            text (str): Input text

        Returns:
            list: Key phrases
        """
        phrases = []

        # Look for strong positive/negative phrases
        positive_phrases = [
            r'very good', r'really good', r'so good', r'really love',
            r'highly recommend', r'best ever', r'absolutely amazing',
            r'exceeded expectations', r'worth the money', r'great quality'
        ]

        negative_phrases = [
            r'very bad', r'really bad', r'so bad', r'waste of money',
            r'not recommend', r'worst ever', r'completely useless',
            r'disappointed', r'don\'t buy', r'poor quality'
        ]

        text_lower = text.lower()

        for pattern in positive_phrases:
            if re.search(pattern, text_lower):
                match = re.search(pattern, text_lower).group()
                phrases.append(('positive', match))

        for pattern in negative_phrases:
            if re.search(pattern, text_lower):
                match = re.search(pattern, text_lower).group()
                phrases.append(('negative', match))

        return phrases
