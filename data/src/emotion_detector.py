"""
Emotion Detection Module
Detects emotions in text beyond basic sentiment
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


class EmotionDetector:
    """
    Emotion detection using keyword matching and machine learning
    Detects: joy, sadness, anger, fear, love, surprise
    """

    def __init__(self):
        self.emotions = ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']

        # Emotion keywords
        self.emotion_keywords = {
            'joy': ['happy', 'excited', 'wonderful', 'amazing', 'excellent', 'perfect',
                   'love', 'great', 'fantastic', 'awesome', 'brilliant', 'delighted',
                   'pleased', 'satisfied', 'glad', 'thrilled', 'enjoy', 'beautiful'],

            'sadness': ['sad', 'disappointed', 'unhappy', 'terrible', 'awful', 'poor',
                       'bad', 'worst', 'regret', 'unfortunate', 'upset', 'depressed',
                       'miserable', 'gloomy', 'sorry', 'pathetic'],

            'anger': ['angry', 'furious', 'frustrated', 'annoyed', 'irritated', 'mad',
                     'outraged', 'hate', 'disgusted', 'horrible', 'rage', 'infuriating',
                     'unacceptable', 'ridiculous', 'terrible', 'worst'],

            'fear': ['afraid', 'scared', 'worried', 'concerned', 'nervous', 'anxious',
                    'frightened', 'terrified', 'alarmed', 'hesitant', 'cautious',
                    'uncertain', 'risky', 'dangerous', 'warning'],

            'love': ['love', 'adore', 'cherish', 'treasure', 'beloved', 'favorite',
                    'passion', 'devoted', 'fond', 'affection', 'romance', 'caring',
                    'wonderful', 'precious', 'best'],

            'surprise': ['surprised', 'unexpected', 'shocking', 'amazed', 'astonished',
                        'stunned', 'incredible', 'unbelievable', 'wow', 'remarkable',
                        'extraordinary', 'sudden', 'startling', 'impressed']
        }

    def detect_emotion(self, text):
        """
        Detect emotion in text using keyword matching

        Args:
            text (str): Input text

        Returns:
            dict: Emotion scores
        """
        text_lower = text.lower()
        emotion_scores = {}

        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = score

        # Normalize scores
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {k: v/total for k, v in emotion_scores.items()}

        return emotion_scores

    def get_dominant_emotion(self, text):
        """
        Get the dominant emotion in text

        Args:
            text (str): Input text

        Returns:
            str: Dominant emotion
        """
        scores = self.detect_emotion(text)

        if sum(scores.values()) == 0:
            return 'neutral'

        return max(scores.items(), key=lambda x: x[1])[0]

    def get_emotion_intensity(self, text):
        """
        Get emotion intensity score (0-1)

        Args:
            text (str): Input text

        Returns:
            float: Intensity score
        """
        scores = self.detect_emotion(text)
        max_score = max(scores.values()) if scores else 0
        return min(max_score * 2, 1.0)  # Scale up for better visualization
