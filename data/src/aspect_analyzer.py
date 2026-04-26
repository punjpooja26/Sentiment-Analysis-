"""
Aspect-Based Sentiment Analysis
Extracts specific aspects and their sentiments from reviews
"""

import re
from collections import defaultdict


class AspectAnalyzer:
    """
    Analyzes sentiment for specific aspects mentioned in reviews
    Example: "The camera quality is excellent but battery life is terrible"
    Extracts: camera -> positive, battery -> negative
    """

    def __init__(self):
        # Common product aspects
        self.aspects = {
            'quality': ['quality', 'build', 'material', 'construction', 'durability'],
            'price': ['price', 'cost', 'value', 'expensive', 'cheap', 'affordable', 'worth'],
            'performance': ['performance', 'speed', 'fast', 'slow', 'efficient', 'power'],
            'design': ['design', 'look', 'appearance', 'style', 'color', 'size', 'aesthetic'],
            'features': ['feature', 'function', 'capability', 'option', 'setting'],
            'usability': ['easy', 'simple', 'difficult', 'complicated', 'user-friendly', 'intuitive'],
            'delivery': ['delivery', 'shipping', 'packaging', 'arrived', 'package'],
            'customer_service': ['service', 'support', 'customer', 'help', 'response'],
            'battery': ['battery', 'charge', 'charging', 'battery life', 'power'],
            'camera': ['camera', 'photo', 'picture', 'video', 'lens'],
            'screen': ['screen', 'display', 'brightness', 'resolution'],
            'sound': ['sound', 'audio', 'speaker', 'volume', 'music', 'bass']
        }

        # Positive and negative indicators
        self.positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'perfect', 'love',
            'best', 'fantastic', 'awesome', 'brilliant', 'outstanding', 'superior',
            'impressive', 'satisfied', 'happy', 'pleased', 'nice', 'fine', 'decent'
        ]

        self.negative_words = [
            'bad', 'poor', 'terrible', 'awful', 'horrible', 'worst', 'hate',
            'disappointing', 'disappointed', 'unsatisfied', 'unhappy', 'displeased',
            'inferior', 'lacking', 'defective', 'broken', 'faulty', 'useless'
        ]

    def extract_aspects(self, text):
        """
        Extract aspects mentioned in text

        Args:
            text (str): Input text

        Returns:
            list: Aspects found in text
        """
        text_lower = text.lower()
        found_aspects = []

        for aspect, keywords in self.aspects.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_aspects.append(aspect)
                    break

        return list(set(found_aspects))  # Remove duplicates

    def analyze_aspect_sentiment(self, text):
        """
        Analyze sentiment for each aspect

        Args:
            text (str): Input text

        Returns:
            dict: Aspect -> sentiment mapping
        """
        text_lower = text.lower()
        aspect_sentiments = {}

        # Extract aspects
        aspects = self.extract_aspects(text)

        # Analyze sentiment for each aspect
        for aspect in aspects:
            # Find keywords for this aspect
            aspect_keywords = self.aspects[aspect]

            # Find context around aspect mention
            for keyword in aspect_keywords:
                if keyword in text_lower:
                    # Get words around the aspect (window of 10 words)
                    pattern = r'\b\w+\b'
                    words = re.findall(pattern, text_lower)

                    try:
                        keyword_idx = words.index(keyword)
                        start = max(0, keyword_idx - 5)
                        end = min(len(words), keyword_idx + 6)
                        context = ' '.join(words[start:end])

                        # Count positive and negative words in context
                        pos_count = sum(1 for word in self.positive_words if word in context)
                        neg_count = sum(1 for word in self.negative_words if word in context)

                        # Determine sentiment
                        if pos_count > neg_count:
                            aspect_sentiments[aspect] = 'positive'
                        elif neg_count > pos_count:
                            aspect_sentiments[aspect] = 'negative'
                        else:
                            aspect_sentiments[aspect] = 'neutral'
                        break
                    except ValueError:
                        continue

        return aspect_sentiments

    def get_aspect_summary(self, text):
        """
        Get a summary of aspects and their sentiments

        Args:
            text (str): Input text

        Returns:
            dict: Summary with positive, negative, and neutral aspects
        """
        aspect_sentiments = self.analyze_aspect_sentiment(text)

        summary = {
            'positive': [],
            'negative': [],
            'neutral': []
        }

        for aspect, sentiment in aspect_sentiments.items():
            summary[sentiment].append(aspect)

        return summary

    def generate_insight(self, text):
        """
        Generate human-readable insights

        Args:
            text (str): Input text

        Returns:
            str: Insight text
        """
        summary = self.get_aspect_summary(text)

        insights = []

        if summary['positive']:
            insights.append(f"Positive aspects: {', '.join(summary['positive'])}")

        if summary['negative']:
            insights.append(f"Negative aspects: {', '.join(summary['negative'])}")

        if summary['neutral']:
            insights.append(f"Neutral aspects: {', '.join(summary['neutral'])}")

        if not insights:
            return "No specific aspects detected in the review."

        return " | ".join(insights)
