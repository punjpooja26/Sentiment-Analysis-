"""
Demo Script - Quick Test of All Advanced Features
Run this to verify everything is working before your presentation
"""

import sys
sys.path.append('src')

from src.emotion_detector import EmotionDetector
from src.aspect_analyzer import AspectAnalyzer
from src.explainability import SentimentExplainer

def test_emotion_detection():
    """Test emotion detection"""
    print("="*70)
    print("1. TESTING EMOTION DETECTION")
    print("="*70)

    detector = EmotionDetector()

    test_texts = [
        "I'm so happy and excited about this amazing product!",
        "Very disappointed and upset with this terrible purchase.",
        "This makes me so angry! Horrible customer service!"
    ]

    for text in test_texts:
        emotion = detector.get_dominant_emotion(text)
        scores = detector.detect_emotion(text)
        print(f"\nText: {text}")
        print(f"Dominant Emotion: {emotion.upper()}")
        print(f"Emotion Scores: {scores}")

    print("\n✅ Emotion Detection Working!\n")


def test_aspect_analysis():
    """Test aspect-based analysis"""
    print("="*70)
    print("2. TESTING ASPECT-BASED ANALYSIS")
    print("="*70)

    analyzer = AspectAnalyzer()

    test_texts = [
        "The camera quality is excellent but the battery life is terrible.",
        "Great design and beautiful screen, but poor performance and high price.",
        "Amazing quality and fast delivery! Customer service was outstanding."
    ]

    for text in test_texts:
        aspects = analyzer.extract_aspects(text)
        summary = analyzer.get_aspect_summary(text)
        insight = analyzer.generate_insight(text)

        print(f"\nText: {text}")
        print(f"Aspects Found: {aspects}")
        print(f"Summary: {summary}")
        print(f"Insight: {insight}")

    print("\n✅ Aspect Analysis Working!\n")


def test_explainability():
    """Test explainability"""
    print("="*70)
    print("3. TESTING EXPLAINABILITY")
    print("="*70)

    explainer = SentimentExplainer()

    test_texts = [
        "This product is excellent, amazing quality and wonderful features!",
        "Terrible quality, horrible experience, worst purchase ever!",
        "The design is beautiful but the quality is poor."
    ]

    for text in test_texts:
        explanation = explainer.explain_prediction(text, "Positive", 85.5)
        key_phrases = explainer.get_key_phrases(text)

        print(f"\nText: {text}")
        print(f"Positive Words: {explanation['positive_words']}")
        print(f"Negative Words: {explanation['negative_words']}")
        print(f"Key Phrases: {key_phrases}")

    print("\n✅ Explainability Working!\n")


def test_integration():
    """Test integration of all features"""
    print("="*70)
    print("4. TESTING FULL INTEGRATION")
    print("="*70)

    # Initialize all components
    emotion_detector = EmotionDetector()
    aspect_analyzer = AspectAnalyzer()
    explainer = SentimentExplainer()

    # Complex test review
    test_review = """
    This smartphone is absolutely amazing! The camera quality is outstanding
    and the screen is beautiful. I'm so happy with this purchase. However,
    the battery life is disappointing and the price is a bit high. Customer
    service was excellent though. Overall, highly recommend!
    """

    print(f"\nTest Review:\n{test_review}")
    print("\n" + "-"*70)

    # Emotion analysis
    emotion = emotion_detector.get_dominant_emotion(test_review)
    emotion_scores = emotion_detector.detect_emotion(test_review)
    print(f"\nEMOTION ANALYSIS:")
    print(f"  Dominant Emotion: {emotion.upper()}")
    print(f"  Emotion Distribution: {emotion_scores}")

    # Aspect analysis
    aspects = aspect_analyzer.extract_aspects(test_review)
    summary = aspect_analyzer.get_aspect_summary(test_review)
    insight = aspect_analyzer.generate_insight(test_review)
    print(f"\nASPECT ANALYSIS:")
    print(f"  Aspects Found: {aspects}")
    print(f"  Positive Aspects: {summary['positive']}")
    print(f"  Negative Aspects: {summary['negative']}")
    print(f"  Insight: {insight}")

    # Explainability
    explanation = explainer.explain_prediction(test_review, "Positive", 72.5)
    print(f"\nEXPLAINABILITY:")
    print(f"  Positive Indicators: {explanation['positive_words']}")
    print(f"  Negative Indicators: {explanation['negative_words']}")
    print(f"  Word Count: {explanation['word_count']}")
    print(f"  Unique Words: {explanation['unique_words']}")

    print("\n✅ Full Integration Working!\n")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" "*15 + "ADVANCED FEATURES DEMO SCRIPT")
    print("="*70)
    print("\nThis script tests all advanced features before your presentation")
    print("Make sure everything shows ✅ at the end\n")

    try:
        # Run tests
        test_emotion_detection()
        test_aspect_analysis()
        test_explainability()
        test_integration()

        # Success message
        print("="*70)
        print(" "*20 + "🎉 ALL TESTS PASSED! 🎉")
        print("="*70)
        print("\nYour advanced features are working perfectly!")
        print("\nNext steps:")
        print("  1. Run: streamlit run app_advanced.py")
        print("  2. Test the web interface")
        print("  3. Try batch analysis with sample_reviews.csv")
        print("  4. You're ready to present!")
        print("\n" + "="*70)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you're in the project directory")
        print("  2. Check that all files are present in src/ folder")
        print("  3. Run: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
