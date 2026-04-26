"""
Advanced Streamlit Web Application for Sentiment Analysis
Enhanced version with emotion detection, aspect analysis, and explainability
"""
import streamlit as st
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Optional imports - wordcloud
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

from src.utils import preprocess_text, numeric_to_sentiment
from src.emotion_detector import EmotionDetector
from src.aspect_analyzer import AspectAnalyzer
from src.explainability import SentimentExplainer

# Page configuration
st.set_page_config(
    page_title="Advanced Sentiment Analysis System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load trained models and vectorizer"""
    try:
        vectorizer = joblib.load('models/saved_models/tfidf_vectorizer.pkl')
        models = {
            'Logistic Regression': joblib.load('models/saved_models/logistic_regression.pkl'),
            'Random Forest': joblib.load('models/saved_models/random_forest.pkl'),
            'Neural Network': joblib.load('models/saved_models/neural_network.pkl')
        }
        return models, vectorizer
    except FileNotFoundError:
        return None, None


@st.cache_resource
def load_analyzers():
    """Load emotion detector and aspect analyzer"""
    emotion_detector = EmotionDetector()
    aspect_analyzer = AspectAnalyzer()
    explainer = SentimentExplainer()
    return emotion_detector, aspect_analyzer, explainer


def predict_sentiment(text, model, vectorizer):
    """
    Predict sentiment for given text

    Args:
        text (str): Input text
        model: Trained model
        vectorizer: TF-IDF vectorizer

    Returns:
        tuple: (sentiment, confidence, probabilities)
    """
    processed_text = preprocess_text(text)
    features = vectorizer.transform([processed_text])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    sentiment = numeric_to_sentiment(prediction)
    confidence = probabilities[prediction] * 100

    return sentiment, confidence, probabilities


def create_emotion_radar_chart(emotion_scores):
    """Create radar chart for emotions"""
    emotions = list(emotion_scores.keys())
    scores = list(emotion_scores.values())

    # Number of variables
    num_vars = len(emotions)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    scores += scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    ax.plot(angles, scores, 'o-', linewidth=2, color='#FF6B6B')
    ax.fill(angles, scores, alpha=0.25, color='#FF6B6B')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(emotions, size=10)
    ax.set_ylim(0, 1)
    ax.set_title('Emotion Analysis', size=14, weight='bold', pad=20)
    ax.grid(True)

    return fig


def create_wordcloud(text):
    """Create word cloud from text"""
    if not WORDCLOUD_AVAILABLE:
        return None

    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='viridis',
        max_words=50
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud', fontsize=14, weight='bold', pad=20)

    return fig


def analyze_batch(df, model, vectorizer, emotion_detector, aspect_analyzer):
    """Analyze batch of reviews"""
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, row in df.iterrows():
        status_text.text(f'Analyzing review {idx + 1}/{len(df)}...')
        progress_bar.progress((idx + 1) / len(df))

        text = str(row.get('review', row.get('text', '')))

        if text:
            sentiment, confidence, _ = predict_sentiment(text, model, vectorizer)
            dominant_emotion = emotion_detector.get_dominant_emotion(text)
            aspects = aspect_analyzer.extract_aspects(text)

            results.append({
                'review': text[:100] + '...' if len(text) > 100 else text,
                'sentiment': sentiment,
                'confidence': f'{confidence:.2f}%',
                'emotion': dominant_emotion,
                'aspects': ', '.join(aspects) if aspects else 'None'
            })

    progress_bar.empty()
    status_text.empty()

    return pd.DataFrame(results)


def main():
    """Main application"""

    # Header
    st.title("🎭 Advanced Sentiment Analysis System")
    st.markdown("### AI-Powered Review Analysis with Emotion Detection & Aspect Extraction")
    st.markdown("---")

    # Load models and analyzers
    models, vectorizer = load_models()

    if models is None:
        st.error("⚠️ Models not found!")
        st.info("Please run the training pipeline first:")
        st.code("python src/data_preprocessing.py")
        st.code("python src/model_training.py")
        st.stop()

    emotion_detector, aspect_analyzer, explainer = load_analyzers()

    # Sidebar
    st.sidebar.title("⚙️ Configuration")

    # Analysis mode
    st.sidebar.markdown("### Analysis Mode")
    mode = st.sidebar.radio(
        "Select mode:",
        ["Single Review", "Batch Analysis", "Dashboard"]
    )

    st.sidebar.markdown("---")

    # Model selection
    st.sidebar.markdown("### Model Selection")
    selected_model_name = st.sidebar.selectbox(
        "Choose a model:",
        list(models.keys()),
        index=2
    )
    selected_model = models[selected_model_name]

    st.sidebar.markdown("---")

    # Features toggle
    st.sidebar.markdown("### Advanced Features")
    show_emotions = st.sidebar.checkbox("Emotion Detection", value=True)
    show_aspects = st.sidebar.checkbox("Aspect Analysis", value=True)
    show_explainability = st.sidebar.checkbox("Explainability", value=True)
    if WORDCLOUD_AVAILABLE:
        show_wordcloud = st.sidebar.checkbox("Word Cloud", value=False)
    else:
        show_wordcloud = False
        st.sidebar.caption("⚠️ Word Cloud: Install wordcloud package")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This advanced system uses multiple AI techniques:\n"
        "- Sentiment Classification (3 classes)\n"
        "- Emotion Detection (6 emotions)\n"
        "- Aspect-Based Analysis\n"
        "- Model Explainability\n"
        "- Batch Processing"
    )

    # Main content based on mode
    if mode == "Single Review":
        # Single review analysis
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 📝 Enter Your Review")
            user_input = st.text_area(
                "Type or paste your review/feedback here:",
                height=150,
                placeholder="Example: This product is amazing! The quality is excellent but the battery life could be better."
            )

            # Sample texts
            st.markdown("#### Quick Examples:")
            col_ex1, col_ex2, col_ex3 = st.columns(3)

            with col_ex1:
                if st.button("😊 Positive Example"):
                    user_input = "This product is amazing! The quality is excellent, design is beautiful, and customer service was outstanding. Highly recommend!"
                    st.rerun()

            with col_ex2:
                if st.button("😐 Mixed Example"):
                    user_input = "The camera quality is excellent and screen is bright, but battery life is disappointing and price is too high."
                    st.rerun()

            with col_ex3:
                if st.button("😞 Negative Example"):
                    user_input = "Very disappointed. Poor quality, terrible customer service, and stopped working after a week. Complete waste of money!"
                    st.rerun()

        with col2:
            st.markdown("### 📊 Model Info")
            st.info(f"**Selected Model:** {selected_model_name}")
            st.metric("Features Enabled",
                     sum([show_emotions, show_aspects, show_explainability, show_wordcloud]))

        # Analyze button
        if st.button("🔍 Analyze Review", type="primary", use_container_width=True):
            if user_input and user_input.strip():
                with st.spinner("Analyzing..."):
                    # Make prediction
                    sentiment, confidence, probabilities = predict_sentiment(
                        user_input, selected_model, vectorizer
                    )

                    # Display main results
                    st.markdown("---")
                    st.markdown("## 🎯 Analysis Results")

                    # Main metrics
                    res_col1, res_col2, res_col3, res_col4 = st.columns(4)

                    with res_col1:
                        st.markdown("### Sentiment")
                        if sentiment == "Positive":
                            st.success(f"## 😊 {sentiment}")
                        elif sentiment == "Negative":
                            st.error(f"## 😞 {sentiment}")
                        else:
                            st.warning(f"## 😐 {sentiment}")

                    with res_col2:
                        st.markdown("### Confidence")
                        st.metric("Score", f"{confidence:.2f}%")

                    with res_col3:
                        if show_emotions:
                            emotion = emotion_detector.get_dominant_emotion(user_input)
                            st.markdown("### Emotion")
                            st.metric("Dominant", emotion.title())

                    with res_col4:
                        aspects = aspect_analyzer.extract_aspects(user_input)
                        st.markdown("### Aspects")
                        st.metric("Detected", len(aspects))

                    # Probability distribution
                    st.markdown("---")
                    col_prob, col_emotion = st.columns(2)

                    with col_prob:
                        st.markdown("### 📈 Sentiment Probabilities")
                        prob_df = pd.DataFrame({
                            'Sentiment': ['Negative', 'Neutral', 'Positive'],
                            'Probability': probabilities * 100
                        })

                        fig, ax = plt.subplots(figsize=(8, 4))
                        colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
                        bars = ax.bar(prob_df['Sentiment'], prob_df['Probability'], color=colors)

                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.1f}%',
                                   ha='center', va='bottom', fontweight='bold')

                        ax.set_ylabel('Probability (%)', fontweight='bold')
                        ax.set_ylim([0, 105])
                        ax.grid(axis='y', alpha=0.3)
                        st.pyplot(fig)

                    # Emotion radar chart
                    with col_emotion:
                        if show_emotions:
                            st.markdown("### 🎭 Emotion Distribution")
                            emotion_scores = emotion_detector.detect_emotion(user_input)
                            fig_emotion = create_emotion_radar_chart(emotion_scores)
                            st.pyplot(fig_emotion)

                    # Aspect analysis
                    if show_aspects and aspects:
                        st.markdown("---")
                        st.markdown("### 🔍 Aspect-Based Analysis")

                        aspect_summary = aspect_analyzer.get_aspect_summary(user_input)

                        col_asp1, col_asp2, col_asp3 = st.columns(3)

                        with col_asp1:
                            if aspect_summary['positive']:
                                st.success("**Positive Aspects:**")
                                for asp in aspect_summary['positive']:
                                    st.write(f"✓ {asp.replace('_', ' ').title()}")

                        with col_asp2:
                            if aspect_summary['neutral']:
                                st.warning("**Neutral Aspects:**")
                                for asp in aspect_summary['neutral']:
                                    st.write(f"○ {asp.replace('_', ' ').title()}")

                        with col_asp3:
                            if aspect_summary['negative']:
                                st.error("**Negative Aspects:**")
                                for asp in aspect_summary['negative']:
                                    st.write(f"✗ {asp.replace('_', ' ').title()}")

                        st.info(f"**Insight:** {aspect_analyzer.generate_insight(user_input)}")

                    # Explainability
                    if show_explainability:
                        st.markdown("---")
                        st.markdown("### 💡 Why This Prediction?")

                        explanation = explainer.explain_prediction(user_input, sentiment, confidence)

                        col_exp1, col_exp2 = st.columns(2)

                        with col_exp1:
                            if explanation['positive_words']:
                                st.success("**Positive Indicators:**")
                                st.write(", ".join(explanation['positive_words']))

                        with col_exp2:
                            if explanation['negative_words']:
                                st.error("**Negative Indicators:**")
                                st.write(", ".join(explanation['negative_words']))

                        # Highlighted text
                        st.markdown("**Highlighted Review:**")
                        highlighted = explainer.highlight_text(user_input, sentiment)
                        st.markdown(highlighted, unsafe_allow_html=True)
                        st.caption("🟢 Green = Positive indicators | 🔴 Pink = Negative indicators")

                    # Word cloud
                    if show_wordcloud:
                        st.markdown("---")
                        st.markdown("### ☁️ Word Cloud")
                        if WORDCLOUD_AVAILABLE:
                            fig_wc = create_wordcloud(user_input)
                            st.pyplot(fig_wc)
                        else:
                            st.warning("⚠️ WordCloud package not installed. Install with: pip install wordcloud")

                    # Preprocessed text
                    with st.expander("🔧 View Preprocessed Text"):
                        preprocessed = preprocess_text(user_input)
                        st.text_area("Preprocessed:", preprocessed, height=100)

            else:
                st.warning("⚠️ Please enter some text to analyze.")

    elif mode == "Batch Analysis":
        # Batch processing mode
        st.markdown("## 📊 Batch Analysis")
        st.markdown("Upload a CSV file with reviews to analyze multiple reviews at once.")

        uploaded_file = st.file_uploader(
            "Upload CSV file (must have 'review' or 'text' column)",
            type=['csv']
        )

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            st.success(f"✓ Loaded {len(df)} reviews")
            st.dataframe(df.head())

            if st.button("🚀 Analyze All Reviews", type="primary"):
                st.markdown("---")
                st.markdown("### Analysis Results")

                results_df = analyze_batch(
                    df, selected_model, vectorizer,
                    emotion_detector, aspect_analyzer
                )

                # Display results
                st.dataframe(results_df, use_container_width=True)

                # Summary statistics
                st.markdown("---")
                st.markdown("### 📈 Summary Statistics")

                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

                with col_stat1:
                    st.metric("Total Reviews", len(results_df))

                with col_stat2:
                    positive_pct = (results_df['sentiment'] == 'Positive').sum() / len(results_df) * 100
                    st.metric("Positive", f"{positive_pct:.1f}%")

                with col_stat3:
                    negative_pct = (results_df['sentiment'] == 'Negative').sum() / len(results_df) * 100
                    st.metric("Negative", f"{negative_pct:.1f}%")

                with col_stat4:
                    neutral_pct = (results_df['sentiment'] == 'Neutral').sum() / len(results_df) * 100
                    st.metric("Neutral", f"{neutral_pct:.1f}%")

                # Sentiment distribution chart
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                # Sentiment counts
                sentiment_counts = results_df['sentiment'].value_counts()
                colors_sent = ['#6bcf7f', '#ff6b6b', '#ffd93d']
                ax1.pie(sentiment_counts.values, labels=sentiment_counts.index,
                       autopct='%1.1f%%', colors=colors_sent, startangle=90)
                ax1.set_title('Sentiment Distribution', fontweight='bold')

                # Emotion counts
                emotion_counts = results_df['emotion'].value_counts().head(5)
                ax2.barh(emotion_counts.index, emotion_counts.values, color='skyblue')
                ax2.set_xlabel('Count', fontweight='bold')
                ax2.set_title('Top Emotions', fontweight='bold')
                ax2.grid(axis='x', alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv"
                )

    else:  # Dashboard mode
        st.markdown("## 📊 Analytics Dashboard")
        st.info("👉 Upload a CSV file in Batch Analysis mode to view analytics dashboard")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p><strong>Advanced Sentiment Analysis System</strong> | Powered by Machine Learning & NLP</p>
            <p>Features: Multi-Model Classification • Emotion Detection • Aspect Analysis • Explainability</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
