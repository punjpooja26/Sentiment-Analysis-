"""
Streamlit Web Application for Sentiment Analysis
Interactive interface for real-time sentiment prediction
"""

import streamlit as st
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import preprocess_text, numeric_to_sentiment

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🎭",

    layout="wide"
)


@st.cache_resource
def load_models():
    try:
        base_path = os.path.join("models", "saved_models")

        vectorizer = joblib.load(os.path.join(base_path, "tfidf_vectorizer.pkl"))

        models = {
            'Logistic Regression': joblib.load(os.path.join(base_path, "logistic_regression.pkl")),
            'Random Forest': joblib.load(os.path.join(base_path, "random_forest.pkl")),
            'Neural Network': joblib.load(os.path.join(base_path, "neural_network.pkl"))
        }

        return models, vectorizer

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None


def predict_sentiment(text, model, vectorizer):
    processed_text = preprocess_text(text)
    features = vectorizer.transform([processed_text])

    prediction = model.predict(features)[0]

    # Handle models without predict_proba
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = probabilities[prediction] * 100
    else:
        probabilities = [0, 0, 0]
        confidence = 0

    sentiment = numeric_to_sentiment(prediction)

    return sentiment, confidence, probabilities


def main():
    """Main application"""

    # Header
    st.title("🎭 Sentiment Analysis System")
    st.markdown("### Analyze sentiment in product reviews and customer feedback")
    st.markdown("---")

    # Check if models exist
    models, vectorizer = load_models()

    if models is None:
        st.error("⚠️ Models not found!")
        st.info("Please run the following commands first:")
        st.code("python src/data_preprocessing.py")
        st.code("python src/model_training.py")
        st.stop()

    # Sidebar
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown("### Select Model")
    selected_model_name = st.sidebar.selectbox(
        "Choose a model",
        list(models.keys()),
        index=2  # Default to Neural Network
    )
    selected_model = models[selected_model_name]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This application uses machine learning to classify sentiment "
        "in text as Positive, Negative, or Neutral. "
        "It was trained on product reviews and can analyze any text input."
    )

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📝 Enter Your Text")
        if "user_input" not in st.session_state:
            st.session_state.user_input = ""



        user_input = st.text_area(
            "Type or paste your review/feedback here:",
            height=150,
            value=st.session_state.user_input,
            placeholder="Example: This product is amazing!"
        )

        # Sample texts
        st.markdown("#### Quick Examples:")
        col_ex1, col_ex2, col_ex3 = st.columns(3)

        with col_ex1:
            if st.button("😊 Positive Example"):
                st.session_state.user_input = "This product is amazing! Exceeded all my expectations."

        with col_ex2:
            if st.button("😐 Neutral Example"):
                st.session_state.user_input = "It's okay. Nothing special but does the job."

        with col_ex3:
            if st.button("😞 Negative Example"):
                st.session_state.user_input = "Very disappointed. Poor quality and waste of money."
    with col2:
        st.markdown("### 📊 Model Info")
        st.info(f"**Selected Model:** {selected_model_name}")

    # Predict button
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if user_input.strip():
            progress = st.progress(0)

            for i in range(100):
                progress.progress(i + 1)

            sentiment, confidence, probabilities = predict_sentiment(
                user_input, selected_model, vectorizer
            )


            # Display results
            st.markdown("---")
            st.markdown("## 🎯 Results")

            # Create columns for results
            res_col1, res_col2, res_col3 = st.columns(3)

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
                st.metric("Confidence Score", f"{confidence:.2f}%")

            with res_col3:
                st.markdown("### Model")
                st.info(f"**{selected_model_name}**")

            # Probability distribution
            st.markdown("---")
            st.markdown("### 📈 Probability Distribution")

            if sum(probabilities) > 0:
                prob_df = pd.DataFrame({
                    'Sentiment': ['Negative', 'Neutral', 'Positive'],
                    'Probability': probabilities * 100
                })
            else:
                prob_df = pd.DataFrame({
                    'Sentiment': ['Negative', 'Neutral', 'Positive'],
                    'Probability': [0, 0, 0]
                })

            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
            bars = ax.bar(prob_df['Sentiment'], prob_df['Probability'], color=colors)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontweight='bold')

            ax.set_ylabel('Probability (%)', fontweight='bold')
            ax.set_xlabel('Sentiment', fontweight='bold')
            ax.set_title('Sentiment Probability Distribution', fontweight='bold', fontsize=14)
            ax.set_ylim([0, 105])
            ax.grid(axis='y', alpha=0.3)

            st.pyplot(fig)

            # Show preprocessed text
            with st.expander("🔧 View Preprocessed Text"):
                preprocessed = preprocess_text(user_input)
                st.text_area("Preprocessed text:", preprocessed, height=100)

        else:
            st.warning("⚠️ Please enter some text to analyze.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Sentiment Analysis System | Powered by Machine Learning</p>
            <p>Models: Logistic Regression, Random Forest, Neural Network</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
