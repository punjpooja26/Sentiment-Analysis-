#!/bin/bash

# Setup script for Sentiment Analysis Project

echo "========================================"
echo "Sentiment Analysis Project Setup"
echo "========================================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

echo ""
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

echo ""
echo "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('omw-1.4')"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Run data preprocessing: python3 src/data_preprocessing.py"
echo "2. Train models: python3 src/model_training.py"
echo "3. Evaluate models: python3 src/model_evaluation.py"
echo "4. Launch web app: streamlit run app.py"
echo ""
