from fastapi import FastAPI
from pydantic import BaseModel
import joblib   # IMPORTANT: better than pickle
import os

app = FastAPI()

# ✅ Load model safely
MODEL_PATH = os.path.join("models", "saved_models", "logistic_regression.pkl")
VECTORIZER_PATH = os.path.join("models", "saved_models", "tfidf_vectorizer.pkl")
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as e:
    print("Error loading model:", e)
    model = None
    vectorizer = None

# ✅ Input schema
class Review(BaseModel):
    text: str

# ✅ Root route
@app.get("/")
def home():
    return {"message": "Sentiment API is running 🚀"}

# ✅ Prediction route
@app.post("/predict")
def predict_sentiment(review: Review):
    if model is None:
        return {"error": "Model not loaded properly"}

    text = [review.text]  # model expects list

    # Transform the text using the vectorizer
    if vectorizer is None:
        return {"error": "Vectorizer not loaded properly"}

    text_vectorized = vectorizer.transform(text)

    prediction = model.predict(text_vectorized)[0]

    return {
        "review": review.text,
        "sentiment": "positive" if prediction == 1 else "negative"
    }
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)