from fastapi import FastAPI
from pydantic import BaseModel
from mlops.sentiment_model import SentimentAnalyzer

app = FastAPI(title="Sentiment Analysis API", description="API for sentiment analysis using RoBERTa model")

analyzer = SentimentAnalyzer()

class TextInput(BaseModel):
    text: str

@app.post("/analyze")
def analyze_sentiment(input: TextInput):
    scores = analyzer.analyze(input.text)
    classification = analyzer.classify(input.text)
    return {"scores": scores, "classification": classification}

@app.get("/")
def read_root():
    return {"message": "Sentiment Analysis API"}