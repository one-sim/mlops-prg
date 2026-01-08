from datasets import load_dataset
from mlops.sentiment_model import SentimentAnalyzer
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import mlflow
import mlflow.sklearn

def load_data():
    # Load a public dataset for sentiment analysis
    dataset = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "all", split="train[:1000]")  # Sample for demo
    return dataset

def evaluate_model():
    mlflow.start_run()
    analyzer = SentimentAnalyzer()
    dataset = load_data()

    predictions = []
    labels = []

    for example in dataset:
        text = example['text']
        label = example['label']  # Assuming 0: negative, 1: neutral, 2: positive

        pred_scores = analyzer.analyze(text)
        # Get the label with max score
        pred_label = max(pred_scores, key=pred_scores.get)
        pred_num = int(pred_label.split('_')[1])  # LABEL_0 -> 0

        predictions.append(pred_num)
        labels.append(label)

    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, target_names=['negative', 'neutral', 'positive'], output_dict=True)

    print(classification_report(labels, predictions, target_names=['negative', 'neutral', 'positive']))

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                mlflow.log_metric(f"{label}_{metric}", value)

    mlflow.end_run()

if __name__ == "__main__":
    evaluate_model()