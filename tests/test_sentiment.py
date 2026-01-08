import pytest
from mlops.sentiment_model import SentimentAnalyzer

def test_analyzer_init():
    analyzer = SentimentAnalyzer()
    assert analyzer.model_name == "cardiffnlp/twitter-roberta-base-sentiment-latest"

def test_analyze():
    analyzer = SentimentAnalyzer()
    scores = analyzer.analyze("I love this!")
    assert isinstance(scores, dict)
    assert 'LABEL_0' in scores
    assert 'LABEL_1' in scores
    assert 'LABEL_2' in scores

def test_classify():
    analyzer = SentimentAnalyzer()
    label = analyzer.classify("I love this!")
    assert label in ['LABEL_0', 'LABEL_1', 'LABEL_2']