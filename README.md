# MLOps Sentiment Analysis Project

Questo progetto implementa un sistema di analisi del sentiment per testi dei social media utilizzando un modello RoBERTa pre-addestrato.

## Motivazione del Progetto

L'analisi del sentiment è cruciale per comprendere l'opinione pubblica sui social media. Questo progetto dimostra l'applicazione di MLOps per sviluppare, testare e deployare un modello di sentiment analysis in modo automatizzato.

## Fase 1: Implementazione del Modello

### Modello Utilizzato
- **Modello**: `cardiffnlp/twitter-roberta-base-sentiment-latest` da HuggingFace.
- **Tipo**: RoBERTa fine-tuned per sentiment su Twitter.
- **Classi**: Positivo (LABEL_2), Neutro (LABEL_1), Negativo (LABEL_0).

### Dataset
- Utilizzato `cardiffnlp/tweet_sentiment_multilingual` per valutazione.
- Dataset pubblico contenente testi etichettati.

## Fase 2: Pipeline CI/CD

La pipeline è implementata con GitHub Actions:
- **CI**: Installazione dipendenze, esecuzione test, linting.
- **CD**: Valutazione modello, deploy su HuggingFace.

File: `.github/workflows/ci-cd.yml`

## Fase 3: Deploy e Monitoraggio

### Deploy
- Applicazione Gradio deployata su HuggingFace Spaces (alternativa a Streamlit).
- API FastAPI per integrazioni.

### Monitoraggio
- MLflow per logging delle metriche di performance.
- Valutazione continua su nuovi dati.

## Struttura del Progetto

```
.
├── src/
│   ├── sentiment_model.py  # Classe per analisi sentiment
│   ├── evaluate.py         # Script valutazione
│   ├── app.py              # API FastAPI
│   └── app_streamlit.py    # App Streamlit
├── tests/
│   └── test_sentiment.py   # Test unitari
├── notebooks/
│   └── sentiment_analysis_colab.ipynb  # Notebook Colab
├── .github/workflows/
│   └── ci-cd.yml           # Pipeline CI/CD
├── requirements.txt        # Dipendenze
└── README.md               # Documentazione
```

## Installazione e Uso

1. Clona il repository:
   ```bash
   git clone https://github.com/yourusername/mlops-prg.git
   cd mlops-prg
   ```

2. Installa dipendenze:
   ```bash
   pip install -r requirements.txt
   ```

3. Esegui valutazione:
   ```bash
   python src/evaluate.py
   ```

4. Avvia API:
   ```bash
   uvicorn src.app:app --reload
   ```

5. Avvia app Streamlit:
   ```bash
   streamlit run src/app_streamlit.py
   ```

## Risultati

Il modello ottiene accuratezza ~80-90% su dataset di test, a seconda del dominio.

## Consegna

- **Repository GitHub**: [Link](https://github.com/yourusername/mlops-prg)
- **Notebook Colab**: notebooks/sentiment_analysis_colab.ipynb