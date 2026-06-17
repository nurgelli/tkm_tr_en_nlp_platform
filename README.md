# Multilingual NLP Platform

Production-oriented NLP platform for Turkish, Turkmen, and English text classification plus semantic search.

## Pipeline

1. Raw corpus: `data/raw/multilingual_corpus.csv`
2. Preprocessing: cleaning, normalization, language detection
3. Baseline model: TF-IDF + Logistic Regression
4. Modern model: multilingual MiniLM sentence embeddings + classifier
5. Semantic search: normalized embedding vectors with cosine similarity
6. Serving: FastAPI API + Streamlit dashboard
7. Quality gate: pytest + GitHub Actions

## Local Setup

```bash
pip install -r requirements.txt
make preprocess
make train-baseline
make build-embeddings
```

Start the API and dashboard:

```bash
make up
```

API: `http://localhost:8000`

Dashboard: `http://localhost:8501`

## API

- `POST /classify`: classify text with `embedding` or `tfidf`
- `POST /search`: return semantically similar corpus documents
- `GET /compare`: return saved baseline and embedding metrics
- `POST /compare`: compare predictions for one text
- `GET /languages`: list supported languages
- `GET /health`: service health

## Data Contract

Raw CSV must include:

- `text`
- `language`
- `category`
- `source`

The preprocessing step writes `data/processed/corpus_cleaned.csv` with normalized columns used by both training pipelines.

## Notes

The first embedding build downloads the sentence-transformers model and can take time. Generated model artifacts live under `app/models/` and are mounted into the Docker API container.
