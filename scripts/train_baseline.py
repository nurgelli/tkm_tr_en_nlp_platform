import pandas as pd
from app.models.baseline.tfidf_classifier import train_baseline

if __name__ == '__main__':
    df = pd.read_csv('data/processed/corpus_cleaned.csv')
    result = train_baseline(df)
    print(result)
