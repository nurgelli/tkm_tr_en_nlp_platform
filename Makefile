.PHONY: help up down build test clean logs preprocess train-baseline build-embeddings

help:
	@echo "Available commands:"
	@echo "  make up      - Start all services"
	@echo "  make down    - Stop all services"
	@echo "  make build   - Build Docker images"
	@echo "  make test    - Run pytest"
	@echo "  make preprocess       - Clean raw corpus"
	@echo "  make train-baseline   - Train TF-IDF classifier"
	@echo "  make build-embeddings - Build semantic index and embedding classifier"


up:
	docker-compose up -d

down:
	docker-compose down

build:
	docker-compose build

test:
	pytest tests/ -v --tb=short

clean:
	docker-compose down -v --remove-orphans

logs:
	docker-compose logs -f

preprocess:
	python scripts/preprocess.py --input data/raw/multilingual_corpus.csv --output data/processed/corpus_cleaned.csv

train-baseline:
	python scripts/train_baseline.py

build-embeddings:
	python scripts/build_embeddings.py
