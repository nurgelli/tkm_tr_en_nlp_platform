import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
DATA_PATH = os.getenv("DATA_PATH", "data/processed/corpus_cleaned.csv")

st.set_page_config(page_title="Multilingual NLP Platform", layout="wide")
st.title("Multilingual NLP Analytics Platform")


def api_post(path: str, payload: dict):
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def api_get(path: str):
    response = requests.get(f"{API_BASE_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


tabs = st.tabs(["Classify", "Semantic Search", "Corpus", "Compare"])

with tabs[0]:
    col_text, col_options = st.columns([3, 1])
    with col_text:
        text = st.text_area(
            "Text", height=180, placeholder="Enter Turkish, Turkmen, or English text"
        )
    with col_options:
        method = st.radio("Model", ["embedding", "tfidf"], horizontal=False)
        language = st.selectbox("Language hint", ["auto", "tr", "tk", "en"])

    if st.button("Classify", type="primary", disabled=not text.strip()):
        payload = {"text": text, "method": method}
        if language != "auto":
            payload["language"] = language
        try:
            result = api_post("/classify", payload)
            st.metric("Category", result["category"])
            st.metric("Detected language", result["detected_language"])
            st.metric("Latency", f"{result['latency_ms']:.2f} ms")
            st.bar_chart(pd.Series(result["probabilities"], name="probability"))
        except requests.HTTPError as exc:
            detail = exc.response.json().get("detail", exc.response.text)
            st.error(detail)
        except requests.RequestException as exc:
            st.error(f"API request failed: {exc}")

with tabs[1]:
    query = st.text_input("Query")
    col_k, col_lang = st.columns(2)
    with col_k:
        top_k = st.slider("Results", min_value=1, max_value=20, value=5)
    with col_lang:
        language_filter = st.selectbox("Language filter", ["all", "tr", "tk", "en"])

    if st.button("Search", type="primary", disabled=not query.strip()):
        payload = {"query": query, "top_k": top_k}
        if language_filter != "all":
            payload["language_filter"] = language_filter
        try:
            results = api_post("/search", payload)
            if results:
                st.dataframe(
                    pd.DataFrame(results), use_container_width=True, hide_index=True
                )
            else:
                st.info("No matching documents found.")
        except requests.HTTPError as exc:
            detail = exc.response.json().get("detail", exc.response.text)
            st.error(detail)
        except requests.RequestException as exc:
            st.error(f"API request failed: {exc}")

with tabs[2]:
    try:
        df = pd.read_csv(DATA_PATH)
        col_lang, col_cat = st.columns(2)
        with col_lang:
            st.subheader("Language Distribution")
            st.bar_chart(df["language"].value_counts())
        with col_cat:
            st.subheader("Category Distribution")
            st.bar_chart(df["category"].value_counts())
        st.dataframe(df.head(100), use_container_width=True, hide_index=True)
    except FileNotFoundError:
        st.warning("Processed corpus not found. Run scripts/preprocess.py first.")

with tabs[3]:
    compare_text = st.text_area("Comparison text", height=140)
    if st.button("Compare predictions", disabled=not compare_text.strip()):
        try:
            result = api_post("/compare", {"text": compare_text})
            st.json(result)
        except requests.HTTPError as exc:
            detail = exc.response.json().get("detail", exc.response.text)
            st.error(detail)
        except requests.RequestException as exc:
            st.error(f"API request failed: {exc}")

    try:
        metrics = api_get("/compare")
        col_base, col_emb = st.columns(2)
        col_base.json(metrics["baseline"])
        col_emb.json(metrics["embedding"])
    except requests.RequestException:
        st.info("Model metrics will appear after training artifacts are generated.")
