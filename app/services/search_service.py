from app.models.embedding.search_index import cosine_search


def search(query: str, top_k: int = 5, language_filter: str = None):
    return cosine_search(query, top_k=top_k, lang_filter=language_filter)
