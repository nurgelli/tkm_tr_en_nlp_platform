from fastapi import APIRouter, HTTPException

from app.api.schemas import SearchRequest, SearchResult
from app.services.search_service import search

router = APIRouter()


@router.post("/", response_model=list[SearchResult], include_in_schema=False)
def semantic_search(request: SearchRequest):
    try:
        results = search(
            request.query, top_k=request.top_k, language_filter=request.language_filter
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return results
