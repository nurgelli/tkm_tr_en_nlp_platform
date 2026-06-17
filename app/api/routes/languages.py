from fastapi import APIRouter

from app.data_utils import SUPPORTED_LANGUAGES

router = APIRouter()


@router.get("")
@router.get("/", include_in_schema=False)
def supported_languages():
    return {
        "supported": [
            {"code": code, "name": name} for code, name in SUPPORTED_LANGUAGES.items()
        ]
    }
