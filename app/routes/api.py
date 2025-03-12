from fastapi import APIRouter, Depends

from app.schemas.nlp import SupportedLanguages, PieLanguage

router = APIRouter()

@router.get("/languages", response_model=SupportedLanguages)
async def get_languages():
    """
    Return a list of supported language schemas.
    """
    languages = [lang.value for lang in PieLanguage]
    return {"languages": languages}