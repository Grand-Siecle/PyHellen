from fastapi import APIRouter, Depends, HTTPException, Request

from app.schemas.nlp import SupportedLanguages, PieLanguage
from app.core.model_manager import get_or_load_model

router = APIRouter()

@router.get("/languages", response_model=SupportedLanguages)
async def get_languages():
    """
    Return a list of supported language schemas.

    **Returns**:
    - **200**: A JSON object with a list of supported language codes.
    - **500**: If retrieving the language list fails due to a server error.

    **Example**:
    ```bash
    curl -X GET http://{address}/languages
    ```

    **Notes**:
    - The list of languages is defined in the `PieLanguage` enum.
    """
    languages = [lang.value for lang in PieLanguage]
    return {"languages": languages}


@router.post("/tag/{model}/")
async def tag_text(model: str, text: str, request: Request):
    """
    Use the model — trigger download if not available.
    """
    try:
        lemmatizer = await get_or_load_model(model, request.app.state)
        if not lemmatizer:
            raise HTTPException(status_code=500, detail=f"Failed to load model '{model}'")

        result = lemmatizer.tag(text)
        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))