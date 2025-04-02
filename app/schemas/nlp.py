from enum import Enum
from typing import Optional, Literal, List
from pydantic import BaseModel


class PieLanguage(str, Enum):
    lasla = "Classical Latin"
    grc = "Ancient Greek"
    fro = "Old French"
    freem = "Early Modern French"
    fr = "Classical French"
    dum = "Old Dutch"
    occ_cont = "Occitan Contemporain"

class SupportedLanguages(BaseModel):
    languages: List[PieLanguage]

class ModelStatusSchema(BaseModel):
    language: str
    status: Literal["loaded", "loading", "not loaded", "downloading"]
    files: Optional[List[str]] = None
    message: Optional[str] = None

class TextInput(BaseModel):
    text: str
    lower: bool = False

class BatchTextInput(BaseModel):
    texts: List[str]
    lower: bool = False