from enum import Enum
from typing import List
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