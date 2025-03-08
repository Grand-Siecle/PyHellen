from typing import Dict

# Global tagger cache
tagger_cache: Dict[str, str] = {}

def get_tagger_status(language: str) -> str:
    """Return the cached tagger status for a given language."""
    return tagger_cache.get(language, "not loaded")

def set_tagger_status(language: str, status: str):
    """Set the status of the tagger for a given language."""
    tagger_cache[language] = status