"""
Pytest configuration and fixtures for PyHellen tests.
"""

import asyncio
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from app.main import app, create_application
from app.core.model_manager import ModelManager
from app.core.cache import LRUCache


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client():
    """Create a test client for synchronous tests."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client():
    """Create an async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def fresh_app():
    """Create a fresh application instance for isolated tests."""
    return create_application()


@pytest.fixture
def mock_model_manager():
    """Create a mock model manager for testing without actual models."""
    manager = ModelManager()
    yield manager


@pytest.fixture
def cache():
    """Create a fresh cache instance for testing."""
    return LRUCache(max_size=100, ttl_seconds=60)


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "Lorem ipsum dolor sit amet.",
        "Gallia est omnis divisa in partes tres.",
        "Arma virumque cano."
    ]


@pytest.fixture
def sample_latin_text():
    """Sample Latin text for testing."""
    return "Gallia est omnis divisa in partes tres."


@pytest.fixture
def sample_greek_text():
    """Sample Ancient Greek text for testing."""
    return "μῆνιν ἄειδε θεά"
