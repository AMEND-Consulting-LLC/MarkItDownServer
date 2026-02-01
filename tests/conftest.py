"""Shared fixtures for MarkItDown Server tests."""

import os
import sys
from io import BytesIO
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
import httpx

# Add the parent directory to the path so we can import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    """Create a FastAPI TestClient for synchronous testing."""
    # Import app here to avoid module-level import issues with env vars
    from app import app
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_client() -> httpx.AsyncClient:
    """Create an async HTTPX client for async testing."""
    from app import app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
def sample_txt_content() -> bytes:
    """Sample text file content."""
    return b"This is a sample text file for testing.\nIt has multiple lines."


@pytest.fixture
def sample_pdf_content() -> bytes:
    """Minimal PDF file content (valid PDF header)."""
    # Minimal valid PDF structure
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer
<< /Size 4 /Root 1 0 R >>
startxref
196
%%EOF"""


@pytest.fixture
def sample_image_content() -> bytes:
    """Minimal PNG image content (1x1 pixel)."""
    # 1x1 transparent PNG
    return bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
        0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,
        0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
        0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
        0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
        0x42, 0x60, 0x82
    ])


@pytest.fixture
def large_file_content() -> bytes:
    """Generate content larger than MAX_FILE_SIZE (50MB)."""
    # 51MB of data
    return b"x" * (51 * 1024 * 1024)


@pytest.fixture
def empty_file_content() -> bytes:
    """Empty file content."""
    return b""


@pytest.fixture
def mock_markitdown():
    """Mock MarkItDown class for testing without actual conversion."""
    mock_result = MagicMock()
    mock_result.text_content = "# Converted Markdown\n\nThis is the converted content."

    mock_instance = MagicMock()
    mock_instance.convert.return_value = mock_result

    with patch('app.MarkItDown', return_value=mock_instance) as mock_class:
        yield mock_class, mock_instance, mock_result


@pytest.fixture
def mock_markitdown_error():
    """Mock MarkItDown that raises an exception on convert."""
    mock_instance = MagicMock()
    mock_instance.convert.side_effect = Exception("Conversion failed")

    with patch('app.MarkItDown', return_value=mock_instance) as mock_class:
        yield mock_class, mock_instance


@pytest.fixture
def env_vars_backup() -> Generator[dict, None, None]:
    """Backup and restore environment variables."""
    original_env = os.environ.copy()
    yield original_env
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def clean_env(env_vars_backup) -> Generator[None, None, None]:
    """Provide a clean environment without LLM/Azure vars."""
    # Remove all LLM and Azure related env vars
    vars_to_remove = [
        'LLM_PROVIDER', 'OPENAI_API_KEY', 'OPENAI_BASE_URL',
        'AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_API_VERSION',
        'LLM_MODEL', 'LLM_PROMPT',
        'AZURE_DOCINTEL_ENDPOINT', 'AZURE_DOCINTEL_API_KEY', 'AZURE_DOCINTEL_FILE_TYPES',
        'ENABLE_PLUGINS', 'EXIFTOOL_PATH',
        'WORKERS', 'PORT', 'ENABLE_RATE_LIMIT', 'RATE_LIMIT'
    ]
    for var in vars_to_remove:
        os.environ.pop(var, None)
    yield


@pytest.fixture
def openai_env(env_vars_backup) -> Generator[None, None, None]:
    """Set up environment for OpenAI LLM testing."""
    os.environ['LLM_PROVIDER'] = 'openai'
    os.environ['OPENAI_API_KEY'] = 'test-api-key'
    os.environ['LLM_MODEL'] = 'gpt-4o'
    yield


@pytest.fixture
def openai_with_base_url_env(env_vars_backup) -> Generator[None, None, None]:
    """Set up environment for OpenAI with custom base URL (LiteLLM)."""
    os.environ['LLM_PROVIDER'] = 'openai'
    os.environ['OPENAI_API_KEY'] = 'test-api-key'
    os.environ['OPENAI_BASE_URL'] = 'http://localhost:4000/v1'
    os.environ['LLM_MODEL'] = 'gpt-4o'
    yield


@pytest.fixture
def azure_openai_env(env_vars_backup) -> Generator[None, None, None]:
    """Set up environment for Azure OpenAI testing."""
    os.environ['LLM_PROVIDER'] = 'azure_openai'
    os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://test.openai.azure.com'
    os.environ['AZURE_OPENAI_API_KEY'] = 'test-azure-key'
    os.environ['LLM_MODEL'] = 'gpt-4o-deployment'
    yield


@pytest.fixture
def azure_docintel_env(env_vars_backup) -> Generator[None, None, None]:
    """Set up environment for Azure Document Intelligence testing."""
    os.environ['AZURE_DOCINTEL_ENDPOINT'] = 'https://test.cognitiveservices.azure.com'
    os.environ['AZURE_DOCINTEL_API_KEY'] = 'test-docintel-key'
    yield


@pytest.fixture
def full_config_env(env_vars_backup) -> Generator[None, None, None]:
    """Set up environment with all features enabled."""
    os.environ['LLM_PROVIDER'] = 'openai'
    os.environ['OPENAI_API_KEY'] = 'test-api-key'
    os.environ['LLM_MODEL'] = 'gpt-4o'
    os.environ['AZURE_DOCINTEL_ENDPOINT'] = 'https://test.cognitiveservices.azure.com'
    os.environ['AZURE_DOCINTEL_API_KEY'] = 'test-docintel-key'
    os.environ['ENABLE_PLUGINS'] = 'true'
    yield
