"""Unit tests for helper functions in app.py."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import sanitize_extension, allowed_file, ALLOWED_EXTENSIONS


class TestSanitizeExtension:
    """Tests for the sanitize_extension function."""

    def test_none_input(self):
        """Test that None input returns empty string."""
        assert sanitize_extension(None) == ""

    def test_empty_string(self):
        """Test that empty string returns empty string."""
        assert sanitize_extension("") == ""

    def test_simple_filename(self):
        """Test extraction from simple filename."""
        assert sanitize_extension("document.pdf") == ".pdf"

    def test_simple_filename_docx(self):
        """Test extraction from docx filename."""
        assert sanitize_extension("report.docx") == ".docx"

    def test_simple_filename_txt(self):
        """Test extraction from txt filename."""
        assert sanitize_extension("readme.txt") == ".txt"

    def test_with_path(self):
        """Test extraction from filename with path."""
        assert sanitize_extension("/path/to/file.pdf") == ".pdf"

    def test_with_windows_path(self):
        """Test extraction from filename with Windows-style path."""
        assert sanitize_extension("C:\\Users\\test\\file.docx") == ".docx"

    def test_multiple_dots(self):
        """Test extraction from filename with multiple dots."""
        assert sanitize_extension("file.name.with.dots.txt") == ".txt"

    def test_no_extension(self):
        """Test filename without extension."""
        assert sanitize_extension("README") == ""

    def test_dot_only(self):
        """Test filename that is just a dot."""
        assert sanitize_extension(".") == ""

    def test_hidden_file_no_extension(self):
        """Test hidden file without extension."""
        assert sanitize_extension(".gitignore") == ".gitignore"

    def test_hidden_file_with_extension(self):
        """Test hidden file with extension."""
        assert sanitize_extension(".config.json") == ".json"

    def test_invalid_characters_at_symbol(self):
        """Test extension with @ symbol is rejected."""
        assert sanitize_extension("file.p@df") == ""

    def test_invalid_characters_hash(self):
        """Test extension with # symbol is rejected."""
        assert sanitize_extension("file.pd#f") == ""

    def test_invalid_characters_space(self):
        """Test extension with space is rejected."""
        assert sanitize_extension("file.pd f") == ""

    def test_invalid_characters_dash(self):
        """Test extension with dash is rejected (not alphanumeric)."""
        assert sanitize_extension("file.tar-gz") == ""

    def test_case_sensitivity_uppercase(self):
        """Test that uppercase extensions are lowercased."""
        assert sanitize_extension("FILE.PDF") == ".pdf"

    def test_case_sensitivity_mixed(self):
        """Test that mixed case extensions are lowercased."""
        assert sanitize_extension("Document.DocX") == ".docx"

    def test_path_traversal_attempt(self):
        """Test path traversal attempt is handled safely - no extension means empty string."""
        # passwd has no dot, so correctly returns empty string
        assert sanitize_extension("../../../etc/passwd") == ""

    def test_path_traversal_with_extension(self):
        """Test path traversal with valid extension."""
        assert sanitize_extension("../../../etc/file.txt") == ".txt"

    def test_numeric_extension(self):
        """Test purely numeric extension."""
        assert sanitize_extension("file.123") == ".123"

    def test_single_char_extension(self):
        """Test single character extension."""
        assert sanitize_extension("file.c") == ".c"

    def test_long_extension(self):
        """Test long extension."""
        assert sanitize_extension("file.dockerfile") == ".dockerfile"

    def test_extension_with_numbers(self):
        """Test extension with numbers (mp3, mp4, etc)."""
        assert sanitize_extension("audio.mp3") == ".mp3"
        assert sanitize_extension("video.mp4") == ".mp4"
        assert sanitize_extension("archive.7z") == ".7z"


class TestAllowedFile:
    """Tests for the allowed_file function."""

    def test_none_input(self):
        """Test that None input returns False."""
        assert allowed_file(None) is False

    def test_empty_string(self):
        """Test that empty string returns False."""
        assert allowed_file("") is False

    # Document format tests
    @pytest.mark.parametrize("ext", ["doc", "docx", "pdf", "ppt", "pptx", "xls", "xlsx", "odt", "ods", "odp", "txt"])
    def test_valid_document_types(self, ext):
        """Test that all document types are allowed."""
        assert allowed_file(f"document.{ext}") is True

    # Image format tests
    @pytest.mark.parametrize("ext", ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", "svg"])
    def test_valid_image_types(self, ext):
        """Test that all image types are allowed."""
        assert allowed_file(f"image.{ext}") is True

    # Audio format tests
    @pytest.mark.parametrize("ext", ["mp3", "wav", "flac", "aac", "ogg", "m4a", "wma"])
    def test_valid_audio_types(self, ext):
        """Test that all audio types are allowed."""
        assert allowed_file(f"audio.{ext}") is True

    # Disallowed types
    @pytest.mark.parametrize("ext", ["exe", "sh", "bat", "zip", "rar", "py", "js", "php", "dll", "so"])
    def test_disallowed_types(self, ext):
        """Test that dangerous/unsupported types are not allowed."""
        assert allowed_file(f"file.{ext}") is False

    def test_case_insensitive_pdf(self):
        """Test that extension matching is case insensitive."""
        assert allowed_file("document.PDF") is True

    def test_case_insensitive_docx(self):
        """Test case insensitivity for docx."""
        assert allowed_file("document.DOCX") is True

    def test_case_insensitive_mixed(self):
        """Test mixed case extension."""
        assert allowed_file("image.JpG") is True

    def test_filename_with_path(self):
        """Test allowed file with path."""
        assert allowed_file("/path/to/document.pdf") is True

    def test_disallowed_with_path(self):
        """Test disallowed file with path."""
        assert allowed_file("/path/to/script.exe") is False

    def test_no_extension(self):
        """Test file without extension is not allowed."""
        assert not allowed_file("README")

    def test_double_extension_allowed_last(self):
        """Test file with double extension, last one allowed."""
        assert allowed_file("document.backup.pdf") is True

    def test_double_extension_disallowed_last(self):
        """Test file with double extension, last one not allowed."""
        assert allowed_file("document.pdf.exe") is False

    def test_all_allowed_extensions_count(self):
        """Verify the count of allowed extensions."""
        # 11 document + 8 image + 7 audio = 26 extensions
        assert len(ALLOWED_EXTENSIONS) == 26


class TestCreateMarkitdownInstance:
    """Tests for the create_markitdown_instance function."""

    def test_default_config(self, clean_env):
        """Test instance creation with default (no LLM) config."""
        # Need to reimport to pick up clean env
        import importlib
        import app
        importlib.reload(app)

        instance = app.create_markitdown_instance()
        assert instance is not None

    def test_with_openai_missing_key(self, env_vars_backup):
        """Test that OpenAI is not configured when API key is missing."""
        os.environ['LLM_PROVIDER'] = 'openai'
        os.environ.pop('OPENAI_API_KEY', None)

        import importlib
        import app
        importlib.reload(app)

        # Should not raise, just skip LLM config
        instance = app.create_markitdown_instance()
        assert instance is not None

    def test_with_azure_openai_missing_endpoint(self, env_vars_backup):
        """Test that Azure OpenAI is not configured when endpoint is missing."""
        os.environ['LLM_PROVIDER'] = 'azure_openai'
        os.environ['AZURE_OPENAI_API_KEY'] = 'test-key'
        os.environ.pop('AZURE_OPENAI_ENDPOINT', None)

        import importlib
        import app
        importlib.reload(app)

        instance = app.create_markitdown_instance()
        assert instance is not None

    def test_with_plugins_enabled(self, env_vars_backup):
        """Test instance creation with plugins enabled."""
        os.environ['ENABLE_PLUGINS'] = 'true'

        import importlib
        import app
        importlib.reload(app)

        instance = app.create_markitdown_instance()
        assert instance is not None

    def test_with_exiftool_path(self, env_vars_backup):
        """Test instance creation with custom ExifTool path."""
        os.environ['EXIFTOOL_PATH'] = '/usr/local/bin/exiftool'

        import importlib
        import app
        importlib.reload(app)

        instance = app.create_markitdown_instance()
        assert instance is not None


class TestConvertToMd:
    """Tests for the convert_to_md function."""

    def test_convert_calls_markitdown(self, mock_markitdown, tmp_path):
        """Test that convert_to_md calls MarkItDown.convert."""
        mock_class, mock_instance, mock_result = mock_markitdown

        # Create a temp file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        from app import convert_to_md
        result = convert_to_md(str(test_file))

        assert result == mock_result.text_content
        mock_instance.convert.assert_called_once_with(str(test_file))

    def test_convert_returns_text_content(self, mock_markitdown, tmp_path):
        """Test that convert_to_md returns text_content from result."""
        mock_class, mock_instance, mock_result = mock_markitdown
        mock_result.text_content = "# Custom Markdown"

        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content")

        from app import convert_to_md
        result = convert_to_md(str(test_file))

        assert result == "# Custom Markdown"
