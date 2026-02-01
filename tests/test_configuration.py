"""Tests for configuration and environment variable handling."""

import importlib
import os
import sys

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestWorkerConfiguration:
    """Tests for worker-related configuration."""

    def test_workers_default_value(self, env_vars_backup):
        """Test WORKERS defaults to 1."""
        os.environ.pop('WORKERS', None)

        import app
        importlib.reload(app)

        assert app.WORKERS == 1

    def test_workers_custom_value(self, env_vars_backup):
        """Test WORKERS can be set via environment."""
        os.environ['WORKERS'] = '4'

        import app
        importlib.reload(app)

        assert app.WORKERS == 4

    def test_workers_string_converted_to_int(self, env_vars_backup):
        """Test WORKERS string is converted to int."""
        os.environ['WORKERS'] = '8'

        import app
        importlib.reload(app)

        assert app.WORKERS == 8
        assert isinstance(app.WORKERS, int)


class TestRateLimitConfiguration:
    """Tests for rate limiting configuration."""

    def test_enable_rate_limit_default_false(self, env_vars_backup):
        """Test ENABLE_RATE_LIMIT defaults to False."""
        os.environ.pop('ENABLE_RATE_LIMIT', None)

        import app
        importlib.reload(app)

        assert app.ENABLE_RATE_LIMIT is False

    def test_enable_rate_limit_true(self, env_vars_backup):
        """Test ENABLE_RATE_LIMIT can be enabled."""
        os.environ['ENABLE_RATE_LIMIT'] = 'true'

        import app
        importlib.reload(app)

        assert app.ENABLE_RATE_LIMIT is True

    def test_enable_rate_limit_case_insensitive(self, env_vars_backup):
        """Test ENABLE_RATE_LIMIT is case insensitive."""
        os.environ['ENABLE_RATE_LIMIT'] = 'TRUE'

        import app
        importlib.reload(app)

        assert app.ENABLE_RATE_LIMIT is True

    def test_enable_rate_limit_mixed_case(self, env_vars_backup):
        """Test ENABLE_RATE_LIMIT works with mixed case."""
        os.environ['ENABLE_RATE_LIMIT'] = 'True'

        import app
        importlib.reload(app)

        assert app.ENABLE_RATE_LIMIT is True

    def test_rate_limit_default_value(self, env_vars_backup):
        """Test RATE_LIMIT default value."""
        os.environ.pop('RATE_LIMIT', None)

        import app
        importlib.reload(app)

        assert app.RATE_LIMIT == '60/minute'

    def test_rate_limit_custom_value(self, env_vars_backup):
        """Test RATE_LIMIT can be customized."""
        os.environ['RATE_LIMIT'] = '100/minute'

        import app
        importlib.reload(app)

        assert app.RATE_LIMIT == '100/minute'


class TestLLMConfiguration:
    """Tests for LLM-related configuration."""

    def test_llm_provider_default_empty(self, env_vars_backup):
        """Test LLM_PROVIDER defaults to empty string."""
        os.environ.pop('LLM_PROVIDER', None)

        import app
        importlib.reload(app)

        assert app.LLM_PROVIDER == ''

    def test_llm_provider_openai(self, env_vars_backup):
        """Test LLM_PROVIDER can be set to openai."""
        os.environ['LLM_PROVIDER'] = 'openai'

        import app
        importlib.reload(app)

        assert app.LLM_PROVIDER == 'openai'

    def test_llm_provider_azure_openai(self, env_vars_backup):
        """Test LLM_PROVIDER can be set to azure_openai."""
        os.environ['LLM_PROVIDER'] = 'azure_openai'

        import app
        importlib.reload(app)

        assert app.LLM_PROVIDER == 'azure_openai'

    def test_llm_provider_case_insensitive(self, env_vars_backup):
        """Test LLM_PROVIDER is lowercased."""
        os.environ['LLM_PROVIDER'] = 'OPENAI'

        import app
        importlib.reload(app)

        assert app.LLM_PROVIDER == 'openai'

    def test_llm_model_default_gpt4o(self, env_vars_backup):
        """Test LLM_MODEL defaults to gpt-4o."""
        os.environ.pop('LLM_MODEL', None)

        import app
        importlib.reload(app)

        assert app.LLM_MODEL == 'gpt-4o'

    def test_llm_model_custom_value(self, env_vars_backup):
        """Test LLM_MODEL can be customized."""
        os.environ['LLM_MODEL'] = 'gpt-4-turbo'

        import app
        importlib.reload(app)

        assert app.LLM_MODEL == 'gpt-4-turbo'

    def test_openai_api_key_not_set(self, env_vars_backup):
        """Test OPENAI_API_KEY is None when not set."""
        os.environ.pop('OPENAI_API_KEY', None)

        import app
        importlib.reload(app)

        assert app.OPENAI_API_KEY is None

    def test_openai_api_key_set(self, env_vars_backup):
        """Test OPENAI_API_KEY can be set."""
        os.environ['OPENAI_API_KEY'] = 'sk-test-key'

        import app
        importlib.reload(app)

        assert app.OPENAI_API_KEY == 'sk-test-key'

    def test_openai_base_url_not_set(self, env_vars_backup):
        """Test OPENAI_BASE_URL is None when not set."""
        os.environ.pop('OPENAI_BASE_URL', None)

        import app
        importlib.reload(app)

        assert app.OPENAI_BASE_URL is None

    def test_openai_base_url_set(self, env_vars_backup):
        """Test OPENAI_BASE_URL can be set for LiteLLM."""
        os.environ['OPENAI_BASE_URL'] = 'http://localhost:4000/v1'

        import app
        importlib.reload(app)

        assert app.OPENAI_BASE_URL == 'http://localhost:4000/v1'

    def test_azure_openai_api_version_default(self, env_vars_backup):
        """Test AZURE_OPENAI_API_VERSION default value."""
        os.environ.pop('AZURE_OPENAI_API_VERSION', None)

        import app
        importlib.reload(app)

        assert app.AZURE_OPENAI_API_VERSION == '2024-02-15-preview'

    def test_azure_openai_api_version_custom(self, env_vars_backup):
        """Test AZURE_OPENAI_API_VERSION can be customized."""
        os.environ['AZURE_OPENAI_API_VERSION'] = '2024-05-01-preview'

        import app
        importlib.reload(app)

        assert app.AZURE_OPENAI_API_VERSION == '2024-05-01-preview'


class TestAzureDocIntelConfiguration:
    """Tests for Azure Document Intelligence configuration."""

    def test_azure_docintel_endpoint_not_set(self, env_vars_backup):
        """Test AZURE_DOCINTEL_ENDPOINT is None when not set."""
        os.environ.pop('AZURE_DOCINTEL_ENDPOINT', None)

        import app
        importlib.reload(app)

        assert app.AZURE_DOCINTEL_ENDPOINT is None

    def test_azure_docintel_endpoint_set(self, env_vars_backup):
        """Test AZURE_DOCINTEL_ENDPOINT can be set."""
        os.environ['AZURE_DOCINTEL_ENDPOINT'] = 'https://test.cognitiveservices.azure.com'

        import app
        importlib.reload(app)

        assert app.AZURE_DOCINTEL_ENDPOINT == 'https://test.cognitiveservices.azure.com'

    def test_azure_docintel_api_key_not_set(self, env_vars_backup):
        """Test AZURE_DOCINTEL_API_KEY is None when not set."""
        os.environ.pop('AZURE_DOCINTEL_API_KEY', None)

        import app
        importlib.reload(app)

        assert app.AZURE_DOCINTEL_API_KEY is None

    def test_azure_docintel_file_types_not_set(self, env_vars_backup):
        """Test AZURE_DOCINTEL_FILE_TYPES is None when not set."""
        os.environ.pop('AZURE_DOCINTEL_FILE_TYPES', None)

        import app
        importlib.reload(app)

        assert app.AZURE_DOCINTEL_FILE_TYPES is None

    def test_azure_docintel_file_types_comma_separated(self, env_vars_backup):
        """Test AZURE_DOCINTEL_FILE_TYPES stores comma-separated string."""
        os.environ['AZURE_DOCINTEL_FILE_TYPES'] = 'pdf,docx,xlsx'

        import app
        importlib.reload(app)

        assert app.AZURE_DOCINTEL_FILE_TYPES == 'pdf,docx,xlsx'


class TestPluginConfiguration:
    """Tests for plugin configuration."""

    def test_enable_plugins_default_false(self, env_vars_backup):
        """Test ENABLE_PLUGINS defaults to False."""
        os.environ.pop('ENABLE_PLUGINS', None)

        import app
        importlib.reload(app)

        assert app.ENABLE_PLUGINS is False

    def test_enable_plugins_true(self, env_vars_backup):
        """Test ENABLE_PLUGINS can be enabled."""
        os.environ['ENABLE_PLUGINS'] = 'true'

        import app
        importlib.reload(app)

        assert app.ENABLE_PLUGINS is True

    def test_enable_plugins_case_insensitive(self, env_vars_backup):
        """Test ENABLE_PLUGINS is case insensitive."""
        os.environ['ENABLE_PLUGINS'] = 'TRUE'

        import app
        importlib.reload(app)

        assert app.ENABLE_PLUGINS is True


class TestExifToolConfiguration:
    """Tests for ExifTool path configuration."""

    def test_exiftool_path_not_set(self, env_vars_backup):
        """Test EXIFTOOL_PATH is None when not set."""
        os.environ.pop('EXIFTOOL_PATH', None)

        import app
        importlib.reload(app)

        assert app.EXIFTOOL_PATH is None

    def test_exiftool_path_set(self, env_vars_backup):
        """Test EXIFTOOL_PATH can be set."""
        os.environ['EXIFTOOL_PATH'] = '/usr/local/bin/exiftool'

        import app
        importlib.reload(app)

        assert app.EXIFTOOL_PATH == '/usr/local/bin/exiftool'


class TestAllowedExtensions:
    """Tests for allowed file extensions configuration."""

    def test_allowed_extensions_is_set(self, env_vars_backup):
        """Test ALLOWED_EXTENSIONS is defined."""
        import app
        importlib.reload(app)

        assert app.ALLOWED_EXTENSIONS is not None
        assert isinstance(app.ALLOWED_EXTENSIONS, set)

    def test_allowed_extensions_contains_documents(self, env_vars_backup):
        """Test ALLOWED_EXTENSIONS contains document types."""
        import app
        importlib.reload(app)

        document_types = {'doc', 'docx', 'pdf', 'ppt', 'pptx', 'xls', 'xlsx', 'txt'}
        assert document_types.issubset(app.ALLOWED_EXTENSIONS)

    def test_allowed_extensions_contains_images(self, env_vars_backup):
        """Test ALLOWED_EXTENSIONS contains image types."""
        import app
        importlib.reload(app)

        image_types = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp', 'svg'}
        assert image_types.issubset(app.ALLOWED_EXTENSIONS)

    def test_allowed_extensions_contains_audio(self, env_vars_backup):
        """Test ALLOWED_EXTENSIONS contains audio types."""
        import app
        importlib.reload(app)

        audio_types = {'mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a', 'wma'}
        assert audio_types.issubset(app.ALLOWED_EXTENSIONS)


class TestMaxFileSize:
    """Tests for max file size configuration."""

    def test_max_file_size_is_50mb(self, env_vars_backup):
        """Test MAX_FILE_SIZE is 50MB."""
        import app
        importlib.reload(app)

        expected_size = 50 * 1024 * 1024  # 50MB in bytes
        assert app.MAX_FILE_SIZE == expected_size
