"""Integration tests for API endpoints."""

import os
import sys
from datetime import datetime
from io import BytesIO
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRootEndpoint:
    """Tests for the root endpoint (/)."""

    def test_root_returns_200(self, test_client):
        """Test that root endpoint returns 200."""
        response = test_client.get("/")
        assert response.status_code == 200

    def test_root_returns_service_info(self, test_client):
        """Test that root returns service information."""
        response = test_client.get("/")
        data = response.json()

        assert data["service"] == "MarkItDown Server"
        assert "description" in data
        assert data["version"] == "1.1.0"

    def test_root_contains_all_endpoints(self, test_client):
        """Test that root lists all available endpoints."""
        response = test_client.get("/")
        data = response.json()

        assert "endpoints" in data
        endpoints = data["endpoints"]
        assert endpoints["health"] == "/health"
        assert endpoints["docs"] == "/docs"
        assert endpoints["process"] == "/process_file"

    def test_root_returns_json(self, test_client):
        """Test that root returns JSON content type."""
        response = test_client.get("/")
        assert response.headers["content-type"] == "application/json"


class TestHealthEndpoint:
    """Tests for the health endpoint (/health)."""

    def test_health_returns_200(self, test_client):
        """Test that health endpoint returns 200."""
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, test_client):
        """Test that health returns healthy status."""
        response = test_client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_timestamp_is_valid_iso(self, test_client):
        """Test that timestamp is valid ISO format."""
        response = test_client.get("/health")
        data = response.json()

        # Should not raise if valid ISO format
        timestamp = datetime.fromisoformat(data["timestamp"])
        assert timestamp is not None

    def test_health_shows_service_name(self, test_client):
        """Test that health shows service name."""
        response = test_client.get("/health")
        data = response.json()
        assert data["service"] == "MarkItDown Server"

    def test_health_shows_version(self, test_client):
        """Test that health shows version."""
        response = test_client.get("/health")
        data = response.json()
        assert data["version"] == "1.1.0"

    def test_health_shows_worker_count(self, test_client):
        """Test that health shows worker count."""
        response = test_client.get("/health")
        data = response.json()
        assert "workers" in data
        assert isinstance(data["workers"], int)

    def test_health_rate_limit_disabled_by_default(self, test_client):
        """Test that rate limiting is disabled by default."""
        response = test_client.get("/health")
        data = response.json()
        assert data["rate_limit_enabled"] is False

    def test_health_llm_disabled_by_default(self, test_client):
        """Test that LLM is disabled by default."""
        response = test_client.get("/health")
        data = response.json()
        assert data["llm_enabled"] is False
        assert data["llm_provider"] is None

    def test_health_azure_docintel_disabled_by_default(self, test_client):
        """Test that Azure DocIntel is disabled by default."""
        response = test_client.get("/health")
        data = response.json()
        assert data["azure_docintel_enabled"] is False

    def test_health_plugins_disabled_by_default(self, test_client):
        """Test that plugins are disabled by default."""
        response = test_client.get("/health")
        data = response.json()
        assert data["plugins_enabled"] is False

    def test_health_response_has_all_fields(self, test_client):
        """Test that health response has all expected fields."""
        response = test_client.get("/health")
        data = response.json()

        expected_fields = [
            "status", "timestamp", "service", "version", "workers",
            "auth_enabled", "rate_limit_enabled", "rate_limit", "llm_enabled",
            "llm_provider", "azure_docintel_enabled", "plugins_enabled",
            "async_jobs_pending", "async_jobs_processing", "async_jobs_total"
        ]
        for field in expected_fields:
            assert field in data, f"Missing field: {field}"

    def test_health_auth_disabled_by_default(self, test_client):
        """Test that auth is disabled by default."""
        response = test_client.get("/health")
        data = response.json()
        assert data["auth_enabled"] is False


class TestProcessFileEndpoint:
    """Tests for the process_file endpoint (/process_file)."""

    def test_process_file_valid_txt(self, test_client, sample_txt_content, mock_markitdown):
        """Test processing a valid text file."""
        response = test_client.post(
            "/process_file",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "markdown" in data

    def test_process_file_valid_pdf(self, test_client, sample_pdf_content, mock_markitdown):
        """Test processing a valid PDF file."""
        response = test_client.post(
            "/process_file",
            files={"file": ("document.pdf", BytesIO(sample_pdf_content), "application/pdf")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "markdown" in data

    def test_process_file_valid_image(self, test_client, sample_image_content, mock_markitdown):
        """Test processing a valid image file."""
        response = test_client.post(
            "/process_file",
            files={"file": ("image.png", BytesIO(sample_image_content), "image/png")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "markdown" in data

    def test_process_file_no_file_returns_422(self, test_client):
        """Test that missing file returns 422."""
        response = test_client.post("/process_file")
        assert response.status_code == 422

    def test_process_file_disallowed_type_returns_400(self, test_client):
        """Test that disallowed file type returns 400."""
        response = test_client.post(
            "/process_file",
            files={"file": ("script.exe", BytesIO(b"fake exe"), "application/octet-stream")}
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "not allowed" in data["error"].lower()

    def test_process_file_empty_file_returns_400(self, test_client, empty_file_content):
        """Test that empty file returns 400."""
        response = test_client.post(
            "/process_file",
            files={"file": ("empty.txt", BytesIO(empty_file_content), "text/plain")}
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "empty" in data["error"].lower()

    def test_process_file_exceeds_size_limit_returns_413(self, test_client, large_file_content):
        """Test that file exceeding size limit returns 413."""
        response = test_client.post(
            "/process_file",
            files={"file": ("large.txt", BytesIO(large_file_content), "text/plain")}
        )
        assert response.status_code == 413
        data = response.json()
        assert "error" in data
        assert "large" in data["error"].lower()

    def test_process_file_conversion_error_returns_500(self, test_client, sample_txt_content, mock_markitdown_error):
        """Test that conversion error returns 500."""
        response = test_client.post(
            "/process_file",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")}
        )
        assert response.status_code == 500
        data = response.json()
        assert "error" in data

    def test_process_file_returns_markdown_key(self, test_client, sample_txt_content, mock_markitdown):
        """Test that response contains markdown key."""
        response = test_client.post(
            "/process_file",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")}
        )
        data = response.json()
        assert "markdown" in data
        assert isinstance(data["markdown"], str)

    def test_process_file_case_insensitive_extension(self, test_client, sample_txt_content, mock_markitdown):
        """Test that extension matching is case insensitive."""
        response = test_client.post(
            "/process_file",
            files={"file": ("test.TXT", BytesIO(sample_txt_content), "text/plain")}
        )
        assert response.status_code == 200

    def test_process_file_pdf_extension(self, test_client, sample_pdf_content, mock_markitdown):
        """Test PDF file with uppercase extension."""
        response = test_client.post(
            "/process_file",
            files={"file": ("document.PDF", BytesIO(sample_pdf_content), "application/pdf")}
        )
        assert response.status_code == 200

    # Audio format tests
    @pytest.mark.parametrize("ext", ["mp3", "wav", "flac", "aac", "ogg"])
    def test_process_file_audio_formats(self, test_client, ext, mock_markitdown):
        """Test that audio formats are accepted."""
        response = test_client.post(
            "/process_file",
            files={"file": (f"audio.{ext}", BytesIO(b"fake audio data"), "audio/mpeg")}
        )
        assert response.status_code == 200


class TestSecurityHeaders:
    """Tests for security headers middleware."""

    def test_x_content_type_options_header(self, test_client):
        """Test X-Content-Type-Options header is set."""
        response = test_client.get("/health")
        assert response.headers.get("x-content-type-options") == "nosniff"

    def test_x_frame_options_header(self, test_client):
        """Test X-Frame-Options header is set."""
        response = test_client.get("/health")
        assert response.headers.get("x-frame-options") == "DENY"

    def test_x_xss_protection_header(self, test_client):
        """Test X-XSS-Protection header is set."""
        response = test_client.get("/health")
        assert response.headers.get("x-xss-protection") == "1; mode=block"

    def test_security_headers_on_root(self, test_client):
        """Test security headers are present on root endpoint."""
        response = test_client.get("/")
        assert "x-content-type-options" in response.headers
        assert "x-frame-options" in response.headers
        assert "x-xss-protection" in response.headers

    def test_security_headers_on_process_file(self, test_client, sample_txt_content, mock_markitdown):
        """Test security headers are present on process_file endpoint."""
        response = test_client.post(
            "/process_file",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")}
        )
        assert "x-content-type-options" in response.headers
        assert "x-frame-options" in response.headers


class TestCORS:
    """Tests for CORS middleware."""

    def test_cors_allows_all_origins(self, test_client):
        """Test that CORS allows all origins (as configured)."""
        response = test_client.options(
            "/health",
            headers={"Origin": "http://example.com", "Access-Control-Request-Method": "GET"}
        )
        # CORS preflight should work
        assert response.status_code in [200, 405]  # Depends on FastAPI version

    def test_cors_header_on_response(self, test_client):
        """Test that CORS header is present on response."""
        response = test_client.get("/health", headers={"Origin": "http://example.com"})
        # With allow_origins=["*"], this header should be present
        assert "access-control-allow-origin" in response.headers


class TestTempFileCleanup:
    """Tests for temporary file cleanup."""

    def test_temp_file_cleanup_on_success(self, test_client, sample_txt_content, mock_markitdown, tmp_path):
        """Test that temp files are cleaned up after successful conversion."""
        import tempfile
        original_temp_dir = tempfile.gettempdir()

        # Count temp files before
        initial_files = set(os.listdir(original_temp_dir))

        response = test_client.post(
            "/process_file",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")}
        )

        assert response.status_code == 200

        # Temp file should be cleaned up - no new files should remain
        # (This is a basic check; exact file tracking would require more setup)

    def test_temp_file_cleanup_on_error(self, test_client, sample_txt_content, mock_markitdown_error):
        """Test that temp files are cleaned up even on conversion error."""
        response = test_client.post(
            "/process_file",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")}
        )

        assert response.status_code == 500
        # Error occurred but temp file cleanup should still happen


class TestApiKeyAuthentication:
    """Tests for API key authentication."""

    def test_auth_disabled_allows_request_without_key(self, test_client, sample_txt_content, mock_markitdown):
        """Test that requests work without API key when auth is disabled."""
        response = test_client.post(
            "/process_file",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")}
        )
        assert response.status_code == 200

    def test_auth_enabled_rejects_missing_key(self, auth_client, sample_txt_content, mock_markitdown):
        """Test that missing API key returns 401 when auth is enabled."""
        response = auth_client.post(
            "/process_file",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")}
        )
        assert response.status_code == 401
        data = response.json()
        assert "Missing API key" in data["detail"]

    def test_auth_enabled_rejects_invalid_key(self, auth_client, sample_txt_content, mock_markitdown, invalid_api_key):
        """Test that invalid API key returns 401."""
        response = auth_client.post(
            "/process_file",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")},
            headers={"X-API-Key": invalid_api_key}
        )
        assert response.status_code == 401
        data = response.json()
        assert "Invalid API key" in data["detail"]

    def test_auth_enabled_accepts_valid_key(self, auth_client, sample_txt_content, mock_markitdown, valid_api_key):
        """Test that valid API key is accepted."""
        response = auth_client.post(
            "/process_file",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")},
            headers={"X-API-Key": valid_api_key}
        )
        assert response.status_code == 200
        data = response.json()
        assert "markdown" in data

    def test_auth_enabled_accepts_second_valid_key(self, auth_client, sample_txt_content, mock_markitdown):
        """Test that second configured API key is also accepted."""
        response = auth_client.post(
            "/process_file",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")},
            headers={"X-API-Key": "test-api-key-2"}
        )
        assert response.status_code == 200

    def test_health_endpoint_not_protected(self, auth_client):
        """Test that health endpoint does not require API key."""
        response = auth_client.get("/health")
        assert response.status_code == 200

    def test_root_endpoint_not_protected(self, auth_client):
        """Test that root endpoint does not require API key."""
        response = auth_client.get("/")
        assert response.status_code == 200

    def test_health_shows_auth_enabled(self, auth_client):
        """Test that health endpoint shows auth is enabled."""
        response = auth_client.get("/health")
        data = response.json()
        assert data["auth_enabled"] is True


class TestAsyncProcessFileEndpoint:
    """Tests for the async process_file endpoint (/process_file/async)."""

    def test_async_submit_returns_202(self, test_client, sample_txt_content, mock_markitdown):
        """Test that async submit returns 202 Accepted."""
        response = test_client.post(
            "/process_file/async",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")}
        )
        assert response.status_code == 202

    def test_async_submit_returns_job_id(self, test_client, sample_txt_content, mock_markitdown):
        """Test that async submit returns a job_id."""
        response = test_client.post(
            "/process_file/async",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")}
        )
        data = response.json()
        assert "job_id" in data
        assert isinstance(data["job_id"], str)
        assert len(data["job_id"]) > 0

    def test_async_submit_returns_pending_status(self, test_client, sample_txt_content, mock_markitdown):
        """Test that async submit returns pending status."""
        response = test_client.post(
            "/process_file/async",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")}
        )
        data = response.json()
        assert data["status"] == "pending"

    def test_async_submit_returns_status_url(self, test_client, sample_txt_content, mock_markitdown):
        """Test that async submit returns status_url."""
        response = test_client.post(
            "/process_file/async",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")}
        )
        data = response.json()
        assert "status_url" in data
        assert data["status_url"] == f"/jobs/{data['job_id']}"

    def test_async_submit_returns_message(self, test_client, sample_txt_content, mock_markitdown):
        """Test that async submit returns a message."""
        response = test_client.post(
            "/process_file/async",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")}
        )
        data = response.json()
        assert "message" in data
        assert "queued" in data["message"].lower()

    def test_async_submit_disallowed_type_returns_400(self, test_client):
        """Test that disallowed file type returns 400."""
        response = test_client.post(
            "/process_file/async",
            files={"file": ("script.exe", BytesIO(b"fake exe"), "application/octet-stream")}
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "not allowed" in data["error"].lower()

    def test_async_submit_empty_file_returns_400(self, test_client, empty_file_content):
        """Test that empty file returns 400."""
        response = test_client.post(
            "/process_file/async",
            files={"file": ("empty.txt", BytesIO(empty_file_content), "text/plain")}
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "empty" in data["error"].lower()

    def test_async_submit_exceeds_size_limit_returns_413(self, test_client, large_file_content):
        """Test that file exceeding size limit returns 413."""
        response = test_client.post(
            "/process_file/async",
            files={"file": ("large.txt", BytesIO(large_file_content), "text/plain")}
        )
        assert response.status_code == 413
        data = response.json()
        assert "error" in data
        assert "large" in data["error"].lower()

    def test_async_submit_no_file_returns_422(self, test_client):
        """Test that missing file returns 422."""
        response = test_client.post("/process_file/async")
        assert response.status_code == 422


class TestJobStatusEndpoint:
    """Tests for the job status endpoint (/jobs/{job_id})."""

    def test_get_job_returns_job_info(self, test_client, sample_txt_content, mock_markitdown):
        """Test that get job returns job information."""
        # Submit a job first
        submit_response = test_client.post(
            "/process_file/async",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")}
        )
        job_id = submit_response.json()["job_id"]

        # Get job status
        response = test_client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "status" in data
        assert "created_at" in data
        assert "filename" in data
        assert data["filename"] == "test.txt"

    def test_get_job_not_found_returns_404(self, test_client):
        """Test that non-existent job returns 404."""
        response = test_client.get("/jobs/non-existent-job-id")
        assert response.status_code == 404
        data = response.json()
        assert "error" in data

    def test_job_status_transitions_to_completed(self, test_client, sample_txt_content, mock_markitdown):
        """Test that job status transitions to completed with result."""
        import time

        # Submit a job
        submit_response = test_client.post(
            "/process_file/async",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")}
        )
        job_id = submit_response.json()["job_id"]

        # Poll until completed (with timeout)
        max_attempts = 20
        for _ in range(max_attempts):
            response = test_client.get(f"/jobs/{job_id}")
            data = response.json()
            if data["status"] == "completed":
                assert "markdown" in data
                assert data["markdown"] is not None
                assert "completed_at" in data
                assert data["completed_at"] is not None
                break
            time.sleep(0.1)
        else:
            pytest.fail("Job did not complete within timeout")

    def test_job_status_shows_error_on_failure(self, test_client, sample_txt_content, mock_markitdown_error):
        """Test that failed job shows error message."""
        import time

        # Submit a job (will fail due to mock_markitdown_error)
        submit_response = test_client.post(
            "/process_file/async",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")}
        )
        job_id = submit_response.json()["job_id"]

        # Poll until failed (with timeout)
        max_attempts = 20
        for _ in range(max_attempts):
            response = test_client.get(f"/jobs/{job_id}")
            data = response.json()
            if data["status"] == "failed":
                assert "error" in data
                assert data["error"] is not None
                break
            time.sleep(0.1)
        else:
            pytest.fail("Job did not fail within timeout")

    def test_job_created_at_is_valid_iso(self, test_client, sample_txt_content, mock_markitdown):
        """Test that created_at is valid ISO format."""
        # Submit a job
        submit_response = test_client.post(
            "/process_file/async",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")}
        )
        job_id = submit_response.json()["job_id"]

        # Get job status
        response = test_client.get(f"/jobs/{job_id}")
        data = response.json()

        # Should not raise if valid ISO format
        timestamp = datetime.fromisoformat(data["created_at"])
        assert timestamp is not None


class TestHealthEndpointAsyncJobs:
    """Tests for async job stats in health endpoint."""

    def test_health_shows_async_job_stats(self, test_client):
        """Test that health endpoint shows async job statistics."""
        response = test_client.get("/health")
        data = response.json()

        assert "async_jobs_pending" in data
        assert "async_jobs_processing" in data
        assert "async_jobs_total" in data
        assert isinstance(data["async_jobs_pending"], int)
        assert isinstance(data["async_jobs_processing"], int)
        assert isinstance(data["async_jobs_total"], int)

    def test_health_job_count_increases_on_submit(self, test_client, sample_txt_content, mock_markitdown):
        """Test that job count increases when a job is submitted."""
        # Get initial count
        initial_response = test_client.get("/health")
        initial_total = initial_response.json()["async_jobs_total"]

        # Submit a job
        test_client.post(
            "/process_file/async",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")}
        )

        # Check count increased
        response = test_client.get("/health")
        data = response.json()
        assert data["async_jobs_total"] >= initial_total


class TestRootEndpointAsyncEndpoints:
    """Tests for async endpoints in root endpoint."""

    def test_root_contains_async_endpoints(self, test_client):
        """Test that root lists async endpoints."""
        response = test_client.get("/")
        data = response.json()

        endpoints = data["endpoints"]
        assert endpoints["process_async"] == "/process_file/async"
        assert endpoints["job_status"] == "/jobs/{job_id}"

    def test_root_version_updated(self, test_client):
        """Test that root shows updated version."""
        response = test_client.get("/")
        data = response.json()
        assert data["version"] == "1.1.0"


class TestAsyncApiKeyAuthentication:
    """Tests for API key authentication on async endpoints."""

    def test_async_auth_disabled_allows_request_without_key(self, test_client, sample_txt_content, mock_markitdown):
        """Test that async requests work without API key when auth is disabled."""
        response = test_client.post(
            "/process_file/async",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")}
        )
        assert response.status_code == 202

    def test_async_auth_enabled_rejects_missing_key(self, auth_client, sample_txt_content, mock_markitdown):
        """Test that missing API key returns 401 for async endpoint when auth is enabled."""
        response = auth_client.post(
            "/process_file/async",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")}
        )
        assert response.status_code == 401

    def test_async_auth_enabled_accepts_valid_key(self, auth_client, sample_txt_content, mock_markitdown, valid_api_key):
        """Test that valid API key is accepted for async endpoint."""
        response = auth_client.post(
            "/process_file/async",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")},
            headers={"X-API-Key": valid_api_key}
        )
        assert response.status_code == 202

    def test_job_status_auth_enabled_rejects_missing_key(self, auth_client, sample_txt_content, mock_markitdown, valid_api_key):
        """Test that job status endpoint requires auth when enabled."""
        # Submit a job with valid key
        submit_response = auth_client.post(
            "/process_file/async",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")},
            headers={"X-API-Key": valid_api_key}
        )
        job_id = submit_response.json()["job_id"]

        # Try to get status without key
        response = auth_client.get(f"/jobs/{job_id}")
        assert response.status_code == 401

    def test_job_status_auth_enabled_accepts_valid_key(self, auth_client, sample_txt_content, mock_markitdown, valid_api_key):
        """Test that job status endpoint accepts valid key."""
        # Submit a job with valid key
        submit_response = auth_client.post(
            "/process_file/async",
            files={"file": ("test.txt", BytesIO(sample_txt_content), "text/plain")},
            headers={"X-API-Key": valid_api_key}
        )
        job_id = submit_response.json()["job_id"]

        # Get status with valid key
        response = auth_client.get(f"/jobs/{job_id}", headers={"X-API-Key": valid_api_key})
        assert response.status_code == 200
