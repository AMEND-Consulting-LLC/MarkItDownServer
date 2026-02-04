import asyncio
import logging
import os
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from markitdown import MarkItDown

# Optional LLM support (OpenAI and Azure OpenAI)
try:
    from openai import OpenAI, AzureOpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Optional Azure Document Intelligence support
try:
    from azure.core.credentials import AzureKeyCredential
    AZURE_DOCINTEL_AVAILABLE = True
except ImportError:
    AZURE_DOCINTEL_AVAILABLE = False

# Optional rate limiting support
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMIT_AVAILABLE = True
except ImportError:
    RATE_LIMIT_AVAILABLE = False

# Configure logging with configurable level
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Environment-based configuration
WORKERS = int(os.getenv('WORKERS', '1'))
ENABLE_RATE_LIMIT = os.getenv('ENABLE_RATE_LIMIT', 'false').lower() == 'true'
RATE_LIMIT = os.getenv('RATE_LIMIT', '60/minute')  # Default: 60 requests per minute

# Authentication Configuration
ENABLE_AUTH = os.getenv('ENABLE_AUTH', 'false').lower() == 'true'
API_KEYS_RAW = os.getenv('API_KEYS', '')
API_KEYS = set(key.strip() for key in API_KEYS_RAW.split(',') if key.strip())

# LLM Configuration (for image captioning and enhanced document processing)
LLM_PROVIDER = os.getenv('LLM_PROVIDER', '').lower()  # 'openai', 'azure_openai', or empty
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')  # For LiteLLM proxy or other OpenAI-compatible APIs
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4o')
LLM_PROMPT = os.getenv('LLM_PROMPT')

# Azure Document Intelligence Configuration
AZURE_DOCINTEL_ENDPOINT = os.getenv('AZURE_DOCINTEL_ENDPOINT')
AZURE_DOCINTEL_API_KEY = os.getenv('AZURE_DOCINTEL_API_KEY')
AZURE_DOCINTEL_FILE_TYPES = os.getenv('AZURE_DOCINTEL_FILE_TYPES')  # Comma-separated: pdf,docx,xlsx,pptx,png,jpg,tiff,bmp

# Other MarkItDown features
ENABLE_PLUGINS = os.getenv('ENABLE_PLUGINS', 'false').lower() == 'true'
EXIFTOOL_PATH = os.getenv('EXIFTOOL_PATH')

# FastAPI app with metadata
app = FastAPI(
    title="MarkItDown Server",
    description="API for converting various document formats to Markdown",
    version="1.1.0",
    contact={
        "name": "El Bruno",
        "url": "https://github.com/elbruno/MarkItDownServer",
    },
    license_info={
        "name": "MIT",
        "url": "https://github.com/elbruno/MarkItDownServer/blob/main/LICENSE",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure rate limiting if enabled and available
if ENABLE_RATE_LIMIT and RATE_LIMIT_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    logger.info(f"Rate limiting enabled: {RATE_LIMIT}")
elif ENABLE_RATE_LIMIT and not RATE_LIMIT_AVAILABLE:
    logger.warning("Rate limiting requested but slowapi not installed. Install with: pip install slowapi")
else:
    logger.info("Rate limiting disabled")

# API Key authentication setup
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str | None = Depends(api_key_header)) -> str | None:
    """Verify the API key if authentication is enabled.

    Args:
        api_key: The API key from the X-API-Key header

    Returns:
        The validated API key if auth is enabled, None if auth is disabled

    Raises:
        HTTPException: If auth is enabled and the key is missing or invalid
    """
    if not ENABLE_AUTH:
        return None

    if not API_KEYS:
        logger.warning("Authentication enabled but no API keys configured")
        raise HTTPException(
            status_code=500,
            detail="Server authentication misconfigured"
        )

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key header."
        )

    # Use secrets.compare_digest for timing-safe comparison
    import secrets
    for valid_key in API_KEYS:
        if secrets.compare_digest(api_key, valid_key):
            return api_key

    raise HTTPException(
        status_code=401,
        detail="Invalid API key"
    )

if ENABLE_AUTH:
    if API_KEYS:
        logger.info(f"API key authentication enabled ({len(API_KEYS)} key(s) configured)")
    else:
        logger.warning("API key authentication enabled but no keys configured!")
else:
    logger.info("API key authentication disabled")

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

# Configuration
ALLOWED_EXTENSIONS = {
    # Document formats
    'doc', 'docx', 'ppt', 'pptx', 'pdf', 'xls', 'xlsx', 'odt', 'ods', 'odp', 'txt',
    # Image formats
    'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp', 'svg',
    # Audio formats
    'mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a', 'wma'
}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Response models
class MarkdownResponse(BaseModel):
    markdown: str

class ErrorResponse(BaseModel):
    error: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    service: str
    version: str
    workers: int
    auth_enabled: bool
    rate_limit_enabled: bool
    rate_limit: str | None
    llm_enabled: bool
    llm_provider: str | None
    azure_docintel_enabled: bool
    plugins_enabled: bool
    async_jobs_pending: int
    async_jobs_processing: int
    async_jobs_total: int


# Async job processing models
JobStatus = Literal["pending", "processing", "completed", "failed"]


class AsyncJobSubmitResponse(BaseModel):
    """Response returned when submitting a file for async processing."""
    job_id: str
    status: JobStatus
    message: str
    status_url: str


class JobStatusResponse(BaseModel):
    """Response returned when checking job status."""
    job_id: str
    status: JobStatus
    created_at: str
    completed_at: str | None = None
    filename: str
    markdown: str | None = None
    error: str | None = None


class Job:
    """Represents an async file processing job."""

    def __init__(self, job_id: str, filename: str, temp_file_path: str):
        self.id = job_id
        self.filename = filename
        self.temp_file_path = temp_file_path
        self.status: JobStatus = "pending"
        self.created_at = datetime.utcnow()
        self.completed_at: datetime | None = None
        self.result: str | None = None
        self.error: str | None = None

    def to_response(self) -> JobStatusResponse:
        """Convert job to API response."""
        return JobStatusResponse(
            job_id=self.id,
            status=self.status,
            created_at=self.created_at.isoformat(),
            completed_at=self.completed_at.isoformat() if self.completed_at else None,
            filename=self.filename,
            markdown=self.result,
            error=self.error
        )


class JobStore:
    """In-memory store for async jobs with TTL-based expiration."""

    JOB_TTL_HOURS = 1  # Jobs expire after 1 hour
    CLEANUP_INTERVAL_MINUTES = 10  # Cleanup runs every 10 minutes

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def create(self, filename: str, temp_file_path: str) -> Job:
        """Create a new job and return it."""
        job_id = str(uuid.uuid4())
        job = Job(job_id, filename, temp_file_path)
        async with self._lock:
            self._jobs[job_id] = job
        logger.info(f"Created async job {job_id} for file: {filename}")
        return job

    async def get(self, job_id: str) -> Job | None:
        """Get a job by ID, or None if not found or expired."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job and self._is_expired(job):
                # Clean up expired job
                del self._jobs[job_id]
                self._cleanup_temp_file(job)
                return None
            return job

    async def update_status(
        self,
        job_id: str,
        status: JobStatus,
        result: str | None = None,
        error: str | None = None
    ) -> None:
        """Update job status and optionally set result or error."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = status
                if result is not None:
                    job.result = result
                if error is not None:
                    job.error = error
                if status in ("completed", "failed"):
                    job.completed_at = datetime.utcnow()
                    # Clean up temp file after processing
                    self._cleanup_temp_file(job)
                logger.debug(f"Job {job_id} status updated to: {status}")

    async def cleanup_expired(self) -> int:
        """Remove expired jobs and return count of removed jobs."""
        removed = 0
        async with self._lock:
            expired_ids = [
                job_id for job_id, job in self._jobs.items()
                if self._is_expired(job)
            ]
            for job_id in expired_ids:
                job = self._jobs.pop(job_id)
                self._cleanup_temp_file(job)
                removed += 1
        if removed > 0:
            logger.info(f"Cleaned up {removed} expired job(s)")
        return removed

    def get_stats(self) -> dict[str, int]:
        """Get job statistics (non-async for health endpoint)."""
        pending = sum(1 for j in self._jobs.values() if j.status == "pending")
        processing = sum(1 for j in self._jobs.values() if j.status == "processing")
        return {
            "pending": pending,
            "processing": processing,
            "total": len(self._jobs)
        }

    def _is_expired(self, job: Job) -> bool:
        """Check if a job has expired based on TTL."""
        expiry_time = job.created_at + timedelta(hours=self.JOB_TTL_HOURS)
        return datetime.utcnow() > expiry_time

    def _cleanup_temp_file(self, job: Job) -> None:
        """Clean up temporary file associated with a job."""
        if job.temp_file_path and os.path.exists(job.temp_file_path):
            try:
                os.remove(job.temp_file_path)
                logger.debug(f"Cleaned up temp file for job {job.id}")
            except OSError as e:
                logger.warning(f"Failed to clean up temp file for job {job.id}: {e}")


# Global job store instance
job_store = JobStore()

# Thread pool for running blocking conversions
thread_pool = ThreadPoolExecutor(max_workers=10)


async def periodic_job_cleanup():
    """Background task that periodically cleans up expired jobs."""
    while True:
        await asyncio.sleep(JobStore.CLEANUP_INTERVAL_MINUTES * 60)
        try:
            await job_store.cleanup_expired()
        except Exception as e:
            logger.error(f"Error during job cleanup: {e}")


# Store cleanup task reference for shutdown
_cleanup_task: asyncio.Task | None = None


@app.on_event("startup")
async def startup_event():
    """Start background tasks on application startup."""
    global _cleanup_task
    _cleanup_task = asyncio.create_task(periodic_job_cleanup())
    logger.info("Started periodic job cleanup task")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    global _cleanup_task
    if _cleanup_task:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
    thread_pool.shutdown(wait=False)
    logger.info("Shutdown complete")

def sanitize_extension(filename: str | None) -> str:
    """Extract and sanitize the file extension from a filename.

    Prevents path traversal by extracting only the basename's extension
    and validating it contains only alphanumeric characters.

    Args:
        filename: The filename to extract extension from

    Returns:
        Sanitized extension with leading dot (e.g., '.pdf') or empty string
    """
    if not filename:
        return ""
    # Get only the basename to prevent path traversal
    basename = Path(filename).name
    if '.' not in basename:
        return ""
    ext = basename.rsplit('.', 1)[1].lower()
    # Validate extension contains only alphanumeric characters
    if ext.isalnum():
        return f".{ext}"
    return ""

def allowed_file(filename: str | None) -> bool:
    """Check if the uploaded file has an allowed extension.

    Args:
        filename: Name of the file to check

    Returns:
        True if file extension is allowed, False otherwise
    """
    if not filename:
        return False
    ext = sanitize_extension(filename)
    return ext and ext[1:] in ALLOWED_EXTENSIONS  # Remove leading dot for comparison

def create_markitdown_instance() -> MarkItDown:
    """Create a configured MarkItDown instance based on environment variables.

    Returns:
        Configured MarkItDown instance with LLM and/or Azure Document Intelligence
        if the appropriate environment variables are set.
    """
    kwargs = {
        'enable_plugins': ENABLE_PLUGINS
    }

    features = []

    # Configure LLM client for image captioning
    if LLM_PROVIDER == 'openai' and LLM_AVAILABLE and OPENAI_API_KEY:
        client_kwargs = {'api_key': OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            client_kwargs['base_url'] = OPENAI_BASE_URL
            features.append(f"LLM(OpenAI-compatible: {OPENAI_BASE_URL})")
        else:
            features.append("LLM(OpenAI)")
        kwargs['llm_client'] = OpenAI(**client_kwargs)
        kwargs['llm_model'] = LLM_MODEL
        if LLM_PROMPT:
            kwargs['llm_prompt'] = LLM_PROMPT
    elif LLM_PROVIDER == 'azure_openai' and LLM_AVAILABLE and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
        kwargs['llm_client'] = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        kwargs['llm_model'] = LLM_MODEL
        if LLM_PROMPT:
            kwargs['llm_prompt'] = LLM_PROMPT
        features.append("LLM(Azure OpenAI)")

    # Configure Azure Document Intelligence
    if AZURE_DOCINTEL_ENDPOINT and AZURE_DOCINTEL_AVAILABLE:
        kwargs['docintel_endpoint'] = AZURE_DOCINTEL_ENDPOINT
        if AZURE_DOCINTEL_API_KEY:
            kwargs['docintel_credential'] = AzureKeyCredential(AZURE_DOCINTEL_API_KEY)
        if AZURE_DOCINTEL_FILE_TYPES:
            kwargs['docintel_file_types'] = set(AZURE_DOCINTEL_FILE_TYPES.lower().split(','))
        features.append("Azure-DocIntel")

    # Configure ExifTool path
    if EXIFTOOL_PATH:
        kwargs['exiftool_path'] = EXIFTOOL_PATH
        features.append("ExifTool")

    if ENABLE_PLUGINS:
        features.append("Plugins")

    if features:
        logger.debug(f"MarkItDown features enabled: {', '.join(features)}")
    else:
        logger.debug("MarkItDown using default configuration (no LLM/Azure features)")

    return MarkItDown(**kwargs)


def convert_to_md(filepath: str) -> str:
    """Convert a file to Markdown format.

    Args:
        filepath: Path to the file to convert

    Returns:
        Markdown content as string
    """
    start_time = time.time()
    logger.info(f"Starting conversion: {filepath}")

    # Log file info
    file_size = os.path.getsize(filepath)
    logger.info(f"File size: {file_size / 1024:.1f} KB")

    # Create instance
    logger.debug("Creating MarkItDown instance...")
    instance_start = time.time()
    markitdown = create_markitdown_instance()
    logger.debug(f"Instance created in {time.time() - instance_start:.2f}s")

    # Convert
    logger.info("Starting document conversion...")
    convert_start = time.time()
    result = markitdown.convert(filepath)
    convert_time = time.time() - convert_start
    logger.info(f"Conversion completed in {convert_time:.2f}s")

    # Log result info
    content_length = len(result.text_content)
    total_time = time.time() - start_time
    logger.info(f"Result: {content_length} chars, total time: {total_time:.2f}s")
    logger.debug(f"Preview: {result.text_content[:200]}...")

    return result.text_content

@app.get("/", summary="Root endpoint", description="Returns service information")
def read_root():
    """Get service information and available endpoints."""
    return {
        "service": "MarkItDown Server",
        "description": "API for converting documents to Markdown",
        "version": "1.1.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "process": "/process_file",
            "process_async": "/process_file/async",
            "job_status": "/jobs/{job_id}"
        }
    }

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
    description="Returns the health status of the service with concurrency information"
)
def health_check():
    """Get service health status."""
    # Determine LLM status
    llm_enabled = False
    llm_provider_name = None
    if LLM_PROVIDER == 'openai' and LLM_AVAILABLE and OPENAI_API_KEY:
        llm_enabled = True
        llm_provider_name = 'openai' if not OPENAI_BASE_URL else 'openai-compatible'
    elif LLM_PROVIDER == 'azure_openai' and LLM_AVAILABLE and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
        llm_enabled = True
        llm_provider_name = 'azure_openai'

    # Determine Azure Document Intelligence status
    azure_docintel_enabled = bool(AZURE_DOCINTEL_ENDPOINT and AZURE_DOCINTEL_AVAILABLE)

    # Get async job stats
    job_stats = job_store.get_stats()

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "MarkItDown Server",
        "version": "1.1.0",
        "workers": WORKERS,
        "auth_enabled": ENABLE_AUTH,
        "rate_limit_enabled": ENABLE_RATE_LIMIT and RATE_LIMIT_AVAILABLE,
        "rate_limit": RATE_LIMIT if (ENABLE_RATE_LIMIT and RATE_LIMIT_AVAILABLE) else None,
        "llm_enabled": llm_enabled,
        "llm_provider": llm_provider_name,
        "azure_docintel_enabled": azure_docintel_enabled,
        "plugins_enabled": ENABLE_PLUGINS,
        "async_jobs_pending": job_stats["pending"],
        "async_jobs_processing": job_stats["processing"],
        "async_jobs_total": job_stats["total"]
    }

@app.post(
    '/process_file',
    response_model=MarkdownResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type or file too large"},
        401: {"model": ErrorResponse, "description": "Unauthorized - Invalid or missing API key"},
        413: {"model": ErrorResponse, "description": "File too large"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Convert document to Markdown",
    description="Upload a document file and receive its content in Markdown format. Requires X-API-Key header when authentication is enabled."
)
async def process_file(
    request: Request,
    file: UploadFile = File(..., description="Document file to convert to Markdown"),
    api_key: str | None = Depends(verify_api_key)
) -> MarkdownResponse:
    """Process an uploaded file and convert it to Markdown."""
    request_start = time.time()
    temp_file_path = None

    # Validate filename
    if not file.filename:
        return JSONResponse(
            content={'error': 'Filename is required'},
            status_code=400
        )

    logger.info(f"=== Processing request: {file.filename} ===")

    if not allowed_file(file.filename):
        logger.warning(f"Rejected file type: {file.filename}")
        return JSONResponse(
            content={'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'},
            status_code=400
        )

    try:
        # Read file content
        logger.debug("Reading uploaded file...")
        read_start = time.time()
        file_content = await file.read()
        logger.debug(f"File read in {time.time() - read_start:.2f}s")

        file_size_kb = len(file_content) / 1024
        logger.info(f"Received: {file.filename} ({file_size_kb:.1f} KB)")

        # Validate file size
        if len(file_content) > MAX_FILE_SIZE:
            logger.warning(f"File too large: {file_size_kb:.1f} KB > {MAX_FILE_SIZE / 1024:.1f} KB")
            return JSONResponse(
                content={'error': f'File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f}MB'},
                status_code=413
            )

        # Validate file is not empty
        if len(file_content) == 0:
            logger.warning("Rejected empty file")
            return JSONResponse(
                content={'error': 'File is empty'},
                status_code=400
            )

        # Save the file to a temporary directory with sanitized extension
        safe_suffix = sanitize_extension(file.filename)
        logger.debug(f"Writing to temp file with suffix: {safe_suffix}")
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=safe_suffix
        ) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
            logger.debug(f"Temp file created: {temp_file_path}")

        # Convert the file to markdown
        markdown_content = convert_to_md(temp_file_path)

        total_time = time.time() - request_start
        logger.info(f"=== Request completed in {total_time:.2f}s ===")

        return JSONResponse(content={'markdown': markdown_content})

    except Exception as e:
        total_time = time.time() - request_start
        logger.error(f"Error after {total_time:.2f}s: {str(e)}")
        return JSONResponse(content={'error': str(e)}, status_code=500)

    finally:
        # Ensure the temporary file is deleted
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.debug(f"Temp file deleted: {temp_file_path}")


async def process_file_background(job: Job) -> None:
    """Background task to process a file and update job status."""
    try:
        await job_store.update_status(job.id, "processing")
        logger.info(f"Job {job.id}: Starting background conversion for {job.filename}")

        # Run blocking conversion in thread pool
        loop = asyncio.get_event_loop()
        markdown_content = await loop.run_in_executor(
            thread_pool,
            convert_to_md,
            job.temp_file_path
        )

        await job_store.update_status(job.id, "completed", result=markdown_content)
        logger.info(f"Job {job.id}: Conversion completed successfully")

    except Exception as e:
        error_msg = str(e)
        await job_store.update_status(job.id, "failed", error=error_msg)
        logger.error(f"Job {job.id}: Conversion failed - {error_msg}")


@app.post(
    '/process_file/async',
    response_model=AsyncJobSubmitResponse,
    status_code=202,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type or empty file"},
        401: {"model": ErrorResponse, "description": "Unauthorized - Invalid or missing API key"},
        413: {"model": ErrorResponse, "description": "File too large"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
    summary="Submit document for async conversion",
    description="Upload a document file for asynchronous processing. Returns a job ID immediately. Poll /jobs/{job_id} to check status and retrieve results."
)
async def process_file_async(
    request: Request,
    file: UploadFile = File(..., description="Document file to convert to Markdown"),
    api_key: str | None = Depends(verify_api_key)
) -> AsyncJobSubmitResponse:
    """Submit a file for asynchronous processing."""
    # Validate filename
    if not file.filename:
        return JSONResponse(
            content={'error': 'Filename is required'},
            status_code=400
        )

    logger.info(f"=== Async processing request: {file.filename} ===")

    if not allowed_file(file.filename):
        logger.warning(f"Rejected file type: {file.filename}")
        return JSONResponse(
            content={'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'},
            status_code=400
        )

    try:
        # Read file content
        file_content = await file.read()
        file_size_kb = len(file_content) / 1024
        logger.info(f"Received: {file.filename} ({file_size_kb:.1f} KB)")

        # Validate file size
        if len(file_content) > MAX_FILE_SIZE:
            logger.warning(f"File too large: {file_size_kb:.1f} KB > {MAX_FILE_SIZE / 1024:.1f} KB")
            return JSONResponse(
                content={'error': f'File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f}MB'},
                status_code=413
            )

        # Validate file is not empty
        if len(file_content) == 0:
            logger.warning("Rejected empty file")
            return JSONResponse(
                content={'error': 'File is empty'},
                status_code=400
            )

        # Save the file to a temporary directory with sanitized extension
        safe_suffix = sanitize_extension(file.filename)
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=safe_suffix
        ) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        # Create job and start background processing
        job = await job_store.create(file.filename, temp_file_path)

        # Schedule background task
        asyncio.create_task(process_file_background(job))

        return AsyncJobSubmitResponse(
            job_id=job.id,
            status="pending",
            message="File queued for processing",
            status_url=f"/jobs/{job.id}"
        )

    except Exception as e:
        logger.error(f"Error submitting async job: {str(e)}")
        return JSONResponse(content={'error': str(e)}, status_code=500)


@app.get(
    '/jobs/{job_id}',
    response_model=JobStatusResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Job not found or expired"},
    },
    summary="Get job status",
    description="Check the status of an async processing job. Returns the result when completed."
)
async def get_job_status(
    job_id: str,
    api_key: str | None = Depends(verify_api_key)
) -> JobStatusResponse:
    """Get the status and result of an async job."""
    job = await job_store.get(job_id)

    if not job:
        return JSONResponse(
            content={'error': 'Job not found or expired'},
            status_code=404
        )

    return job.to_response()


if __name__ == "__main__":
    import uvicorn

    # Log startup configuration
    logger.info("=" * 50)
    logger.info("MarkItDown Server Starting")
    logger.info("=" * 50)
    logger.info(f"Workers: {WORKERS}")
    logger.info(f"Log level: {LOG_LEVEL}")
    logger.info(f"Authentication: {'enabled' if ENABLE_AUTH else 'disabled'}")
    logger.info(f"LLM Provider: {LLM_PROVIDER or 'disabled'}")
    if LLM_PROVIDER:
        logger.info(f"LLM Model: {LLM_MODEL}")
        if OPENAI_BASE_URL:
            logger.info(f"OpenAI Base URL: {OPENAI_BASE_URL}")
    logger.info(f"Azure DocIntel: {'enabled' if AZURE_DOCINTEL_ENDPOINT else 'disabled'}")
    logger.info(f"Plugins: {'enabled' if ENABLE_PLUGINS else 'disabled'}")
    if ENABLE_RATE_LIMIT and RATE_LIMIT_AVAILABLE:
        logger.info(f"Rate limiting: {RATE_LIMIT}")
    logger.info("=" * 50)
    
    # Run with configured number of workers
    # When workers > 1, we need to pass the app as a string
    if WORKERS > 1:
        uvicorn.run(
            "app:app",
            host="0.0.0.0", 
            port=int(os.getenv('PORT', '8490')),
            workers=WORKERS
        )
    else:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=int(os.getenv('PORT', '8490'))
        )     