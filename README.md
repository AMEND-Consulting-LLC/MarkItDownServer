# MarkItDown Web Server

A production-ready web server application built using FastAPI that receives binary data from various document formats and converts them to Markdown using the MarkItDown library.

> **💡 Quick Answer: Is there a concurrency limit?**  
> By default, the server runs with 1 worker and no rate limiting. You can configure workers and rate limits using environment variables.  
> See [docs/CONCURRENCY_SUMMARY.md](./docs/CONCURRENCY_SUMMARY.md) for a quick guide or [docs/CONCURRENCY.md](./docs/CONCURRENCY.md) for detailed information.

## 🚀 Features

- **Multiple Format Support**: Convert 26+ file types including documents, images, and audio
  - **Documents**: DOC, DOCX, PPT, PPTX, PDF, XLS, XLSX, ODT, ODS, ODP, TXT
  - **Images**: JPG, JPEG, PNG, GIF, BMP, TIFF, WEBP, SVG
  - **Audio**: MP3, WAV, FLAC, AAC, OGG, M4A, WMA
- **LLM Integration**: Enhanced image captioning and document processing
  - OpenAI API support
  - Azure OpenAI support
  - LiteLLM proxy support (for self-hosted or alternative LLM providers)
- **Azure Document Intelligence**: Advanced PDF and document analysis
- **FastAPI Framework**: Modern, fast, and well-documented REST API
- **Health Checks**: Built-in health monitoring with feature status
- **Input Validation**: Comprehensive file size, type, and content validation
- **Error Handling**: Robust error handling with detailed error messages
- **CORS Support**: Configurable CORS for web client integration
- **Security Headers**: Built-in security headers middleware
- **API Key Authentication**: Optional API key authentication for protected endpoints
- **Docker Support**: Docker and Docker Compose deployment ready
- **Azure Compatible**: Ready for Azure Container Apps deployment
- **Comprehensive Testing**: 154+ pytest tests covering all functionality
- **AI Chat Web App Sample**: Full-featured .NET Aspire application with document upload and RAG chat capabilities (see [samples/AiChatWebApp](./samples/AiChatWebApp/README.md))

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Samples](#samples)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Documentation](#documentation)
- [License](#license)

## ⚡ Quick Start

### Using Docker Compose (Recommended)

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your configuration (optional - defaults work out of the box)
# nano .env

# Build and start the server
docker compose up -d

# Test the health endpoint
curl http://localhost:8490/health
```

### Using Docker

```bash
# Build the Docker image
docker build -t markitdownserver .

# Run the container
docker run -d --name markitdownserver -p 8490:8490 markitdownserver

# Test the health endpoint
curl http://localhost:8490/health
```

### Using Python (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

The server will be available at `http://localhost:8490`

## 📦 Installation

### Prerequisites

- **Python 3.12+** (for local development)
- **Docker** (for containerized deployment)
- **.NET 9.0 SDK** (for running C# client examples)

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/elbruno/MarkItDownServer.git
   cd MarkItDownServer
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server**:
   ```bash
   python app.py
   ```

   The server will start on `http://0.0.0.0:8490`

### Docker Deployment

1. **Build the Docker image**:
   ```bash
   docker build -t markitdownserver:latest .
   ```

2. **Run the container**:
   ```bash
   docker run -d \
     --name markitdownserver \
     -p 8490:8490 \
     markitdownserver:latest
   ```

3. **Verify the container is running**:
   ```bash
   docker ps | grep markitdownserver
   ```

4. **View logs**:
   ```bash
   docker logs markitdownserver
   ```

5. **Stop the container**:
   ```bash
   docker stop markitdownserver
   docker rm markitdownserver
   ```

## 📖 Usage

### API Endpoints

#### Root Endpoint
```http
GET /
```

Returns service information and available endpoints.

**Response**:
```json
{
  "service": "MarkItDown Server",
  "description": "API for converting documents to Markdown",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "docs": "/docs",
    "process": "/process_file"
  }
}
```

#### Health Check
```http
GET /health
```

Returns the health status of the service including feature configuration.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-07T12:00:00",
  "service": "MarkItDown Server",
  "version": "1.0.0",
  "workers": 1,
  "auth_enabled": false,
  "rate_limit_enabled": false,
  "rate_limit": "60/minute",
  "llm_enabled": false,
  "llm_provider": null,
  "azure_docintel_enabled": false,
  "plugins_enabled": false
}
```

#### Process File
```http
POST /process_file
```

Upload a document file and receive its content in Markdown format.

**Parameters**:
- `file`: The document file to convert (multipart/form-data)

**Supported File Types** (26 formats):
- **Documents**: DOC, DOCX, XLS, XLSX, PPT, PPTX, PDF, ODT, ODS, ODP, TXT
- **Images**: JPG, JPEG, PNG, GIF, BMP, TIFF, WEBP, SVG
- **Audio**: MP3, WAV, FLAC, AAC, OGG, M4A, WMA

**File Size Limit**: 50MB (configurable via `MAX_FILE_SIZE`)

**Response**:
```json
{
  "markdown": "# Document Title\n\nContent in markdown format..."
}
```

**Headers** (when authentication is enabled):
- `X-API-Key`: Your API key

**Error Responses**:
- `400 Bad Request`: Invalid file type or empty file
- `401 Unauthorized`: Missing or invalid API key (when auth enabled)
- `413 Payload Too Large`: File exceeds 50MB
- `500 Internal Server Error`: Conversion error

### Client Examples

#### Simple Console Application

Located in `samples/SimpleConsole/`, this is a basic example showing minimal code to use the API.

```bash
cd samples/SimpleConsole
dotnet run
```

**Code**:
```csharp
using System.Net.Http.Headers;

HttpClient client = new HttpClient();
string url = "http://127.0.0.1:8490/process_file";
string filePath = "Benefit_Options.pdf";

using (var content = new MultipartFormDataContent())
{
    byte[] fileBytes = File.ReadAllBytes(filePath);
    var fileContent = new ByteArrayContent(fileBytes);
    fileContent.Headers.ContentType = MediaTypeHeaderValue.Parse("application/pdf");
    content.Add(fileContent, "file", Path.GetFileName(filePath));

    var response = await client.PostAsync(url, content);
    if (response.IsSuccessStatusCode)
    {
        string responseBody = await response.Content.ReadAsStringAsync();
        Console.WriteLine($"MarkDown for {filePath}\n\n{responseBody}");
    }
}
```

#### Detailed Console Application

Located in `samples/DetailedConsole/`, this includes comprehensive error handling, configuration, and features.

```bash
cd samples/DetailedConsole
dotnet run
```

**Features**:
- Configuration file support (`appsettings.json`)
- Comprehensive error handling
- Timeout configuration
- File validation
- Colored console output
- Automatic markdown file saving
- Content type detection

**Configuration** (`appsettings.json`):
```json
{
  "MarkItDownServer": {
    "Url": "http://127.0.0.1:8490/process_file",
    "FilePath": "Benefit_Options.pdf",
    "TimeoutMinutes": "5"
  }
}
```

#### AI Chat Web App (Full-Featured Sample)

Located in `samples/AiChatWebApp/`, this is a complete .NET Aspire application with:
- Blazor Server UI with modern chat interface
- Document upload with drag-and-drop support
- Integration with GitHub Models for AI chat
- Retrieval-Augmented Generation (RAG) with vector search
- Real-time document processing and ingestion

```bash
cd samples/AiChatWebApp
# See QUICKSTART.md for detailed setup instructions
dotnet run --project AiChatWebApp.AppHost
```

**Features**:
- Upload documents (PDF, Word, PowerPoint, Excel, Text) through the web UI
- Documents are automatically converted to Markdown via MarkItDown
- Chat with your documents using AI
- Semantic search with citations
- .NET Aspire orchestration with health monitoring

For complete documentation, see [samples/AiChatWebApp/README.md](./samples/AiChatWebApp/README.md) or [QUICKSTART.md](./samples/AiChatWebApp/QUICKSTART.md).

#### cURL Example

```bash
# Without authentication (when ENABLE_AUTH=false)
curl -X POST "http://localhost:8490/process_file" \
  -F "file=@document.pdf"

# With authentication (when ENABLE_AUTH=true)
curl -X POST "http://localhost:8490/process_file" \
  -H "X-API-Key: your-api-key" \
  -F "file=@document.pdf"
```

#### Python Example

```python
import requests

url = "http://localhost:8490/process_file"
files = {"file": open("document.pdf", "rb")}

response = requests.post(url, files=files)
if response.status_code == 200:
    markdown = response.json()["markdown"]
    print(markdown)
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

#### PowerShell Example

```powershell
$url = "http://localhost:8490/process_file"
$filePath = "document.pdf"

$fileContent = [System.IO.File]::ReadAllBytes($filePath)
$boundary = [System.Guid]::NewGuid().ToString()
$LF = "`r`n"

$bodyLines = (
    "--$boundary",
    "Content-Disposition: form-data; name=`"file`"; filename=`"$(Split-Path $filePath -Leaf)`"",
    "Content-Type: application/pdf$LF",
    [System.Text.Encoding]::UTF8.GetString($fileContent),
    "--$boundary--$LF"
) -join $LF

Invoke-RestMethod -Uri $url -Method Post -ContentType "multipart/form-data; boundary=$boundary" -Body $bodyLines
```

## ⚙️ Configuration

### Quick Setup with .env File

The easiest way to configure the server is using the `.env` file:

```bash
# Copy the example configuration
cp .env.example .env

# Edit with your settings
nano .env

# Start with Docker Compose
docker compose up -d
```

### Environment Variables

#### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8490` | Server port |
| `WORKERS` | `1` | Number of uvicorn workers |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `MAX_FILE_SIZE` | `52428800` | Maximum file size in bytes (50MB) |
| `ENABLE_RATE_LIMIT` | `false` | Enable rate limiting |
| `RATE_LIMIT` | `60/minute` | Rate limit per IP |

#### Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_AUTH` | `false` | Enable API key authentication |
| `API_KEYS` | *(empty)* | Comma-separated list of valid API keys |

#### LLM Configuration (for image captioning)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | *(empty)* | `openai`, `azure_openai`, or empty to disable |
| `LLM_MODEL` | `gpt-4o` | Model name or deployment name |
| `LLM_PROMPT` | *(empty)* | Custom prompt for image descriptions |
| `OPENAI_API_KEY` | *(empty)* | OpenAI API key (when using `openai` provider) |
| `OPENAI_BASE_URL` | *(empty)* | Custom base URL for OpenAI-compatible APIs (e.g., LiteLLM) |
| `AZURE_OPENAI_ENDPOINT` | *(empty)* | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | *(empty)* | Azure OpenAI API key |
| `AZURE_OPENAI_API_VERSION` | `2024-02-15-preview` | Azure OpenAI API version |

#### Azure Document Intelligence

| Variable | Default | Description |
|----------|---------|-------------|
| `AZURE_DOCINTEL_ENDPOINT` | *(empty)* | Azure Document Intelligence endpoint |
| `AZURE_DOCINTEL_API_KEY` | *(empty)* | Azure Document Intelligence API key |
| `AZURE_DOCINTEL_FILE_TYPES` | *(empty)* | Comma-separated file types (e.g., `pdf,docx,xlsx`) |

#### Other Features

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_PLUGINS` | `false` | Enable third-party MarkItDown plugins |
| `EXIFTOOL_PATH` | *(empty)* | Custom path to ExifTool binary |

### Docker Compose (Recommended)

Use Docker Compose with the provided `docker-compose.yml`:

```bash
# Basic startup
docker compose up -d

# With LLM enabled (edit .env first)
LLM_PROVIDER=openai OPENAI_API_KEY=sk-... docker compose up -d

# View logs
docker compose logs -f
```

### Docker Environment

```bash
docker run -d \
  --name markitdownserver \
  -p 8490:8490 \
  -e PORT=8490 \
  -e MAX_FILE_SIZE=104857600 \
  -e WORKERS=4 \
  -e ENABLE_RATE_LIMIT=true \
  -e RATE_LIMIT=100/minute \
  markitdownserver:latest
```

### Enabling LLM Features

#### With OpenAI

```bash
docker run -d \
  --name markitdownserver \
  -p 8490:8490 \
  -e LLM_PROVIDER=openai \
  -e OPENAI_API_KEY=sk-your-api-key \
  -e LLM_MODEL=gpt-4o \
  markitdownserver:latest
```

#### With LiteLLM Proxy

```bash
docker run -d \
  --name markitdownserver \
  -p 8490:8490 \
  -e LLM_PROVIDER=openai \
  -e OPENAI_API_KEY=your-litellm-key \
  -e OPENAI_BASE_URL=http://localhost:4000/v1 \
  -e LLM_MODEL=gpt-4o \
  markitdownserver:latest
```

#### With Azure OpenAI

```bash
docker run -d \
  --name markitdownserver \
  -p 8490:8490 \
  -e LLM_PROVIDER=azure_openai \
  -e AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com \
  -e AZURE_OPENAI_API_KEY=your-azure-key \
  -e LLM_MODEL=your-deployment-name \
  markitdownserver:latest
```

### Enabling Azure Document Intelligence

```bash
docker run -d \
  --name markitdownserver \
  -p 8490:8490 \
  -e AZURE_DOCINTEL_ENDPOINT=https://your-resource.cognitiveservices.azure.com \
  -e AZURE_DOCINTEL_API_KEY=your-docintel-key \
  -e AZURE_DOCINTEL_FILE_TYPES=pdf,docx \
  markitdownserver:latest
```

### Enabling API Key Authentication

Protect the `/process_file` endpoint with API key authentication:

```bash
# Generate secure API keys
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Run with authentication enabled
docker run -d \
  --name markitdownserver \
  -p 8490:8490 \
  -e ENABLE_AUTH=true \
  -e API_KEYS="key1,key2,key3" \
  markitdownserver:latest
```

**Using with Docker Compose** (recommended):

```bash
# Add to your .env file
ENABLE_AUTH=true
API_KEYS=your-secure-key-1,your-secure-key-2

# Start the server
docker compose up -d
```

**Making authenticated requests**:

```bash
curl -X POST "http://localhost:8490/process_file" \
  -H "X-API-Key: your-secure-key-1" \
  -F "file=@document.pdf"
```

**Notes**:
- The `/health` and `/` endpoints remain public (no authentication required)
- Multiple API keys can be configured (comma-separated)
- Uses timing-safe comparison to prevent timing attacks

## 🚦 Concurrency and Performance

### Default Behavior

By default, the server runs with:
- **1 worker process** (single worker)
- **Async request handling** via FastAPI
- **No rate limiting**

### Configuring Concurrency

**Multi-worker setup** for better performance:
```bash
# Run with 4 workers
docker run -d -p 8490:8490 -e WORKERS=4 markitdownserver:latest
```

**Enable rate limiting** to prevent abuse:
```bash
# Limit to 100 requests per minute per IP
docker run -d -p 8490:8490 \
  -e ENABLE_RATE_LIMIT=true \
  -e RATE_LIMIT=100/minute \
  markitdownserver:latest
```

**Note**: Rate limiting requires `slowapi` package. Install with:
```bash
pip install slowapi
```

### Performance Recommendations

- **Small scale** (< 100 req/min): 1-2 workers
- **Medium scale** (100-1000 req/min): 2-4 workers  
- **Large scale** (> 1000 req/min): Use horizontal scaling with load balancer

**📚 For detailed concurrency information**, see [docs/CONCURRENCY.md](./docs/CONCURRENCY.md)

## 🧪 Testing

### Running the Test Suite

The project includes a comprehensive pytest test suite with 154+ tests:

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=app --cov-report=term-missing

# Run specific test file
pytest tests/test_endpoints.py
pytest tests/test_helpers.py
pytest tests/test_configuration.py
```

**Running tests in Docker**:
```bash
docker compose run --rm markitdown-server pytest
```

### Test Categories

- **`tests/test_helpers.py`**: Unit tests for helper functions (sanitize_extension, allowed_file, etc.)
- **`tests/test_endpoints.py`**: Integration tests for API endpoints (/, /health, /process_file)
- **`tests/test_configuration.py`**: Environment variable and configuration tests

### Manual Testing

1. **Start the server**:
   ```bash
   python app.py
   ```

2. **Run health check**:
   ```bash
   curl http://localhost:8490/health
   ```

3. **Test file conversion**:
   ```bash
   curl -X POST "http://localhost:8490/process_file" \
     -F "file=@samples/SimpleConsole/Benefit_Options.pdf"
   ```

### Run Client Examples

**Simple Console**:
```bash
cd samples/SimpleConsole
dotnet run
```

**Detailed Console**:
```bash
cd samples/DetailedConsole
dotnet run
```

## 🚀 Deployment

### Local Deployment

For development and testing:

```bash
# Using Python
python app.py

# Using uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8490 --reload
```

### Docker Deployment

For production:

```bash
# Build
docker build -t markitdownserver:1.0.0 .

# Run
docker run -d \
  --name markitdownserver \
  -p 8490:8490 \
  --restart unless-stopped \
  markitdownserver:1.0.0

# View logs
docker logs -f markitdownserver
```

### Azure Container Apps

See the comprehensive [docs/CODE_QUALITY_IMPROVEMENTS.md](./docs/CODE_QUALITY_IMPROVEMENTS.md) document for detailed Azure deployment instructions, including:

- Multi-stage Dockerfile optimization
- Azure CLI deployment scripts
- Bicep templates for Infrastructure as Code
- Environment configuration
- Security best practices
- Cost estimation and optimization

**Quick Azure Deployment**:

```bash
# Set variables
RESOURCE_GROUP="rg-markitdown"
LOCATION="eastus"
CONTAINER_APP_NAME="markitdown-server"

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Deploy (see docs/CODE_QUALITY_IMPROVEMENTS.md for complete script)
az containerapp up \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --source .
```

## 👨‍💻 Development

### Project Structure

```
MarkItDownServer/
├── app.py                          # Main FastAPI application
├── requirements.txt                # Python dependencies
├── requirements-test.txt           # Test dependencies
├── pytest.ini                      # Pytest configuration
├── dockerfile                      # Docker configuration
├── docker-compose.yml              # Docker Compose configuration
├── .env.example                    # Example environment variables
├── README.md                       # This file
├── docs/                           # Comprehensive documentation
├── tests/                          # Test suite
│   ├── conftest.py                 # Shared test fixtures
│   ├── test_endpoints.py           # API endpoint tests
│   ├── test_helpers.py             # Helper function tests
│   └── test_configuration.py       # Configuration tests
├── samples/
│   ├── SimpleConsole/              # Basic C# client example
│   │   ├── Program.cs
│   │   ├── SimpleConsole.csproj
│   │   └── Benefit_Options.pdf
│   ├── DetailedConsole/            # Advanced C# client example
│   │   ├── Program.cs
│   │   ├── DetailedConsole.csproj
│   │   ├── appsettings.json
│   │   └── Benefit_Options.pdf
│   └── AiChatWebApp/               # Full .NET Aspire RAG application
├── src/                            # Legacy client (preserved)
│   └── ...
└── utils/
    └── file_handler.py             # Utility functions
```

### Adding New Features

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Quality

The project follows Python best practices:
- Type hints for better code clarity
- Comprehensive error handling
- Input validation
- Security headers
- Structured logging

## 📚 API Documentation

The server provides automatic interactive API documentation:

- **Swagger UI**: http://localhost:8490/docs
- **ReDoc**: http://localhost:8490/redoc

These interfaces allow you to:
- Explore all available endpoints
- Test API calls directly from the browser
- View request/response schemas
- See example requests and responses

## 📦 Dependencies

### Python Dependencies

- **fastapi** (0.128.0): Modern web framework for building APIs
- **uvicorn[standard]** (0.40.0): ASGI server for FastAPI
- **python-multipart** (0.0.22): Multipart form data support
- **markitdown[all]** (≥0.1.4): Document to Markdown conversion with all optional features
- **pydantic** (2.12.5): Data validation using Python type hints
- **openai** (≥1.0.0): OpenAI and Azure OpenAI client
- **azure-identity** (≥1.15.0): Azure credential handling

### Optional Dependencies

- **slowapi**: Rate limiting (install with `pip install slowapi`)

### Test Dependencies

Install with `pip install -r requirements-test.txt`:
- pytest, pytest-asyncio, pytest-cov, httpx, pytest-mock

### System Requirements

- Python 3.9 or higher (3.12 recommended)
- 512MB RAM minimum (1GB recommended)
- 100MB disk space

## 🔍 Troubleshooting

### Server won't start

**Issue**: Port already in use
```
Error: [Errno 48] Address already in use
```

**Solution**: Change the port or stop the conflicting service
```bash
# Find process using port 8490
lsof -i :8490

# Kill the process
kill -9 <PID>

# Or use a different port
python app.py --port 8491
```

### File conversion fails

**Issue**: "File type not allowed"

**Solution**: Ensure your file has a supported extension (doc, docx, pdf, etc.)

**Issue**: "File too large"

**Solution**: Files must be under 50MB. Compress or split large files.

### Docker issues

**Issue**: Cannot connect to Docker daemon

**Solution**: Ensure Docker Desktop is running
```bash
docker ps  # Test Docker connection
```

**Issue**: Container exits immediately

**Solution**: Check container logs
```bash
docker logs markitdownserver
```

## 📞 Support

For issues, questions, or contributions:

- **GitHub Issues**: [Create an issue](https://github.com/elbruno/MarkItDownServer/issues)
- **Documentation**: See [docs/](./docs/)

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Powered by [MarkItDown](https://github.com/microsoft/markitdown)
- Developed by [El Bruno](https://github.com/elbruno)

## 📚 Documentation

Comprehensive documentation is available in the [docs](./docs/) directory:

- **[Quick Reference](./docs/QUICK_REFERENCE.md)** - Common commands and API usage
- **[Developer Manual](./docs/DEVELOPER_MANUAL.md)** - Integration guide for developers
- **[Concurrency Guide](./docs/CONCURRENCY.md)** - Performance and scaling information
- **[Code Quality](./docs/CODE_QUALITY_IMPROVEMENTS.md)** - Best practices and improvements
- **[Implementation Plans](./docs/plans/)** - Detailed feature implementation plans

### Sample Applications

- **[AI Chat Web App](./samples/AiChatWebApp/)** - Full .NET Aspire application with:
  - Document upload and conversion
  - RAG-based chat with semantic search
  - Vector store integration
  - Real-time markdown preview
  - [Quick Start Guide](./samples/AiChatWebApp/QUICKSTART.md)
  - [User Manual](./samples/AiChatWebApp/docs/USER_MANUAL.md) *(coming soon)*

## 📈 Version History

- **v1.2.0** (2025-02): Security and authentication release
  - API key authentication for protected endpoints
  - Timing-safe key comparison to prevent timing attacks
  - Multiple API keys support
  - Health endpoint shows authentication status

- **v1.1.0** (2025-01): Enhanced capabilities release
  - LLM integration (OpenAI, Azure OpenAI, LiteLLM proxy support)
  - Azure Document Intelligence integration
  - Image format support (JPG, PNG, GIF, BMP, TIFF, WEBP, SVG)
  - Audio format support (MP3, WAV, FLAC, AAC, OGG, M4A, WMA)
  - Docker Compose and .env configuration
  - Comprehensive test suite (154+ tests)
  - Enhanced logging with timing metrics
  - Plugin support

- **v1.0.0** (2025-01): Initial release with production-ready features
  - Multi-format document conversion
  - Comprehensive error handling
  - Health check endpoints
  - Docker support
  - Azure deployment ready
  - AI Chat Web App sample with .NET Aspire

---

**Ready to convert your documents to Markdown?** 🚀

Start the server and visit http://localhost:8490/docs to explore the interactive API documentation!