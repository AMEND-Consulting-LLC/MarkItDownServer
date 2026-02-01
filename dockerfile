# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt

# Environment variables for configuration
ENV WORKERS=1
ENV PORT=8490
ENV LOG_LEVEL=INFO
ENV ENABLE_RATE_LIMIT=false
ENV RATE_LIMIT=60/minute

# LLM Configuration (for image captioning and enhanced document processing)
# Set LLM_PROVIDER to 'openai' or 'azure_openai' to enable
ENV LLM_PROVIDER=""
ENV LLM_MODEL="gpt-4o"
# For OpenAI or LiteLLM proxy: set OPENAI_API_KEY and optionally OPENAI_BASE_URL at runtime
# For Azure OpenAI: set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY at runtime
ENV AZURE_OPENAI_API_VERSION="2024-02-15-preview"

# Azure Document Intelligence (for enhanced PDF/document processing)
# Set AZURE_DOCINTEL_ENDPOINT and AZURE_DOCINTEL_API_KEY at runtime to enable
ENV AZURE_DOCINTEL_ENDPOINT=""

# Other MarkItDown features
ENV ENABLE_PLUGINS=false

# Make port 8490 available to the world outside this container
EXPOSE 8490

# Run app.py when the container launches
CMD ["python", "./app.py"]