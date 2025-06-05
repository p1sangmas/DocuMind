FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies (combine RUN commands to reduce layers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    tesseract-ocr \
    poppler-utils \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Configure nginx
COPY nginx.conf /etc/nginx/nginx.conf
RUN mkdir -p /var/log/nginx && \
    mkdir -p /var/lib/nginx && \
    mkdir -p /run/nginx

# Create volume and cache directories with proper permissions
RUN mkdir -p /app/data/documents /app/data/vectorstore

# Create a non-root user to run the application
RUN groupadd -r docuuser && useradd --no-log-init -r -g docuuser docuuser \
    && mkdir -p /home/docuuser/.cache/huggingface \
    && chown -R docuuser:docuuser /app /home/docuuser \
    && chmod -R 755 /home/docuuser/.cache \
    && mkdir -p /var/log/nginx /var/lib/nginx /run/nginx \
    && touch /var/log/nginx/error.log /var/log/nginx/access.log \
    && chmod 777 /var/run \
    && chown -R docuuser:docuuser /var/log/nginx /var/lib/nginx /run/nginx \
    && chmod -R 755 /var/log/nginx /var/lib/nginx /run/nginx

# Set the cache directory environment variables
ENV HF_HOME=/home/docuuser/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/docuuser/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/docuuser/.cache/huggingface \
    CHROMADB_CACHE_DIR=/home/docuuser/.cache/chroma

# Copy the application code
COPY . .

# Make entrypoint script executable
RUN chmod +x /app/docker-entrypoint.sh

# Switch to non-root user
USER docuuser

# Expose the ports for both web UI and API
EXPOSE 80 8080

# Set the entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]
