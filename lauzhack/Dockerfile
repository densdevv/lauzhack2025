# Swiss Weather Intelligence System - Docker Container
# LauzHack 2025 - Judge Evaluation Ready

FROM python:3.11-slim

LABEL maintainer="LauzHack 2025 Team"
LABEL description="Swiss Weather Intelligence System with ML-Enhanced Predictions"

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash weather
RUN chown -R weather:weather /app
USER weather

# Run the application
CMD ["streamlit", "run", "weather_app.py", "--server.headless", "true", "--server.address", "0.0.0.0", "--server.port", "8501"]