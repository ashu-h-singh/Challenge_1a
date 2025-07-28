# Use slim Python base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file first for better caching
COPY requirements.txt .

# Install system dependencies for spaCy and PDF tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Download spaCy model
RUN python -m spacy download xx_ent_wiki_sm

# Copy full project
COPY . .

# Set multiprocessing-friendly environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the script
CMD ["python", "process_pdfs.py"]
