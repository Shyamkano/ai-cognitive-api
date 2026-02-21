FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed by deepface/opencv/soundfile
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD gunicorn app:app --bind 0.0.0.0:7860 --timeout 600