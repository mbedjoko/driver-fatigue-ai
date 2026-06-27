FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libx11-6 \
    libsdl2-dev \
    libsdl2-mixer-dev \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir \
    opencv-python \
    mediapipe \
    numpy \
    pygame

CMD ["python", "app.py"]
