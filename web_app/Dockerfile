FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and ffmpeg (H.264 browser encoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY web_app/requirements-web.txt /app/requirements-web.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /app/requirements-web.txt

COPY . /app

# Hugging Face Spaces listens on port 7860
EXPOSE 7860

CMD ["streamlit", "run", "web_app/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.maxUploadSize=25"]
