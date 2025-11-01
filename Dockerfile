FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_ENV=production
ENV FLASK_DEBUG=0
ENV TF_ENABLE_ONEDNN_OPTS=0
# TensorFlow memory optimization
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_GPU_MEMORY_LIMIT=256
ENV OMP_NUM_THREADS=1
ENV TF_NUM_INTEROP_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=1

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

CMD ["python", "app.py"]