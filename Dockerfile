# CPU image — the frozen-backbone pipeline doesn't need a GPU.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src:/app/scripts

WORKDIR /app

# System libs needed by Pillow / matplotlib.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libjpeg62-turbo libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU PyTorch first (kept out of requirements.txt — hardware specific).
RUN pip install torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
# Model weights are cached on first run; mount ./models and ./data as volumes.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
