FROM node:20-bookworm AS frontend

WORKDIR /app
COPY package*.json index.html ./
COPY src ./src
RUN npm ci && npm run build

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY app.py .
COPY cnn_crack_model_final.h5 .
COPY --from=frontend /app/dist ./frontend/dist

EXPOSE 7860

CMD ["sh", "-c", "uvicorn backend.main:api --host 0.0.0.0 --port ${PORT}"]
