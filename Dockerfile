FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FACE_MODEL=buffalo_sc
ENV LITE_MODE=true

RUN python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider']); app.prepare(ctx_id=0, det_size=(320,320)); print('Model ready')"

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
