FROM python:3.10-slim

WORKDIR /app

COPY requirements-prod.txt .
RUN pip install --upgrade pip && pip install -r requirements-prod.txt

COPY api/ ./api/
COPY inference/predict_tags_occx.py ./inference/predict_tags_occx.py
COPY inference/audio_processor.py ./inference/audio_processor.py
COPY weights/model.onnx ./weights/model.onnx

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
