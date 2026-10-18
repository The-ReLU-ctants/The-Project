# Deploys the species classifier's inference server + upload UI as a
# Hugging Face Space (Docker SDK). Hugging Face Spaces expects the app to
# listen on port 7860.

FROM python:3.11-slim

WORKDIR /app

COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

COPY src/ src/
COPY models/ models/
COPY data/processed/label_map.json data/processed/label_map.json

EXPOSE 7860

CMD ["python", "src/inference_server.py", "--host", "0.0.0.0", "--port", "7860"]
