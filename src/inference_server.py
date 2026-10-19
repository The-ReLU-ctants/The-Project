"""REST inference server for the species classifier.

Start it (from the project root, venv active):

    python src/inference_server.py                 # http://127.0.0.1:8000
    python src/inference_server.py --port 9000     # different port
    python src/inference_server.py --host 0.0.0.0  # reachable from other devices on your network

Then open http://127.0.0.1:8000 for the upload page, or call the API directly.

Endpoints
---------
GET  /health    -> {"status": "ok", "classes": [...], "device": "...", "checkpoint_epoch": N,
                     "baseline_loaded": true}
POST /predict   multipart/form-data with one file field named "file" (JPEG/PNG/WebP/BMP, max 10 MB)
                -> {
                     "vit": {
                       "species": "polar_bear",
                       "confidence": 0.9731,              # probability of the predicted species, 0-1
                       "probabilities": {"polar_bear": 0.9731, "...": 0.0123, ...}  # sorted, highest first
                     },
                     "baseline": { ...same shape as "vit"... } | null   # null if no baseline model is trained yet
                   }
Errors are JSON: {"detail": "..."} with status 400 (not an image / empty file),
413 (file too large) or 503 (model not loaded).
Interactive API docs: http://127.0.0.1:8000/docs

Example
-------
    curl -F "file=@some_animal.jpg" http://127.0.0.1:8000/predict

Model and label-map paths can be overridden with the SPECIES_CHECKPOINT and
SPECIES_LABEL_MAP environment variables (defaults: models/vit_species.pt and
data/processed/label_map.json). The baseline model path can be overridden
with SPECIES_BASELINE (default: models/baseline_logreg.joblib) — if that
file doesn't exist, "baseline" in the response is just null, nothing breaks.
"""

import argparse
import io
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import get_transforms, load_label_map  # noqa: E402
from src.model import build_model  # noqa: E402

CHECKPOINT_PATH = os.environ.get("SPECIES_CHECKPOINT", str(ROOT / "models" / "vit_species.pt"))
LABEL_MAP_PATH = os.environ.get("SPECIES_LABEL_MAP", str(ROOT / "data" / "processed" / "label_map.json"))
BASELINE_PATH = os.environ.get("SPECIES_BASELINE", str(ROOT / "models" / "baseline_logreg.joblib"))
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

log = logging.getLogger("inference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

state = {}  # filled in at startup: model, transform, idx_to_species, device, epoch, baseline


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_everything():
    device = get_device()
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model = build_model(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    label_map = load_label_map(LABEL_MAP_PATH)
    if label_map != ckpt.get("label_map", label_map):
        log.warning("label_map.json differs from the label map saved in the checkpoint; "
                    "using label_map.json. Predictions will be mislabelled if the class order changed.")
    if len(label_map) != ckpt["config"]["num_classes"]:
        raise RuntimeError(f"label_map has {len(label_map)} classes but the model outputs "
                           f"{ckpt['config']['num_classes']}")

    state.update(
        model=model,
        transform=get_transforms("val", ckpt["config"]["image_size"]),  # same preprocessing as evaluation
        idx_to_species={v: k for k, v in label_map.items()},
        device=device,
        epoch=ckpt.get("epoch"),
    )
    log.info("Model loaded from %s (epoch %s) on %s; classes: %s",
             CHECKPOINT_PATH, ckpt.get("epoch"), device, list(label_map))

    # Baseline is optional — the ViT is the real model, the baseline is just a
    # reference point, so a missing/untrained baseline shouldn't block startup.
    if os.path.exists(BASELINE_PATH):
        baseline = joblib.load(BASELINE_PATH)
        state["baseline"] = baseline
        state["baseline_idx_to_species"] = {v: k for k, v in baseline["label_map"].items()}
        log.info("Baseline model loaded from %s (test accuracy %.4f)",
                 BASELINE_PATH, baseline["test_accuracy"])
    else:
        log.warning("No baseline model found at %s; /predict will return baseline: null. "
                    "Run scripts/baseline.py to train one.", BASELINE_PATH)


def predict_baseline(image: Image.Image):
    """Flatten-pixel Logistic Regression baseline — same preprocessing as scripts/baseline.py."""
    baseline = state["baseline"]
    image_size = baseline["image_size"]
    x = (np.asarray(image.resize((image_size, image_size)), dtype=np.float32) / 255.0).ravel()
    probs = baseline["model"].predict_proba(x.reshape(1, -1))[0]
    idx_to_species = state["baseline_idx_to_species"]
    ranked = sorted(
        ((idx_to_species[i], float(p)) for i, p in enumerate(probs)),
        key=lambda t: t[1],
        reverse=True,
    )
    return {
        "species": ranked[0][0],
        "confidence": round(ranked[0][1], 4),
        "probabilities": {name: round(p, 4) for name, p in ranked},
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # FastAPI's startup hook: runs load_everything() once when the server
    # starts, not on every request, so the model stays in memory between calls.
    load_everything()
    yield


app = FastAPI(title="Wildlife species classifier", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    if "model" not in state:
        raise HTTPException(503, "Model not loaded")
    return {
        "status": "ok",
        "classes": [state["idx_to_species"][i] for i in range(len(state["idx_to_species"]))],
        "device": str(state["device"]),
        "checkpoint_epoch": state["epoch"],
        "baseline_loaded": "baseline" in state,
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if "model" not in state:
        raise HTTPException(503, "Model not loaded")

    # Read one byte past the limit: if that byte exists, the file is too big,
    # without having to read a potentially huge file fully into memory first.
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Could not read the file as an image (use JPEG, PNG, WebP or BMP)")

    x = state["transform"](image).unsqueeze(0).to(state["device"])
    with torch.inference_mode():
        probs = torch.softmax(state["model"](x), dim=1)[0].cpu()

    ranked = sorted(
        ((state["idx_to_species"][i], float(p)) for i, p in enumerate(probs)),
        key=lambda t: t[1],
        reverse=True,
    )
    vit_result = {
        "species": ranked[0][0],
        "confidence": round(ranked[0][1], 4),
        "probabilities": {name: round(p, 4) for name, p in ranked},
    }

    baseline_result = predict_baseline(image) if "baseline" in state else None

    return {"vit": vit_result, "baseline": baseline_result}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Species classifier inference server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
