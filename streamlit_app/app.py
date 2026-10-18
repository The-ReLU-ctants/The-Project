"""Public deployment front-end for the species classifier (Streamlit Community Cloud).

Loads the same ViT checkpoint and baseline model used by src/inference_server.py
and runs the same prediction logic, just through a Streamlit UI instead of a
FastAPI REST call, since Streamlit Cloud is the free host used for this demo.

Run locally with: streamlit run streamlit_app/app.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import get_transforms, load_label_map  # noqa: E402
from src.model import build_model  # noqa: E402

CHECKPOINT_PATH = ROOT / "models" / "vit_species.pt"
LABEL_MAP_PATH = ROOT / "data" / "processed" / "label_map.json"
BASELINE_PATH = ROOT / "models" / "baseline_logreg.joblib"


@st.cache_resource
def load_everything():
    device = torch.device("cpu")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model = build_model(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    label_map = load_label_map(str(LABEL_MAP_PATH))
    idx_to_species = {v: k for k, v in label_map.items()}
    transform = get_transforms("val", ckpt["config"]["image_size"])

    baseline = None
    if BASELINE_PATH.exists():
        baseline = joblib.load(BASELINE_PATH)

    return {
        "model": model,
        "transform": transform,
        "idx_to_species": idx_to_species,
        "epoch": ckpt.get("epoch"),
        "baseline": baseline,
    }


def predict_vit(state, image):
    x = state["transform"](image).unsqueeze(0)
    with torch.inference_mode():
        probs = torch.softmax(state["model"](x), dim=1)[0]
    ranked = sorted(
        ((state["idx_to_species"][i], float(p)) for i, p in enumerate(probs)),
        key=lambda t: t[1],
        reverse=True,
    )
    return ranked


def predict_baseline(baseline, image):
    image_size = baseline["image_size"]
    x = (np.asarray(image.resize((image_size, image_size)), dtype=np.float32) / 255.0).ravel()
    probs = baseline["model"].predict_proba(x.reshape(1, -1))[0]
    idx_to_species = {v: k for k, v in baseline["label_map"].items()}
    return sorted(
        ((idx_to_species[i], float(p)) for i, p in enumerate(probs)),
        key=lambda t: t[1],
        reverse=True,
    )


st.set_page_config(page_title="Wildlife Species Classifier", page_icon="\U0001F993")
st.title("\U0001F993 Wildlife Species Classifier")
st.caption("The ReLU-ctants — Vision Transformer trained from scratch on WildlifeReID-10k (22 species)")

state = load_everything()
st.success(f"Model loaded (epoch {state['epoch']}) — {len(state['idx_to_species'])} species")

uploaded = st.file_uploader("Upload a photo of an animal", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        vit_ranked = predict_vit(state, image)
        st.subheader("Vision Transformer")
        st.metric("Predicted species", vit_ranked[0][0], f"{vit_ranked[0][1] * 100:.1f}% confidence")
        st.bar_chart({name: p for name, p in vit_ranked[:5]})

        if state["baseline"] is not None:
            baseline_ranked = predict_baseline(state["baseline"], image)
            st.subheader("Baseline (Logistic Regression)")
            st.metric("Predicted species", baseline_ranked[0][0], f"{baseline_ranked[0][1] * 100:.1f}% confidence")
            st.bar_chart({name: p for name, p in baseline_ranked[:5]})
        else:
            st.info("Baseline model not found; run scripts/baseline.py to train it.")
