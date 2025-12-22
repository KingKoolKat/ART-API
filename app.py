from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
from pathlib import Path
from urllib.request import urlretrieve

from model import load_model, load_label_map, predict_pil

app = FastAPI(title="Art Style Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Model config
# -------------------------
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "efficientnet_b3_best.pt"

# Set this in Render → Environment Variables
MODEL_URL = os.environ.get("MODEL_URL")

idx_to_style = load_label_map("idx_to_style.json")

_model = None


def ensure_model_file():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return

    if not MODEL_URL:
        raise RuntimeError("MODEL_URL env var is not set")

    tmp_path = MODEL_PATH.with_suffix(".pt.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    urlretrieve(MODEL_URL, tmp_path)
    tmp_path.replace(MODEL_PATH)


def get_model():
    global _model
    if _model is not None:
        return _model

    ensure_model_file()

    model = load_model(
        str(MODEL_PATH),
        num_classes=len(idx_to_style),
    )

    # Safety check
    assert len(idx_to_style) == model.classifier[1].out_features

    _model = model
    return _model


# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/styles")
def styles():
    return idx_to_style


@app.post("/predict-style")
async def predict_style(
    file: UploadFile = File(...),
    top_k: int = 5,
):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(400, "Unsupported file type")

    img_bytes = await file.read()
    try:
        img = Image.open(io.BytesIO(img_bytes))
    except Exception:
        raise HTTPException(400, "Invalid image")

    model = get_model()
    return predict_pil(model, img, top_k, idx_to_style)
