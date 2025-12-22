from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from model import load_model, load_label_map, predict_pil

app = FastAPI(title="Art Style Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "efficientnet_b3_best.pt"

idx_to_style = load_label_map("idx_to_style.json")
model = load_model(MODEL_PATH, num_classes=len(idx_to_style))

# Safety check: refuse to start if mismatched
assert len(idx_to_style) == model.classifier[1].out_features

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

    return predict_pil(model, img, top_k, idx_to_style)
