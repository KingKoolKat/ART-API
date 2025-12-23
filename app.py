from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
from pathlib import Path
from urllib.request import urlretrieve
from typing import Optional

import psycopg2
import psycopg2.extras

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

    # Safety check (adjust if your model architecture differs)
    assert len(idx_to_style) == model.classifier[1].out_features

    _model = model
    return _model


# -------------------------
# DB config (Postgres)
# -------------------------
DATABASE_URL = os.environ.get("DATABASE_URL")


def get_conn():
    """
    Create a new DB connection. (Simple + fine for now.)
    Later you can switch to a connection pool.
    """
    if not DATABASE_URL:
        raise HTTPException(
            status_code=500,
            detail="DATABASE_URL env var is not set (needed for /gallery).",
        )
    return psycopg2.connect(DATABASE_URL)


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


@app.get("/gallery")
def gallery(
    style: str = Query(..., description="Style/category name (e.g., Impressionism)"),
    limit: int = Query(24, ge=1, le=60),
    exclude_id: Optional[str] = Query(None, description="Optional artwork id to exclude"),
):
    """
    Returns up to `limit` artworks from the `artworks` table that match the given style.
    Expects you ran seed_wikiart.py to populate:
      - table: artworks(id, title, artist, style, image_url)
      - directory with files referenced by image_url 
    """
    sql = """
    SELECT id, title, artist, style, image_url
    FROM artworks
    WHERE style = %s
      AND (%s IS NULL OR id <> %s)
    ORDER BY RANDOM()
    LIMIT %s;
    """

    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (style, exclude_id, exclude_id, limit))
                rows = cur.fetchall()
    except HTTPException:
        raise
    except Exception as e:
        # Avoid leaking creds; keep error generic but helpful
        raise HTTPException(status_code=500, detail=f"DB query failed: {type(e).__name__}")

    return {"style": style, "items": rows}
