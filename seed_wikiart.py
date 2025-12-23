import os
import re
import io
import hashlib
from typing import Optional

import psycopg2
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

import cloudinary
import cloudinary.uploader


# -------------------------
# Config (edit as you want)
# -------------------------
PER_STYLE = int(os.environ.get("PER_STYLE", "300"))   # how many images per style to upload
FOLDER = os.environ.get("CLOUDINARY_FOLDER", "wikiart")  # Cloudinary folder name
DATASET_SPLIT = os.environ.get("WIKIART_SPLIT", "train")  # usually "train"

# If you want to only seed the styles your model supports, set STYLE_WHITELIST_FILE
# to a JSON file with idx_to_style mapping, or a newline-separated list of style names.
STYLE_WHITELIST_FILE = os.environ.get("STYLE_WHITELIST_FILE")  # optional
# -------------------------


def slugify(x: str) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def load_style_whitelist(path: str) -> Optional[set]:
    """
    Optional: only seed styles you support. Supports:
      - JSON dict { "0": "Impressionism", ... } (idx_to_style.json)
      - JSON list ["Impressionism", ...]
      - newline-separated text
    """
    import json
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()

    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return set(str(v) for v in obj.values())
        if isinstance(obj, list):
            return set(str(v) for v in obj)
    except Exception:
        pass

    # fallback: newline-separated
    return set(line.strip() for line in txt.splitlines() if line.strip())


def ensure_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS artworks (
          id TEXT PRIMARY KEY,
          title TEXT,
          artist TEXT,
          style TEXT NOT NULL,
          image_url TEXT NOT NULL
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_artworks_style ON artworks(style);")
    conn.commit()


def compute_id(style: str, img: Image.Image) -> str:
    """
    Deterministic-ish ID: style_slug + sha1 of JPEG bytes (so duplicates won't reupload/insert).
    """
    buf = io.BytesIO()
    rgb = img.convert("RGB")
    rgb.save(buf, format="JPEG", quality=95)
    h = hashlib.sha1(buf.getvalue()).hexdigest()[:16]
    return f"{slugify(style)}_{h}"


def upload_to_cloudinary(img: Image.Image, public_id: str) -> str:
    """
    Uploads full-quality JPEG to Cloudinary and returns secure URL.
    """
    buf = io.BytesIO()
    rgb = img.convert("RGB")
    rgb.save(buf, format="JPEG", quality=95)
    buf.seek(0)

    res = cloudinary.uploader.upload(
        buf,
        folder=FOLDER,
        public_id=public_id,
        overwrite=False,      # don’t overwrite if it already exists
        resource_type="image"
    )
    return res["secure_url"]


def main():
    # --- env checks ---
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME")
    api_key = os.environ.get("CLOUDINARY_API_KEY")
    api_secret = os.environ.get("CLOUDINARY_API_SECRET")
    if not (cloud_name and api_key and api_secret):
        raise RuntimeError("Set CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET")

    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        secure=True,
    )

    whitelist = load_style_whitelist(STYLE_WHITELIST_FILE) if STYLE_WHITELIST_FILE else None
    if whitelist:
        print(f"Using style whitelist ({len(whitelist)} styles).")

    print("Loading dataset...")
    ds = load_dataset("huggan/wikiart", split=DATASET_SPLIT)

    # style int -> name mapping (ClassLabel)
    style_feature = ds.features.get("style")
    id_to_style = getattr(style_feature, "names", None)

    # connect DB
    conn = psycopg2.connect(database_url)
    ensure_table(conn)

    # progress tracking
    saved_per_style = {}  # style -> count uploaded/inserted this run

    # optional: speed up by checking existing ids in DB? (not necessary; ON CONFLICT handles it)
    print("Uploading to Cloudinary + inserting into Postgres...")
    with conn:
        with conn.cursor() as cur:
            for ex in tqdm(ds):
                style_val = ex.get("style")
                if style_val is None:
                    continue

                # convert style label to string
                if isinstance(style_val, int) and id_to_style is not None:
                    style = id_to_style[style_val]
                else:
                    style = str(style_val)

                if whitelist and style not in whitelist:
                    continue

                c = saved_per_style.get(style, 0)
                if c >= PER_STYLE:
                    continue

                img = ex.get("image")
                if img is None:
                    continue

                # title/artist (may be missing)
                title = ex.get("title")
                title = title.strip() if isinstance(title, str) and title.strip() else None

                artist = ex.get("artist")
                artist = artist.strip() if isinstance(artist, str) and artist.strip() else None

                # deterministic id from image bytes
                art_id = compute_id(style, img)

                # Upload (use art_id as public_id so reruns don't create duplicates)
                try:
                    image_url = upload_to_cloudinary(img, public_id=art_id)
                except Exception as e:
                    # If upload fails, skip and continue
                    # (You can print e for debugging if needed)
                    continue

                # Insert row
                cur.execute(
                    """
                    INSERT INTO artworks (id, title, artist, style, image_url)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING;
                    """,
                    (art_id, title, artist, style, image_url),
                )

                # We only increment if we attempted; even if conflict, it's fine
                saved_per_style[style] = c + 1

    conn.close()
    print("Done.")
    print(f"Seeded styles this run: {len(saved_per_style)}")
    # show top few
    top = sorted(saved_per_style.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("Top styles seeded:", top)


if __name__ == "__main__":
    main()
