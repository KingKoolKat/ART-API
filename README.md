# Art Style Classifier API

ML-backed API that predicts an artwork’s style from an input image (27 WikiArt-style categories). It also includes an optional Postgres-backed “gallery” endpoint for serving example artworks by style.

## For recruiters
This repo is a compact example of:
- Serving a PyTorch model behind a production-friendly FastAPI service (lazy model loading, simple input validation, Dockerized deployment).
- Basic backend hygiene (health endpoint, env-driven config, non-leaky error handling).
- A small data pipeline (`seed_wikiart.py`) to populate a Postgres table from a public dataset, storing images in Cloudinary for fast delivery.

## Tech stack
- **API**: Python 3.11, FastAPI, Uvicorn
- **ML inference**: PyTorch, torchvision (EfficientNet-B3), Pillow
- **Data**: Postgres (`psycopg2`)
- **Optional seeding**: HuggingFace `datasets`, `tqdm`, Cloudinary

## API endpoints
| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/styles` | Returns the supported style labels |
| POST | `/predict-style` | Upload an image and get top-k style predictions |
| GET | `/gallery` | Returns artworks matching a style (requires Postgres) |

### `POST /predict-style`
- **Body**: `multipart/form-data` with field `file` (`image/jpeg`, `image/png`, or `image/webp`)
- **Query params**: `top_k` (default `5`)
- **Response**: the best prediction plus the top-k list with probabilities

Example:
```bash
curl -F "file=@/path/to/image.jpg" "http://localhost:10000/predict-style?top_k=5"
```

### `GET /gallery` (optional)
Returns random artworks for a given style.

- **Query params**:
  - `style` (required) — style/category name (e.g., `Impressionism`)
  - `limit` (default `24`, min `1`, max `60`)
  - `exclude_id` (optional) — omit a specific artwork id
- **Requires**: `DATABASE_URL`

This endpoint expects a table:
`artworks(id, title, artist, style, image_url)`

## Running locally
### 1) Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Provide model weights
At startup the API ensures `models/efficientnet_b3_best.pt` exists:
- **Recommended**: set `MODEL_URL` to a downloadable `.pt` file (the server downloads it on first run).
- **Or**: place the weights file at `models/efficientnet_b3_best.pt`.

### 3) Start the server
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 10000
```

### 4) Quick checks
```bash
curl http://localhost:10000/health
curl http://localhost:10000/styles
```

## Docker
```bash
docker build -t art-style-api .
docker run -p 10000:10000 -e MODEL_URL="https://..." art-style-api
```

## Gallery seeding (optional)
`seed_wikiart.py` can populate the `artworks` table used by `/gallery` by:
1) pulling images from the HuggingFace WikiArt dataset, then
2) uploading them to Cloudinary, and
3) upserting metadata into Postgres.

This script uses extra dependencies that aren’t in `requirements.txt`:
```bash
pip install datasets tqdm cloudinary
```

Required env vars:
- `DATABASE_URL`
- `CLOUDINARY_CLOUD_NAME`
- `CLOUDINARY_API_KEY`
- `CLOUDINARY_API_SECRET`

Optional env vars:
- `PER_STYLE` (default `300`)
- `CLOUDINARY_FOLDER` (default `wikiart`)
- `WIKIART_SPLIT` (default `train`)
- `STYLE_WHITELIST_FILE` (optional path to `idx_to_style.json`, a JSON list, or a newline-separated list of styles)

Run:
```bash
python seed_wikiart.py
```

## Project layout
- `app.py`: FastAPI routes (`/predict-style`, `/gallery`, etc.)
- `model.py`: model loading + preprocessing + prediction helpers
- `idx_to_style.json`: class label map
- `seed_wikiart.py`: dataset → Cloudinary + Postgres seeding pipeline
- `Dockerfile`: container build for deployment

## Notes / next improvements
- CORS is currently wide-open for easier demo; production should restrict `allow_origins`.
- `/gallery` uses a new DB connection per request; a connection pool would be the next step for higher throughput.
- If exposing publicly: add auth, rate limiting, and request size limits.
