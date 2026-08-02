# Deployment guide

How to build and run HarmonyMix, either locally or with Docker, and what to
expect from each option. For loading the real Spotify dataset instead of the
demo data, see [RUNNING_WITH_REAL_DATA.md](RUNNING_WITH_REAL_DATA.md). If
something breaks, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Two modes, always

The app never fails to start just because data is missing. It checks for
`data/processed/tracks_cleaned.csv` or `data/raw/tracks.csv` on startup:

* **Found** → loads the real catalogue, header says "Real Spotify Data".
* **Not found** → generates a small synthetic sample in memory, header says
  "Running in demo mode."

This matters for deployment because the dataset (~600 MB) and fitted models
(~270 MB) are intentionally not committed to GitHub (see
[RUNNING_WITH_REAL_DATA.md](RUNNING_WITH_REAL_DATA.md) for why). A fresh
clone or a plain Docker build will run in demo mode until you add the data.

## Running locally (no Docker)

```bash
python -m venv venv
source venv/Scripts/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

Open `http://localhost:8501`. Works immediately in demo mode; add the
dataset under `data/raw/` for real data (see the other doc).

## Running with Docker

### Plain `docker build` / `docker run`

```bash
docker build -t harmonymix .
docker run -p 8501:8501 harmonymix
```

`.dockerignore` excludes `data/raw/`, `data/processed/`, `data/artifacts/`,
and `models/` from the build context — they are never baked into the image.
**A container started this way, with no volumes, only ever runs in demo
mode**, even if you have the real dataset sitting on your host machine,
because the container can't see it.

To run with real data using plain `docker run`, mount the folders in:

```bash
docker run -p 8501:8501 \
  -v "$(pwd)/data:/workspace/data" \
  -v "$(pwd)/models:/workspace/models" \
  harmonymix
```

### `docker-compose` (recommended)

```bash
docker build -t harmonymix .        # or: docker-compose build
docker-compose up
```

`docker-compose.yml` already mounts the whole project directory
(`.:/workspace`) into the container. If `data/raw/tracks.csv` and
`data/raw/playlists.csv` exist on your host, the containerized app sees them
automatically and runs in real-data mode — no extra flags needed. This also
means code edits on the host take effect on container restart without
rebuilding the image, since the bind mount overlays the copied source.

### Healthcheck

The image defines `HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health`,
Streamlit's built-in health endpoint. `docker ps` will show `healthy` /
`unhealthy` once the app has finished starting.

## Expected folder structure at runtime

```
data/
  raw/          tracks.csv, playlists.csv   (not committed — see RUNNING_WITH_REAL_DATA.md)
  processed/    tracks_cleaned.csv           (written by the preprocessing step, or by the app itself)
  artifacts/    reserved for pipeline outputs (currently unused, kept for structure)
models/
  content_engine.pkl   (not committed — built automatically on first run)
  collab_engine.pkl    (not committed — built automatically on first run)
```

Every one of these folders is present in the repo as an empty placeholder
(`.gitkeep`) so `git clone` always creates the right directory tree, even
though the files inside are gitignored.

## First run is slower than later runs

`app/app.py` fits the content-based and collaborative engines once, then
caches them to `models/*.pkl` (via `src/utils/model_persistence.py`). First
launch (or first launch after changing the dataset size) takes noticeably
longer while it fits; the "Re-fit engines" button in the sidebar forces this
to happen again after you swap in new data.

## CI

`.github/workflows/ci.yml` runs on every push/PR to `main`/`master`:

1. `pytest tests/ -v --tb=short` (27 tests) on Python 3.9.
2. `docker build -t harmonymix:test .` — confirms the image builds; it does
   not start the container or hit the healthcheck endpoint.

## Deployment blockers found during this audit

None that block running the app, tests, or the Docker build. One thing worth
your attention before a public release — doesn't affect demo-mode
functionality:

* The README's Docker "Running the app" example (`docker run -p 8501:8501
  harmonymix`) is correct but will only ever show demo mode, since it
  doesn't mount data volumes — see the "Plain `docker run`" note above.
