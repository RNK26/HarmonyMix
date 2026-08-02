# Troubleshooting

Common problems when deploying or running HarmonyMix, and what to do about
them. See also [DEPLOYMENT.md](DEPLOYMENT.md) and
[RUNNING_WITH_REAL_DATA.md](RUNNING_WITH_REAL_DATA.md).

## App shows "Running in demo mode" but I added the dataset

* Check the files are at exactly `data/raw/tracks.csv` and
  `data/raw/playlists.csv` (relative to the project root, not inside a
  subfolder).
* If running in Docker with plain `docker run` (no `-v` flags), the
  container can't see host files — real data is invisible to it by design
  (`.dockerignore` excludes it from the image). Use `docker-compose up`
  instead, or add the volume mounts described in
  [DEPLOYMENT.md](DEPLOYMENT.md).
* Restart the app after adding the files — `st.cache_data` only re-checks
  the filesystem on a fresh process start or after "Re-fit engines."

## Recommendations look unchanged after I replaced the dataset

The engines are cached to `models/*.pkl` and only auto-rebuild when the row
count changes. If your new dataset happens to have the same number of rows,
click **"Re-fit engines (after new data)"** in the sidebar, or delete
`models/content_engine.pkl` and `models/collab_engine.pkl` and restart.

## "Couldn't start the recommendation engines" error in the app

This is a caught, generic error shown for any startup failure (malformed
CSV, out-of-memory while fitting, etc.) — the real exception is logged
server-side, not shown to the user. Check the terminal / container logs
(`docker logs harmonymix_container`) for the actual traceback.

Common causes:
* A `tracks.csv` missing one of the required columns listed in
  [RUNNING_WITH_REAL_DATA.md](RUNNING_WITH_REAL_DATA.md).
* Out of memory — the collaborative engine's co-occurrence matrix and the
  ~250 MB fitted pickle are the most memory-hungry part; make sure the
  container/host has a few GB of headroom free on first run.

## `dvc repro` / `dvc pull` doesn't work

* `dvc repro` (regenerate `data/processed/tracks_cleaned.csv` locally)
  should work as long as `data/raw/tracks.csv` exists — it doesn't need a
  remote.
* `dvc pull` will not work in this repo. `.dvc/config` is committed but has
  no remote configured, since the dataset isn't hosted anywhere DVC can
  reach. This is expected, not a bug — get the dataset directly (see
  [RUNNING_WITH_REAL_DATA.md](RUNNING_WITH_REAL_DATA.md)) rather than via
  `dvc pull`.

## `pip install category_encoders` fails or is skipped

`src/preprocessing/preprocess.py` has a built-in fallback `CountEncoder` used
automatically if `category_encoders` isn't importable, so this doesn't break
anything — it's a soft dependency, not a hard requirement.

## Docker build is slow / seems to recompile things

It shouldn't need to: numpy, pandas, scikit-learn, and scipy all install
from prebuilt wheels for `python:3.9-slim`, so `pip install` in the
Dockerfile doesn't compile from source. If a build is unusually slow, it's
most likely a cold Docker layer cache (first build after a `requirements.txt`
change) — later builds reuse cached layers.

## Container is `healthy` in `docker ps` but the page won't load

Confirm the port mapping: the app always listens on `8501` inside the
container (`Dockerfile`/`docker-compose.yml` both use `-p 8501:8501`). If
another local process is already using host port 8501, either stop it or
remap, e.g. `docker run -p 8502:8501 harmonymix` and open `:8502`.

## Editing code but the running container doesn't pick it up

Depends on how you started it:
* **`docker-compose up`** — the whole project directory is bind-mounted
  (`.:/workspace`), so host-side edits are visible inside the container
  immediately; Streamlit's own file-watcher will pick up `.py` changes
  without a rebuild.
* **Plain `docker run` (no volumes)** — the container only has whatever was
  copied in at build time. You must `docker build` again after code changes.

## Tests fail locally but passed in CI (or vice versa)

CI runs on Python 3.9 with a clean `pip install -r requirements.txt` — no
real dataset, no pre-existing `models/*.pkl`. If your local environment has
a different Python version, stale cached models, or extra packages
installed, try reproducing the CI environment: a fresh virtualenv on Python
3.9, `pip install -r requirements.txt`, then `pytest tests/ -v --tb=short`.
