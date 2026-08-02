# Running with the real dataset

By default HarmonyMix runs in **demo mode** with a small synthetic sample —
this doc covers switching it over to the real ~50k-track Spotify dataset.

## Why the data and models aren't in the repo

* **Dataset (~600 MB combined)** — far too large for a normal Git repo, and
  there's no configured DVC remote to pull it from (`.dvc/config` is
  committed but empty — `dvc repro`/local preprocessing works, `dvc pull`
  does not, because there's nothing to pull from).
* **Fitted models (~270 MB, `models/*.pkl`)** — regenerated automatically the
  first time the app runs against real data; committing derived, rebuildable
  binaries wastes repo space for no benefit.

Both are excluded via `.gitignore` (and `.dockerignore` for image builds),
with `.gitkeep` placeholders so the folders themselves still exist after
`git clone`.

## Required files

Place two CSVs under `data/raw/`:

| File | Rows | Required columns |
|---|---|---|
| `tracks.csv` | ~50,000 | `track_id`/`spotify_id`, `name`, `artist`, `year`, `duration_ms`, `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `time_signature`, `tags` (optional: `genre`, `popularity`) |
| `playlists.csv` | ~9.7M | `user_id` (or `playlist_id`), `track_id` (or `song_id`) |

The schema matches Kaggle's Spotify audio-features-plus-listening-history
datasets, commonly distributed as `Music Info.csv` (rename to `tracks.csv`)
and `User Listening History.csv` (rename to `playlists.csv`). Any dataset
with the same column names will work — the loading code
(`src/recommenders/collaborative_filtering.py`) already accepts either
`user_id`/`playlist_id` and `song_id`/`track_id` naming.

This project does not bundle a download link — source the dataset yourself
and place the two files as described above.

## Steps

1. Copy the two CSVs into `data/raw/`:
   ```
   data/raw/tracks.csv
   data/raw/playlists.csv
   ```
2. (Optional) Pre-clean the tracks file via the local DVC stage:
   ```bash
   dvc repro
   ```
   This runs `src/pipelines/data_pipeline.py` locally and writes
   `data/processed/tracks_cleaned.csv`. It's optional — if you skip it, the
   app cleans the raw CSV itself at startup, just slightly slower.
3. Start the app as usual (`streamlit run app/app.py`, or Docker/
   docker-compose — see [DEPLOYMENT.md](DEPLOYMENT.md) for the difference
   between plain `docker run` and `docker-compose` here).
4. Confirm it picked up real data: the header under "HarmonyMix" should read
   "...Real Spotify Data" instead of "...Running in demo mode."

## Regenerating models after changing the dataset

The app detects a changed row count automatically (it compares the row count
stored on the cached engine against the current CSV) and refits. If you
replace the dataset with a *same-sized* file, that check won't trigger —
use the **"Re-fit engines (after new data)"** button in the sidebar, or
delete the cache files directly:

```bash
rm models/content_engine.pkl models/collab_engine.pkl
```

## Switching back to demo mode

Delete or move the files out of `data/raw/`. There's nothing else to undo —
the app checks for their presence on every startup.
