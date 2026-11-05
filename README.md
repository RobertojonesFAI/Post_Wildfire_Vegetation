# Post-Fire Vegetation Recovery

CS 497/597 Computer Vision final project, Boise State University, Fall 2026.
Roberto Jones, Angie Bui, Santiago Vargas.

We fine-tune Prithvi-EO-2.0 on HLS time series to track and forecast vegetation recovery in burn scars across the western US. As a stretch goal, we test if an updated burn map makes the USGS M1 debris-flow model better.

## Repo layout

```
notebooks/   step-by-step notebooks (run in order)
data/fires/  fires.csv and candidates.csv (small tables only, no imagery)
```

The image chips are too big for git. They live on Borah, with a small sample on Google Drive for Colab.

## Steps

| # | Notebook | What it does | Output |
|---|----------|--------------|--------|
| 1 | `01_select_fires.ipynb` | Picks ~15 MTBS wildfires (2015–2023), 3 per region, plus the 2024 Wapiti Fire | `fires.csv`, `candidates.csv` |

## Setup

Each of us uses our own Google Cloud project with Earth Engine enabled. Put its ID in a file called `ee_project.txt` at the repo root (git ignores it):

```bash
echo "your-project-id" > ee_project.txt
```

**Locally (VS Code):**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt ipykernel
```

Then open a notebook, pick the `.venv` kernel and Run All. The first time, it asks you to log in to Earth Engine.

**Colab:** open the notebook from GitHub, set `PROJECT_ID` in the first cell and Run All. It also asks to mount Google Drive.

Please clear the notebook outputs before committing, so diffs stay small.

## Data sources

- MTBS burned area boundaries and burn severity mosaics (USFS/USGS)
- HLS v2.0, Landsat 30 m (HLSL30) and Sentinel-2 30 m (HLSS30) (NASA)
- NLCD 2019 release land cover (USGS)
- gridMET (weather) and USGS 3DEP (terrain), added in step 2

All of them are read through Google Earth Engine.
