# Running on Borah

Everything on Borah is run by hand from a normal SSH terminal. Big data goes in your scratch or project folder, not in your home folder (check the Research Computing docs for where that is). Below, `$CHIPS` and `$DATASET` are those folders, for example:

```bash
export CHIPS=/path/to/scratch/cs497/hls_chips
export DATASET=/path/to/scratch/cs497/dataset
```

## 1. Get the code (once)

```bash
git clone https://github.com/RobertojonesFAI/Post_Wildfire_Vegetation.git
cd Post_Wildfire_Vegetation
module avail python            # pick a Python 3.10+ module
module load <that module>
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas rasterio
```

Later, `git pull` gets the new scripts.

## 2. Copy the chips from Google Drive (rclone)

The chips from notebook 02 are in Google Drive, folder `cs497_hls_chips` (about 10 GB).

**Install rclone on Borah (once)**, if `module avail rclone` doesn't have it:

```bash
cd ~
curl -O https://downloads.rclone.org/rclone-current-linux-amd64.zip
unzip rclone-current-linux-amd64.zip
mkdir -p ~/bin && cp rclone-*-linux-amd64/rclone ~/bin/
export PATH=$HOME/bin:$PATH    # also add this line to ~/.bashrc
```

**Connect it to your Drive (once).** Borah has no browser, so the login happens on your laptop:

1. On your laptop, install rclone (`brew install rclone` on a Mac).
2. On Borah run `rclone config`, then: `n` (new remote) → name `gdrive` → storage `drive` → leave client id and secret empty → scope `drive.readonly` → leave the rest empty → advanced config `n` → auto config `n`.
3. rclone prints a command like `rclone authorize "drive" "..."`. Run it **on your laptop**, log in with your BSU Google account, and paste the token it gives you back into Borah.
4. Shared drive `n`, then `y` to save.

**Copy:**

```bash
rclone copy gdrive:cs497_hls_chips "$CHIPS" --progress --transfers 8
ls "$CHIPS" | wc -l            # 148 .tif + gridmet_monthly.csv
```

## 3. Build the dataset (step 3)

```bash
cd Post_Wildfire_Vegetation
git pull
mkdir -p logs
sbatch scripts/build_dataset.sbatch "$CHIPS" "$DATASET"
squeue -u $USER                # is it running?
tail -f logs/build_dataset-*.out
```

When it ends, `$DATASET` has one `.npz` per chip plus `chips.csv`, `chip_months.csv` and `splits.json` (see the top of `scripts/build_dataset.py` for what's inside).

## Colab instead

The same script runs in Colab with the chips straight from Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
!git clone https://github.com/RobertojonesFAI/Post_Wildfire_Vegetation.git
%cd Post_Wildfire_Vegetation
!pip install -q rasterio
!python scripts/build_dataset.py --chips /content/drive/MyDrive/cs497_hls_chips --out /content/drive/MyDrive/cs497_dataset --workers 2
```
