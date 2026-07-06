# Data Layout

This repository supports two model families:

- **TabFM** (tabular classification/regression)
- **TimesFM** (univariate time-series forecasting)

We keep **tiny, redistributable samples** in git for instant demos, and store full downloads in `data/raw/` (gitignored).

## Folders

- `data/samples/`
  - `tabular/<dataset>/train.csv`: features + target column
  - `tabular/<dataset>/predict.csv`: features only (no target)
  - `timeseries/<dataset>/history.csv`: `timestamp,value`
- `data/raw/` (gitignored)
  - full downloads and intermediate normalization outputs

## Tabular Contract (TabFM)

Train CSV:
- Contains a **target column** (you select it in the UI)
- All other columns are treated as features

Predict CSV:
- Must contain **the same feature columns** as the training CSV
- Must **not** include the target column

## Time-Series Contract (TimesFM)

History CSV must include:
- `timestamp`: parseable datetime
- `value`: numeric

The app coerces types, drops invalid rows, and sorts by `timestamp`.

## Provenance

Full datasets are downloaded by `scripts/download_datasets.py` and written under `data/raw/`.

Source URLs are also recorded in `data/manifest.json` (committed).
