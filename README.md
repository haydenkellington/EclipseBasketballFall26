# EclipseBasketballFall26
Fall 2026 Eclipse Analytics Basketball Project

## NBA shot prediction: data preparation and EDA

Research question: **Can we predict whether an NBA player makes a shot from its location, distance, and game situation, while accounting for offensive and defensive skill?**

Open [`HaydenEDA.ipynb`](HaydenEDA.ipynb) for the executed analysis. The notebook reads and combines all four supplied `.tar.xz` archives directly, preserves source and season labels, audits and cleans the records, creates plots, and prepares features for later modeling. The archive year identifies the season's **start year**; `po` identifies playoff records.

The combined data contain **467,537 shots**, **2,629 games**, and **684 players**, from October 22, 2024 through June 13, 2026. No rows were removed. The notebook flags 33 shot-type/zone inconsistencies and identifies spelling variants for three player IDs; IDs remain the basis for player histories.

### Run locally

Keep the four original archives beside the notebook, then run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyterlab HaydenEDA.ipynb
```

Select the Python kernel associated with `.venv` and choose **Run All** to regenerate the analysis. If that kernel is not listed, register it in the project environment:

```bash
.venv/bin/python -m ipykernel install --sys-prefix --name eclipse-basketball --display-name "Python (Eclipse Basketball)"
```

### What the notebook produces

- `data/processed/nba_shots_clean.parquet` and `nba_shots_clean.csv.gz`: combined, cleaned shots and derived features.
- `data/processed/nba_shots_model_ready.parquet`: selected predictors, target, identifiers, and chronological split assignments.
- `reports/*.csv`: source coverage, data-quality checks, summaries, and feature documentation.
- `figures/*.png`: EDA charts saved separately from their notebook outputs.

The original datasets, executed notebook, generated data, reports, and figures are tracked in Git. Running the notebook regenerates the derived outputs. The local Python environment remains ignored.

### Feature interpretation and modeling limits

Offensive skill is represented by a player's earlier shooting results and zone-specific history. Defense is represented by the opponent team's earlier FG% allowed, including zone-specific rates. These histories expand across seasons and exclude **the entire current date**. Smoothed rates retain supporting attempt counts. Team FG% allowed is a limited proxy affected by opponents and shot selection, not a direct measurement of individual defensive ability.

The files lack defender identity, contest distance, score margin, and shot-clock time. Available game situation features therefore describe period, game clock, home/away, and regular-season/playoff context. Late-game indicators do not establish whether a game was close.

The notebook ends with approximately **70%/15%/15% of unique dates** assigned chronologically to training, validation, and test sets, plus explicit predictor allowlists. Shot proportions can differ from those date proportions. Direct outcome fields such as `EVENT_TYPE` are excluded from predictors, and `ACTION_TYPE` is omitted from the initial allowlists because its recorded labels can depend on the completed play. Histories on later dates incorporate earlier observed outcomes, including earlier validation/test dates, matching daily updating. **No model is trained** in this EDA notebook.
