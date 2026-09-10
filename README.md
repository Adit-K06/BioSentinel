# BioGas-Sentinel

BioGas-Sentinel is a predictive safety layer for anaerobic digestion / biogas
production. Since no real bioreactor or hardware was available, we built a
physics-informed digital twin (growth kinetics, off-gas stoichiometry, foam
dynamics) to generate realistic sensor and synthetic camera data. On top of
that, LightGBM models forecast foam overflow and gas threshold breaches
(CH4/CO2/O2) 5-15 minutes ahead, an OpenCV pipeline extracts real visual
features from generated frames, and an LLM-based explanation engine (with an
offline template fallback) gives mechanism-based reasoning for every alert.
A virtual membrane actuator responds automatically. Everything is demoed
live through a Streamlit dashboard.

Built in 18-24 hours for [Hackathon Name].

## Project layout

```
simulator/              # Digital twin: growth, off-gas, foam, anomalies
vision/                 # Synthetic camera frames + OpenCV extraction
ml_core/                # LightGBM models (Job 1: foam, Job 2: gas)
explanation_engine/     # LLM explainer + offline template fallback
actuator/               # Virtual membrane (OPEN/CLOSED) state logic
dashboard/               # Streamlit app (entrypoint: dashboard/app.py)
data/
  raw/                  # Reference public datasets (Kaggle, Mendeley, etc.)
  synthetic/            # Generated batch logs (gitignored)
  formulas/             # Formula sheets / calibration notes
notebooks/              # Scratch / EDA notebooks
tests/
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file (not committed) with:
```
ANTHROPIC_API_KEY=your_key_here
```

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

## Team workflow

- Single `main` branch, direct push — no PRs (time constraint).
- `git pull` before you start a session, commit often, `git push` when done.
- Never `git push --force`. If you might overwrite someone's file, message first.

## Team split

- **ML / simulation** — `simulator/`, `ml_core/`, `vision/`
- **Dashboard / integration** — `dashboard/`, `actuator/`, `explanation_engine/`
- **Formulas / research** — `data/formulas/`, validates equations in `simulator/`

## Datasets referenced

- Kaggle — Fermentation Optimization Dataset (5L bioreactor)
- Kaggle — AgSTAR Livestock Anaerobic Digester Database
- MDPI Processes (2021) — Off-Gas-Based Soft Sensor methodology
- Mendeley Data — anaerobic acidogenic digestion (xylose) biogas dataset
- Synthetic data (self-generated) — foam/video/batch time-series

## Note on this skeleton

Every `.py` file in this repo currently contains ONLY comments describing
what that file is for, what it will contain, and what it will do — no
actual implementation code yet. This is intentional (Stage 1 of the build:
structure first, code from Stage 2 onward).
