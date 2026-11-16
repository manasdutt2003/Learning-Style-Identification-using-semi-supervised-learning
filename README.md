# Learning Style Identification using Semi-Supervised Learning

Predicts a student's learning style on the four Felder-Silverman (FSLSM)
dimensions from their behaviour on an e-learning platform, using
semi-supervised and tabular deep-learning models.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://learning-style-identification-using-semi-supervised-learning-6.streamlit.app/)

## Learning dimensions

| Dimension | Question it answers |
| --- | --- |
| Visual / Verbal | How do you prefer to receive information? |
| Active / Reflective | How do you process it? |
| Sensing / Intuitive | Do you prefer concrete facts or abstract ideas? |
| Sequential / Global | Do you learn step by step or from the big picture? |

## What is in the repository

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit app: behaviour questionnaire, per-dimension prediction with confidence, radar and bar charts, PDF report, faculty dashboard |
| `main.py` | FastAPI service exposing `POST /predict` with SHAP-based explanations |
| `index.html` | Static front end that calls the FastAPI service |
| `model_definitions.py`, `sota_models.py` | KAN, TabNet, NAM and FT-Transformer wrappers with a scikit-learn interface |
| `train_semi_supervised.py` | Self-training pipeline; picks the best model per dimension and writes `sota_semi_supervised_models.joblib` |
| `train_sota.py`, `train_model.py` | Earlier training pipelines (tabular deep models; CatBoost with fuzzy c-means features) |
| `generate_*.py`, `extract_pdf*.py` | Scripts that produced the plots, architecture diagram, slides and paper extracts |
| `data_fs1.csv` | Training data: 17 behavioural features per learner |
| `*.joblib`, `*.pt`, `*.zip` | Trained model artifacts |
| `tests/` | pytest tests |
| `docs/ARCHITECTURE.md` | Architecture notes and known limitations |
| `IEEE_journal paper.pdf`, `Architecture_Description.pdf`, `Research_Project_Presentation.pptx` | Paper, architecture write-up and slides |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the parts fit together.

## Run the Streamlit app

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Sign in with the demo faculty account shown on the login form, or press
**Demo Mode**.

## Run the API

```bash
uvicorn main:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"T_video": 40, "T_read": 10, "N_msgs_posted": 12}'
```

Open `index.html` in a browser to use the form-based front end against the
local API.

## Retrain

```bash
python train_semi_supervised.py
```

This reads `data_fs1.csv`, trains each candidate model per dimension with
self-training on the unlabelled rows, and saves the best one per dimension.

## Tests

```bash
pip install pytest
python -m pytest tests -q
```

## Utility scripts

```bash
# Extract text from a PDF (pdfminer.six)
python extract_pdf_miner.py "IEEE_journal paper.pdf" -o paper.txt

# Write placeholder daily stats (override path with DAILY_STATS_FILE)
python generate_daily_stats.py
```

## Docker

```bash
docker build -t learning-style .
docker run -p 8501:8501 learning-style
```

## Notes

- Predictions come from the saved models in `sota_semi_supervised_models.joblib`.
  If that file cannot be loaded the app falls back to a neutral rule-based
  result and says so.
- The model-comparison and confusion-matrix charts on the Analytics and
  Visualizations tabs use fixed reference values defined in `app.py`. They are
  not recomputed from the saved models.
- `data/daily_stats.json` is produced by `generate_daily_stats.py` from random
  numbers and is marked `"synthetic": true`. It is not real usage or accuracy
  data.

## License

MIT

## Author

Manas Dutt
