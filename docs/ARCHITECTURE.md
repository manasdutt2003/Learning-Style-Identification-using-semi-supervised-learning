# Architecture

This document summarises how the pieces of the repository fit together, based
on the README and the architecture write-up (`Architecture_Description.pdf`).

## Components

| Component | Role |
| --- | --- |
| `app.py` (Streamlit) | Questionnaire input, per-dimension prediction with confidence, charts, PDF report, faculty dashboard |
| `main.py` (FastAPI) | `POST /predict` with SHAP-based explanations; used by `index.html` |
| `model_definitions.py`, `sota_models.py` | KAN, TabNet, NAM and FT-Transformer wrappers with a scikit-learn style interface |
| `train_semi_supervised.py` | Self-training pipeline; selects the best candidate per FSLSM dimension |
| `sota_semi_supervised_models.joblib` | Saved per-dimension models used for inference |
| `data_fs1.csv` | Training data (17 behavioural features per learner) |

## Data flow

1. Input: slider/questionnaire values (Streamlit), a JSON body (API) or a CSV.
2. Features are mapped to the training schema.
3. For each of the four FSLSM dimensions (Visual/Verbal, Active/Reflective,
   Sensing/Intuitive, Sequential/Global) the dimension's model returns class
   probabilities.
4. Confidence is derived from those probabilities and shown with the result.
5. Output: UI cards and charts, a PDF report, or JSON from the API.

## Training

`train_semi_supervised.py` reads `data_fs1.csv`, trains candidate models per
dimension with self-training on rows treated as unlabelled, and keeps the best
one per dimension. Earlier pipelines live in `train_sota.py` and
`train_model.py`.

## Failure behaviour

If the saved model file cannot be loaded, the Streamlit app falls back to a
neutral rule-based result and tells the user it is doing so.

## Known limitations

- The model-comparison and confusion-matrix charts in the app use fixed
  reference values defined in `app.py`; they are not recomputed from the saved
  models.
- `data/daily_stats.json` from `generate_daily_stats.py` is random placeholder
  data (flagged `"synthetic": true`), not a record of real usage.
- `extract_pdf.py`, `extract_pdf_ocr.py` and `generate_prompt_pdf.py` still
  contain hard-coded local paths and need extra dependencies that are not in
  `requirements.txt`.
