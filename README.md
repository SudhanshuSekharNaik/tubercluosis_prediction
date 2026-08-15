# Tuberculosis Risk Prediction — Notebook + Flask Deployment

This package trains and deploys a Logistic Regression model that estimates tuberculosis (TB)
risk from six clinical/demographic factors, based on
[SudhanshuSekharNaik/tubercluosis_prediction](https://github.com/SudhanshuSekharNaik/tubercluosis_prediction).

**Features used:** Age, Sex, Smoking Habits, Alcohol Consumption, Diabetic Status, Asthma History

> ⚠️ Trained on **synthetic data** for educational purposes only. Not a diagnostic tool —
> should never replace clinical judgment or microbiological confirmation.

---

## 📁 What's in this package

```
tb_project/
├── notebook/
│   ├── tb_risk_prediction_training.ipynb   ← upload this to Kaggle
│   ├── tb_model.pkl                        ← trained model (already generated)
│   ├── tb_scaler.pkl                       ← fitted StandardScaler
│   ├── feature_names.json
│   ├── synthetic_tb_data.csv               ← generated dataset
│   └── model_evaluation.png                ← confusion matrix / ROC / feature importance
│
├── flask_app/
│   ├── app.py                              ← Flask API + web form
│   ├── templates/index.html                ← risk-assessment UI
│   ├── model/                              ← copy of the trained model artifacts
│   ├── requirements.txt
│   └── Dockerfile
│
└── README.md
```

---

## 1. Run the notebook on Kaggle

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**.
2. **File → Upload Notebook** → select `notebook/tb_risk_prediction_training.ipynb`.
3. Click **Run All** (or **Save Version → Save & Run All**).
4. The notebook generates a synthetic TB dataset, trains the Logistic Regression model,
   evaluates it (accuracy/precision/recall/F1/AUC), plots a confusion matrix, ROC curve
   and feature-importance chart, and saves:
   - `tb_model.pkl`
   - `tb_scaler.pkl`
   - `feature_names.json`

   These appear under **Output** in the right-hand Kaggle panel — download them from there
   if you want to regenerate the model artifacts already included in `flask_app/model/`.

No dataset needs to be attached — the notebook synthesizes its own data (no internet access
required, so it will run fine on Kaggle's default environment).

---

## 2. Run the Flask API locally

```bash
cd flask_app
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** for the web form, or call the JSON API directly:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":65,"sex":1,"smoking":2,"alcohol":2,"diabetes":1,"asthma":1}'
```

Response:
```json
{
  "probability": 0.999,
  "probability_pct": "99.9%",
  "prediction": "TB PATIENT",
  "risk_level": "HIGH RISK",
  "recommendation": "Immediate medical attention required"
}
```

**Field encoding**

| Field    | Values |
|----------|--------|
| `age`      | integer, 0–120 |
| `sex`      | 0 = Female, 1 = Male |
| `smoking`  | 0 = Never, 1 = Former, 2 = Current |
| `alcohol`  | 0 = Never, 1 = Former, 2 = Current |
| `diabetes` | 0 = No, 1 = Yes |
| `asthma`   | 0 = No, 1 = Yes |

Endpoints:
- `GET  /` — HTML intake form
- `POST /predict` — JSON API
- `POST /predict-form` — form submission (renders the result in the page)
- `GET  /health` — health check

---

## 3. Deploy with Docker

```bash
cd flask_app
docker build -t tb-risk-api .
docker run -p 5000:5000 tb-risk-api
```

---

## 4. Deploy to a hosting platform (Render / Railway / Fly.io / Heroku-style)

The `flask_app/` folder is self-contained and production-ready via `gunicorn`
(already in `requirements.txt` and referenced by the `Dockerfile`). Typical steps:

1. Push `flask_app/` to its own GitHub repo (or a subfolder of your existing repo).
2. On the hosting platform, point it at that repo/folder.
3. Set the start command to: `gunicorn --bind 0.0.0.0:$PORT app:app`
4. Deploy — the model files in `model/` are loaded automatically at startup.

---

## 5. Retraining with your own data

If you get access to a real (de-identified, ethically sourced) TB dataset, replace the
`generate_tb_data()` call in the notebook with `pd.read_csv('your_data.csv')`, keeping the
same six column names (`Age, Sex, Smoking, Alcohol, Diabetes, Asthma, TB_Status`), then
re-run the notebook and copy the new `tb_model.pkl` / `tb_scaler.pkl` into `flask_app/model/`.
