# Detroit Blight Ticket Compliance Prediction

Predicting whether a property-violation ticket issued by the City of Detroit will be paid on time, so that limited enforcement and follow-up resources can be prioritized toward tickets that are actually at risk of going unpaid.

## Problem Statement

The City of Detroit issues "blight tickets" — fines for property violations such as illegal dumping, unmaintained buildings, or failure to obtain required permits. A large share of these tickets are never paid, and the city has limited resources to chase down non-compliant violators.

This project frames compliance prediction as a binary classification problem: given the details of a ticket at the time it is issued, predict whether the violator will pay it (compliant) or not (non-compliant), so the city can direct follow-up effort where it is most likely to matter.

## Dataset

Data was sourced from the Detroit Open Data Portal (the same underlying data used in the University of Michigan "Applied Machine Learning in Python" course and the original MDST/Kaggle blight ticket challenge). Each row represents one ticket, with fields covering:

- Violation details (ordinance code, description, disposition, issuing agency)
- Dates (ticket issued, hearing, judgment, payment)
- Financial fields (fine amount, fees, discounts, judgment amount)
- Violator and property information (address, mailing address, owner details)
- Location (neighborhood, council district, latitude/longitude)

The raw dataset spans tickets issued from 2004 through 2026 (~900,000 rows).

## Target Variable Construction

The dataset does not ship with a ready-made label, so `compliance` was derived directly:

- If `Disposition` contains "Not responsible" → excluded from supervised training (per the standard convention for this dataset — these cases are not evaluated even in the original competition).
- If no `Payment Date` exists → `compliance = 0` (never paid).
- If paid within 30 days of the `Hearing Date` → `compliance = 1`.
- If paid more than 30 days after the hearing → `compliance = 0`.

After filtering out "not responsible" tickets, the dataset is **~85% non-compliant / ~15% compliant** — a significant class imbalance that shaped most of the downstream modeling decisions.

## Data Cleaning

- **Leakage removal**: Columns only known *after* the outcome — `Payment Date`, `Payment Amount`, `Payment Status`, `Balance Due`, `Collection Status`, `Judgement Date`, `Ticket Updated At` — were dropped before training. Including them would let the model trivially "cheat" using information that doesn't exist at prediction time for a new ticket.
- **Identifier and unusable columns dropped**: internal IDs, raw high-cardinality free text (violator name, raw address, inspector name), and columns that were over 90% empty (e.g. `Property Owner Non US Address`, `Street Prefix`).
- **Redundant location fields removed**: `x`/`y` projected coordinates were found to correlate ~1.0 with `Longitude`/`Latitude` and were dropped in favor of the latter.
- **Outlier handling on `days_to_hearing`**: negative values (hearing scheduled before the ticket was even issued) and extreme values (>1000 days) were identified as data errors, affecting roughly 0.1% of rows, and removed.
- **Missing values**: rows missing key location fields (`Neighborhood`, `Council District`, `Longitude`, `Latitude`) were dropped — under 2% of the dataset in every case.

## Feature Engineering

- `is_out_of_state_owner`: whether the property owner's mailing address is outside Michigan.
- `owner_lives_in_detroit`: whether the owner's mailing city is Detroit.
- `owner_lives_at_violation_property`: whether the owner's mailing address matches the violation address exactly.
- `days_to_hearing`: days between ticket issue date and hearing date.
- High-cardinality categoricals (`Ordinance Description`, `Neighborhood`) were bucketed to their top-N most frequent categories plus an "Other" bucket, then one-hot encoded, along with `Disposition` and `Agency Name`.

`Ordinance Law` was dropped in favor of `Ordinance Description`, since the two carried largely overlapping information and `Ordinance Description` was more granular.

## Handling Redundant / Circular Features

A correlation analysis revealed:

- `Admin Fee` and `State Fee` were correlated at r=1.00 (near-constant, duplicate values).
- `Fine Amount`, `Late Fee`, and `Judgement Amount` were correlated at r=0.98 — `Late Fee` and `Judgement Amount` are arithmetic derivatives of `Fine Amount` and the underlying fee schedule.

`Late Fee` in particular stood out in early SHAP analysis as the single strongest predictor — but it only accumulates once a payment is already late, making it a **circular signal** rather than a genuine pre-outcome predictor. To validate this, the model was retrained without `Late Fee`, `State Fee`, and `Judgement Amount`: performance did not drop (ROC-AUC actually improved slightly, from 0.8632 to 0.8644), confirming the model did not depend on this circular feature and was learning from genuinely independent signals instead.

A Variance Inflation Factor (VIF) check was also run on the numeric features. Unscaled `Latitude`/`Longitude` and the near-constant `Admin Fee` initially showed extremely high VIF values (in the hundreds of thousands), which turned out to be a measurement artifact of unscaled/near-constant features rather than genuine multicollinearity — after standardization, all VIF values fell to a healthy 1–2.2 range. `Admin Fee` was ultimately dropped for having near-zero variance and no meaningful predictive contribution.

## Train / Validation Split

A random split was avoided in favor of a **time-based split**, since randomly shuffling ticket data would leak future patterns into validation:

- **Training set**: tickets issued 2004–2023 (~76.5% of data)
- **Validation set**: tickets issued 2024–2026 (~23.5% of data)

This mirrors how the model would actually be used — predicting on new tickets it has never seen, using only patterns learned from the past.

## Modeling

Three models were trained and compared:

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression (baseline, scaled features, `class_weight='balanced'`) | 0.858 | 0.734 |
| **XGBoost** (`scale_pos_weight` for imbalance) | 0.855–0.864 | 0.70–0.747 |
| LightGBM | 0.855 | 0.696 |

All three models scored within roughly 0.003 ROC-AUC of one another. This suggests the predictive ceiling for this feature set had essentially been reached — algorithm choice mattered far less than the upstream feature engineering and leakage removal. **XGBoost** was selected as the final model based on its PR-AUC, which is the more informative metric given the class imbalance.

### Imbalance Handling: SMOTE vs. Class Weighting

Two approaches to the class imbalance were directly compared:

| Approach | ROC-AUC | PR-AUC |
|---|---|---|
| Class weighting (`scale_pos_weight`) | 0.855 | 0.734 |
| SMOTE (synthetic oversampling) | 0.846 | 0.678 |

Class weighting outperformed SMOTE on both metrics. This is likely because SMOTE's interpolation-based oversampling is not well suited to a feature space dominated by one-hot encoded categorical variables — synthetic samples can end up as unrealistic combinations of categorical flags. Class weighting was used in the final model.

### Threshold Tuning

The default 0.5 decision threshold is not appropriate for an ~85/15 imbalanced problem. Thresholds from 0.10 to 0.85 were evaluated on precision, recall, and F1:

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.50 (default) | 0.63 | 0.66 | 0.645 |
| **0.70 (chosen)** | 0.62 | 0.64 | 0.632 |
| 0.85 | 0.78 | 0.52 | 0.62 |

A threshold of **0.70** was selected as the operating point, balancing precision and recall rather than defaulting to 0.5.

## Model Validation

- **Overfitting check**: training ROC-AUC (0.892) vs. validation ROC-AUC (0.855) — a gap of roughly 3.7%, indicating mild but not severe overfitting, and reasonable generalization to unseen years.
- **Confusion matrix** (validation set, threshold = 0.70):

  |  | Predicted Non-Compliant | Predicted Compliant |
  |---|---|---|
  | **Actual Non-Compliant** | 103,506 | 10,038 |
  | **Actual Compliant** | 6,448 | 13,038 |

- **SHAP analysis** identified the strongest predictors as: disposition type (particularly "Responsible by Default"), fine amount, discount amount, geographic location (latitude/longitude), whether the owner lives in Detroit, and violation/ordinance type — all genuine, pre-outcome signals.

## Business Impact

False Positives — tickets the model predicts as "likely compliant" that turn out to be non-compliant — represent real, avoidable revenue loss if the city deprioritizes follow-up on them. In the validation set alone:

- **10,038 False Positives**, with an average fine amount of **$243.90**
- **~$2.45 million in potential missed revenue**, if these tickets were deprioritized based on the model's prediction alone

This framing is the basis for treating the model as a **decision-support tool rather than a fully automated filter** — particularly for high-fine-value tickets, where the cost of a False Positive is highest.

## Deployment

The final model was serialized with `joblib` (model, feature list, and decision threshold saved separately) and wrapped in a Streamlit application (`app.py`) that:

- Accepts ticket details through numeric inputs, checkboxes, and dropdowns
- Dynamically builds its categorical dropdown options from the saved feature list, so the app always stays in sync with whatever categories the model was actually trained on
- Returns a compliance probability and a compliant / high-risk classification based on the tuned threshold

## Repository Structure

```
├── detroit_blight_project.ipynb   # full pipeline: cleaning, feature engineering, modeling, evaluation
├── app.py                         # Streamlit prediction app
├── blight_compliance_model.pkl    # trained XGBoost model
├── model_features.pkl             # feature column order used at training time
├── model_threshold.pkl            # tuned decision threshold (0.70)
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn xgboost lightgbm shap statsmodels imbalanced-learn streamlit joblib

streamlit run app.py
```

## Tech Stack

Python, pandas, scikit-learn, XGBoost, LightGBM, SHAP, statsmodels (VIF), imbalanced-learn (SMOTE), Streamlit, joblib.

## Key Takeaways

- Feature engineering and leakage removal drove far more of the model's quality than algorithm choice — three different model families converged to within 0.003 ROC-AUC of each other.
- A feature with strong SHAP importance is not automatically a genuine predictor — `Late Fee` looked like the top driver of compliance but was actually a circular signal, and removing it did not hurt performance.
- Class weighting outperformed SMOTE for this feature space, which is worth testing empirically rather than assuming SMOTE is always the default answer to class imbalance.
- Framing model errors in business terms (a $2.45M revenue-impact estimate from False Positives) turns a technical classification exercise into something a non-technical stakeholder can act on.

## Limitations & Future Work

- The model has not been tested against a true out-of-time holdout beyond 2026; performance on further future years is unverified.
- Hyperparameter tuning (grid/Bayesian search) was not exhaustively performed — default-to-lightly-tuned parameters were used throughout, given that model choice was already shown to matter less than feature quality.
- The Streamlit app is a local prototype; a production version would wrap the model behind a REST API (FastAPI), containerize it, and add basic prediction logging and drift monitoring.
