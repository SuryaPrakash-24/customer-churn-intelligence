# 📉 Customer Churn Intelligence Dashboard

An interactive Streamlit dashboard for analysing customer churn, identifying at-risk accounts, and evaluating model performance. Built with a class-balanced Random Forest classifier.

---

## How it works

1. Customer records are feature-engineered from the raw dataset (tenure, onboard date features, state extraction, account manager flag)
2. A Random Forest classifier is trained with class balancing on first run and cached in memory
3. The optimal probability threshold is tuned on a validation set, maximising recall while maintaining acceptable precision
4. All existing customers and any new customers (from `new_customers_1.csv`) are scored and ranked by churn probability
5. The dashboard serves three interactive tabs — overview, an at-risk customer list, and model/threshold diagnostics

---

## Tech stack

| Layer | Technology |
|---|---|
| Data processing | Python, pandas, NumPy |
| ML pipeline | scikit-learn (Random Forest, ColumnTransformer, Pipeline, calibration) |
| Dashboard | Streamlit |
| Visualisations | Plotly |

---

## Project structure

```
customer-churn-intelligence/
├── app/
│   └── app.py                  # Main Streamlit application
├── app/data/                   # Place customer_churn.csv and new_customers_1.csv here (git-ignored)
├── assets/                     # Screenshots and other static assets
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Getting started

### Prerequisites

- Python 3.9+
- The customer churn dataset CSVs (see [Dataset](#dataset) section below)

### 1. Clone the repository

```bash
git clone https://github.com/SuryaPrakash-24/customer-churn-intelligence.git
cd customer-churn-intelligence
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the datasets

Place both CSV files inside `app/` (alongside `app.py`):

```
app/
├── app.py
├── customer_churn.csv        ← training data
└── new_customers_1.csv       ← new customers to score (optional)
```

**`customer_churn.csv` required columns:**

| Column | Description |
|---|---|
| `Names` | Customer name |
| `Age` | Customer age |
| `Total_Purchase` | Total spend |
| `Account_Manager` | 1 = assigned, 0 = not assigned |
| `Years` | Tenure in years |
| `Num_Sites` | Number of product sites |
| `Onboard_date` | Account creation date |
| `Location` | Address string (state code is extracted automatically) |
| `Company` | Company name |
| `Churn` | Target label — 1 = churned, 0 = retained |

**`new_customers_1.csv`** uses the same columns minus `Churn`. If this file is absent, the new-customer scoring panel is hidden gracefully.

### 4. Run the dashboard

```bash
cd app
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

The model trains on first load (~15–30 seconds) and is cached for the session — subsequent interactions are instant.

---

## Dashboard tabs

| Tab | What it shows |
|---|---|
| 📊 Churn Overview | Churn rate, class balance, breakdowns by account manager / state / tenure, feature distributions split by churn class, permutation importance chart |
| 🚨 At-Risk Customers | Existing customers ranked by churn probability; searchable by name or company; new-customer scoring from `new_customers_1.csv`; adjustable probability threshold |
| 🎯 Threshold & Performance | Precision/Recall/F1 vs threshold curve, ROC curve, Precision-Recall curve, calibration curve, confusion matrix, test-set summary metrics |

---

## Dataset

This project was built using a synthetic B2B customer churn dataset. A structurally compatible dataset can be sourced from [Kaggle — Customer Churn Dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset).

---

## Roadmap

- [ ] Deploy to Streamlit Community Cloud
- [ ] Add SHAP-based individual customer explanations
- [ ] Integrate with CRM export (Salesforce / HubSpot CSV)
- [ ] Add model retraining trigger when drift is detected
- [ ] Email alerting for newly flagged high-risk accounts

---

## License

MIT
