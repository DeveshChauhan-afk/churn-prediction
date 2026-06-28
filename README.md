# 📊 Customer Churn Prediction & Business Insights

An end-to-end Machine Learning project that predicts customer churn using classification models and provides actionable business insights through a Flask API. The project covers the complete ML lifecycle, including data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, and deployment.

---

## 🚀 Project Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses. This project aims to identify customers who are likely to leave a telecom service by analyzing customer demographics, service usage, and billing information.

The final system predicts customer churn probability and categorizes customers into risk levels to support proactive retention strategies.

---

## 🎯 Objectives

- Analyze customer behavior to identify churn patterns.
- Build machine learning models to predict customer churn.
- Compare multiple classification algorithms.
- Identify the most influential factors contributing to churn.
- Deploy the trained model using a Flask REST API.
- Generate actionable business recommendations based on prediction results.

---

## 📂 Dataset

**Dataset Used:** IBM Telco Customer Churn Dataset

The dataset contains customer information including:

- Customer demographics
- Service subscriptions
- Billing information
- Contract details
- Payment methods
- Customer tenure
- Churn status

### Target Variable

- **Churn Value**
  - `0` → Customer stays
  - `1` → Customer churns

---

# 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Flask
- Joblib
- Jupyter Notebook

---

# 📊 Exploratory Data Analysis

The project includes extensive EDA to understand customer behavior.

Key analyses include:

- Customer churn distribution
- Churn vs Tenure
- Churn vs Contract Type
- Churn vs Payment Method
- Churn vs Monthly Charges
- Feature Importance Analysis

---

# ⚙️ Data Preprocessing

The preprocessing pipeline includes:

- Handling missing values
- Converting data types
- Removing unnecessary columns
- Removing leakage features
- One-hot encoding categorical variables
- Feature selection

### Removed Leakage Features

The following columns were excluded because they contain post-outcome information:

- Churn Label
- Churn Score
- Churn Reason
- CLTV

Additionally, geographic features such as Latitude, Longitude, and Zip Code were removed after identifying them as non-causal predictors that reduced model interpretability.

---

# 🤖 Machine Learning Models

Two classification models were trained:

### Logistic Regression
- Used as the baseline model

### Random Forest Classifier
- Selected as the final model due to superior performance

---

# 📈 Model Performance

| Model | ROC-AUC |
|--------|---------|
| Logistic Regression | ~0.71 |
| Random Forest | ~0.85 |

### Random Forest Evaluation

- Accuracy: ~81%
- Precision (Churn): ~72%
- Recall (Churn): ~53%
- Improved Recall (Threshold Tuning): ~70%

---

# 🔥 Key Features Influencing Churn

The model identified the following as the strongest predictors:

- Total Charges
- Tenure Months
- Monthly Charges
- Internet Service
- Contract Type
- Payment Method
- Online Security
- Tech Support

---

# 🌐 Flask API

The trained model is deployed using Flask.

### Endpoint

```
POST /predict
```

### Sample Request

```json
{
  "Gender": "Male",
  "Senior Citizen": "Yes",
  "Partner": "No",
  "Dependents": "No",
  "Tenure Months": 1,
  "Phone Service": "Yes",
  "Multiple Lines": "Yes",
  "Internet Service": "Fiber optic",
  "Online Security": "No",
  "Online Backup": "No",
  "Device Protection": "No",
  "Tech Support": "No",
  "Streaming TV": "Yes",
  "Streaming Movies": "Yes",
  "Contract": "Month-to-month",
  "Paperless Billing": "Yes",
  "Payment Method": "Electronic check",
  "Monthly Charges": 110,
  "Total Charges": 110
}
```

### Sample Response

```json
{
    "churn": 0,
    "churn_probability": 0.47,
    "risk_level": "Medium Risk",
    "recommended_action": "Offer discount"
}
```

---

# 📁 Project Structure

```
Customer-Churn-Prediction/
│
├── data/
│   └── churn.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── model/
│   ├── model.pkl
│   └── columns.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 🚀 How to Run

### Clone Repository

```bash
git clone <repository-url>
cd Customer-Churn-Prediction
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python src/train.py
```

### Run Flask API

```bash
python app.py
```

---

# 📌 Challenges Faced

During development, several real-world machine learning challenges were encountered:

- Dataset schema mismatch
- Missing value handling
- Data type inconsistencies
- Feature leakage
- Feature alignment between training and inference
- Model serialization
- API schema consistency
- Removal of non-causal geographic features
- Threshold tuning for improving churn detection

These issues helped build a more robust and production-ready ML pipeline.

---

# 💼 Business Impact

This system enables organizations to:

- Identify customers at risk of churning
- Prioritize retention campaigns
- Improve customer lifetime value
- Reduce revenue loss
- Support data-driven business decisions

---

# 🔮 Future Improvements

- Hyperparameter tuning using GridSearchCV
- XGBoost and LightGBM implementation
- SHAP explainability
- Probability calibration
- Interactive dashboard using Streamlit
- Docker deployment
- Cloud deployment on Render or AWS
- CI/CD integration

---

# 👨‍💻 Author

Developed as an end-to-end Machine Learning project demonstrating data preprocessing, model development, evaluation, deployment, and business insight generation.

---

## ⭐ If you found this project useful, consider giving it a star!
