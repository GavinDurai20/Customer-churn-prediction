# Customer Churn Prediction

## Project Overview
This project predicts whether a telecom customer will churn using machine learning.
The goal is to analyze customer behavior and identify the factors that influence churn.

## Dataset
- Telco Customer Churn Dataset (Kaggle)
- https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## Project Structure
Customer-Churn-Prediction/
├── data/
│ ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│ └── cleaned_churn_data.csv
├── src/
│ ├── data_analysis.py
│ └── churn_ml_visuals.py
├── reports/
│ ├── churn_distribution.png
│ ├── monthly_charges_vs_churn.png
│ ├── churn_by_contract.png
│ ├── confusion_matrix.png
│ └── feature_importance.png
├── requirements.txt
└── README.md


## Exploratory Data Analysis

### Customer Churn Distribution
![Churn Distribution](reports/churn_distribution.png)

### Monthly Charges vs Churn
![Monthly Charges vs Churn](reports/monthly_charges_vs_churn.png)

### Churn by Contract Type
![Churn by Contract](reports/churn_by_contract.png)

## Machine Learning Model

### Model Used
- Logistic Regression

### Model Results
- Accuracy: ~0.79
- The model performs well in identifying non-churn customers and reasonably predicts churn behavior.

### Confusion Matrix
![Confusion Matrix](reports/confusion_matrix.png)

### Feature Importance
![Feature Importance](reports/feature_importance.png)

## Key Insights
- Customers on month-to-month contracts show higher churn rates
- Higher monthly charges are associated with increased churn risk
- Long-term contracts reduce the likelihood of churn

## How to Run This Project
```bash
pip install -r requirements.txt
python src/data_analysis.py
python src/model_traning.py


Author

Gavin Durai
GitHub: https://github.com/GavinDurai20
LinkedIn: https://www.linkedin.com/in/gavin-durai/