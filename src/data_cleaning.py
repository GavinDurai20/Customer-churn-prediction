import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Handle missing values (FIXED)
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Drop customer ID
df.drop("customerID", axis=1, inplace=True)

# Encode categorical variables
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# Save cleaned data
df.to_csv("data/cleaned_churn_data.csv", index=False)

print("✅ Data cleaning completed successfully.")
