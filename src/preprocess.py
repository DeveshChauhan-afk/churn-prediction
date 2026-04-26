import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    # Drop ID column (correct name)
    df.drop("CustomerID", axis=1, inplace=True)

    # Convert Total Charges (if needed)
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    # Fill numeric columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    # Fill categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    # Target variable
    y = df["Churn Value"]   # 0 / 1 already perfect

    # Drop target + leakage columns
    df.drop(["Zip Code", "Latitude", "Longitude"], axis=1, inplace=True)
    df.drop(["Churn Value", "Churn Label", "Churn Score", "CLTV", "Churn Reason"], axis=1, inplace=True)

    # One-hot encoding
    X = pd.get_dummies(df, drop_first=True)

    return X, y