def prepare_features(df):
    df = df.copy()

    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(["Churn", "customerID"], axis=1)

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    return X, y, cat_cols


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("data/cleaned_telco.csv")

    X, y, cat_cols = prepare_features(df)

    X["Churn"] = y

    X.to_csv("data/features_telco.csv", index=False)
