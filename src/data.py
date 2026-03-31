import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    return df

if __name__ == "__main__":
    df = load_data("data/telco.csv")
    df = clean_data(df)
    df.to_csv("data/cleaned_telco.csv", index=False)
