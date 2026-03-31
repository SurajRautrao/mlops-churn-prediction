'''
import shap
import joblib
import pandas as pd

# Load trained pipeline
model = joblib.load("model/pipeline.pkl")

def explain_prediction(data: dict):
    df = pd.DataFrame([data])

    # Extract model and preprocessing
    preprocessor = model.named_steps["preprocessing"]
    classifier = model.named_steps["model"]

    # Transform input
    X_transformed = preprocessor.transformer(df)

    # SHAP explainer
    explainer = shap.TreeExplainer(classifier)

    shap_values = explainer.shap_values(X_transformed)

    return shap_values
'''
import shap
import joblib
import pandas as pd
from catboost import Pool

# Load model
model = joblib.load("models/catboost_model.pkl")

def explain_prediction(data: dict):
    df = pd.DataFrame([data])

    # Identify categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Ensure correct types
    for col in cat_cols:
        df[col] = df[col].astype(str)

    # Create Pool
    pool = Pool(df, cat_features=cat_cols)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pool)

    # Handle output
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
    else:
        shap_vals = shap_values[0]

    shap_vals = shap_vals.flatten()

    return df.columns, shap_vals
