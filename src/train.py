import mlflow
import mlflow.catboost
import joblib

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.data import load_data, clean_data
from src.features import prepare_features


def train_model(data_path: str):

    # ------------------ Load & Clean ------------------
    df = load_data(data_path)
    df = clean_data(df)

    # ------------------ Features ------------------
    X, y, cat_cols = prepare_features(df)

    # ------------------ Split ------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------ Model ------------------
    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=0
    )

    # ------------------ Training ------------------
    with mlflow.start_run():

        model.fit(
            X_train,
            y_train,
            cat_features=cat_cols,
            eval_set=(X_test, y_test),
            verbose=0
        )

        # ------------------ Evaluation ------------------
        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)

        mlflow.log_metric("roc_auc", auc)
        mlflow.catboost.log_model(model, "model")

        # ------------------ Save ------------------
        joblib.dump(model, "models/catboost_model.pkl")

        print(f"AUC: {auc:.4f}")


if __name__ == "__main__":
    train_model("data/cleaned_telco.csv")
