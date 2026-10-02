import mlflow
import mlflow.catboost
import joblib
import pandas as pd

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
    # 60/20/20 train/val/test, stratified on churn. The val set drives
    # CatBoost's best-iteration selection, so the test set stays unseen.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=42
    )

    # ------------------ Model ------------------
    params = dict(
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=0
    )
    model = CatBoostClassifier(iterations=300, **params)

    # ------------------ Training ------------------
    with mlflow.start_run():

        model.fit(
            X_train,
            y_train,
            cat_features=cat_cols,
            eval_set=(X_val, y_val),
            verbose=0
        )

        best_iteration = model.get_best_iteration()

        # ------------------ Final Model ------------------
        # Refit on train + val with the selected iteration count
        final_model = CatBoostClassifier(iterations=best_iteration + 1, **params)
        final_model.fit(
            pd.concat([X_train, X_val]),
            pd.concat([y_train, y_val]),
            cat_features=cat_cols,
            verbose=0
        )

        # ------------------ Evaluation ------------------
        preds = final_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)

        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("val_roc_auc", model.get_best_score()["validation"]["AUC"])
        mlflow.log_metric("best_iteration", best_iteration)
        mlflow.catboost.log_model(final_model, "model")

        # ------------------ Save ------------------
        joblib.dump(final_model, "models/catboost_model.pkl")

        print(f"AUC: {auc:.4f}")


if __name__ == "__main__":
    train_model("data/telco.csv")
