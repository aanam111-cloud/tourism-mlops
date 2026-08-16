#!/usr/bin/env python3
"""
Model Training & Experiment Tracking (MLflow)
- Loads train/test splits from the previous job
- Defines model(s) + hyperparameter grid and tunes them
- Logs parameters & metrics to MLflow
- Evaluates the best model
- Saves the best model into tourism_project/deployment/
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEPLOY_DIR = PROJECT_ROOT / "deployment"


def load_splits(data_dir: Path):
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Train/test not found in {data_dir}. Run prep.py first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    target = "ProdTaken"
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]
    return X_train, X_test, y_train, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )


def get_model_configs():
    configs = {
        "DecisionTree": (
            DecisionTreeClassifier(random_state=42),
            {
                "clf__max_depth": [3, 5, 7, 10, None],
                "clf__min_samples_split": [2, 5, 10],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__criterion": ["gini", "entropy"],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "clf__n_estimators": [100, 200, 300],
                "clf__max_depth": [5, 10, 15, None],
                "clf__min_samples_split": [2, 5, 10],
                "clf__min_samples_leaf": [1, 2],
                "clf__max_features": ["sqrt", "log2"],
            },
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.01, 0.05, 0.1],
                "clf__max_depth": [3, 5, 7],
                "clf__subsample": [0.8, 1.0],
            },
        ),
        "AdaBoost": (
            AdaBoostClassifier(random_state=42),
            {
                "clf__n_estimators": [50, 100, 200],
                "clf__learning_rate": [0.01, 0.1, 0.5, 1.0],
            },
        ),
        "Bagging": (
            BaggingClassifier(random_state=42, n_jobs=-1),
            {
                "clf__n_estimators": [50, 100, 200],
                "clf__max_samples": [0.5, 0.8, 1.0],
                "clf__max_features": [0.5, 0.8, 1.0],
            },
        ),
    }
    if HAS_XGB:
        configs["XGBoost"] = (
            XGBClassifier(
                random_state=42, eval_metric="logloss",
                use_label_encoder=False, n_jobs=-1
            ),
            {
                "clf__n_estimators": [100, 200, 300],
                "clf__max_depth": [3, 5, 7],
                "clf__learning_rate": [0.01, 0.05, 0.1],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.8, 1.0],
            },
        )
    return configs


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    except Exception:
        pass
    return metrics


def train():
    print("=" * 70)
    print("MODEL TRAINING + MLflow EXPERIMENT TRACKING")
    print("=" * 70)

    X_train, X_test, y_train, y_test = load_splits(DATA_DIR)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    preprocessor = build_preprocessor(X_train)
    configs = get_model_configs()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("tourism_wellness_package")

    best_f1 = -1.0
    best_name = None
    best_model = None
    best_params = None
    all_results = {}

    for name, (estimator, param_grid) in configs.items():
        print(f"\n>>> Tuning {name} ...")
        pipe = Pipeline([("preprocessor", preprocessor), ("clf", estimator)])

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grid,
            n_iter=12,
            scoring="f1",
            cv=cv,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)

        model = search.best_estimator_
        params = search.best_params_
        cv_f1 = float(search.best_score_)
        test_metrics = evaluate(model, X_test, y_test)

        with mlflow.start_run(run_name=name):
            mlflow.log_params({k.replace("clf__", ""): v for k, v in params.items()})
            mlflow.log_param("model_type", name)
            mlflow.log_metric("cv_f1", cv_f1)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            # Log model with trusted types for XGBoost
            log_kwargs = {"artifact_path": "model"}
            if name == "XGBoost":
                log_kwargs["skops_trusted_types"] = [
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier"
                ]
            mlflow.sklearn.log_model(model, **log_kwargs)

            print(f"  CV F1 = {cv_f1:.4f} | Test F1 = {test_metrics['f1']:.4f}")
            print(f"  Best params: {params}")

        all_results[name] = {
            "cv_f1": cv_f1,
            "test_metrics": test_metrics,
            "best_params": params,
        }

        if test_metrics["f1"] > best_f1:
            best_f1 = test_metrics["f1"]
            best_name = name
            best_model = model
            best_params = params

    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_name}  (Test F1 = {best_f1:.4f})")
    print("=" * 70)
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save best model so the workflow can commit it
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    model_path = DEPLOY_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"\nBest model saved → {model_path}")

    meta = {
        "best_model_name": best_name,
        "best_params": {k: str(v) for k, v in best_params.items()},
        "best_test_f1": best_f1,
        "all_results": {
            k: {"cv_f1": v["cv_f1"], "test_f1": v["test_metrics"]["f1"]}
            for k, v in all_results.items()
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(DEPLOY_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved  → {DEPLOY_DIR / 'model_meta.json'}")

    print("\n--- Experiment Summary ---")
    for name, res in all_results.items():
        print(f"{name:20s} | CV F1: {res['cv_f1']:.4f} | Test F1: {res['test_metrics']['f1']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        DATA_DIR = Path(sys.argv[1])
    train()
