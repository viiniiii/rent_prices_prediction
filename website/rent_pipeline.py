from __future__ import annotations

import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 16

# Schema Definitions
TARGET_COL = "cold_rent"
DROP_COLS = ["listing_url"]

NUMERIC_COLS = ["area", "rooms"]
FLOOR_COL = ["floor"]      # Optional at prediction time -> imputed
YEAR_COL = ["year_built"]  # Optional at prediction time -> imputed
BOOL_COLS = [
    "has_balkon", "has_terrasse", "has_garten",
    "elevator", "parking", "has_basement",
    "is_barrier-free", "has_built-in_kitchen",
    "has_bathtub", "has_shower",
]
CATEGORICAL_COLS = [
    "district", "energy_source", "heating_type",
    "flooring_type", "property_condition", "free_from",
]

ALL_FEATURE_COLS = NUMERIC_COLS + FLOOR_COL + YEAR_COL + BOOL_COLS + CATEGORICAL_COLS
REQUIRED_COLS = NUMERIC_COLS + BOOL_COLS + CATEGORICAL_COLS
NULLABLE_COLS = FLOOR_COL + YEAR_COL


# Preprocessing & Pipeline Construction
def build_preprocessor(feature_cols: Optional[List[str]] = None) -> ColumnTransformer:
    if feature_cols is None:
        feature_cols = ALL_FEATURE_COLS

    numeric_cols = [c for c in NUMERIC_COLS if c in feature_cols]
    floor_cols = [c for c in FLOOR_COL if c in feature_cols]
    year_cols = [c for c in YEAR_COL if c in feature_cols]
    bool_cols = [c for c in BOOL_COLS if c in feature_cols]
    categorical_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]

    transformers = []

    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))

    if floor_cols:
        transformers.append(
            ("floor", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), floor_cols)
        )

    if year_cols:
        transformers.append(
            ("year", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), year_cols)
        )

    if bool_cols:
        transformers.append(("bool", "passthrough", bool_cols))

    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ), categorical_cols)
        )

    return ColumnTransformer(transformers=transformers)


def build_pipeline(model: Any, feature_cols: Optional[List[str]] = None) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(feature_cols)),
        ("model", model),
    ])


def get_feature_names(pipeline: Pipeline) -> List[str]:
    """Extract human-readable feature names from a fitted pipeline."""
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    return [n.split("__", 1)[1] if "__" in n else n for n in names]


# Data Loading & Splitting
def load_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = set(ALL_FEATURE_COLS + [TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[ALL_FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Model Evaluation & CV
def make_cv(n_splits: int = 5, random_state: int = RANDOM_STATE) -> KFold:
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def evaluate_model(
    name: str,
    model: Any,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: Optional[KFold] = None,
    search: str = "grid",
    n_iter: int = 20,
    random_state: int = RANDOM_STATE,
    verbose: int = 0,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Pipeline]:
    """Tune, fit, and evaluate a single model pipeline."""
    cv = cv or make_cv(random_state=random_state)
    pipeline = build_pipeline(model, feature_cols)

    if search == "grid":
        searcher = GridSearchCV(
            pipeline, param_grid, cv=cv,
            scoring="neg_mean_squared_error", n_jobs=-1, verbose=verbose,
        )
    elif search == "random":
        searcher = RandomizedSearchCV(
            pipeline, param_grid, n_iter=n_iter, cv=cv,
            scoring="neg_mean_squared_error", n_jobs=-1,
            random_state=random_state, verbose=verbose,
        )
    else:
        raise ValueError("search must be 'grid' or 'random'")

    searcher.fit(X_train, y_train)
    best_pipeline = searcher.best_estimator_

    y_pred_train = best_pipeline.predict(X_train)
    y_pred_test = best_pipeline.predict(X_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    metrics = {
        "model": name,
        "best_params": searcher.best_params_,
        "cv_rmse": float(np.sqrt(-searcher.best_score_)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
        "overfit_gap_r2": float(train_r2 - test_r2),
    }
    return metrics, best_pipeline


# Serialization, Validation & Inference
def save_bundle(
    pipeline: Pipeline, path: str | Path = "model_bundle.pkl", extra_metadata: Optional[dict] = None
) -> Path:
    """Serialize a fitted pipeline alongside feature metadata and runtime environments."""
    preprocessor = pipeline.named_steps["preprocessor"]
    if not hasattr(preprocessor, "transformers_"):
        raise ValueError("save_bundle() requires a fitted pipeline instance.")

    # Dynamically extract features actually used by the fitted transformer
    pipeline_feature_columns = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        if isinstance(columns, str):
            columns = [columns]
        pipeline_feature_columns.extend(list(columns))

    bundle = {
        "pipeline": pipeline,
        "feature_columns": pipeline_feature_columns,
        "required_columns": [c for c in REQUIRED_COLS if c in pipeline_feature_columns],
        "nullable_columns": [c for c in NULLABLE_COLS if c in pipeline_feature_columns],
        "target_col": TARGET_COL,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "metadata": extra_metadata or {},
    }
    try:
        import xgboost
        bundle["xgboost_version"] = xgboost.__version__
    except ImportError:
        pass

    path = Path(path)
    joblib.dump(bundle, path)

    reloaded = joblib.load(path)
    if set(reloaded["feature_columns"]) != set(pipeline_feature_columns):
        raise RuntimeError("Saved bundle's schema does not match the fitted pipeline schema.")
    
    try:
        _smoke_test_predict(reloaded)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Saved bundle failed prediction smoke test: {exc}") from exc

    return path


def load_bundle(path: str | Path = "model_bundle.pkl") -> dict:
    bundle = joblib.load(path)
    current = sklearn.__version__
    saved = bundle.get("sklearn_version")
    if saved and saved != current:
        warnings.warn(
            f"Model was saved with scikit-learn {saved} but current environment has {current}. "
            "Pin serving environments to match training conditions.",
            stacklevel=2,
        )
    return bundle


def validate_listing(raw: dict, bundle: dict) -> pd.DataFrame:
    """Validate payload keys and construct a standard 1-row DataFrame."""
    required = bundle["required_columns"]
    nullable = bundle["nullable_columns"]
    all_cols = bundle["feature_columns"]

    missing_required = [c for c in required if c not in raw or raw[c] is None]
    if missing_required:
        raise ValueError(f"Missing required field(s): {missing_required}")

    row = {}
    for col in all_cols:
        if col in raw and raw[col] is not None:
            row[col] = raw[col]
        elif col in nullable:
            row[col] = np.nan 
        else:
            raise ValueError(f"Missing required field: {col!r}")

    return pd.DataFrame([row], columns=all_cols)


def predict_listing(raw: dict, bundle: dict) -> float:
    df_row = validate_listing(raw, bundle)
    prediction = bundle["pipeline"].predict(df_row)[0]
    return float(prediction)


def _smoke_test_predict(bundle: dict) -> None:
    dummy = {c: False for c in BOOL_COLS}
    dummy.update({c: 0.0 for c in NUMERIC_COLS})
    dummy.update({c: "Unknown" for c in CATEGORICAL_COLS})
    predict_listing(dummy, bundle)