"""
rent_pipeline.py
----------------
Reusable, function-based pipeline for the Berlin rent-price model.

This replaces the notebook's inline, copy-pasted blocks with tested functions,
and specifically fixes the bugs that caused "save model -> load model -> predict"
to break:

  1. The original notebook built ONE ColumnTransformer instance and reused the
     same mutable object across every Pipeline. `build_preprocessor()` below
     returns a brand-new, unfitted transformer every time it's called, so
     nothing is ever accidentally shared or half-fitted.

  2. There was no schema validation on the way in. If a caller's dict was
     missing a key (or a client sent JSON with a slightly different shape),
     sklearn raised a cryptic KeyError deep inside ColumnTransformer.
     `validate_listing()` checks the input against the exact schema the model
     was trained on and raises one clear, specific error message.

  3. Nothing recorded *which* library versions the model was trained with.
     `save_bundle()` stores that alongside the pipeline and `load_bundle()`
     warns you if the runtime environment doesn't match - the single most
     common real-world cause of "it worked in the notebook, breaks on load".

  4. `save_bundle()` immediately reloads what it just wrote and sanity-checks
     it, so a broken save is caught at save time, not three days later in
     production.

  5. Explicit `random_state` everywhere instead of relying on a single global
     `np.random.seed()` call and hoping cells run in the same order forever.
"""
from __future__ import annotations

import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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

# ---------------------------------------------------------------------------
# Schema - the single source of truth for every column the model touches.
# Change a column here and every function below picks it up automatically.
# ---------------------------------------------------------------------------
TARGET_COL = "cold_rent"
DROP_COLS = ["listing_url"]

NUMERIC_COLS = ["area", "rooms"]
FLOOR_COL = ["floor"]      # allowed to be missing at prediction time -> imputed
YEAR_COL = ["year_built"]  # allowed to be missing at prediction time -> imputed
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
# Columns a prediction request MUST provide.
REQUIRED_COLS = NUMERIC_COLS + BOOL_COLS + CATEGORICAL_COLS
# Columns that may legitimately be absent/None - the pipeline's own imputer handles them.
NULLABLE_COLS = FLOOR_COL + YEAR_COL


# ---------------------------------------------------------------------------
# Preprocessing / pipeline construction
# ---------------------------------------------------------------------------
def build_preprocessor(feature_cols=None) -> ColumnTransformer:
    if feature_cols is None:
        feature_cols = ALL_FEATURE_COLS

    numeric_cols = [c for c in NUMERIC_COLS if c in feature_cols]
    floor_cols = [c for c in FLOOR_COL if c in feature_cols]
    year_cols = [c for c in YEAR_COL if c in feature_cols]
    bool_cols = [c for c in BOOL_COLS if c in feature_cols]
    categorical_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]

    transformers = []

    if numeric_cols:
        transformers.append(
            ("num", StandardScaler(), numeric_cols)
        )

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
        transformers.append(
            ("bool", "passthrough", bool_cols)
        )

    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ), categorical_cols)
        )

    return ColumnTransformer(transformers=transformers)


def build_pipeline(model, feature_cols=None) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(feature_cols)),
        ("model", model),
    ])


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Pull human-readable feature names out of an already-FITTED pipeline.

    (Deriving these from some other, unrelated pipeline object - as the
    original notebook did - only works by accident. Always derive them from
    the specific fitted pipeline you actually care about.)
    """
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    return [n.split("__", 1)[1] if "__" in n else n for n in names]


# ---------------------------------------------------------------------------
# Data loading / splitting
# ---------------------------------------------------------------------------
def load_data(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = set(ALL_FEATURE_COLS + [TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    X = df[ALL_FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def make_cv(n_splits: int = 5, random_state: int = RANDOM_STATE) -> KFold:
    """One shared CV strategy so every model is compared on identical folds."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def evaluate_model(
    name: str,
    model,
    param_grid: dict,
    X_train, y_train, X_test, y_test,
    cv: Optional[KFold] = None,
    search: str = "grid",
    n_iter: int = 20,
    random_state: int = RANDOM_STATE,
    verbose: int = 0,
    feature_cols=None,
):
    """Tune, fit, and score one model with one call.

    Returns (metrics_dict, fitted_best_pipeline). Append metrics_dict to a
    list as you try more models and you have a ready-made comparison table
    (and the input for a radar chart) instead of scattered print statements.
    """
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


# ---------------------------------------------------------------------------
# Save / load / predict - this is the part that was breaking.
# ---------------------------------------------------------------------------
def save_bundle(pipeline: Pipeline, path="model_bundle.pkl", extra_metadata: Optional[dict] = None) -> Path:
    """Save the fitted pipeline TOGETHER with the schema and library versions
    it depends on, then immediately reload it to catch a broken save before
    you ever hand the file to anyone else."""
    bundle = {
        "pipeline": pipeline,
        "feature_columns": ALL_FEATURE_COLS,
        "required_columns": REQUIRED_COLS,
        "nullable_columns": NULLABLE_COLS,
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

    # Self-check: reload right now, in this process, and confirm the schema
    # round-tripped correctly. Fail loudly here rather than silently later.
    reloaded = joblib.load(path)
    if set(reloaded["feature_columns"]) != set(ALL_FEATURE_COLS):
        raise RuntimeError("Saved bundle's schema does not match the current schema - save aborted.")
    try:
        _smoke_test_predict(reloaded)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Saved bundle failed a round-trip prediction smoke test: {exc}") from exc

    return path


def load_bundle(path="model_bundle.pkl") -> dict:
    bundle = joblib.load(path)
    current = sklearn.__version__
    saved = bundle.get("sklearn_version")
    if saved and saved != current:
        warnings.warn(
            f"Model was saved with scikit-learn {saved} but this environment has "
            f"{current}. Predictions will usually still work, but pin your serving "
            "environment to match the training environment to be safe.",
            stacklevel=2,
        )
    return bundle


def validate_listing(raw: dict, bundle: dict) -> pd.DataFrame:
    """Turn a raw dict (e.g. an API payload) into the one-row DataFrame the
    pipeline expects, or raise ONE clear error instead of a cryptic sklearn
    KeyError several layers down."""
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
            row[col] = np.nan  # the pipeline's own imputer will fill this in
        else:
            raise ValueError(f"Missing required field: {col!r}")

    return pd.DataFrame([row], columns=all_cols)


def predict_listing(raw: dict, bundle: dict) -> float:
    df_row = validate_listing(raw, bundle)
    prediction = bundle["pipeline"].predict(df_row)[0]
    return float(prediction)


def _smoke_test_predict(bundle: dict) -> None:
    """Build a harmless, schema-valid dummy row and run it through predict()
    purely to prove the loaded bundle can actually produce a prediction."""
    dummy = {c: False for c in BOOL_COLS}
    dummy.update({c: 0.0 for c in NUMERIC_COLS})
    dummy.update({c: "Unknown" for c in CATEGORICAL_COLS})
    predict_listing(dummy, bundle)
