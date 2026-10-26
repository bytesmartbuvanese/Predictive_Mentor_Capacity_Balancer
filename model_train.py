import json
import joblib
import warnings
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =====================================================
# CONFIG
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "engineered_data" / "student_model_training.csv"

MODEL_DIR = BASE_DIR / "models"

TARGET = "is_at_risk"

RANDOM_STATE = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

print("=" * 60)
print("Loading dataset...")
print("=" * 60)

if not DATA_PATH.is_file():
    raise FileNotFoundError(
        f"Training dataset not found: {DATA_PATH}. "
        "Run clean_oulad.py and feature_extraction.py first."
    )

df = pd.read_csv(DATA_PATH)

print(f"Dataset Shape: {df.shape}")

if df.empty:
    raise ValueError(f"Training dataset is empty: {DATA_PATH}")

if TARGET not in df.columns:
    raise ValueError(
        f"Training dataset must contain the target column '{TARGET}'."
    )

# =====================================================
# FEATURES / TARGET
# =====================================================

X = df.drop(columns=[TARGET])
y = df[TARGET]

if y.nunique() < 2:
    raise ValueError(
        f"Training requires at least two classes in '{TARGET}'."
    )

feature_names = list(X.columns)

print(f"Features: {len(feature_names)}")

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

X_train_values = X_train.to_numpy()
X_test_values = X_test.to_numpy()

print(f"Train Shape: {X_train.shape}")
print(f"Test Shape : {X_test.shape}")

# =====================================================
# MODELS
# =====================================================

models = {

    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss"
    )
}

best_model = None
best_name = None
best_f1 = -1

results = []

# =====================================================
# TRAIN & EVALUATE
# =====================================================

print("\n")
print("=" * 60)
print("MODEL TRAINING")
print("=" * 60)

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train_values, y_train)

    y_pred = model.predict(X_test_values)

    y_prob = model.predict_proba(X_test_values)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    result = {

        "model": name,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4)
    }

    results.append(result)

    print(result)

    if f1 > best_f1:

        best_f1 = f1
        best_name = name
        best_model = model

# =====================================================
# BEST MODEL
# =====================================================

print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"Best Model : {best_name}")
print(f"Best F1    : {best_f1:.4f}")

# =====================================================
# SAVE MODEL
# =====================================================

model_path = MODEL_DIR / "risk_model.pkl"

joblib.dump(
    best_model,
    model_path
)

print(f"Saved: {model_path}")

# =====================================================
# SAVE FEATURE LIST
# =====================================================

feature_path = MODEL_DIR / "risk_features.pkl"

joblib.dump(
    feature_names,
    feature_path
)

print(f"Saved: {feature_path}")

# =====================================================
# SAVE MODEL RESULTS
# =====================================================

metrics_path = MODEL_DIR / "model_metrics.json"

with metrics_path.open("w") as f:
    json.dump(
        results,
        f,
        indent=4
    )

print(f"Saved: {metrics_path}")

# =====================================================
# SAVE FEATURE IMPORTANCE
# =====================================================

if hasattr(best_model, "feature_importances_"):

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance":
            best_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    importance_path = MODEL_DIR / "feature_importance.csv"

    importance_df.to_csv(
        importance_path,
        index=False
    )

    print(f"Saved: {importance_path}")

    print("\nTop 20 Features\n")

    print(
        importance_df.head(20)
    )

print("\nTraining Completed Successfully")