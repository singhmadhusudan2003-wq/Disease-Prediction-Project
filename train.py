# train.py
# Main script to train and compare multiple models for heart disease
# prediction. Run with: python train.py

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('src')
from preprocessing import clean_and_prepare
from train_utils import (
    train_model,
    evaluate_model,
    plot_roc_curve,
    plot_confusion_matrix,
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# xgboost is an optional dependency; the pipeline still works without it.
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("xgboost not installed, skipping it. Run: pip install xgboost")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = Path('data/heart.csv')
MODELS_DIR = Path('models')
OUTPUTS_DIR = Path('outputs')

MODEL_PATH = MODELS_DIR / 'best_model.pkl'
SCALER_PATH = MODELS_DIR / 'scaler.pkl'
FEATURE_NAMES_PATH = MODELS_DIR / 'feature_names.pkl'

COMPARISON_CSV_PATH = OUTPUTS_DIR / 'model_comparison.csv'
ROC_CURVE_PATH = OUTPUTS_DIR / 'roc_curve.png'
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / 'confusion_matrix.png'
FEATURE_IMPORTANCE_PATH = OUTPUTS_DIR / 'feature_importance.png'


def ensure_output_dirs():
    """Create the models/ and outputs/ directories if they don't exist yet."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def get_models():
    """Return a dict mapping model_name -> unfitted estimator instance."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': SVC(probability=True, kernel='rbf'),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, random_state=42
        ),
    }

    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            use_label_encoder=False, eval_metric='logloss', random_state=42
        )

    return models


def train_and_evaluate_models(models, X_train, y_train, X_test, y_test):
    """
    Train and evaluate each model in `models`.

    Any model that raises an exception during training/evaluation is
    skipped (with a warning printed) instead of crashing the whole run,
    so a single misbehaving model doesn't block the others.

    Returns
    -------
    tuple
        (trained_models, all_results) containing only the models that
        trained successfully.

    Raises
    ------
    RuntimeError
        If every model fails to train.
    """
    trained_models = {}
    all_results = []

    for name, model in models.items():
        try:
            trained_model = train_model(model, X_train, y_train)
            results = evaluate_model(
                trained_model, X_test, y_test, model_name=name
            )
        except Exception as exc:
            print(
                f"Warning: '{name}' failed to train/evaluate and will be "
                f"skipped. Reason: {exc}"
            )
            continue

        trained_models[name] = trained_model
        all_results.append(results)

    if not all_results:
        raise RuntimeError(
            "All models failed to train. Check the dataset and model "
            "configuration."
        )

    return trained_models, all_results


def build_comparison_table(all_results):
    """Build the model comparison table, sorted by ROC AUC (best first)."""
    comparison_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Accuracy': round(r['accuracy'], 4),
            'Precision': round(r['precision'], 4),
            'Recall': round(r['recall'], 4),
            'F1 Score': round(r['f1_score'], 4),
            'ROC AUC': round(r['roc_auc'], 4),
        }
        for r in all_results
    ])

    return comparison_df.sort_values(
        by='ROC AUC', ascending=False
    ).reset_index(drop=True)


def plot_feature_importance(rf_model, feature_names, save_path):
    """Plot and save a horizontal bar chart of Random Forest feature
    importances. Only applicable to tree-based models."""
    importances = rf_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(8, 6))
    plt.barh(feat_imp_df['feature'], feat_imp_df['importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance - Random Forest')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    try:
        plt.savefig(save_path)
        print(f"Feature importance plot saved to {save_path}")
    except OSError as exc:
        print(f"Warning: could not save feature importance plot: {exc}")
    finally:
        plt.close()


def main():
    ensure_output_dirs()

    print("Loading and preparing data...")
    try:
        X_train, X_test, y_train, y_test, scaler, feature_names = (
            clean_and_prepare(str(DATA_PATH))
        )
    except (FileNotFoundError, KeyError) as exc:
        print(f"Error: could not prepare the dataset. {exc}")
        sys.exit(1)

    models = get_models()
    trained_models, all_results = train_and_evaluate_models(
        models, X_train, y_train, X_test, y_test
    )

    comparison_df = build_comparison_table(all_results)

    print("\n===== MODEL COMPARISON =====")
    print(comparison_df)

    try:
        comparison_df.to_csv(COMPARISON_CSV_PATH, index=False)
    except OSError as exc:
        print(f"Warning: could not save model comparison CSV: {exc}")

    # Picking the best model based on ROC AUC
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    best_results = next(
        r for r in all_results if r['model_name'] == best_model_name
    )

    print(f"\nBest model is: {best_model_name}")

    # Saving best model and scaler for later use in predict.py / app.py
    try:
        joblib.dump(best_model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(list(feature_names), FEATURE_NAMES_PATH)
        print(f"Saved best model to {MODEL_PATH}")
        print(f"Saved scaler to {SCALER_PATH}")
        print(f"Saved feature names to {FEATURE_NAMES_PATH}")
    except OSError as exc:
        print(f"Error: could not save model artifacts. {exc}")
        sys.exit(1)

    # Plotting ROC curve and confusion matrix for the best model
    try:
        plot_roc_curve(
            y_test,
            best_results['y_prob'],
            best_model_name,
            save_path=str(ROC_CURVE_PATH),
        )
    except OSError as exc:
        print(f"Warning: could not save ROC curve plot: {exc}")

    try:
        plot_confusion_matrix(
            best_results['confusion_matrix'],
            best_model_name,
            save_path=str(CONFUSION_MATRIX_PATH),
        )
    except OSError as exc:
        print(f"Warning: could not save confusion matrix plot: {exc}")

    # Feature importance only works for tree-based models (Random Forest here)
    if 'Random Forest' in trained_models:
        plot_feature_importance(
            trained_models['Random Forest'],
            feature_names,
            save_path=str(FEATURE_IMPORTANCE_PATH),
        )

    print("\nDone! Check the outputs/ folder for saved plots and comparison CSV.")


if __name__ == '__main__':
    main()
