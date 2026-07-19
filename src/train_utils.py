# train_utils.py
# Helper functions used by train.py to train and evaluate models, and to
# plot evaluation results.

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)


def train_model(model, X_train, y_train):
    """Fit `model` on the training data and return the fitted model."""
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name='model'):
    """
    Evaluate a fitted model on the test set and print a summary.

    Uses predict_proba() for probability scores where available (needed
    for ROC AUC); falls back to decision_function() for models such as an
    SVC without probability=True.

    Returns
    -------
    dict
        Keys: model_name, accuracy, precision, recall, f1_score, roc_auc,
        confusion_matrix, classification_report, y_prob.
    """
    y_pred = model.predict(X_test)

    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\n----- {model_name} results -----")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

    results = {
        'model_name': model_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_prob': y_prob
    }

    return results


def plot_roc_curve(y_test, y_prob, model_name, save_path=None):
    """Plot the ROC curve for one model and optionally save it to disk."""
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=model_name)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # random guess line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")

    plt.close()


def plot_confusion_matrix(cm, model_name, save_path=None):
    """Plot a confusion matrix heatmap for one model and optionally save it."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")

    plt.close()
