import logging
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, name: str = "Model") -> dict:
    """
    Evaluate a model and return a metrics dict.
    Also logs a full classification report.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  round(accuracy_score(y_test,  y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test,    y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test,        y_pred, zero_division=0), 4),
    }

    logger.info(f"\n{name} Performance:")
    for k, v in metrics.items():
        logger.info(f"  {k.capitalize()}: {v}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")

    return metrics