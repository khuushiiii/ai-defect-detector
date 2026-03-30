import logging
import os
from modules.data_loader import load_data
from modules.preprocessing import preprocess_data
from modules.model import train_models
from modules.prediction import predict
from modules.evaluation import evaluate_model
from modules.visualization import (
    plot_defect_probability,
    plot_heatmap,
    plot_feature_importance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

os.makedirs("output", exist_ok=True)


def main():
    # 1. Load
    df = load_data("data/data.csv")

    # 2. Preprocess
    df = preprocess_data(df)

    # 3. Train
    rf, lr, X_test, y_test = train_models(df)

    # 4. Evaluate
    logger.info("=== Random Forest ===")
    evaluate_model(rf, X_test, y_test, "Random Forest")

    logger.info("=== Logistic Regression ===")
    evaluate_model(lr, X_test, y_test, "Logistic Regression")

    # 5. Predict on full dataset
    results = predict(df, rf)
    results = results.sort_values("Probability", ascending=False)

    # 6. Print top risk modules
    logger.info("\nTop Risk Modules:")
    print(results[["Module", "Probability", "Risk", "Severity"]].head(10).to_string(index=False))

    # 7. Save reports
    results.to_csv("output/prediction_report.csv",  index=False)
    results.to_excel("output/report.xlsx",           index=False)
    logger.info("Reports saved to output/")

    # 8. Visualisations
    plot_defect_probability(results)
    plot_heatmap(results)
    plot_feature_importance(rf, results)
    logger.info("Plots saved to output/")


if __name__ == "__main__":
    main()
