import pandas as pd
import logging

logger = logging.getLogger(__name__)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the input DataFrame.
    - Fill missing numeric values with column medians.
    - Drop rows where the target column is missing.
    - Log a summary of changes.
    """
    original_len = len(df)

    # Drop rows missing the target
    if "Defect" in df.columns:
        df = df.dropna(subset=["Defect"])

    # Fill missing numeric features with median
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Filled {missing} missing values in '{col}' with median {median_val:.2f}")

    logger.info(f"Preprocessing complete: {original_len} → {len(df)} rows")
    return df.reset_index(drop=True)