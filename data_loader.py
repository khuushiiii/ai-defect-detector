import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV and return a DataFrame."""
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows from '{path}'")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise