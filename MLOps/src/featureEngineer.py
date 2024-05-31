import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)

class FeatureEngineer:
    def __init__(self):
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data['BMI'] = data['BMI'].fillna(data['BMI'].mean())
            logger.info("Feature engineering completed.")
            return data
        except Exception as e:
            logger.error(f"Failed to engineer features: {e}")
            raise
