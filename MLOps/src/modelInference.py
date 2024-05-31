import pandas as pd
import joblib
from src.utils import get_logger

logger = get_logger(__name__)

class ModelInference:
    def __init__(self, config):
        self.config = config

    def predict(self, data: pd.DataFrame) -> pd.Series:
        try:
            model = joblib.load(self.config['model']['output_path'])
            predictions = model.predict(data)
            logger.info("Model inference completed.")
            return predictions
        except Exception as e:
            logger.error(f"Failed to perform inference: {e}")
            raise
