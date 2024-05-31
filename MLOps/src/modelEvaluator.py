import pandas as pd
from sklearn.metrics import classification_report
import joblib
from src.utils import get_logger

logger = get_logger(__name__)

class ModelEvaluator:
    def __init__(self, config):
        self.config = config

    def evaluate(self, data: pd.DataFrame):
        try:
            X = data.drop('Outcome', axis=1)
            y = data['Outcome']
            model = joblib.load(self.config['model']['output_path'])

            y_pred = model.predict(X)
            report = classification_report(y, y_pred)
            logger.info("Model evaluation completed.")
            return report
        except Exception as e:
            logger.error(f"Failed to evaluate model: {e}")
            raise
