import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from src.utils import get_logger

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def train(self, data: pd.DataFrame):
        try:
            X = data.drop('Outcome', axis=1)
            y = data['Outcome']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config['training']['test_size'], random_state=self.config['training']['random_state'])

            model = RandomForestClassifier(n_estimators=self.config['training']['n_estimators'], max_depth=self.config['training']['max_depth'])
            model.fit(X_train, y_train)

            os.makedirs(os.path.dirname(self.config['model']['output_path']), exist_ok=True)
            joblib.dump(model, self.config['model']['output_path'])
            logger.info("Model training completed and saved.")
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            raise
