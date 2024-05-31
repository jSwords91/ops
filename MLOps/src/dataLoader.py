import pandas as pd
from dataclasses import dataclass
from src.utils import get_logger

logger = get_logger(__name__)

@dataclass
class DataLoader:
    file_path: str

    def load_data(self) -> pd.DataFrame:
        try:
            data = pd.read_csv(self.file_path)
            logger.info("Data loaded successfully.")
            return data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
