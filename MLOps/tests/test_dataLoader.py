import unittest
import pandas as pd
from src.dataLoader import DataLoader

class TestDataLoader(unittest.TestCase):

    def test_load_data(self):
        data_loader = DataLoader(file_path="data/raw/diabetes.csv")
        data = data_loader.load_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)

if __name__ == '__main__':
    unittest.main()
