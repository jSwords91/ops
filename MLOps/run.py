from src.dataLoader import DataLoader
from src.featureEngineer import FeatureEngineer
from src.modelTrainer import ModelTrainer
from src.modelEvaluator import ModelEvaluator
from src.modelInference import ModelInference
from src.utils import load_config

def main():
    config = load_config('config/config.yaml')

    # Load data
    data_loader = DataLoader(file_path=config['data']['path'])
    data = data_loader.load_data()

    # Feature engineering
    feature_engineer = FeatureEngineer()
    data = feature_engineer.transform(data)

    # Train model
    model_trainer = ModelTrainer(config=config)
    model_trainer.train(data)

    # Evaluate model
    model_evaluator = ModelEvaluator(config=config)
    report = model_evaluator.evaluate(data)
    print(report)

    # Perform inference
    model_inference = ModelInference(config=config)
    predictions = model_inference.predict(data.drop('Outcome', axis=1))
    print(predictions)

if __name__ == "__main__":
    main()
