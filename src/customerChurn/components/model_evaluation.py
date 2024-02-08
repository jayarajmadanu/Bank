from src.logger import logger
from src.customerChurn.entity.config_entity import ModelEvaluationConfig

import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score
import numpy as np

from src.utils.common import load_object
 
class ModelEvaluation:
    def __init__(self, config:ModelEvaluationConfig):
        self.config = config
        
    def eval_metrics(self,actual, pred):
        accuracy = accuracy_score(y_true=actual, y_pred=pred)
        precision = precision_score(y_true=actual, y_pred=pred) 
        recall = recall_score(y_true=actual, y_pred=pred)
        f1 = f1_score(y_true=actual, y_pred=pred)
        return accuracy, precision, recall, f1
    
    def model_evaluation(self, best_params=None):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment('Bank-Customer-Churn-Prediction')
        mlflow.autolog()
        test_data_path = self.config.test_data_path
        model = load_object(self.config.model_path)
        
        test_df = pd.read_csv(test_data_path)
        X_test = test_df.iloc[:,0:13]
        y_test = test_df.iloc[:,13]
        
        with mlflow.start_run():
            predicted_values = model.predict(X_test)
            (accuracy, precision, recall, f1) = self.eval_metrics(y_test, predicted_values)
            mlflow.log_metrics({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            mlflow.log_params(model.get_params())
        logger.info("END")