from src.customerChurn.entity.config_entity import PredictionConfig
from src.customerChurn.logger import logger
from src.utils.common import load_object

import pandas as pd

class Predict:
    def __init__(self, config: PredictionConfig):
        self.config = config
    
    def predict(self, data):
        df = self.get_data_as_dataframe(data)
        preprocessor = load_object(self.config.preprocessor_path)
        model = load_object(self.config.model_path)
        
        processed_data = preprocessor.transform(df)
        logger.info(f"received data is {data} \n processed data is {processed_data}")
        y_pred = model.predict(processed_data)
        logger.info(f"prediction of {data} is {y_pred}")
        
        return y_pred
    
    def get_data_as_dataframe(self, data):
        try:
            dataframe = {
                'CreditScore': data['CreditScore'],
                'Geography': data['Geography'],
                'Gender': data['Gender'],
                'Age': data['Age'],
                'Tenure': data['Tenure'],
                'Balance': data['Balance'],
                'NumOfProducts': data['NumOfProducts'],
                'HasCrCard': data['HasCrCard'],
                'IsActiveMember': data['IsActiveMember'],
                'EstimatedSalary': data['EstimatedSalary']
            }
            df = pd.DataFrame(dataframe, index=[0])
            return df
        except Exception as e:
            logger.info(f'Exception Occured in prediction pipeline, ERROR: {e}')
            
        
        