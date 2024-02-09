from src.customerChurn.components.predict import Predict
from src.customerChurn.config.configuration import ConfigurationManager
from src.customerChurn.entity.config_entity import PredictionConfig

class PredictionPipeline:
    def __init__(self, config:PredictionConfig=None):
        if config==None:
            config = ConfigurationManager().get_prediction_config()
        self.config = config
        
    def main(self, data=None):
        model_evaluation = Predict(config = self.config)
        y_pred = model_evaluation.predict(data=data)
        return y_pred
        
