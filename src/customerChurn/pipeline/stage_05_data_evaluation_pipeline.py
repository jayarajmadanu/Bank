from customerChurn.config.configuration import ConfigurationManager
from src.customerChurn.entity.config_entity import ModelEvaluationConfig
from src.customerChurn.components.model_evaluation import ModelEvaluation

class ModelEvaluationPipeline:
    def __init__(self, config:ModelEvaluationConfig=None):
        if config==None:
            config = ConfigurationManager().get_model_evaluation_config()
        self.config = config
        
    def main(self, best_params=None):
        model_evaluation = ModelEvaluation(config = self.config)
        model_evaluation.model_evaluation(best_params=best_params)
        
if __name__ == '__main__':
    try:
        obj = ModelEvaluationPipeline()
        obj.main()
    except Exception as e:
        raise e