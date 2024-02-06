from src.customerChurn.config.configuration import ConfigurationManager
from src.logger import logger
from src.customerChurn.entity.config_entity import DataValidationConfig
from src.customerChurn.components.data_validation import DataValidation

class DataValidationPipeline:
    def __init__(self, config: DataValidationConfig=None):
        if config == None:
            config = ConfigurationManager().get_data_validation_config()
        self.data_validation_config = config
        
    def main(self):
        data_validation = DataValidation(config= self.data_validation_config)
        validation_status = data_validation.validate_all_columns()
        
        if not validation_status:
            raise Exception("InvalidDatasetError: Passed Invalid Dataset, doesn't contains colunms specified in schema ")
    

if __name__ == '__main__':
    try:
        obj = DataValidationPipeline()
        obj.main()
    except Exception as e:
        raise e