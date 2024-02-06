from src.customerChurn.entity.config_entity import DataIngestionConfig
from src.customerChurn.logger import logger
import os
from src.utils.common import create_directories

import pandas as pd

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def downloadDataSet(self):
        if not os.path.exists(self.config.local_data_file_path):
            df = pd.read_csv(self.config.source_url)
            logger.info(f'Reading data from {self.config.source_url}')
            
            logger.info(f'Creating directory {self.config.root_dir}')
            create_directories([self.config.root_dir])
            
            logger.info(f'Writing data into {self.config.local_data_file_path}')
            df.to_csv(self.config.local_data_file_path, index=False, header=True)
            
        else:
            logger.info("File already exists")
            
    
    