from src.customerChurn.entity.config_entity import DataTransformationConfig
from src.customerChurn.logger import logger
from src.utils.common import create_directories, save_object

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        create_directories([self.config.root_dir])
        
    def transform_data(self) -> ColumnTransformer:
        try:
            ## NOTE: Colunm Transformer will change the order of colunms after applying transformation, so check the value of index mentioned in colunmTransformer after eact CT        
            tr1 = ColumnTransformer([
                ("oneHotEncoder",OneHotEncoder(drop='first', handle_unknown='ignore'), [1,2,6,7,8])
            ], remainder='passthrough')
            
            tr2 = ColumnTransformer([
                ('scalar', StandardScaler(), slice(0,14))
            ], remainder='passthrough')
            pipeline = Pipeline(
                steps=[
                    ('tr1', tr1),
                    ('tr2', tr2)
                ]
            )
            return pipeline
        except Exception as e:
            logger.info(e)
            
    def initiate_data_transformation(self):
        dataset_file_path = self.config.dataset_file_path
        logger.info(f"Reading dataset from {dataset_file_path}")
        df = pd.read_csv(dataset_file_path)
        logger.info(df.head())
        
        logger.info("Droping ['id','CustomerId','Surname'] from dataset")
        df.drop(['id','CustomerId','Surname'], inplace=True, axis=1)
        
        X = df.drop(self.config.targer_colunm, axis=1)
        y = df[self.config.targer_colunm]
        X_train, X_test, y_train, y_test  = train_test_split(X,y, test_size=self.config.test_size, random_state=self.config.random_state)
        
        preprocessor = self.transform_data()
        
        logger.info("Preprocessing data")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Oversampling the training data
        logger.info(f"Before Sample value count is {Counter(y_train)}")
        smt = SMOTE(sampling_strategy=0.8)
        X_train_sm, y_train_sm = smt.fit_resample(X_train_processed, y_train)
        logger.info(f"After Sample value count is {Counter(y_train_sm)}")
        
        train_dataset = np.c_[X_train_sm, np.array(y_train_sm)]
        test_dataset = np.c_[X_test_processed, np.array(y_test)]
        
        train_dataset = pd.DataFrame(train_dataset)
        logger.info(f"Created train dataset at location {self.config.train_dataset_file_path} with shape {train_dataset.shape}")
        train_dataset.to_csv(self.config.train_dataset_file_path, index=False)
        test_dataset = pd.DataFrame(test_dataset)
        logger.info(f"Created test dataset at location {self.config.test_dataset_file_path} with shape {test_dataset.shape}")
        test_dataset.to_csv(self.config.test_dataset_file_path, index=False)
        
        save_object(self.config.preprocessor_obj_path, preprocessor)
        