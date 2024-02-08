from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    source_url: Path
    local_data_file_path: Path
    root_dir: Path
    
@dataclass
class DataValidationConfig:
    root_dir: Path
    dataset_file_path: Path
    validation_status_file_path: Path
    dataset_schema: dict
    
@dataclass
class DataTransformationConfig:
    root_dir: Path
    dataset_file_path: Path
    processed_dataset_file_path: Path
    preprocessor_obj_path: Path
    targer_colunm: str
    test_size: float
    train_dataset_file_path: Path
    test_dataset_file_path: Path
    random_state: float
    
@dataclass
class DataTrainingConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    params: dict
    model_name: str
    best_parsms: Path
    
@dataclass
class ModelEvaluationConfig:
    root_dir:Path
    mlflow_uri: str
    test_data_path: Path
    model_path: Path
    preprocessor_path: Path
    
@dataclass
class PredictionConfig:
    preprocessor_path: Path
    model_path: Path
    