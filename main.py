from src.customerChurn.pipeline.stage_01_data_ingestion_pipeline import DataIngestionPipeline
from src.customerChurn.config.configuration import ConfigurationManager
from src.customerChurn.logger import logger

config = ConfigurationManager()

STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion_pipeline = DataIngestionPipeline(data_ingestion_config)
    data_ingestion_pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e