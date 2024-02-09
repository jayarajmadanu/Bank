from src.customerChurn.pipeline.predict_pipeline import PredictionPipeline
from src.logger import logger
data = {
    'CreditScore': 300,
    'Geography': 'Spain',
    'Gender': 'Female',
    'Age': 24,
    'Tenure': 1,
    'Balance': 0,
    'NumOfProducts': 0,
    'HasCrCard': 0,
    'IsActiveMember': 0,
    'EstimatedSalary': 0
}

predict = PredictionPipeline()
res = predict.main(data=data)
