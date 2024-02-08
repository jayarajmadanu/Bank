'''MLFLOW_TRACKING_URI=https://dagshub.com/jayarajmadanu/Bank.mlflow \
set MLFLOW_TRACKING_USERNAME=jayarajmadanu \
set MLFLOW_TRACKING_PASSWORD= \
python script.py
'''

'''
model_trainer:
  params:
    Random Forest: 
      max_features: ['sqrt','log2']
      n_estimators: [64,128,256]
      max_depth : [7,9,11, 13]
      n_jobs: [-1]
      class_weight: [{0: 1, 1: 1},{0: 1, 1: 2}]
    SVR:
      kernel: ['linear', 'poly', 'rbf']
      C: [1, 25, 57]
      epsilon: [0.1, 0.01, 0.001]
    Gradient Boosting :
      learning_rate: [.1,.01,.05,.001]
      subsample: [0.75,0.8,0.85,0.9]
      max_features: ['sqrt','log2']
      n_estimators: [128,256, 512]
      max_depth: [3,5,6]
    XGBClassifier:
      learning_rate: [.05,.08, 0.1]
      n_estimators: [128,256, 512, 800]
      max_depth: [5,7,10, 12]
    AdaBoost Classifier: 
      learning_rate: [.1,.01,0.5,0.9]
      n_estimators: [128,256, 512]
    
'''