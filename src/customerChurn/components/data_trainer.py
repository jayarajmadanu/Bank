from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.svm import SVR
from src.logger import logger
from src.customerChurn.entity.config_entity import DataTrainingConfig
from src.utils.common import create_directories, save_object

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer, precision_score
import pandas as pd
from xgboost import XGBClassifier
import os

class DataTrainer:
    def __init__(self, config: DataTrainingConfig):
        self.config = config
        create_directories([self.config.root_dir])
        
    def train(self):
        train_df = pd.read_csv(self.config.train_data_path, )
        logger.info(f"train_df Shape = {train_df.shape}")
        test_df = pd.read_csv(self.config.test_data_path)
        logger.info(f"test_df Shape = {test_df.shape}")
        X_train = train_df.iloc[:,0:13]
        logger.info(f"X_train Shape = {X_train.shape}")
        y_train = train_df.iloc[:,13].astype(int)
        logger.info(f"y_train Shape = {y_train.shape}")
        X_test = test_df.iloc[:,0:13]
        y_test = test_df.iloc[:,13].astype(int)
        
        models = {
            "Random Forest": RandomForestClassifier(),
            #"SVR":SVR(),
            #"XGBClassifier": XGBClassifier(),
            #"AdaBoost Classifier": AdaBoostClassifier(),
            #"Gradient Boosting": GradientBoostingClassifier(),
        }
        logger.info("TRAINING MODELS")
        report = self.evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=self.config.params)
        logger.info(f"report is {report}")
        estomators = []
        for rep in report.items():
            tup = (rep[1]['model_name'],rep[1]['model'])
            estomators.append(tup)
            
        stack = StackingClassifier(estimators=estomators, cv=3, n_jobs=-1)
        stack.fit(X_train, y_train)
        y_pred = stack.predict(X_test)
        cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
        logger.info(f"Final model ConfunsionMatrix is \n{cm}")
        logger.info(f"Final model Classification Report is  \n {classification_report(y_true=y_test, y_pred=y_pred)}")
        test_model_score = accuracy_score(y_test, y_pred)
        logger.info(f"Final Model Accuracy score is {test_model_score}")
        save_object(os.path.join(self.config.root_dir, self.config.model_name), stack)
        return ""
        
    def evaluate_models(self,X_train,y_train, X_test,y_test, models:dict, params:dict):
        model_keys = models.keys()
        report = {}
        
        for model_name in model_keys:
            model = models[model_name]
            parameters = params[model_name]
            logger.info(f"Training {model_name}")
            # GridSearchCV will get best hypermaters for each model
            custom_scorer = make_scorer(precision_score, greater_is_better=True,  pos_label=1)
            gs = GridSearchCV(estimator=model, param_grid=parameters, cv=3, scoring=custom_scorer)
            gs.fit(X_train, y_train)

            # now test the model with training data

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            y_test_pred = model.predict(X_test)
            try:
                with open(self.config.best_parsms, 'a') as f:
                    f.write(f"Best Params for {model_name} are \n {gs.best_params_}\n")
            except Exception as e:
                raise e
            cm = confusion_matrix(y_test, y_test_pred)
            logger.info(f"Confusion Matrif for model {model_name} is \n {cm}")
            logger.info(f"Classification Report for model {model_name} is  \n {classification_report(y_true=y_test, y_pred=y_test_pred)}")
            test_model_score = accuracy_score(y_test, y_test_pred)
            logger.info(f"Accuracy score for {model_name} is {test_model_score}")

            y_train_pred = model.predict(X_train)
            train_model_score = accuracy_score(y_train, y_train_pred)
            report[model_name] = {
                'model' : model,
                'model_name': model_name,
                'accuracy_score_test' : test_model_score,
                'accuracy_score_train' : train_model_score,
                'best_params': gs.best_params_
            }
        logger.info(f'Model Evaluation report: \n{report}')
        return report