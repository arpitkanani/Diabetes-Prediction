import warnings

from sklearn.metrics import confusion_matrix, roc_auc_score
warnings.filterwarnings('ignore')

import numpy as np
import os,sys

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model,evaluate_metrics

from urllib.parse import urlparse
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from dataclasses import dataclass
import dagshub
dagshub.init(repo_owner='arpitkanani', repo_name='Diabetes-Prediction', mlflow=True)#type: ignore

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('model training imitiated.')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                'LogisticRegression':LogisticRegression(),
                'SVC':SVC(),
                'DecisionTree':DecisionTreeClassifier(),
                'RandomForest':RandomForestClassifier(),
                'XGBClassifier':XGBClassifier(),
                'CatBoostClassifier':CatBoostClassifier(),
                'AdaBoostClassifier':AdaBoostClassifier(),
                'GradientBoostingClassifier':GradientBoostingClassifier(),
                'KNNClassifier':KNeighborsClassifier(),
                'GaussianNB':GaussianNB()
            }

            params = {

                'LogisticRegression': {
                    'penalty': ['l2'],
                    'C': np.logspace(-3,3,7),
                    'solver': ['lbfgs','liblinear'],
                
                },

                'SVC': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                },

                'DecisionTree': {
                    'criterion':['gini','entropy','log_loss'],
                    'splitter':['best','random'],
                    'max_depth':[1,2,3,4,5],
                    'max_features':['sqrt','log2', None]   
                }

                ,

                'RandomForest': {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },

                'XGBClassifier': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 6],
                    'subsample': [0.8, 1.0],
                    'eval_metric': ['logloss']
                },

                'CatBoostClassifier': {
                    'iterations': [200, 500],
                    'depth': [6, 8],
                    'verbose': [False]
                },

                'AdaBoostClassifier': {
                    'n_estimators': [50, 100,200],
                    #'learning_rate': [0.01, 0.1, 1.0]
                },

                'GradientBoostingClassifier': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                },

                'KNNClassifier': {
                    'n_neighbors': [3, 5, 7, 9,2],
                    'weights': ['uniform', 'distance'],
                    #'metric': ['euclidean', 'manhattan']
                },

                'GaussianNB': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7]
                }

            }
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models,param=params)
            logging.info(f"model report :{model_report}")

            best_model_name = max(
                model_report,
                key=lambda x: model_report[x]["score"]
            )

            best_model_score = model_report[best_model_name]["score"]
            best_model = model_report[best_model_name]["model"]

            print(f"best model is :{best_model_name},accuracy score : {best_model_score}")
            print("="*12)

            logging.info(f"best Model Found, Model Name :{best_model_name}, accuracy_score : {best_model_score}")

            print("this is the best model")
            print(best_model_name)

            models_names=list(params.keys())
            
            actual_model=""

            for model in models_names:
                if best_model_name == model:
                    actual_model=actual_model + model
            
            
            best_params=params[actual_model]

            mlflow.set_registry_uri("https://dagshub.com/arpitkanani/Diabetes-Prediction.mlflow")
            tracking_url_type_store= urlparse(mlflow.get_tracking_uri()).scheme

            #ml flow pipe line
            with mlflow.start_run():
                y_pred = best_model.predict(X_test)
                y_proba = best_model.predict_proba(X_test)[:, 1]

                (score, precision, recall, f1) = evaluate_metrics(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_proba)

                mlflow.log_metric("accuracy", float(score))
                mlflow.log_metric("precision", float(precision))
                mlflow.log_metric("recall", float(recall))
                mlflow.log_metric("f1_score", float(f1))
                mlflow.log_metric("roc_auc", float(roc_auc))

                if tracking_url_type_store != 'file':
                    mlflow.sklearn.log_model(best_model, artifact_path="model") # type: ignore

                else:
                    mlflow.sklearn.log_model(best_model,'model') # type: ignore

                
            

           
            if best_model_score<0.6:
                logging.info(f"not a best model found in training and test dataset")
                raise CustomException("No best Model Found") # type: ignore
            
            

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, 
                obj=best_model
            )

        except Exception as e:
            logging.info("error occured in initiate model trainer method")
            raise CustomException(e,sys) #type:ignore

