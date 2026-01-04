
import dill
import pickle
import sys,os
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        logging.info("error occured in save object method")
        raise CustomException(sys,e) #type:ignore
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        logging.info("error occured in load object")
        raise CustomException(e,sys) #type:ignore
    
def evaluate_metrics(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try:

        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            params = param[list(models.keys())[i]]

            gs = gs = GridSearchCV(
                        model,
                        param_grid=params,
                        cv=5,
                        n_jobs=-1,
                        error_score='raise'
                      )

            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            score = evaluate_metrics(y_test, y_test_pred)
            report[list(models.keys())[i]] = {
                "score": score,
                "model": model
            }


        return report
    
    except Exception as e:
        logging.info("error occured in evalute model method.")
        raise CustomException(e,sys) #type:ignore
