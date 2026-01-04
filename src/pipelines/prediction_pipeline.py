import os,sys
import pandas as pd
import warnings 
warnings.filterwarnings('ignore')
from src.utils import load_object
from src.exception import CustomException
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
             
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)

            predict=model.predict(data_scaled)

            return predict
        
        except Exception as e:
            logging.info("error occured in predict pipeline predict method.")
            raise CustomException(e,sys) #type: ignore


class CustomData:

    def __init__(self,
                 Pregnancies:int,
                 Glucose:float,
                 BloodPressure:float,
                 Insulin:float,
                 BMI:float,
                 DiabetesPedigreeFunction:float,
                 Age:int
            ):
            self.Pregnancies=Pregnancies
            self.Glucose=Glucose
            self.BloodPressure=BloodPressure
            self.Insulin=Insulin
            self.BMI=BMI
            self.DiabetesPedigreeFunction=DiabetesPedigreeFunction
            self.Age=Age
        
    def get_data_as_dataframe(self):
        try:

            custom_data_input={
                    'Pregnancies':[self.Pregnancies],
                    'Glucose':[self.Glucose],
                    'BloodPressure':[self.BloodPressure],
                    'Insulin':[self.Insulin],
                    'BMI':[self.BMI],
                    'DiabetesPedigreeFunction':[self.DiabetesPedigreeFunction],
                    'Age':[self.Age]
                }
            return pd.DataFrame(custom_data_input)
        except Exception as e:
             logging.info("error occured in data transform on dataframe.")
             raise CustomException(e,sys)#type:ignore

