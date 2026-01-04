import os,sys
import  pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass 
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    
    def initiate_data_ingestion(self):
        logging.info("data ingestion initate.")
        try:
            df=pd.read_csv(r"E:\Data Analysis\Classification-project\notebooks\data\diabetes.csv")
            logging.info(f"Data read as Dataframe.\n{df.head(1)}")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info("split data into train and test array.")

            df['Glucose']=df['Glucose'].replace(0,df['Glucose'].mean())
            df['BloodPressure']=df['BloodPressure'].replace(0,df['BloodPressure'].mean())
            df = df.drop(columns=['SkinThickness'])
            df['Insulin']=df['Insulin'].replace(0,df['Insulin'].mean())
            df['BMI']=df['BMI'].replace(0,df['BMI'].mean())
            train_set,test_set=train_test_split(df,test_size=0.3,random_state=10)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of Data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("error occured in data ingestion method.")
            raise CustomException(e,sys)# type:ignore

# if __name__ == "__main__":
#     obj = DataIngestion()
#     obj.initiate_data_ingestion()
