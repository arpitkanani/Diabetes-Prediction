import pandas as pd
import numpy as np

import os ,sys

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path: str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation(self):
        logging.info("data transformation initiated.")
        try:
            num_cols=['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin',
                    'BMI', 'DiabetesPedigreeFunction', 'Age']
            cat_cols=[]

            
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder',OneHotEncoder()),
                ('scaler',StandardScaler(with_mean=False))
                ]
            )

            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,num_cols),
                    ('cat_pipeline',cat_pipeline,cat_cols)
                ]
            )

            logging.info("pipeline completed.")

            return preprocessor
        except Exception as e:
            logging.info("error occured in get data transformation method")
            raise CustomException(e,sys)#type:ignore

    def initiate_data_transformation(self,train_path,test_path):
        logging.info("data transformation in array's initated in data transformer method.")
        try:
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)

            logging.info("Reading of train and test data is completed.")

            logging.info("obtaining preprocessor object.")

            preprocessor_obj=self.get_data_transformation()

            target_col='Outcome'

            input_feature_train_df=train_data.drop(columns=[target_col])
            target_feature_train_df=train_data[target_col]

            input_feature_test_df=test_data.drop(columns=[target_col])
            target_feature_test_df=test_data[target_col]

            logging.info("Applying preprocessing on both dataset.")

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessor_obj                
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path
            )

        except Exception as e:
            logging.info("error occured in initate data transformation method.")
            raise CustomException(e,sys)#type:ignore

# if __name__ == "__main__":
#     obj = DataTransformation()
#     obj.initiate_data_transformation(os.path.join('artifacts','train.csv'),os.path.join('artifacts','test.csv'))
