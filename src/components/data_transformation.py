import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging 
import os
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","Preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_confi=DataTransformationConfig()

    def create_data_transformer_object(self):
        numerical_columns=["writing score","reading score"]
        categorical_columns=["gender","race/ethnicity","parental level of education","lunch","test preparation course"]
        try:
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"categorical columns{categorical_columns}")

            logging.info(f"numerical columns{numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("Numerical transformation",num_pipeline,numerical_columns),
                    ("Categorical transformation",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise e
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Train and test data read completed")

            logging.info("Using and calling preprocessor object")

            preprocessor_obj=self.create_data_transformer_object()

            target_column="math score"
            numerical_columns=["writing score","reading score"]

            input_features_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]

            input_features_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]

            logging.info("applying preprocessor object on the train and test data")

            input_features_train_arr=preprocessor_obj.fit_transform(input_features_train_df)
            input_features_test_arr=preprocessor_obj.transform(input_features_test_df)

            train_arr=np.c_[
                input_features_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[input_features_test_arr,np.array(target_feature_test_df)]
            logging.info("saved preprocessing object")

            save_object(
                file_path=self.data_transformation_confi.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            return(
                train_arr,test_arr,self.data_transformation_confi.preprocessor_obj_file_path
            )
        except Exception as e:
            raise e

    