import os 
import sys 
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split

from dataclasses import dataclass

# it basically gives the info to system where we have to store our data(In which folder we have to save)
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts","train.csv")
    test_data_path: str=os.path.join("artifacts","test.csv")
    raw_data_path: str=os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_confi=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion component")
        try:
            df=pd.read_csv("notebook\data\StudentsPerformance.csv")
            logging.info("fetched dataset as dataframe")

            os.makedirs(os.path.dirname(self.data_ingestion_confi.train_data_path),exist_ok=True)

            df.to_csv(self.data_ingestion_confi.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.data_ingestion_confi.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_confi.test_data_path,index=False,header=True)

            logging.info("Ingestion completed")

            return(
                self.data_ingestion_confi.train_data_path,
                self.data_ingestion_confi.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
