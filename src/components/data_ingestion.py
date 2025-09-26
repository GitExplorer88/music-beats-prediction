import os
import sys
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.data_access.proj1_data import Proj1Data
from src.logger import logging
from src.exception import MyException

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        try:
            self.data_ingestion_config= DataIngestionConfig()
        except Exception as e:
            raise MyException
        
        
    def export_data_into_feature_store(self)->DataFrame:
        try:
            logging.info("Exporting data from mongodb")
            my_data=Proj1Data()
            Data=my_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            logging.info(f"Shape of dataframe: {Data.shape}")
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            dir_path= os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            Data.to_csv(feature_store_file_path,index=False,header=True)
            return Data
        except Exception as e:
            raise MyException(e,sys)
        
    def split_data_as_train_test(self,Data:DataFrame)->None:
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

        try:
            train_set,test_set=train_test_split(Data,test_size=self.data_ingestion_config.train_test_split_ratio,random_state=42)
            logging.info("Performed train test split on the dataframe")
            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )

            training_file_path=self.data_ingestion_config.training_file_path
            dir_path1=os.path.dirname(training_file_path)
            os.makedirs(dir_path1,exist_ok=True)
            testing_file_path=self.data_ingestion_config.testing_file_path
            dir_path2=os.path.dirname(testing_file_path)
            os.makedirs(dir_path2,exist_ok=True)
            logging.info(f"Exporting train and test file path.")
            train_set.to_csv(training_file_path,index=False,header=True)
            test_set.to_csv(testing_file_path,index=False,header=True)
            logging.info(f"Exported train and test file path.")
        except Exception as e:
            raise MyException(e,sys)
    
    def initiate_data_ingestion(self)->DataIngestionArtifact:
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")
        try:
            dataframe= self.export_data_into_feature_store()
            logging.info("Got the data from mongodb")
            self.split_data_as_train_test(dataframe)
            logging.info("Performed train test split on the dataset")

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )
            data_ingestion_artifact= DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,test_file_path=self.data_ingestion_config.testing_file_path)
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e,sys)

        