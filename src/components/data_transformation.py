import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object,save_numpy_array_data, read_yaml_file 

class DataTransformation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_artifact:DataValidationArtifact,data_transformation_config:DataTransformationConfig):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_artifact=data_validation_artifact
            self.data_transformation_config=data_transformation_config
            self._schema_config=read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e,sys)
        
    @staticmethod
    def read_data(file_path)-> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e,sys)
        
    def get_data_transformer_object(self)-> Pipeline:
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            numeric_transformer= StandardScaler()
            min_max_scaler= MinMaxScaler()
            logging.info("Transformers Initialized: StandardScaler-MinMaxScaler")
            num_features=self._schema_config['num_features']
            mm_columns=self._schema_config['mm_columns']
            logging.info("Cols loaded from schema.")
            preprocessor= ColumnTransformer(transformers=[('num',numeric_transformer,num_features),('mm',min_max_scaler,mm_columns)],remainder='passthrough')
            final_pipeline= Pipeline(steps=[("preprocessor",preprocessor)])
            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline
        except Exception as e:
            raise MyException(e,sys)
        
    def _drop_id_column(self,df):
        try:
            logging.info("Dropping 'id' column")
            drop_col=self._schema_config['drop_columns']
            if drop_col in df.columns:
                df=df.drop(drop_col,axis=1)
                return df
        except Exception as e:
            raise MyException(e,sys)
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            train_df_arr = train_df.drop("BeatsPerMinute", axis=1)
            test_df_arr = test_df.drop("BeatsPerMinute", axis=1)
            train_df_target = train_df["BeatsPerMinute"]
            test_df_target = test_df["BeatsPerMinute"]
            logging.info("Input and Target cols defined for both train and test df.")

            pipeline = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            train_df_arr = self._drop_id_column(df=train_df_arr)
            test_df_arr = self._drop_id_column(df=test_df_arr)
            logging.info("Custom transformations applied to train and test data")

            logging.info("Starting data transformation")
            logging.info("Initializing transformation for Training-data")
            train_df_arr_transformed = pipeline.fit_transform(train_df_arr)
            logging.info("Initializing transformation for Testing-data")
            test_df_arr_transformed = pipeline.transform(test_df_arr)
            logging.info("Transformation done end to end to train-test df.")

            train_transformed = np.c_[train_df_arr_transformed, train_df_target]
            test_transformed = np.c_[test_df_arr_transformed, test_df_target]
            logging.info("feature-target concatenation done for train-test df.")

            save_object(file_path=self.data_transformation_config.transformed_object_file_path, obj=pipeline)
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_file_path, array=train_transformed)
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_file_path, array=test_transformed)
            logging.info("Saving transformation object and transformed files.")

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path
                )
        except Exception as e:
            logging.exception("Exception occurred in initiate_data_transformation")
            raise MyException(e, sys) from e
