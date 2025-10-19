from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelEvaluationArtifact,ModelTrainerArtifact, DataIngestionArtifact
from sklearn.metrics import mean_squared_error
from src.exception import MyException
from src.logger import logging
from src.constants import TARGET_COLUMN
from src.utils.main_utils import load_object
import sys
import pandas as pd
from typing import Optional
from src.entity.s3_estimator import Proj1Estimator 
from dataclasses import dataclass

@dataclass
class EvaluateModelResponse:
    trained_model_root_mean_squared_error:float
    best_model_root_mean_squared_error:float
    is_model_accepted:bool
    difference:float
    
class ModelEvaluation:
    def __init__(self,model_eval_config:ModelEvaluationConfig,data_ingestion_artifact:DataIngestionArtifact,model_trainer_artifact:ModelTrainerArtifact):
        try:
            self.model_eval_config=model_eval_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.model_trainer_artifact=model_trainer_artifact
        except Exception as e:
            raise MyException(e,sys)
        
    def get_best_model(self)->Optional[Proj1Estimator]:
        try:
            bucket_name=self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_model_key_path
            proj1estimator= Proj1Estimator(bucket_name=bucket_name,model_path=model_path)
            if proj1estimator.is_model_present(model_path=model_path):
                return proj1estimator
            return None
        except Exception as e:
            raise MyException(e,sys)
        
    def _drop_id_column(self,df):
        logging.info("Dropping id column if present")
        if "_id" in df.columns:
            df=df.drop("_id",axis=1)
        return df 
    
    def evaluate_model(self)->EvaluateModelResponse:
        try:
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x,y=test_df.drop(TARGET_COLUMN,axis=1),test_df[TARGET_COLUMN]
            logging.info("Test data loaded and now transforming it for prediction...")
            x=self._drop_id_column(x)
            trained_model=load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded/exists.")
            trained_model_root_mean_squared_error=self.model_trainer_artifact.metric_artifact.root_mean_squared_error
            logging.info(f"F1_Score for this model: {trained_model_root_mean_squared_error}")
            best_model=self.get_best_model()
            best_model_root_mean_squared_error=None
            if best_model is not None:
                logging.info(f"Computing F1_Score for production model..")
                y_hat_best_model=best_model.predict(x)
                best_model_root_mean_squared_error= mean_squared_error(y,y_hat_best_model)**0.5
                logging.info(f"F1_Score-Production Model: {best_model_root_mean_squared_error}, F1_Score-New Trained Model: {best_model_root_mean_squared_error}")
            
            tmp_best_model_score=0 if best_model_root_mean_squared_error is None else best_model_root_mean_squared_error
            result= EvaluateModelResponse(trained_model_root_mean_squared_error=trained_model_root_mean_squared_error,
                                          best_model_root_mean_squared_error=best_model_root_mean_squared_error,
                                          is_model_accepted=trained_model_root_mean_squared_error>tmp_best_model_score,
                                          difference=trained_model_root_mean_squared_error-tmp_best_model_score)
            logging.info(f"Result: {result}")
            return result
        except Exception as e:
            raise MyException(e,sys)
        
    def initiate_model_evaluation(self)->ModelEvaluationArtifact:
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response=self.evaluate_model()
            s3_model_path=self.model_eval_config.s3_model_key_path
            model_evaluation_artifact=ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                changed_accuracy=evaluate_model_response.difference,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path
            )
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e,sys)
            

                