import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact,metric_artifactss
from src.entity.estimator import MyModel
from typing import Tuple


class ModelTrainer:
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,model_trainer_config:ModelTrainerConfig):
        self.data_transformation_artifact=data_transformation_artifact
        self.model_trainer_config=model_trainer_config
    
    def get_model_object_and_report(self, train:np.array ,test:np.array)->Tuple[object,object]:
        try:
            logging.info("Training RandomForestClassifier with specified parameters")
            x_train,y_train,x_test,y_test= train[:,:-1],train[:,-1], test[:,:-1],test[:,-1]
            logging.info("train-test split done.")
            model= LinearRegression(fit_intercept=self.model_trainer_config.fit_intercept,copy_X=self.model_trainer_config.copy_x,n_jobs=self.model_trainer_config.n_jobs,positive=self.model_trainer_config.positive)
            logging.info("Model training going on...")
            model.fit(x_train,y_train)
            logging.info("Model training done.")
            y_pred= model.predict(x_test)
            mean_absolute_error_score=mean_absolute_error(y_test,y_pred)
            root_mean_squared_error_score=mean_squared_error(y_test,y_pred)**0.5
            r2_score_score=r2_score(y_test,y_pred)
            metric_artifact=metric_artifactss(mean_absolute_error=mean_absolute_error_score,root_mean_squared_error=root_mean_squared_error_score,R2score=r2_score_score)
            return model, metric_artifact
        except Exception as e:
            raise MyException(e,sys)
        
    def initiate_model_trainer(self) ->ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try: 
            print("------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component")
            train_arr= load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr=load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("train-test data loaded")
            trained_model,metric_artifact= self.get_model_object_and_report(train=train_arr,test=test_arr)
            logging.info("Model object and artifact loaded.")
            preprocessing_object= load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing obj loaded.")
            if (mean_squared_error(train_arr[:,-1],trained_model.predict(train_arr[:,:-1]))**0.5)<self.model_trainer_config.expected_accuracy:
                logging.info("No model found with score above the base score")
                raise Exception("No model found with score above the base score")
            logging.info("Saving new model as performace is better than previous one.")
            my_model= MyModel(preprocessing_object=preprocessing_object,trained_model_object=trained_model)
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info("Saved final model object that includes both preprocessing and the trained model")
            
            model_trainer_artifact= ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,metric_artifact=metric_artifact)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise MyException(e,sys)










    