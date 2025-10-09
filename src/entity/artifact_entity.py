from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    trained_file_path:str
    test_file_path:str
    
@dataclass
class DataValidationArtifact:
    validation_status:bool
    message:str
    validation_report_file_path:str
    
@dataclass
class DataTransformationArtifact:
    transformed_train_file_path:str
    transformed_test_file_path:str
    transformed_object_file_path:str

@dataclass
class metric_artifactss:
    mean_absolute_error:float
    root_mean_squared_error:float
    R2score:float  
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str
    metric_artifact:metric_artifactss
    
    

    