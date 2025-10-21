import os
import datetime


TARGET_COLUMN= "BeatsPerMinute"
CURRENT_YEAR= datetime.date.today().year
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"


DATABASE_NAME= "musix"
COLLECTION_NAME="Muxix_data"
MONGODB_URL_KEY= "MONGODB_URL"

PIPELINE_NAME=""
ARTIFACT_DIR:str="artifact"


DATA_INGESTION_COLLECTION_NAME: str= "Muxix_data"
DATA_INGESTION_DIR_NAME:str= "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR:str= "feature_store"
DATA_INGESTION_INGESTED_DIR:str= "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO:float=0.25

FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

DATA_VALIDATION_DIR_NAME:str= "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME:str= "report.yaml"
SCHEMA_FILE_PATH= os.path.join("config","schema.yaml")


DATA_TRANSFORMATION_DIR_NAME:str= "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR:str= "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR:str= "transformed_object"

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 1
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")
FIT_INTERCEPT:bool= True
COPY_X:bool= True
N_JOBS:int=1
POSITIVE:bool=False

AWS_ACCESS_KEY_ID_ENV_KEY="AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY="AWS_SECRET_ACCESS_KEY"
REGION_NAME="us-east-1"

MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE:float= 0.02
MODEL_BUCKET_NAME= "harharmahadev123"
MODEL_PUSHER_S3_KEY="model-registery"
MODEL_FILE_NAME="model.pkl"


APP_HOST="0.0.0.0"
APP_PORT=5000
