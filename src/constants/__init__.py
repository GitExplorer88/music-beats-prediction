import os
import datetime

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
