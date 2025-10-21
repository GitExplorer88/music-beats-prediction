import sys
from pandas import DataFrame 
from src.logger import logging
from src.exception import MyException 
from src.entity.s3_estimator import Proj1Estimator
from src.entity.config_entity import Beatspredictor 

class Beatsdata:
    def __init__(self, RhythmScore,AudioLoudness,VocalContent,AcousticQuality,InstrumentalScore,LivePerformanceLikelihood,MoodScore,TrackDurationMs,Energy):
        try:
            self.RhythmScore=RhythmScore
            self.AudioLoudness=AudioLoudness
            self.VocalContent=VocalContent
            self.AcousticQuality=AcousticQuality
            self.InstrumentalScore=InstrumentalScore
            self.LivePerformanceLikelihood=LivePerformanceLikelihood
            self.MoodScore=MoodScore
            self.TrackDurationMs=TrackDurationMs
            self.Energy=Energy
        except Exception as e:
            raise MyException(e,sys)
    
    def get_beats_data_as_dict(self):
        logging.info("Entered get_usvisa_data_as_dict method as VehicleData class")
        try:
            input_dict= {"RhythmScore":self.RhythmScore,
                         "AudioLoudness":self.AudioLoudness,
                         "VocalContent":self.VocalContent,
                         "AcousticQuality":self.AcousticQuality,
                         "InstrumentalScore":self.InstrumentalScore,
                         "LivePerformanceLikelihood":self.LivePerformanceLikelihood,
                         "MoodScore":self.MoodScore,
                         "TrackDurationMs":self.TrackDurationMs,
                         "Energy":self.Energy}
            logging.info("Created vehicle data dict")
            logging.info("Exited get_vehicle_data_as_dict method as VehicleData class")
            return input_dict
        except Exception as e:
            raise MyException(e,sys)
        
    
    def get_beats_input_data_frame(self)->DataFrame:
        try:
            input_data_frame=self.get_beats_data_as_dict()
            return DataFrame([input_data_frame])
        except Exception as e:
            raise MyException(e,sys)
        
class BeatsDataClassifier:
    def __init__(self,prediction_pipeline_config:Beatspredictor=Beatspredictor(),):
        try:
            self.prediction_pipeline_config=prediction_pipeline_config
        except Exception as e:
            raise MyException(e,sys)
        
    def predict(self,dataframe:DataFrame)->str:
        try:
            model=Proj1Estimator(bucket_name=self.prediction_pipeline_config.model_bucket_name,model_path=self.prediction_pipeline_config.model_file_path)
            result=model.predict(dataframe)
            return result
        except Exception as e:
            raise MyException(e,sys)
        