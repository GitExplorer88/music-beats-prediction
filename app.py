from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import Beatsdata ,BeatsDataClassifier 
from src.pipline.training_pipeline import Traininpipeline 
from typing import Optional 
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run 
app=FastAPI()

app.mount("/static", StaticFiles(directory="static"),name="static")

templates= Jinja2Templates(directory='templates')
origins=["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class DataForm:
    """
    DataForm class to handle and process incoming form data.
    This class defines the vehicle-related attributes expected from the form.
    """
    def __init__(self, request: Request):
        self.request: Request = request
        self.RhythmScore: Optional[float] = None
        self.AudioLoudness: Optional[float] = None
        self.VocalContent: Optional[float] = None
        self.AcousticQuality: Optional[float] = None
        self.InstrumentalScore: Optional[float] = None
        self.LivePerformanceLikelihood: Optional[float] = None
        self.MoodScore: Optional[float] = None
        self.TrackDurationMs: Optional[float] = None
        self.Energy: Optional[float] = None
    
    async def get_music_data(self):
        form = await self.request.form()
        self.RhythmScore = form.get("RhythmScore")
        self.AudioLoudness= form.get("AudioLoudness")
        self.VocalContent= form.get("VocalContent")
        self.AcousticQuality= form.get("AcousticQuality")
        self.InstrumentalScore = form.get("InstrumentalScore")
        self.LivePerformanceLikelihood= form.get("LivePerformanceLikelihood")
        self.MoodScore= form.get("MoodScore")
        self.TrackDurationMs= form.get("TrackDurationMs")
        self.Energy= form.get("Energy")
        
    @app.get("/", tags=["authentication"])
    async def index(request: Request):
        return templates.TemplateResponse("beats.html",{"request": request, "context": "Rendering"})
    
    @app.get("/train")
    async def trainRouteClient():
        try:
            train_pipeline = Traininpipeline()
            train_pipeline.run_pipeline()
            return Response("Training successful!!!")
        except Exception as e:
            return Response(f"Error Occurred! {e}")
        
    @app.post("/")
    async def predictRouteClient(request: Request):
        try:
            form = DataForm(request)
            await form.get_music_data()
            Beats_data = Beatsdata(
                                RhythmScore= form.RhythmScore,
                                AudioLoudness = form.AudioLoudness,
                                VocalContent = form.VocalContent,
                                AcousticQuality = form.AcousticQuality,
                                InstrumentalScore = form.InstrumentalScore,
                                LivePerformanceLikelihood = form.LivePerformanceLikelihood,
                                MoodScore = form.MoodScore,
                                TrackDurationMs = form.TrackDurationMs,
                                Energy = form.Energy)
            beats_df = Beats_data.get_beats_input_data_frame()
            model_predictor = BeatsDataClassifier()
            value = model_predictor.predict(dataframe=beats_df)[0]
            return value
        
        except Exception as e:
            return {"status": False, "error": f"{e}"}

# Main entry point to start the FastAPI server
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)




