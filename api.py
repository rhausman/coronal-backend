import os
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Alec's basemodel
from basemodel_v1 import RHRAD_online, resultsProcesser

# Project-level imports
# from utils import make_prediction_on_data_with_appropriate_model, load_appropriate_model, extract_data_from_file

# from app.api.routes import router as api_router

# to run server, uvicorn api:app --reload
def get_application():
    app = FastAPI(title="Phresh", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # *
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # app.include_router(api_router, prefix="/api")
    return app


app = get_application()


class Item(BaseModel):
    name: str
    price: float
    is_offer: Optional[bool] = None

# an example definition of a post operation which will take in a file
@app.post("/analyze") #/{option}
async def analyze_data(
    file: UploadFile = File(...)#, option: Optional[str] = "30day"
):
    
    # dictionary to hold display messages
    display_strings = {"low":"You may not be infected with COVID-19, though you may be asymptomatic. Maintain normal testing routine, and continue social distancing.",
                       "medium":"Your resting heart rate has been elevated for a few hours. Monitor your symptoms closely and consider isolation.",
                       "high":"Your resting heart rate has been elevated for many hours. You may have COVID-19 or another serious condition. Consider isolation and contact your healthcare provider if symptoms worsen."}

    """
    #----------MODELS REQUIRE .csv FILE PATHS AS SHOWN BELOW-------------
    
    model1 = RHRAD_online(hr="data/IndividualRed_hr.csv", # path to heart rate csv
                     steps="data/IndividualRed_steps.csv", # path to steps csv
                     baseline_window=480, # number of hours to use as baseline (if baseline_window > data length, will fail)
                     last_day_only=True, # if True, only the most recent day is checked for anomalous heartrates
                     myphd_id_anomalies="results/IndividualRed_anomalies.csv", # where to put anomalies csv
                     myphd_id_alerts = "results/IndividualRed_alerts.csv", # where to put alerts csv
                    )
    # using results paths from model1
    resultsModel = resultsProcesser(anomaliesCSV="results/IndividualRed_anomalies.csv",
                                    alertsCSV="results/IndividualRed_alerts.csv")
    #---------------------------------------------------------------------                           

    alertLevel = resultsModel.getAlertLevel() # returns a string: low, medium, or high
    anomalousHours = resultsModel.getAnomalousHours() # returns a list of strings
    hours_msg = "During these hours you had a high resting heart rate, relative to your baseline, which triggered the alert: " + ", ".join(anomalousHours)
    """
    
    # setting alertLevel and hours_msg for test purposes
    # TODO: delete these when the above code is working
    alertLevel = "low"
    hours_msg = "During these hours you had a high resting heart rate, relative to your baseline, which triggered the alert: 5:00, 6:00, 12:00"
    
    return {"threat_level":alertLevel, "disp_str":display_strings[alertLevel]}
    
    # alternative return statement
    return {"threat_level":alertLevel, "disp_str":display_strings[alertLevel], "hours_msg":hours_msg}


@app.get("/get_operations")
def get_operations():
    return {"operations": ["analyze", "something else"]}
