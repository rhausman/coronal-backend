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
    # INSERT PROCESSING HERE
    """
    data = extract_data_from_file(file)
    model = load_appropriate_model(path)
    prediction = make_prediction_on_data_using_appropriate_model(data,model)
    etc.

    model1 = RHRAD_online(hr="data/AJWW3IY_hr.csv", # path to heart rate csv
                     steps="data/AJWW3IY_steps.csv", # path to steps csv
                     baseline_window=480, # number of hours to use as baseline (if baseline_window > data length, will fail)
                     last_day_only=True, # if True, only the most recent day is checked for anomalous heartrates
                     myphd_id_anomalies="results/AJWW3IY_anomalies.csv", # where to put anomalies csv
                     myphd_id_alerts = "results/AJWW3IY_alerts.csv", # where to put alerts csv
                    )
    # using results paths from model1
    resultsModel = resultsProcesser(anomaliesCSV="results/AJWW3IY_anomalies.csv",
                                    alertsCSV="results/AJWW3IY_alerts.csv")

    alertLevel = resultsModel.getAlertLevel()    
    if(alertLevel == "low"):
        return {"threat_level":alertLevel,  "disp_str":"You probably don't have COVID. Maintain normal testing routine, and continue social distancing."}
    elif(alertLevel == "medium"):
        return {"threat_level":alertLevel,  "disp_str":"Your resting heart rate has been elevated for a few hours."}
    else:
        return {"threat_level":alertLevel,  "disp_str":"Your resting heart rate has been elevated for many"}
    """
    
    return {"threat_level":"low", "disp_str":"You probably don't have COVID. Maintain normal testing routine, and continue social distancing."}


@app.get("/get_operations")
def get_operations():
    return {"operations": ["analyze", "something else"]}
