import os
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Alec's basemodel
from basemodel_v1 import RHRAD_online, resultsProcesser

TEMP_PATH = os.environ.get("DATA_PATH", default="data_cache")
# dictionary to hold display messages
display_strings = {
    "low": "You may not be infected with COVID-19, though you may be asymptomatic. Maintain normal testing routine, and continue social distancing.",
    "medium": "Your resting heart rate has been elevated for a few hours. Monitor your symptoms closely and consider isolation.",
    "high": "Your resting heart rate has been elevated for many hours. You may have COVID-19 or another serious condition. Consider isolation and contact your healthcare provider if symptoms worsen.",
    "unknown": "Something went wrong - please try again. Make sure that both files are properly formatted.",
}

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
@app.post("/analyze")  # /{option}
async def analyze_data(
    heart_file: UploadFile = File(...),
    step_file: UploadFile = Body(...),
):
    # default response - will be modified and returned
    response = {
        "threat_level": "unknown",
        "disp_str": "Please upload a file to recieve an analysis",
    }
    # check for filetype
    if heart_file.content_type != "text/csv" or step_file.content_type != "text/csv":
        response["disp_str"] = "Please be sure to upload a CSV file."
    else:
        # save the uploaded files to temporary files
        hr_path, step_path = os.path.join(TEMP_PATH, "hr.csv"), os.path.join(
            TEMP_PATH, "step.csv"
        )
        anomalies_path, alerts_path = (
            os.path.join(TEMP_PATH, "out_anomalies.csv"),
            os.path.join(TEMP_PATH, "out_alerts"),
        )
        with open(hr_path, "wb") as hr_file_tmp:
            hr_file_tmp.write(heart_file.file.read())
        with open(step_path, "wb") as step_file_tmp:
            step_file_tmp.write(step_file.file.read())
        try:
            # load the model with the appropriate files
            model1 = RHRAD_online(
                hr=hr_path,  # path to heart rate csv
                steps=step_path,  # path to steps csv
                myphd_id_anomalies=anomalies_path,  # where to put anomalies csv
                myphd_id_alerts=alerts_path,  # "results/IndividualRed_alerts.csv",  # where to put alerts csv
            )
            # using results paths from model1
            resultsModel = resultsProcesser(
                anomaliesCSV=anomalies_path,  # "results/IndividualRed_anomalies.csv",
                alertsCSV=alerts_path,  # "results/IndividualRed_alerts.csv",
            )
            alertLevel = (
                resultsModel.getAlertLevel()
            )  # returns a string: low, medium, or high
            anomalousHours = (
                resultsModel.getAnomalousHours()
            )  # returns a list of strings
            hours_msg = (
                "During these hours you had a high resting heart rate, relative to your baseline, which triggered the alert: "
                + ", ".join(anomalousHours)
            )
            print("RESULTS: ", alertLevel)

            response = {
                "threat_level": alertLevel,
                "disp_str": display_strings[alertLevel],
                "hours_msg": hours_msg,
            }
        except:
            response["disp_str"] = "Something was wrong with the files you uploaded."
        # alternative return statement
        # return {"threat_level":alertLevel, "disp_str":display_strings[alertLevel], "hours_msg":hours_msg}
    return response


@app.get("/get_operations")
def get_operations():
    return {"operations": ["analyze", "something else"]}
