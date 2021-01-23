import os
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

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
    print("string: ", heart_file.__dict__, step_file.__dict__)
    resp = {
        "threat_level": "unknown",
        "disp_str": "Please upload a file to recieve an analysis",
    }
    # INSERT PROCESSING HERE
    """
    data = extract_data_from_file(file)
    model = load_appropriate_model(path)
    prediction = make_prediction_on_data_using_appropriate_model(data,model)
    etc.
    """
    if heart_file.content_type != "text/csv":
        resp["disp_str"] = "Please be sure to upload a CSV file."
    elif True:
        resp["threat_level"] = "low"
        resp[
            "disp_str"
        ] = "You probably don't have COVID. Maintain normal testing routine, and continue social distancing."
    return resp


@app.get("/get_operations")
def get_operations():
    return {"operations": ["analyze", "something else"]}
