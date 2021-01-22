import os
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
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
@app.post("/analyze/{option}")
async def style_the_image(
    file: UploadFile = File(...), option: Optional[str] = "30day"
):
    # INSERT PROCESSING HERE
    """
    data = extract_data_from_file(file)
    model = load_appropriate_model(path)
    prediction = make_prediction_on_data_using_appropriate_model(data,model)
    etc.
    """
    
    return {"threat_level":"low", "disp_str":"You probably don't have COVID. Maintain normal testing routine, and continue social distancing."}


@app.get("/get_operations")
def get_operations():
    return {"operations": ["analyze", "something else"]}
