from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseConfig
from data_controller.load_data import load_data
from model.dataset import DatasetModel

app = FastAPI()
BaseConfig.arbitrary_types_allowed = True


origins = [
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    dq_data = load_data()

    return dq_data


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
