import json
from typing import List

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from icecream import ic
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from jitx_backend.worker import celery, predict_number


app = FastAPI()
app.celery_app = celery
all_active_connections = {}
task_id_to_user_id = {}


async def connect(active_connections: dict, websocket: WebSocket, user_id: str):
    await websocket.accept()
    if user_id not in active_connections:
        active_connections[user_id] = {}
    active_connections[user_id] = websocket


async def disconnect(active_connections: dict, websocket: WebSocket, user_id: str):
    try:
        await websocket.close()
    except Exception as _:
        ic("Websocket already closed")
    del active_connections[user_id]


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await connect(all_active_connections, websocket, user_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ic(f"Websocket closed for user {user_id}")
        await disconnect(all_active_connections, websocket, user_id)


class PredictionRequest(BaseModel):
    array: List[int]
    user_id: str


class PredictionResponse(BaseModel):
    prediction: int
    task_id: str


@app.post("/predict")
async def make_prediction(prediction_request: PredictionRequest):
    async_res = predict_number.delay(prediction_request.array)
    task_id_to_user_id[async_res.id] = prediction_request.user_id
    return async_res.id


@app.get("/predict/{task_id}")
async def get_prediction(task_id: str):
    task_result = celery.AsyncResult(task_id)
    return task_result.status


@app.post("/predict/webhook")
async def task_webhook(prediction_response: PredictionResponse):
    try:
        task_result = celery.AsyncResult(prediction_response.task_id)
        user_id = task_id_to_user_id[prediction_response.task_id]
        await all_active_connections[user_id].send_text(
            json.dumps({"type": "prediction", "prediction": task_result.get()})
        )
        # remove task id from dict
        del task_id_to_user_id[prediction_response.task_id]
        return task_result.status
    except Exception as e:
        ic(e)
        return "Error, " + str(e)


middleware = CORSMiddleware(
    app=app,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def start():
    uvicorn.run("jitx_backend.main:app", host="0.0.0.0", port=8004, reload=False)
