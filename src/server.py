import os
import json
from datetime import datetime

import pandas as pd
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi import WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()
# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("./src/index.html", "r") as f:
    html = f.read()

"""
##class DataProcessor:
    def __init__(self):
        self.data_buffer = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = f"fall_data_{timestamp}.csv"

    def add_data(self, data):
        self.data_buffer.append(data)

    def save_to_csv(self):
        df = pd.DataFrame.from_dict(self.data_buffer)
        self.data_buffer = []
        # Append the new row to the existing DataFrame
        df.to_csv(
            self.file_path,
            index=False,
            mode="a",
            header=not os.path.exists(self.file_path),
        )
        # print(f"DataFrame saved to {self.file_path}")##
"""

class DataProcessor:
    def __init__(self):
        self.data_buffer = []
        self.is_collecting = False  # To control start/stop of data collection
        self.file_path = ""  # Dynamic file naming

    def start_collecting(self):
        self.is_collecting = True
        self.data_buffer = []  # Reset buffer when starting

    def stop_collecting(self, file_name):
        self.is_collecting = False
        if file_name:
            self.file_path = file_name
        self.save_to_csv()
        self.data_buffer = []  # Optionally reset buffer after saving

    def add_data(self, data):
        if self.is_collecting:
            self.data_buffer.append(data)

    def save_to_csv(self):
        if self.file_path:  # Only save if a path is set
            df = pd.DataFrame.from_dict(self.data_buffer)
            df.to_csv(self.file_path, index=False)
            print(f"Data saved to {self.file_path}")


data_processor = DataProcessor()


def load_model():
    # you should modify this function to return your model
    model = None
    return model


def predict_label(model=None, data=None):
    # you should modify this to return the label
    if model is not None:
        label = model(data)
        return label
    return 0


class WebSocketManager:
    def __init__(self):
        self.active_connections = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        print("WebSocket connected")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print("WebSocket disconnected")

    async def broadcast_message(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                # Handle disconnect if needed
                self.disconnect(connection)


websocket_manager = WebSocketManager()
model = load_model()


@app.post("/start")
async def start_collection():
    data_processor.start_collecting()
    return {"message": "Data collection started."}

@app.post("/stop")
async def stop_collection(file_name: str = ''):
    data_processor.stop_collecting(file_name)
    return {"message": f"Data collection stopped and saved to {file_name}."}



@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while data_processor.is_collecting:
            data = await websocket.receive_text()
            print("hi")

            # Broadcast the incoming data to all connected clients
            json_data = json.loads(data)

            # use raw_data for prediction
            raw_data = list(json_data.values())

            # Add time stamp to the last received data
            json_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            data_processor.add_data(json_data)
            # this line save the recent 100 samples to the CSV file. you can change 100 if you want.
            if len(data_processor.data_buffer) >= 100:
                data_processor.save_to_csv()

            """  
            In this line we use the model to predict the labels.
            Right now it only return 0.
            You need to modify the predict_label function to return the true label
            """
            label = predict_label(model, raw_data)
            json_data["label"] = label

            # print the last data in the terminal
            print(json_data)

            # broadcast the last data to webpage
            await websocket_manager.broadcast_message(json.dumps(json_data))

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
