import os
import json
from datetime import datetime

import pandas as pd
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi import WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware

import requests
import random

from tensorflow.keras.models import load_model
import pickle
import numpy as np

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


class DataProcessor:
    def __init__(self):
        self.data_buffer = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19] # new time stamp to include mili sec wait probly not needed since this is just the file name
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
        # print(f"DataFrame saved to {self.file_path}")


data_processor = DataProcessor()


def load_model_LSTM():
    # Load the trained LSTM model
    model = load_model('Model detection/LSTM_model.keras')
    # Load the scaler
    with open('Model detection/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    # Load the encoder
    with open('Model detection/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, scaler, encoder



def predict_label(model, scaler, encoder, data):

    #features = {key: data[key] for key in data.keys() - {'timestamp', 'label'}} #maybe fix for error

    df = pd.DataFrame([data])
    # Preprocess data
    scaled_data = scaler.transform(df)
    reshaped_data = np.reshape(scaled_data, (1, 1, scaled_data.shape[1]))  # Reshape for LSTM input
    # Predict
    prediction = model.predict(reshaped_data)
    predicted_label_index = np.argmax(prediction, axis=1)
    # Decode prediction
    predicted_label = encoder.inverse_transform(predicted_label_index)
    return predicted_label[0]  # Return the label 


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

model, scaler, encoder = load_model_LSTM() #load model here

@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")

async def websocket_endpoint(websocket: WebSocket):
    collection = 0
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            #print(data)

            # Broadcast the incoming data to all connected clients
            json_data = json.loads(data)

            # use raw_data for prediction
            raw_data = list(json_data.values())

            # Add time stamp to the last received data
            json_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:23]

            data_processor.add_data(json_data)
            # this line save the recent 100 samples to the CSV file. you can change 100 if you want.
            #if len(data_processor.data_buffer) >= 20: #saves more often originally 100
                #data_processor.save_to_csv()

            label = predict_label(model, scaler, encoder, raw_data)
            json_data["label"] = label
            collection = collection + 1
            #print(collection)
            # print the last data in the terminal
            #if collection == 100:
            #    json_data["patientinfo"] = getpatientdata()
            #    break
            #print(json_data)
            #print(raw_data)
            print(label)

            if label == 1: #simple check to test fall detection
                await websocket_manager.broadcast_message(json.dumps(getpatientdata()))
                break
            
            # broadcast the last data to webpage

            await websocket_manager.broadcast_message(json.dumps(json_data))


    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

def getpatientdata():
    sussy = requests.get('https://fhirsandbox.healthit.gov/open/r4/fhir/Patient?_format=json')
    jsus = sussy.json()
    patients = jsus["entry"]
    pnr = random.randint(0,len(patients)-1)
    patient = patients[pnr]["resource"]
    patientid = patient["id"]
    patientname = patient["name"][0]["given"][0] + " " + patient["name"][0]["family"]
    patientphone = patient["telecom"][0]["value"]
    patientadress = patient["address"][0]["line"][0] + ", " + patient["address"][0]["city"] + " " + patient["address"][0]["state"]
    patientinfo = {"name": patientname, "phonenumber": patientphone, "adress": patientadress}

    condcon = requests.get('https://fhirsandbox.healthit.gov/open/r4/fhir/Condition?_format=json')
    patientconds = condcon.json()["entry"]
    patientcond = (item for item in patientconds if item["resource"]["subject"]["reference"] == "Patient/" + patientid)
    conds = []
    for item in patientcond:
        if item["resource"]["clinicalStatus"]["coding"][0]["code"] == "active":
            conds.append(item["resource"]["code"]["coding"][0]["display"])
    medcon = requests.get('https://fhirsandbox.healthit.gov/open/r4/fhir/MedicationRequest?_format=json')
    patientmeds = medcon.json()["entry"]
    patientmed = (item for item in patientmeds if item["resource"]["subject"]["reference"] == "Patient/" + patientid)
    meds = []
    for item in patientmed:
        if item["resource"]["status"] == "active":
            if "medicationCodeableConcept" in item["resource"]:
                meds.append(item["resource"]["medicationCodeableConcept"]["coding"][0]["display"])
            if "medicationReference" in item["resource"]:
                meds.append(item["resource"]["medicationReference"]["display"])
    patientinfo["conditions"] = conds
    patientinfo["medications"] = meds
    return patientinfo


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
