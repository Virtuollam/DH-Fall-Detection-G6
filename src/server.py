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
from sklearn.preprocessing import StandardScaler

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
        # Specify the column labels
        columns = ['acceleration_x', 'acceleration_y', 'acceleration_z', 
           'gyroscope_x', 'gyroscope_y', 'gyroscope_z']

        # Initialize an empty DataFrame with these column labels
        self.df = pd.DataFrame(columns=columns)
        self.data_buffer = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = f"fall_data_{timestamp}.csv"
        #self.scaler = StandardScaler()
        with open('Model detection/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

    def add_data(self, data):
        # Ensure the buffer doesn't exceed desired elements
        if len(self.df) >= 60:
            #self.data_buffer.pop(0)
            self.df = self.df.iloc[1:]  # Remove the oldest element
        #self.data_buffer.append(data)
        new_row = pd.DataFrame([data])
        self.df = pd.concat([self.df, new_row], ignore_index=True)

    def prepare_data_for_lstm(self):
        if len(self.data_buffer) == 0:
            print("Data buffer is empty. No data to prepare.")
            return np.empty((0, 1, 0))

        # Convert list of dictionaries to a 2D numpy array
        features = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyroscope_x', 'gyroscope_y', 'gyroscope_z']
        data_2d = np.array([[observation[feature] for feature in features] for observation in self.data_buffer])

        # Ensure the data is 2D before scaling
        if data_2d.ndim != 2:
            print("Error: Data is not 2-dimensional.")
            return None

        # Scale the 2D data
        data_scaled = self.scaler.fit_transform(data_2d)  # Use fit_transform or transform as appropriate

        # Reshape for LSTM input: (samples, time steps, features)
        lstm_compliant_data = data_scaled.reshape(-1, 1, len(features))
        return lstm_compliant_data

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



def predict_label(model, encoder, data):

    #features = {key: data[key] for key in data.keys() - {'timestamp', 'label'}} #maybe fix for error

    #df = pd.DataFrame([data])
    # Preprocess data
    scaled_data = scaler.transform(data)
    print(scaled_data)
    reshaped_data = np.reshape(scaled_data, (-1, 1, 6))  # Reshape for LSTM input
    print(reshaped_data)
    # Predict
    prediction = model.predict(reshaped_data)
    # prediction = model.predict(data)
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

    await websocket_manager.connect(websocket)
    try:
        while True:
            #recieve line of data
            data = await websocket.receive_text()

            # Load data into JSON format
            json_data = json.loads(data)

            # use raw_data for prediction
            # raw_data = list(json_data.values())

            #data_processor.add_data(json_data)
            # this line save the recent 100 samples to the CSV file. you can change 100 if you want.
            # if len(data_processor.data_buffer) >= 20: #saves more often originally 100
            #    data_processor 
            #print(data_processor.data_buffer)

            #call compliance enforcer for data bundle
            # LSTM_bundle = data_processor.prepare_data_for_lstm()
            #scaled_data = scaler.transform(data_processor.df)
            #print(scaled_data)
            #reshaped_data = np.reshape(scaled_data, (-1, 1, 6))  # Reshape for LSTM input
            #print(reshaped_data)

            #prediction = model.predict(reshaped_data)
            #predicted_label_index = np.argmax(prediction, axis=1)
            #predicted_label = encoder.inverse_transform(predicted_label_index)
            #print(predicted_label)

            #print(data_processor.data_buffer)
            #print(LSTM_bundle)
            #print(data_processor.df)
            #print(json_data)

            #label = predict_label(model, scaler, encoder, json_data)
            #label = predict_label(model, scaler, encoder, data_processor.df)
            #label = predict_label(model, scaler, encoder, data_processor.data_buffer)
            #print(label)
            #json_data["label"] = label


            #print(json_data)
            #print(raw_data)
            #print(json_data)

            #if label == 1: #simple check to test fall detection
                #await websocket_manager.broadcast_message(json.dumps(getpatientdata()))
                #break
            
            # broadcast the last data to webpage
            print(json_data)
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
