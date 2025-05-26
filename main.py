from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model and encoders
model = joblib.load("model.pkl")
provider_encoder = joblib.load("provider_encoder.pkl")
suggested_encoder = joblib.load("suggested_encoder.pkl")

# Define the input model
class ServerInput(BaseModel):
    provider_server_type: str
    duration: float
    number_of_server_answers: int
    points: int

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Server Recommendation API is running!"}

@app.post("/predict")
def predict_server(data: ServerInput):
    # Encoding the input provider type
    encoded_provider = provider_encoder.transform([data.provider_server_type])[0]
    features = np.array([[encoded_provider, data.duration, data.number_of_server_answers, data.points]])
    
    # Making the prediction
    prediction = model.predict(features)[0]
    result = suggested_encoder.inverse_transform([prediction])[0]
    
    return {"Suggested Server": result}
