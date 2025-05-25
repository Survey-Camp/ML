from fastapi import FastAPI, Form
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn

# Load the model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Define the input data model using Pydantic
class InputData(BaseModel):
    Scrolling_Behavior: int
    Idle_Time: float
    Typing_Patterns: float
    Total_Time: float

# Create the FastAPI app
app = FastAPI()

# Define the prediction endpoint
@app.post("/predict")
async def predict(
    Scrolling_Behavior: int = Form(...),
    Idle_Time: float = Form(...),
    Typing_Patterns: float = Form(...),
    Total_Time: float = Form(...)
):
    # Extract features from the input data
    features = [
        Scrolling_Behavior,
        Idle_Time,
        Typing_Patterns,
        Total_Time
    ]
    
    # Convert to NumPy array and reshape
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]
    
    # Make prediction
    prediction = prediction.item()
    
    # Return prediction
    return {"prediction": prediction}

# Run the app using Uvicorn (for Colab)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)