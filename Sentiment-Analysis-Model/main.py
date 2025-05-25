from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Initialize FastAPI app
app = FastAPI()

# Input data model
class FeedbackInput(BaseModel):
    feedback: str

@app.post("/predict")
def predict_sentiment(input_data: FeedbackInput):
    input_vector = vectorizer.transform([input_data.feedback])
    prediction = model.predict(input_vector)
    return {"feedback": input_data.feedback, "predicted_label": prediction[0]}




# app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"message": "Hello, World!"}

