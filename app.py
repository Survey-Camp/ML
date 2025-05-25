import os
from dotenv import load_dotenv
from fastapi import FastAPI, Form
from pydantic import BaseModel
import json
import google.generativeai as genai

# from IPython import get_ipython
# from IPython.display import display
# %%
# %%
# import os
# import google.generativeai as genai
# %%
# from google.colab import userdata
# userdata.get('GOOGLE_API_KEY')
# genai.configure(api_key=userdata.get('GOOGLE_API_KEY'))

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not found in environment variables")
# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)

# %%
# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 128,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash-exp",
  generation_config=generation_config,
  system_instruction="You are a system that rephrases a question while keeping its meaning and expected answer unchanged.",
)

chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        "Rephrase the following question while keeping its meaning and expected answer the same. Ensure minimal alteration.",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Understood. Please provide the question, and I will generate a reworded version that preserves the original intent and answer.",
        
      ],
    },
  ]
)
# %%
from fastapi import FastAPI
from typing import Optional

app = FastAPI()

@app.get("/generate_similar_question/")
async def generate_similar_question(prompt: Optional[str] = None):
  if prompt is None:
    return {"error": "Please provide a prompt"}

  response = chat_session.send_message({prompt})
  similar_question = response.text

  return {"similar_question": similar_question}

import uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
# %%