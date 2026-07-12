import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, create_model
from typing import Optional

from google import genai
from google.genai import types

# 1. Load the .env file automatically
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")

# 2. Initialize the Client
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client() if api_key else None

app = FastAPI()

# Enable CORS for the Grader
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DynamicExtractRequest(BaseModel):
    text: str
    schema_mapping: dict[str, str] = Field(alias="schema")

def build_dynamic_model(schema_mapping: dict[str, str]):
    """Dynamically creates a Pydantic model based on the requested types."""
    fields = {}
    for key, type_string in schema_mapping.items():
        type_string = type_string.lower()
        
        if type_string == "integer":
            fields[key] = (Optional[int], Field(default=None, description="Extract as an integer JSON number. Return null if missing."))
        elif type_string == "float":
            fields[key] = (Optional[float], Field(default=None, description="Extract as a float JSON number. Return null if missing."))
        elif type_string == "date":
            fields[key] = (Optional[str], Field(default=None, description="Extract as a date strictly in ISO format YYYY-MM-DD. Return null if missing."))
        elif type_string == "time":
            fields[key] = (Optional[str], Field(default=None, description="Extract strictly as a time string (e.g., HH:MM or HH:MM:SS). Return null if missing."))
        else: 
            fields[key] = (Optional[str], Field(default=None, description="Extract as a string. Return null if missing."))
            
    return create_model('DynamicSchemaModel', **fields)

@app.post("/dynamic-extract")
async def dynamic_extract(request: DynamicExtractRequest):
    if not client:
        raise HTTPException(status_code=500, detail="API Key missing on server.")
        
    try:
        # 1. Build the Pydantic model on the fly
        DynamicModel = build_dynamic_model(request.schema_mapping)
        
        # 2. Instruct the LLM with rigorous boundary constraints
        prompt = f"""
        Extract the requested information from the text based strictly on the provided schema keys and types.
        
        CRITICAL RULES:
        - If a field cannot be found or is not explicitly present in the text, you MUST return null.
        - Pay close attention to fields asking for TIME versus fields asking for DATE. 
        - Do NOT populate a time field (like 'event_time') with a date string (YYYY-MM-DD). If a specific time (like HH:MM) is expected but missing, return null.
        - Dates MUST be formatted strictly as YYYY-MM-DD.
        - Times should be formatted as HH:MM or HH:MM:SS strings.
        - Do not include any extra keys.
        
        TEXT TO EXTRACT FROM:
        {request.text}
        """

        # 3. Call Gemini
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DynamicModel,
                temperature=0.0
            )
        )
        
        return json.loads(response.text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
