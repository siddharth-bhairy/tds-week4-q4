import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, create_model
from typing import Optional, Any

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

# Request Model: Notice how we capture the dynamic schema as a dictionary
class DynamicExtractRequest(BaseModel):
    text: str
    # We use alias="schema" because 'schema' is a special keyword in OpenAPI
    schema_mapping: dict[str, str] = Field(alias="schema")

def build_dynamic_model(schema_mapping: dict[str, str]):
    """Dynamically creates a Pydantic model based on the requested types."""
    fields = {}
    for key, type_string in schema_mapping.items():
        type_string = type_string.lower()
        
        # Map the requested string types to actual Python types
        if type_string == "integer":
            fields[key] = (Optional[int], Field(default=None, description="Extract as an integer. Return null if missing."))
        elif type_string == "float":
            fields[key] = (Optional[float], Field(default=None, description="Extract as a float. Return null if missing."))
        elif type_string == "date":
            fields[key] = (Optional[str], Field(default=None, description="Extract as a date strictly in ISO format YYYY-MM-DD. Return null if missing."))
        else: # Default to string
            fields[key] = (Optional[str], Field(default=None, description="Extract as a string. Return null if missing."))
            
    # Create and return the dynamic class
    return create_model('DynamicSchemaModel', **fields)

@app.post("/dynamic-extract")
async def dynamic_extract(request: DynamicExtractRequest):
    if not client:
        raise HTTPException(status_code=500, detail="API Key missing on server.")
        
    try:
        # 1. Build the Pydantic model on the fly
        DynamicModel = build_dynamic_model(request.schema_mapping)
        
        # 2. Instruct the LLM
        prompt = f"""
        Extract the requested information from the text based strictly on the provided schema.
        - If a field cannot be found, you MUST return null.
        - Dates MUST be formatted strictly as YYYY-MM-DD.
        - Do not include any extra keys.
        
        TEXT:
        {request.text}
        """

        # 3. Call Gemini, passing our on-the-fly model to response_schema
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