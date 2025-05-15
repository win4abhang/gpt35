from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
from openai import OpenAIError

load_dotenv()  # Load environment variables

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

class QueryInput(BaseModel):
    prompt: str

@app.post("/intent")
def get_intent(data: QueryInput):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a classifier that returns the most relevant service category."},
                {"role": "user", "content": f"Classify this query: {data.prompt}"}
            ],
            max_tokens=30,
            temperature=0.2,
        )
        tag = response.choices[0].message.content.strip()
        return {"input": data.prompt, "predicted_tag": tag}

        except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
