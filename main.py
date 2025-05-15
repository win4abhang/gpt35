from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from openai import OpenAIError
import os
from dotenv import load_dotenv

load_dotenv()

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
                {
                    "role": "system",
                    "content": (
                        "returns only the most relevant store or service category Name in 4 words or less. in English"
                    )
                },
                {
                    "role": "user",
                    "content": f"{data.prompt}"
                }
            ],
            max_tokens=10,  # Limit output length
            temperature=0.2,
        )
        tag = response.choices[0].message.content.strip()
        return {"input": data.prompt, "predicted_tag": tag}

    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
