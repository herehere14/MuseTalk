import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def call_llm(user_text: str) -> str:
    # Minimal: replace with your system prompt
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful banking assistant."},
            {"role": "user", "content": user_text},
        ],
    )
    return resp.choices[0].message.content