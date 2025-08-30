import os

import tiktoken
from fastapi import FastAPI
from pydantic import BaseModel
from together import Together

# FastAPI app
app = FastAPI()

# Together client
api_key = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=api_key)

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
SYSTEM_PROMPT = "You are a fed up and sassy assistor who hates answering questions."
MESSAGES = [{"role": "system", "content": SYSTEM_PROMPT}]
TOKEN_BUDGET = 100
TEMPERATURE = 0.7
MAX_TOKENS = 100

# Tokenizer fallback
def get_encoding(model):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

ENCODING = get_encoding(MODEL)

def count_tokens(text): return len(ENCODING.encode(text))
def total_tokens_used(messages): return sum(count_tokens(m["content"]) for m in messages)
def enforce_token_budget(messages, budget=TOKEN_BUDGET):
    while total_tokens_used(messages) > budget and len(messages) > 2:
        messages.pop(1)

# Request/Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    tokens_used: int

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    MESSAGES.append({"role": "user", "content": req.message})

    response = client.chat.completions.create(
        model=MODEL,
        messages=MESSAGES,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

    reply = response.choices[0].message.content
    MESSAGES.append({"role": "assistant", "content": reply})
    enforce_token_budget(MESSAGES)

    return ChatResponse(reply=reply, tokens_used=total_tokens_used(MESSAGES))