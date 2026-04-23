from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq 
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
import time, os

load_dotenv()
app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

REQUEST_COUNT = Counter(
    "agent_requests_total",
    "Total requests to the AI agent",
    ["status"]
)
REQUEST_LATENCY = Histogram(
    "agent_request_latency_seconds",
    "Response latency of AI agent"
)

class Question(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "ok", "service": "AI Agent"}

@app.post("/ask")
def ask(body: Question):
    start = time.time()
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": body.question}]
        )
        answer = response.choices[0].message.content
        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(time.time() - start)
        return {"answer": answer}
    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        return {"error": str(e)}

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest())

