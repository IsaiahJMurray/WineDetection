from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import asyncio
import time
from typing import Dict
from fastapi.responses import JSONResponse
from fastapi.responses import PlainTextResponse
from concurrent.futures import ThreadPoolExecutor



MODEL_DIR = "outputs\distilbert-base-uncased_top4_20251217_144633\model\checkpoint-2582"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

# --- Rate limit / queue configuration ---
# Per-IP token bucket: capacity tokens and refill interval (seconds per token)
RATE_LIMIT_TOKENS = 10          # burst capacity
RATE_REFILL_SECONDS = 6         # one token every 6 seconds (~10 tokens per minute)

# Per-IP queue: how many requests we allow queued waiting for processing
PER_IP_QUEUE_SIZE = 8

# Global concurrency for running model in threadpool
MODEL_WORKERS = 2

# Token limit for text (counted in tokenizer tokens)
MAX_TOKENS_PER_REQUEST = 256

# Executor for running model inference without blocking event loop
executor = ThreadPoolExecutor(max_workers=MODEL_WORKERS)


# Per-IP state
class _ClientState:
    def __init__(self):
        self.tokens = RATE_LIMIT_TOKENS
        self.last_refill = time.time()
        self.queue = asyncio.Queue(maxsize=PER_IP_QUEUE_SIZE)


clients: Dict[str, _ClientState] = {}
last_seen: Dict[str, float] = {}


def _refill_tokens(state: _ClientState):
    now = time.time()
    elapsed = now - state.last_refill
    if elapsed <= 0:
        return
    # how many tokens to add
    add = int(elapsed / RATE_REFILL_SECONDS)
    if add > 0:
        state.tokens = min(RATE_LIMIT_TOKENS, state.tokens + add)
        state.last_refill += add * RATE_REFILL_SECONDS


async def _process_queue_for_client(state: _ClientState):
    # run forever as long as items exist — called per-request when needed
    while not state.queue.empty():
        item = await state.queue.get()
        fut, inp_text = item
        try:
            # run inference in threadpool to avoid blocking
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(executor, _sync_model_predict, inp_text)
            fut.set_result(res)
        except Exception as e:
            fut.set_exception(e)
        finally:
            state.queue.task_done()


def _sync_model_predict(text: str):
    # Synchronous inference helper; used inside threadpool
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS_PER_REQUEST).to(DEVICE)
    with torch.inference_mode():
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

    id2label = model.config.id2label
    out = {id2label[str(i)] if isinstance(id2label, dict) and str(i) in id2label else id2label[i]: float(probs[i])
           for i in range(len(probs))}
    out = {k: round(v * 100, 2) for k, v in out.items()}
    return {"device": DEVICE, "predictions": out}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local demo; tighten for real deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictIn(BaseModel):
    text: str

@app.on_event("startup")
def warmup():
    # Warm-up to avoid first-request latency spikes
    dummy = "Warm up the model."
    inputs = tokenizer(dummy, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    with torch.inference_mode():
        _ = model(**inputs)

@app.post("/predict")
async def predict(request: Request, inp: PredictIn):
    # Identify client by device token (header `x-client-id`) falling back to IP
    client_id = request.headers.get("x-client-id")
    if not client_id:
        client_id = request.client.host if request.client else "unknown"

    # Enforce token limit using tokenizer to count tokens
    try:
        token_count = len(tokenizer.encode(inp.text, truncation=False))
    except Exception:
        token_count = None

    if token_count is not None and token_count > MAX_TOKENS_PER_REQUEST:
        raise HTTPException(status_code=413, detail=f"Request too long: {token_count} tokens (max {MAX_TOKENS_PER_REQUEST})")

    # Get or create client state keyed by device id
    state = clients.get(client_id)
    if state is None:
        state = _ClientState()
        clients[client_id] = state

    # update last seen for "users in past hour" metric
    last_seen[client_id] = time.time()

    # Refill tokens
    _refill_tokens(state)

    # If token available -> consume and run immediately
    if state.tokens > 0:
        state.tokens -= 1
        # Run model in threadpool so we don't block event loop
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(executor, _sync_model_predict, inp.text)
        return res

    # No tokens: try to enqueue (up to PER_IP_QUEUE_SIZE). If queue full, reject.
    fut = asyncio.get_running_loop().create_future()
    try:
        state.queue.put_nowait((fut, inp.text))
    except asyncio.QueueFull:
        raise HTTPException(status_code=429, detail="Rate limit exceeded; please retry later")

    # Start background processing for this client's queue if not already running
    # We schedule a task to drain the queue — it will process until empty
    asyncio.create_task(_process_queue_for_client(state))

    # Wait for the queued inference to complete and return its result
    try:
        res = await fut
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
def client_status(request: Request):
    """Return the client's token budget, queue length, and number of unique users in the past hour.

    Header: `x-client-id` is required to identify the device client.
    """
    client_id = request.headers.get("x-client-id")
    if not client_id:
        client_id = request.client.host if request.client else "unknown"

    state = clients.get(client_id)
    if state is None:
        # empty state
        tokens = RATE_LIMIT_TOKENS
        queue_len = 0
    else:
        _refill_tokens(state)
        tokens = state.tokens
        queue_len = state.queue.qsize()

    # count unique users seen in the past hour
    cutoff = time.time() - 3600
    users_past_hour = sum(1 for t in last_seen.values() if t >= cutoff)

    return {"client_id": client_id, "tokens": tokens, "queue_len": queue_len, "users_past_hour": users_past_hour}


@app.get("/api/examples")
def get_examples():
    """Return the examples.json contents for frontend use.

    The file is expected to live next to `app.py`.
    """
    p = Path(__file__).parent / "examples.json"
    if not p.exists():
        return JSONResponse(status_code=404, content={"error": "examples.json not found"})

    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/examples")
async def post_examples(request: Request):
    """Accept a POST (plain/text or JSON) and return examples.json.

    This variant allows clients to send a device id in the request body
    without using custom headers, avoiding some CORS preflight behavior
    with certain tunnels/proxies.
    """
    p = Path(__file__).parent / "examples.json"
    if not p.exists():
        return JSONResponse(status_code=404, content={"error": "examples.json not found"})

    try:
        # consume body but we don't require its content
        try:
            _ = await request.body()
        except Exception:
            pass
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/status")
async def client_status_post(request: Request):
    """POST variant of /api/status. Reads device id from plain text body or JSON payload.
    Useful for clients that avoid custom headers and send the device id in the body.
    """
    body = None
    try:
        raw = await request.body()
        if raw:
            s = raw.decode("utf-8").strip()
            # try parse JSON if possible
            try:
                j = json.loads(s)
                body = j.get("client_id") if isinstance(j, dict) else s
            except Exception:
                body = s
    except Exception:
        body = None

    client_id = None
    if body:
        client_id = body
    else:
        client_id = request.headers.get("x-client-id")
    if not client_id:
        client_id = request.client.host if request.client else "unknown"

    state = clients.get(client_id)
    if state is None:
        tokens = RATE_LIMIT_TOKENS
        queue_len = 0
    else:
        _refill_tokens(state)
        tokens = state.tokens
        queue_len = state.queue.qsize()

    cutoff = time.time() - 3600
    users_past_hour = sum(1 for t in last_seen.values() if t >= cutoff)

    return {"client_id": client_id, "tokens": tokens, "queue_len": queue_len, "users_past_hour": users_past_hour}


def _cors_preflight_response():
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, X-Client-Id, Authorization",
        "Access-Control-Allow-Credentials": "true",
    }
    return PlainTextResponse("OK", status_code=200, headers=headers)


@app.options("/predict")
def predict_options():
    return _cors_preflight_response()


@app.options("/api/examples")
def examples_options():
    return _cors_preflight_response()


@app.options("/api/status")
def status_options():
    return _cors_preflight_response()
