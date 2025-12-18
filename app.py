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



# Use Path to normalize the path (handles Windows backslashes correctly)
MODEL_DIR = str(Path("outputs\\distilbert-base-uncased_top4_20251217_150923\\model\\checkpoint-2582").resolve())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True).to(DEVICE)
model.eval()

# --- Rate limit / queue configuration ---
# Per-IP token bucket: capacity tokens and refill interval (seconds per token)
RATE_LIMIT_TOKENS = 100          # burst capacity
RATE_REFILL_SECONDS = 5         # one token every 5 seconds (~12 tokens per minute)

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
        self.processor_running = False  # Flag to prevent multiple queue processors
        self.lock = asyncio.Lock()  # Lock for token operations


clients: Dict[str, _ClientState] = {}
last_seen: Dict[str, float] = {}


def _refill_tokens(state: _ClientState):
    """Refill tokens based on elapsed time. Thread-safe when called with lock."""
    now = time.time()
    elapsed = now - state.last_refill
    if elapsed <= 0:
        return
    # how many tokens to add
    add = int(elapsed / RATE_REFILL_SECONDS)
    if add > 0:
        state.tokens = min(RATE_LIMIT_TOKENS, state.tokens + add)
        state.last_refill += add * RATE_REFILL_SECONDS


async def _wait_for_token(state: _ClientState):
    """Wait until at least one token is available. Returns True when token is available."""
    while True:
        async with state.lock:
            _refill_tokens(state)
            if state.tokens > 0:
                return True
        
        # No tokens available, wait for next refill
        await asyncio.sleep(RATE_REFILL_SECONDS)


async def _process_queue_for_client(state: _ClientState, client_id: str):
    """Continuously process queue items for a client, respecting rate limits.
    
    This function runs as a background task and processes items from the queue
    one at a time, waiting for tokens to become available before processing each item.
    """
    consecutive_empty_checks = 0
    MAX_EMPTY_CHECKS = 3  # Exit after 3 consecutive empty checks (3 seconds)
    
    while True:
        try:
            # Wait for an item in the queue (with timeout to allow checking if we should exit)
            try:
                item = await asyncio.wait_for(state.queue.get(), timeout=1.0)
                consecutive_empty_checks = 0  # Reset counter when we get an item
            except asyncio.TimeoutError:
                # Check if queue is empty and we should stop
                if state.queue.empty():
                    consecutive_empty_checks += 1
                    if consecutive_empty_checks >= MAX_EMPTY_CHECKS:
                        # Queue has been empty for a while, exit processor
                        async with state.lock:
                            # Double-check queue is still empty before exiting
                            if state.queue.empty():
                                state.processor_running = False
                                return
                        consecutive_empty_checks = 0  # Reset if queue got items
                continue
            
            fut, inp_text = item
            
            # Wait for a token to be available before processing
            await _wait_for_token(state)
            
            # Consume token and process
            token_consumed = False
            async with state.lock:
                _refill_tokens(state)  # Refill one more time before consuming
                if state.tokens <= 0:
                    # This shouldn't happen after _wait_for_token, but handle gracefully
                    token_consumed = False
                else:
                    state.tokens -= 1
                    token_consumed = True
            
            if not token_consumed:
                # Rare edge case - put item back and retry
                await asyncio.sleep(0.5)
                await state.queue.put((fut, inp_text))
                state.queue.task_done()  # Mark current get() as done since we're putting it back
                continue
            
            # Run inference in threadpool to avoid blocking event loop
            try:
                loop = asyncio.get_running_loop()
                res = await loop.run_in_executor(executor, _sync_model_predict, inp_text)
                fut.set_result(res)
            except Exception as e:
                fut.set_exception(e)
            finally:
                state.queue.task_done()
                
        except Exception as e:
            # Log error but continue processing
            print(f"Error in queue processor for {client_id}: {e}")
            if 'fut' in locals() and not fut.done():
                fut.set_exception(e)
            if 'item' in locals():
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

    # Check for available tokens with lock and consume if available
    has_token = False
    async with state.lock:
        _refill_tokens(state)
        if state.tokens > 0:
            state.tokens -= 1
            has_token = True

    # If we have a token, process immediately
    if has_token:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(executor, _sync_model_predict, inp.text)
        return res

    # No tokens: try to enqueue (up to PER_IP_QUEUE_SIZE). If queue full, reject.
    fut = asyncio.get_running_loop().create_future()
    try:
        state.queue.put_nowait((fut, inp.text))
    except asyncio.QueueFull:
        raise HTTPException(status_code=429, detail="Rate limit exceeded; queue full. Please retry later.")

    # Start background queue processor if not already running
    if not state.processor_running:
        state.processor_running = True
        asyncio.create_task(_process_queue_for_client(state, client_id))

    # Wait for the queued inference to complete and return its result
    try:
        res = await fut
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def client_status(request: Request):
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
        async with state.lock:
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
        async with state.lock:
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
