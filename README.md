# Wine variety classification from sommelier descriptions

## Quick start

### 1) Train (top 4 classes, cap rows for quick iteration)
python train.py --csv winemag-data_first150k.csv --top_n 4 --max_rows 60000 --model distilbert-base-uncased

### 2) Train on all rows for those classes (no cap)
python train.py --csv winemag-data_first150k.csv --top_n 4 --model distilbert-base-uncased

### Outputs
- outputs/<run_name>/metrics.json
- outputs/<run_name>/confusion_matrix.png
- outputs/<run_name>/label_counts.csv
- Hugging Face model checkpoint in outputs/<run_name>/model/

## Serve locally (backend) and use a static frontend

1. Install dependencies (use the project's `requirements.txt`):

```cmd
python -m pip install -r requirements.txt
```

2. Run the FastAPI app (Windows `cmd.exe`):

```cmd
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

3. Open `index.html` in a browser (or host it on any static host). The page's `API URL` input defaults to `http://127.0.0.1:8000/predict` — change it if your backend is reachable at a different address.

Expose your local backend to the public (optional): use tools like `ngrok` or Cloudflare Tunnel, then set the `API URL` on the frontend to `<your-tunnel-url>/predict`.

Notes:
- The app exposes an endpoint `GET /api/examples` that serves `examples.json` to the frontend for the example picker.
- CORS is permissive in `app.py` for local development. Tighten `allow_origins` for production.

Rate limiting and queueing (defaults)
- Per-client (by IP) token-bucket: `10` tokens capacity, refills 1 token every `6` seconds (~10 tokens/minute).
- If a client uses up tokens, requests are placed in a per-client queue (max queued requests: `8`).
- Global model concurrency: `2` worker threads process inference to avoid saturating CPU/GPU.
- Text token limit: `256` tokenizer tokens per request. Requests exceeding this return HTTP `413`.

You can tune the constants in `app.py` near the top: `RATE_LIMIT_TOKENS`, `RATE_REFILL_SECONDS`, `PER_IP_QUEUE_SIZE`, `MODEL_WORKERS`, `MAX_TOKENS_PER_REQUEST`.

Client/device IDs
- The frontend generates a stable device ID stored in `localStorage` under `qea_client_id` and sends it in the `x-client-id` header. This enables device-based rate limiting.
- You can clear the stored device ID in your browser to get a new one.

Status endpoint
- The server exposes `GET /api/status` which returns the requesting client's remaining tokens, queue length, and the number of unique users seen in the past hour. The frontend polls this endpoint and displays the values.
