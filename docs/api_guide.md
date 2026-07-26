# MyGPT API Guide & OpenAPI Documentation

The MyGPT API server exposes endpoints for text generation, multi-turn chat completions, system health checks, and model specifications.

## Server Launch Command

Start the API server locally using Uvicorn:

```powershell
uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

- **Interactive Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`

---

## Endpoint Specifications

### 1. Health Check
`GET /api/v1/health`

Returns system health status and active hardware device.

#### Example Request:
```bash
curl -X GET "http://127.0.0.1:8000/api/v1/health"
```

#### Example Response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu"
}
```

---

### 2. Model Information
`GET /api/v1/info`

Returns parameter counts and layer configurations.

#### Example Request:
```bash
curl -X GET "http://127.0.0.1:8000/api/v1/info"
```

#### Example Response:
```json
{
  "model_name": "MyGPT-Micro",
  "num_parameters": 814720,
  "vocab_size": 300,
  "d_model": 128,
  "n_layer": 4,
  "n_head": 4,
  "max_seq_len": 64
}
```

---

### 3. Text Generation
`POST /api/v1/generate`

Generates text continuations for a given prompt string.

#### Example Request:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "Building a custom GPT model",
           "max_new_tokens": 30,
           "temperature": 0.8,
           "top_k": 40,
           "top_p": 0.95
         }'
```

#### Example Response:
```json
{
  "prompt": "Building a custom GPT model",
  "generated_text": "Building a custom GPT model from scratch using PyTorch...",
  "tokens_generated": 30
}
```

---

### 4. Chat Completion
`POST /api/v1/chat`

Generates multi-turn assistant responses for conversational prompts.

#### Example Request:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat" \
     -H "Content-Type: application/json" \
     -d '{
           "messages": [
             {"role": "system", "content": "You are a helpful AI coding assistant."},
             {"role": "user", "content": "Explain self-attention briefly."}
           ],
           "max_new_tokens": 40,
           "temperature": 0.7
         }'
```

#### Example Response:
```json
{
  "message": {
    "role": "assistant",
    "content": "Self-attention allows tokens in a sequence to compute pairwise similarity scores..."
  }
}
```
