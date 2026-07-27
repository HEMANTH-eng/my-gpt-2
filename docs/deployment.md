# Production Deployment Guide

Comprehensive deployment documentation for orchestrating the MyGPT FastAPI model backend and Next.js web application.

---

## 1. Quickstart Container Deployment

The project provides multi-stage Docker builds and Docker Compose orchestration for one-command deployment.

### Prerequisites:
- [Docker Engine](https://docs.docker.com/engine/install/) $\ge$ 20.10
- [Docker Compose](https://docs.docker.com/compose/install/) $\ge$ 2.0

### Step-by-Step Deployment:

1. **Clone & Configure Environment**:
   ```bash
   cp .env.example .env
   ```

2. **Execute Deployment Script**:
   * **Linux / macOS**:
     ```bash
     chmod +x scripts/deploy.sh
     ./scripts/deploy.sh
     ```
   * **Windows PowerShell**:
     ```powershell
     .\scripts\deploy.ps1
     ```

3. **Verify Active Services**:
   - **Next.js Web Interface**: `http://localhost:3000`
   - **FastAPI Documentation**: `http://localhost:8000/docs`
   - **API Health Check**: `http://localhost:8000/api/v1/health`

---

## 2. Docker Architecture & Services

The `docker-compose.yml` file configures two decoupled services linked via an isolated bridge network (`gpt-net`):

```text
                               ┌───────────────────────────┐
                               │       Reverse Proxy       │
                               │      (Nginx / Traefik)    │
                               └─────────────┬─────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       │                                           │
         ┌─────────────▼─────────────┐               ┌─────────────▼─────────────┐
         │     Web Container         │               │     API Container         │
         │   Node 20 / Next.js       │               │   Python 3.11 / FastAPI   │
         │   Port 3000               │               │   Port 8000               │
         └───────────────────────────┘               └─────────────┬─────────────┘
                                                                   │
                                                     ┌─────────────▼─────────────┐
                                                     │ Model Checkpoint Storage  │
                                                     │ Docker Volume / Path      │
                                                     └───────────────────────────┘
```

| Service | Dockerfile | Internal Port | Environment Variables | Volumes |
| :--- | :--- | :--- | :--- | :--- |
| `api` | `Dockerfile.api` | `8000` | `HOST`, `PORT`, `DEVICE`, `MODEL_CONFIG` | `./checkpoints`, `./runs` |
| `web` | `Dockerfile.web` | `3000` | `NODE_ENV`, `NEXT_PUBLIC_API_URL` | None |

---

## 3. Environment Variables Reference

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `HOST` | `0.0.0.0` | Network binding interface for Uvicorn. |
| `PORT` | `8000` | API server listening port. |
| `DEVICE` | `cpu` | PyTorch execution device (`cpu` or `cuda`). |
| `MODEL_CONFIG` | `gpt_micro` | Model preset configuration (`gpt_micro`, `gpt2_small`, `gpt2_medium`). |
| `CORS_ORIGINS` | `*` | Allowed CORS origins for cross-domain API access. |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000/api/v1` | Public API endpoint URL consumed by Next.js frontend. |

---

## 4. Production Nginx Reverse Proxy Setup

To expose the services securely over SSL (`https://`), configure an Nginx server block:

```nginx
server {
    listen 80;
    server_name mygpt.example.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name mygpt.example.com;

    ssl_certificate /etc/letsencrypt/live/mygpt.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mygpt.example.com/privkey.pem;

    # Next.js Frontend
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # FastAPI Model Backend
    location /api/v1/ {
        proxy_pass http://127.0.0.1:8000/api/v1/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Enable HTTP streaming for real-time tokens
        proxy_buffering off;
        proxy_read_timeout 600s;
    }
}
```

---

## 5. Systemd Service Unit File Setup

For standalone Linux server deployments without Docker Compose, create `/etc/systemd/system/mygpt-api.service`:

```ini
[Unit]
Description=MyGPT FastAPI Model Server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/my-gpt
ExecStart=/opt/my-gpt/.venv/bin/python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=5
Environment=DEVICE=cuda
Environment=MODEL_CONFIG=gpt2_small

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now mygpt-api
```

---

## 6. GPU Acceleration (CUDA Docker)

To run the API container with NVIDIA GPU acceleration, ensure `nvidia-container-toolkit` is installed on the host and update `docker-compose.yml`:

```yaml
services:
  api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - DEVICE=cuda
```
