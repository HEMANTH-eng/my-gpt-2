#!/usr/bin/env bash
# ==============================================================================
# MyGPT Production Deployment Script (POSIX Bash)
# ==============================================================================

set -e

echo "=== [MyGPT Deployment Engine] ==="

# 1. Verify Docker & Docker Compose installation
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed or not available in PATH."
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo "ERROR: 'docker compose' plugin is not available."
    exit 1
fi

# 2. Check for .env file
if [ ! -f ".env" ]; then
    echo "[!] .env file not found. Copying default configuration from .env.example..."
    cp .env.example .env
fi

# 3. Build and launch Docker Compose services
echo "[+] Building and launching containers via Docker Compose..."
docker compose build --parallel
docker compose up -d

# 4. Perform Healthcheck Verification
echo "[+] Verifying service health..."
MAX_RETRIES=15
RETRY_COUNT=0

until curl -s http://localhost:8000/api/v1/health | grep -q '"status":"ok"'; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: API Healthcheck failed after $MAX_RETRIES attempts."
        docker compose logs api
        exit 1
    fi
    echo "    Waiting for API server to initialize... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

echo "=== [Deployment Successful!] ==="
echo "  • API Endpoint: http://localhost:8000/api/v1/health"
echo "  • Swagger UI:   http://localhost:8000/docs"
echo "  • Web Interface: http://localhost:3000"
