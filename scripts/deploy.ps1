# ==============================================================================
# MyGPT Production Deployment Script (PowerShell for Windows)
# ==============================================================================

$ErrorActionPreference = "Stop"

Write-Host "=== [MyGPT Deployment Engine (PowerShell)] ===" -ForegroundColor Cyan

# 1. Check Docker & Compose
try {
    $null = docker --version
} catch {
    Write-Error "ERROR: Docker is not installed or not running."
    exit 1
}

# 2. Check .env file
if (-not (Test-Path ".env")) {
    Write-Host "[!] .env file not found. Copying from .env.example..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
}

# 3. Build & Launch Containers
Write-Host "[+] Building and launching containers via Docker Compose..." -ForegroundColor Green
docker compose build
docker compose up -d

# 4. Health Check Verification
Write-Host "[+] Verifying API service health..." -ForegroundColor Green
$maxRetries = 15
$retries = 0
$healthy = $false

while ($retries -lt $maxRetries) {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/health" -Method Get -TimeoutSec 2
        if ($response.status -eq "ok") {
            $healthy = $true
            break
        }
    } catch {
        # Retry loop
    }
    $retries++
    Write-Host "    Waiting for API server initialization... ($retries/$maxRetries)" -ForegroundColor Yellow
    Start-Sleep -Seconds 2
}

if ($healthy) {
    Write-Host "=== [Deployment Successful!] ===" -ForegroundColor Cyan
    Write-Host "  • API Endpoint:  http://localhost:8000/api/v1/health"
    Write-Host "  • Swagger UI:    http://localhost:8000/docs"
    Write-Host "  • Web Interface: http://localhost:3000"
} else {
    Write-Host "ERROR: Health check failed after $maxRetries attempts." -ForegroundColor Red
    docker compose logs api
}
