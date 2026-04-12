#!/usr/bin/env bash
# ── deploy-aws.sh ────────────────────────────────────────────────────────────
# Quick-start script for AWS g5.2xlarge (NVIDIA A10G) deployment.
#
# Prerequisites on the EC2 instance:
#   1. NVIDIA driver + Container Toolkit installed
#   2. Docker + Docker Compose v2 installed
#   3. .env file configured (cp .env.aws .env && edit)
#
# Usage:
#   chmod +x deploy-aws.sh
#   ./deploy-aws.sh          # build + deploy
#   ./deploy-aws.sh --pull   # pull latest, rebuild, deploy
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COMPOSE_FILE="docker-compose.aws.yml"

# ── Pre-flight checks ────────────────────────────────────────────────────────
echo "==> Pre-flight checks"

if ! command -v docker &>/dev/null; then
  echo "ERROR: docker not found. Install Docker first."
  exit 1
fi

if ! docker compose version &>/dev/null; then
  echo "ERROR: docker compose v2 not found."
  exit 1
fi

if ! nvidia-smi &>/dev/null; then
  echo "WARNING: nvidia-smi not available. GPU may not be accessible."
fi

if [[ ! -f .env ]]; then
  echo "WARNING: .env not found. Copying from .env.aws template…"
  cp .env.aws .env
  echo "  → Edit .env to set HF_TOKEN and other secrets, then re-run."
  exit 1
fi

# ── Optional: git pull ───────────────────────────────────────────────────────
if [[ "${1:-}" == "--pull" ]]; then
  echo "==> Pulling latest code"
  git pull --ff-only
fi

# ── Build + deploy ───────────────────────────────────────────────────────────
echo "==> Building and starting services (compose file: $COMPOSE_FILE)"
docker compose -f "$COMPOSE_FILE" up --build -d

echo ""
echo "==> Deployment started. Checking service health…"
sleep 5

docker compose -f "$COMPOSE_FILE" ps

echo ""
echo "==> Backend health check:"
curl -sf http://localhost:8000/health 2>/dev/null && echo "" || echo "(backend still starting — check 'docker compose -f $COMPOSE_FILE logs -f backend')"

echo ""
echo "==> Frontend available at: http://$(curl -sf http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo '<instance-ip>'):80"
echo "==> Logs: docker compose -f $COMPOSE_FILE logs -f"
