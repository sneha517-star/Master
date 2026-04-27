#!/usr/bin/env zsh
# MASTER — start backend + frontend

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"

# Kill stale processes
lsof -ti :5001 | xargs kill -9 2>/dev/null
lsof -ti :5173 | xargs kill -9 2>/dev/null
sleep 0.5

echo "Starting backend..."
(cd "$ROOT/backend" && "$PYTHON" app.py) &
BACKEND_PID=$!

echo "Starting frontend..."
(cd "$ROOT/frontend" && npm run dev) &
FRONTEND_PID=$!

echo ""
echo "MASTER is running:"
echo "  Backend  → http://localhost:5001"
echo "  Frontend → http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers."

wait
