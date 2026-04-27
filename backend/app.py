"""
Flask API for MASTER — Multi-Agent Stockmarket Trading Environment for Research.

DevHack 2026 Phase-1 architecture:

    React Frontend  ──REST──►  Flask API (this file)
                                    │
                              Simulation (facade)
                                    │
                           OrchestratorAgent (Head Agent)
                           ┌────────┼────────┐
                           ▼        ▼        ▼
                        Agents  Regulator  SQLite DB
                       (5 autonomous       (persistent
                        decision makers)    storage)

Endpoints:
    POST /api/init          – initialise simulation with ticker/period/interval/agents/params
    POST /api/step          – advance one or N simulation steps
    POST /api/auto-step     – advance N steps at once
    POST /api/jump          – jump to a specific step
    POST /api/trigger-crash – trigger a market crash event
    GET  /api/state         – return current simulation snapshot
"""

import sys
import os
import json
import requests as http_requests
import pandas as pd

# Ensure the backend directory is on the Python path so relative imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS
from simulation.simulation import Simulation

app = Flask(__name__)
CORS(app)  # allow React frontend requests

# Global in-memory simulation instance
simulation = Simulation()


@app.route("/")
def home():
    return "Server is running successfully 🚀"


# ------------------------------------------------------------------ #
# POST /api/init
# ------------------------------------------------------------------ #
@app.route("/api/init", methods=["POST"])
def init_simulation():
    """
    Initialize (or re-initialize) the simulation.

    Request body (JSON):
        {
            "ticker":        "AAPL",
            "period":        "5d",
            "interval":      "5m",
            "active_agents": ["conservative", "momentum", ...],  // optional
            "agent_params":  { "conservative": { "risk_pct": 0.05 }, ... }  // optional
        }
    """
    is_multipart = "multipart/form-data" in (request.content_type or "")

    custom_data_df = None
    start_date = None
    end_date = None

    if is_multipart:
        ticker = request.form.get("ticker", "AAPL")
        period = request.form.get("period", "5d")
        interval = request.form.get("interval", "5m")
        start_date = request.form.get("start_date") or None
        end_date = request.form.get("end_date") or None

        active_agents_raw = request.form.get("active_agents")
        agent_params_raw = request.form.get("agent_params")
        active_agents = json.loads(active_agents_raw) if active_agents_raw else None
        agent_params = json.loads(agent_params_raw) if agent_params_raw else None

        csv_file = request.files.get("custom_data")
        if csv_file and csv_file.filename:
            custom_data_df = pd.read_csv(csv_file)
    else:
        data = request.get_json(force=True, silent=True) or {}
        ticker = data.get("ticker", "AAPL")
        period = data.get("period", "5d")
        interval = data.get("interval", "5m")
        active_agents = data.get("active_agents", None)
        agent_params = data.get("agent_params", None)
        start_date = data.get("start_date")
        end_date = data.get("end_date")

    try:
        snapshot = simulation.init_simulation(
            ticker, period, interval,
            active_agents=active_agents,
            agent_params=agent_params,
            start_date=start_date,
            end_date=end_date,
            custom_data_df=custom_data_df,
        )
        return jsonify(snapshot)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


# ------------------------------------------------------------------ #
# POST /api/optimize
# ------------------------------------------------------------------ #
@app.route("/api/optimize", methods=["POST"])
def optimize_simulation():
    """
    Run headless parameter sweep optimization and return trial metrics.

    Request body (JSON):
        {
            "ticker": "AAPL",
            "period": "5d",
            "interval": "5m",
            "active_agents": [...],
            "agent_params": {...},
            "parameter": "custom.basic.position_size_pct",
            "min": 0.05,
            "max": 0.25,
            "step": 0.05,
            "start_date": "2020-02-01",   // optional
            "end_date": "2020-04-30"       // optional
        }
    """
    data = request.get_json(force=True, silent=True) or {}
    try:
        results = simulation.optimize(
            ticker=data.get("ticker", "AAPL"),
            period=data.get("period", "5d"),
            interval=data.get("interval", "5m"),
            active_agents=data.get("active_agents"),
            agent_params=data.get("agent_params"),
            parameter=data.get("parameter"),
            min_value=data.get("min"),
            max_value=data.get("max"),
            step_value=data.get("step"),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
        )
        if not isinstance(results, dict):
            results = {"results": []}
        if not isinstance(results.get("results"), list):
            results["results"] = []
        if "error" in results and not results["results"]:
            results["results"].append({"error": results["error"]})
        return jsonify(results)
    except Exception as e:
        return jsonify({
            "error": str(e),
            "results": [{"error": str(e)}],
        }), 500


# ------------------------------------------------------------------ #
# POST /api/step
# ------------------------------------------------------------------ #
@app.route("/api/step", methods=["POST"])
def step_simulation():
    """
    Advance simulation by one or N steps.

    Query params:
        ?n=5   → batch-step 5 bars at once (default 1)
    """
    n = request.args.get("n", 1, type=int)
    try:
        if n <= 1:
            snapshot = simulation.step_simulation()
        else:
            snapshot = simulation.batch_step(n)
        if "error" in snapshot:
            return jsonify(snapshot), 400
        return jsonify(snapshot)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------ #
# POST /api/auto-step
# ------------------------------------------------------------------ #
@app.route("/api/auto-step", methods=["POST"])
def auto_step_simulation():
    """
    Run N simulation steps in one call.

    Request body (JSON):
        { "steps": 10 }      (default 10)
    """
    data = request.get_json(force=True, silent=True) or {}
    n = data.get("steps", 10)
    n = min(int(n), 200)

    try:
        snapshot = simulation.batch_step(n)
        if snapshot is None:
            return jsonify({"error": "No steps executed."}), 400
        return jsonify(snapshot)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------ #
# POST /api/jump
# ------------------------------------------------------------------ #
@app.route("/api/jump", methods=["POST"])
def jump_to_step():
    """
    Jump (fast-forward or rewind) to a specific simulation step.

    Request body (JSON):
        { "step": 42 }
    """
    data = request.get_json(force=True, silent=True) or {}
    target = data.get("step", 0)
    try:
        snapshot = simulation.jump_to_step(int(target))
        if "error" in snapshot:
            return jsonify(snapshot), 400
        return jsonify(snapshot)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------ #
# POST /api/trigger-crash
# ------------------------------------------------------------------ #
@app.route("/api/trigger-crash", methods=["POST"])
def trigger_crash():
    """
    Trigger a market crash: 15-20% price drop, tripled volatility,
    circuit breakers on vulnerable agents.
    """
    try:
        snapshot = simulation.trigger_crash()
        if "error" in snapshot:
            return jsonify(snapshot), 400
        return jsonify(snapshot)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------ #
# POST /api/set-agents
# ------------------------------------------------------------------ #
@app.route("/api/set-agents", methods=["POST"])
def set_agents():
    """
    Enable / disable agents mid-simulation without re-initialising.

    Request body (JSON):
        { "active_agents": ["conservative", "momentum", ...] }
    """
    data = request.get_json(force=True, silent=True) or {}
    active_agents = data.get("active_agents", [])
    try:
        snapshot = simulation.set_active_agents(active_agents)
        if "error" in snapshot:
            return jsonify(snapshot), 400
        return jsonify(snapshot)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------ #
# POST /api/liquidate-agent
# ------------------------------------------------------------------ #
@app.route("/api/liquidate-agent", methods=["POST"])
def liquidate_agent():
    """
    Sell all open positions for a specific agent at current market price.

    Request body (JSON):
        { "agent_key": "momentum" }
    """
    data = request.get_json(force=True, silent=True) or {}
    agent_key = data.get("agent_key", "")
    if not agent_key:
        return jsonify({"error": "agent_key is required."}), 400
    try:
        snapshot = simulation.liquidate_agent(agent_key)
        if "error" in snapshot:
            return jsonify(snapshot), 400
        return jsonify(snapshot)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------ #
# GET /api/state
# ------------------------------------------------------------------ #
@app.route("/api/state", methods=["GET"])
def get_state():
    """Return the current simulation snapshot."""
    try:
        snapshot = simulation.get_snapshot()
        if "error" in snapshot:
            return jsonify(snapshot), 400
        return jsonify(snapshot)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------ #
# GET /api/ollama-models
# ------------------------------------------------------------------ #
@app.route("/api/ollama-models", methods=["GET"])
def get_ollama_models():
    """Return list of locally available Ollama models."""
    try:
        resp = http_requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        result = [
            {
                "name": m["name"],
                "size_gb": round(m.get("size", 0) / 1e9, 1),
                "family": m.get("details", {}).get("family", ""),
                "parameters": m.get("details", {}).get("parameter_size", ""),
            }
            for m in models
        ]
        return jsonify({"models": result, "available": True})
    except Exception:
        return jsonify({"models": [], "available": False, "error": "Ollama not running"})


# ------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("Starting MASTER server on http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)
