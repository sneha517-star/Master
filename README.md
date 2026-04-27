# MASTER — Multi-Agent Stockmarket Trading Environment for Research

A simulated financial ecosystem populated by **multiple autonomous trading agents** operating in a shared market. Each agent analyses real market data (via **yfinance**), trades under uncertainty, manages its portfolio & risk, and adapts over time — including an adversarial "whale" agent that attempts pump-and-dump manipulation and a **Custom Agent** powered by a local LLM via **Ollama**.

A **Regulator** module enforces compliance constraints, detects contrarian trades during crashes, and blocks or warns on suspicious behaviour. An **OrchestratorAgent** (Head Agent) coordinates the simulation clock, triggers whale-crash cascade scenarios, and activates global circuit breakers when systemic risk thresholds are breached.

Every decision includes a structured **reasoning** string for transparent, interpretable, post-hoc auditing.

---

## Tech Stack

| Layer    | Technology |
|----------|------------|
| Backend  | Python 3 · Flask · yfinance · pandas · numpy · SQLite |
| Frontend | React 18 (Vite) · Lightweight Charts · Recharts · jsPDF · Axios |
| LLM      | Ollama (local) · qwen2.5:3b / qwen2.5:7b (configurable) |
| Data     | SQLite (devhack.db) + in-memory dual-write logging |

---

## Agents

| Agent | Strategy | Goal |
|-------|----------|------|
| **Conservative** | Low volatility filter, small positions, tight stop-loss | Preserve capital |
| **Momentum** | SMA20/SMA50 crossover trend following | Ride established trends |
| **MeanReversion** | Bollinger Band mean-reversion (buy BB_LOW, sell BB_UP) | Exploit price extremes |
| **NoiseTrader** | Random small trades to inject realistic noise | Simulate retail activity |
| **Adversarial** | Pump-and-dump: burst buys in low-volume, dumps on gain | Stress-test the Regulator |
| **Custom (LLM)** | User-defined via Builder — basic rules, advanced IF-THEN, or **LLM-powered** | User's custom goal |

All agents return a **structured decision dict**:
```python
{"action": "BUY"|"SELL"|"HOLD", "quantity": int, "reasoning": str}
```

### Custom Agent — LLM Mode

The Custom Agent can operate in three modes, configured via the **Builder** panel:

| Mode | Description |
|------|-------------|
| **Basic** | Simple entry/exit rules (SMA crossover, Bollinger oversold, stop-loss) |
| **Advanced** | IF-THEN condition blocks with custom indicators, operators, and actions |
| **LLM** | Queries a local Ollama model (e.g. qwen2.5:3b) with market context; model returns JSON trading decisions |

LLM mode features:
- **Model selection** — choose from locally installed Ollama models
- **Trading style** — conservative / balanced / aggressive risk profiles
- **Custom instructions** — free-text prompt injection (e.g. "Only buy on dips")
- **Response caching** — LLM queried every N steps (configurable), cached in between for speed
- **GPU pre-warming** — model loaded into VRAM on init to eliminate cold-start latency

---

## Key Features

- **Structured decisions with reasoning** — every agent explains why it acted
- **6 autonomous agents** — 5 built-in strategies + 1 user-customisable (with LLM support)
- **Local LLM integration** — Ollama-powered trading decisions with zero cloud dependency
- **Strategy Builder UI** — visual drag-and-drop strategy creation (basic / advanced / LLM modes)
- **5 compliance rules** — MaxPosition, MaxOrder, BurstTrading, AdversarialFlag, ContrarianCrashDetection
- **Whale Manipulation / Cascade Crash** — one-click demo: whale dumps → price crash → momentum agents panic-sell → global circuit breaker halts trading
- **Global circuit breaker** — trading halted system-wide at −15 % drawdown
- **Per-agent circuit breakers** — individual agents halted at −10 % drawdown
- **Endogenous price impact** — agent order flow moves the simulated price
- **Follower multiplier** — each agent can have configurable followers that amplify market impact
- **Live agent toggle** — enable/disable agents mid-simulation with liquidation options
- **Simulation summary & PDF export** — auto-generated report with KPIs, agent performance table, trade log, and risk metrics
- **Candlestick chart** — TradingView-style chart (Lightweight Charts) with SMA, Bollinger Bands, and trade markers
- **Auto-recovery** — frontend automatically re-initialises if the backend restarts mid-simulation
- **Error boundary** — React error boundary prevents blank screens on render failures
- **SQLite audit trail** — every trade and regulation event persisted
- **Dark / Light theme** — toggle between themes

---

## Project Structure

```
backend/
  app.py                       # Flask API server (9 endpoints)
  db.py                        # SQLite database layer
  requirements.txt
  market/
    market.py                  # Market data download, indicators & endogenous price model
  agents/
    base_agent.py              # TradingAgent base class (perceive→reason→act pipeline)
    conservative_agent.py
    momentum_agent.py
    mean_reversion_agent.py
    noise_trader.py
    adversarial_agent.py
    custom_agent.py            # Custom agent — basic/advanced/LLM modes via JSON recipe
  regulator/
    regulator.py               # 5 compliance rules incl. crash-contrarian detection
  simulation/
    simulation.py              # Facade delegating to OrchestratorAgent
    orchestrator.py            # Head Agent: clock, whale crash, cascade, circuit breakers
  logging_utils/
    logger.py                  # Dual-write audit trail (memory + SQLite)
frontend/
  src/
    api/client.js              # Axios API client (180 s timeout)
    components/
      TopBar.jsx               # Ticker selector, period, theme toggle, balance display
      LeftSidebar.jsx          # Tab navigation (Trades, Market, Agents, Stats, Help, Builder)
      RightTradePanel.jsx      # Controls: step, auto-run, pause, speed, agent toggles
      PriceChart.jsx           # TradingView-style candlestick chart (Lightweight Charts)
      AgentsPanel.jsx          # Agent cards with reasoning + risk metrics
      BuilderPanel.jsx         # Custom Strategy Builder (basic / advanced / LLM)
      MarketPanel.jsx          # Market overview & price data
      StatsPanel.jsx           # Detailed statistics
      PerformanceCharts.jsx    # Portfolio + violation charts (Recharts)
      RiskOverviewPanel.jsx    # System-wide risk dashboard
      TradeLogTable.jsx        # Filterable trade log
      RegulationLogTable.jsx   # Regulation event log
      SettingsModal.jsx        # Per-agent parameter tuning
      DisableAgentModal.jsx    # Confirm disable with liquidation option
      SimulationSummaryModal.jsx # End-of-sim report + PDF download (jsPDF)
      HelpPanel.jsx            # User guide
      ErrorBoundary.jsx        # React error boundary for crash recovery
    App.jsx                    # Main app — state management, auto-run loop, recovery logic
    main.jsx                   # Entry point with ErrorBoundary wrapper
    index.css                  # Full theme system (dark/light)
README.md
PROJECT_REPORT.txt
```

---

## How to Run

### Prerequisites

- **Python 3.10+** with pip
- **Node.js 18+** with npm
- **Ollama** (optional, for LLM agent) — [install from ollama.com](https://ollama.com)

### 1. Backend

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Start Flask server (port 5001)
python app.py
```

### 2. Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start Vite dev server (port 5173)
npm run dev
```

### 3. Ollama (optional — for LLM Custom Agent)

```bash
# Install a small model (recommended for fast inference)
ollama pull qwen2.5:3b

# Or a larger model for better reasoning
ollama pull qwen2.5:7b

# Ollama runs automatically on localhost:11434
```

---

## How to Use

1. Open the React UI at **http://localhost:5173** (or the port Vite assigns).
2. Select a **ticker** (e.g. AAPL, TSLA, RELIANCE.NS), **period**, and **interval** in the top bar.
3. Click **Init** to download market data and create agents.
4. Click **Step** to advance one bar at a time, or **Auto-Run** to continuously simulate.
5. Adjust **speed** (ms per step) and **batch size** in the right panel.
6. Toggle individual agents on/off mid-simulation (with optional position liquidation).
7. Observe:
   - **Price Chart** — TradingView-style candlesticks with SMA, Bollinger Bands & trade markers.
   - **Agent cards** — cash, positions, portfolio value, last action, reasoning, risk metrics.
   - **Trade Log** — all trades with regulator decisions, filterable by agent.
   - **Regulation Events** — compliance blocks and warnings.
   - **Performance Charts** — portfolio value over time per agent + violations bar chart.
   - **Risk Overview** — AUM, exposure, drawdown, circuit breakers.
8. **Builder tab** — create a custom strategy:
   - **Basic mode**: pick entry/exit rules + position size.
   - **Advanced mode**: build IF-THEN condition blocks with indicators & operators.
   - **LLM mode**: select an Ollama model, trading style, and custom instructions.
   - Click **Save & Initialize Now** to deploy and run your custom agent.
9. Click **Trigger Crash** to demo the whale manipulation cascade:
   - Adversarial whale dumps 100 % of holdings.
   - Price drops 15-20 %, indicators recalculate.
   - Momentum / other agents react with stop-loss sells.
   - If global drawdown exceeds −15 %, **all trading is halted**.
10. When the simulation finishes, a **Summary Modal** appears with full KPIs — click **Download PDF** for a report.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/init` | Init simulation `{"ticker","period","interval","active_agents?","agent_params?"}` |
| POST | `/api/step` | Advance one or N steps `?n=5` |
| POST | `/api/auto-step` | Batch-run N steps `{"steps": 10}` |
| POST | `/api/jump` | Jump to step `{"step": 42}` |
| POST | `/api/trigger-crash` | Trigger whale dump + cascade crash |
| POST | `/api/set-agents` | Enable/disable agents mid-sim `{"active_agents": [...]}` |
| POST | `/api/liquidate-agent` | Sell all positions for an agent `{"agent_key": "momentum"}` |
| GET  | `/api/state` | Get current simulation snapshot |
| GET  | `/api/ollama-models` | List locally available Ollama models |

Snapshot includes `trading_status: "ACTIVE" | "HALTED_BY_CIRCUIT_BREAKER"` and full per-agent status (`ACTIVE` / `DISABLED` / `HALTED`).

---

## Architecture

```
React Frontend ──REST──► Flask API (app.py, port 5001)
                              │
                        Simulation (facade)
                              │
                     OrchestratorAgent (Head Agent)
                     ┌────────┼────────┐
                     ▼        ▼        ▼
                  Agents   Regulator  SQLite DB
                (6 autonomous        (persistent
                 decision makers)     audit trail)
                     │
              Custom Agent ──HTTP──► Ollama (localhost:11434)
                                      └── qwen2.5:3b / 7b
```

---

## License

MIT  / educational purposes.
