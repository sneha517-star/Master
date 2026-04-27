from __future__ import annotations

"""
Market engine module — Endogenous Price Impact Model.

Downloads real market data via yfinance, computes technical indicators,
and provides a step-based environment where **prices respond to net
buy/sell volume from agents** (not just a passive replay engine).

Price impact formula (per step):
    impact   = sensitivity_factor × net_volume          (capped ±20 %)
    sim_next = historical_close_next × (1 + impact)

When agents are net buyers the simulated price rises above the historical
baseline; when they are net sellers it drops below.  With net_volume == 0
the model degrades to pure historical replay, preserving full backward
compatibility.
"""

import yfinance as yf
import pandas as pd
import numpy as np


def _merge_custom_columns(market_df: pd.DataFrame, custom_data_df: pd.DataFrame | None) -> pd.DataFrame:
    """Merge user-uploaded custom CSV columns into market data by calendar date."""
    if custom_data_df is None or custom_data_df.empty:
        return market_df

    df_market = market_df.copy()
    df_custom = custom_data_df.copy()

    date_col = None
    for c in df_custom.columns:
        if str(c).strip().lower() == "date":
            date_col = c
            break
    if date_col is None:
        raise ValueError("Custom CSV must include a 'Date' column.")

    df_custom[date_col] = pd.to_datetime(df_custom[date_col], errors="coerce")
    df_custom = df_custom.dropna(subset=[date_col]).copy()
    if df_custom.empty:
        return df_market

    # Build date-only merge key to support daily signals on intraday bars.
    df_market["__merge_date"] = pd.to_datetime(df_market["Datetime"], errors="coerce").dt.date
    df_custom["__merge_date"] = pd.to_datetime(df_custom[date_col], errors="coerce").dt.date

    drop_cols = {date_col, "__merge_date"}
    custom_cols = [c for c in df_custom.columns if c not in drop_cols]

    # If duplicates exist on same date, keep the last uploaded row.
    df_custom = df_custom.drop_duplicates(subset=["__merge_date"], keep="last")
    merged = df_market.merge(
        df_custom[["__merge_date", *custom_cols]],
        on="__merge_date",
        how="left",
    )

    if custom_cols:
        merged[custom_cols] = merged[custom_cols].ffill()

    merged.drop(columns=["__merge_date"], inplace=True)
    return merged


def download_market_data(
    ticker: str,
    period: str,
    interval: str,
    start_date: str | None = None,
    end_date: str | None = None,
    custom_data_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Download historical market data using yfinance.

    Args:
        ticker:   e.g. "AAPL", "NSEI", "RELIANCE.NS"
        period:   e.g. "1d", "5d", "1mo"
        interval: e.g. "5m", "15m", "1h", "1d"

    Returns:
        pandas DataFrame with columns:
        Datetime, Open, High, Low, Close, Volume

    Raises:
        ValueError if no data could be downloaded.
    """
    ticker_obj = yf.Ticker(ticker)
    requested_start = pd.to_datetime(start_date).tz_localize(None) if start_date else None

    if start_date and end_date:
        # Pull an additional lookback window so indicators are valid from day 1.
        lookback_start = (requested_start - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
        df = ticker_obj.history(start=lookback_start, end=end_date, interval=interval)
    else:
        df = ticker_obj.history(period=period, interval=interval)

    if df is None or df.empty:
        raise ValueError(
            f"No data downloaded for ticker='{ticker}', "
            f"period='{period}', interval='{interval}'. "
            "Check that the ticker symbol is valid and yfinance is reachable."
        )

    # Reset index so that the datetime becomes a regular column
    df = df.reset_index()

    # Normalise the datetime column name
    if "Date" in df.columns:
        df.rename(columns={"Date": "Datetime"}, inplace=True)
    elif "index" in df.columns:
        df.rename(columns={"index": "Datetime"}, inplace=True)

    # Keep only the columns we care about
    keep = [c for c in ["Datetime", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()

    # Drop duplicate timestamps (yfinance can return duplicates at day boundaries)
    df = df.drop_duplicates(subset=["Datetime"], keep="last").reset_index(drop=True)

    # Merge optional user-supplied alternative dataset.
    df = _merge_custom_columns(df, custom_data_df)

    if end_date:
        requested_end = pd.to_datetime(end_date).tz_localize(None)
        dt = pd.to_datetime(df["Datetime"], errors="coerce")
        df = df.loc[dt <= requested_end].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"No rows available for ticker='{ticker}' in requested date range."
        )

    return df


class MarketEnvironment:
    """
    Endogenous Price Impact Model: prices respond to net buy/sell volume
    from agents.

    Downloads historical data, pre-computes technical indicators on the
    **unmodified** historical Close series, and maintains a separate
    **simulated price** that diverges from the historical baseline in
    proportion to net agent order flow.

    Key attributes
    --------------
    current_price           – latest simulated price (the actual trade price)
    price_history_simulated – full series of simulated prices so far
    sensitivity_factor      – how strongly net volume moves the price
    df["Close"]             – raw historical closes (never mutated)
    """

    # Default sensitivity: 1 unit of net volume moves the price by
    # 0.001 % of its historical value.  Tune for realistic results.
    DEFAULT_SENSITIVITY = 1e-5

    def __init__(
        self,
        ticker: str,
        period: str,
        interval: str,
        start_date: str | None = None,
        end_date: str | None = None,
        custom_data_df: pd.DataFrame | None = None,
        sensitivity_factor: float | None = None,
    ):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.current_step = 0

        # ----- Endogenous Price Impact – sensitivity coefficient ------
        # Configurable: pass in constructor or modify after creation.
        self.sensitivity_factor: float = (
            sensitivity_factor
            if sensitivity_factor is not None
            else self.DEFAULT_SENSITIVITY
        )

        # Download raw historical data (with optional date range / custom columns)
        self.df = download_market_data(
            ticker,
            period,
            interval,
            start_date=start_date,
            end_date=end_date,
            custom_data_df=custom_data_df,
        )

        # ---------- Pre-compute technical indicators (on historical) --
        # NOTE: indicators are always computed on the raw historical
        # Close so they do not drift with the simulated price.
        close = self.df["Close"]

        # Simple Moving Averages
        self.df["SMA20"] = close.rolling(window=20, min_periods=1).mean()
        self.df["SMA50"] = close.rolling(window=50, min_periods=1).mean()

        # Bollinger Bands (20-period)
        rolling_std = close.rolling(window=20, min_periods=1).std()
        self.df["BB_MID"] = self.df["SMA20"]
        self.df["BB_UP"] = self.df["SMA20"] + 2 * rolling_std
        self.df["BB_LOW"] = self.df["SMA20"] - 2 * rolling_std

        # Rolling volatility – 20-period std of log returns
        log_returns = np.log(close / close.shift(1))
        self.df["Volatility"] = log_returns.rolling(window=20, min_periods=1).std()

        # Fill any remaining NaNs with 0
        self.df.fillna(0, inplace=True)

        # For explicit historical stress tests, keep only rows from requested
        # start date onward *after* indicators are computed with lookback data.
        if start_date:
            requested_start = pd.to_datetime(start_date).tz_localize(None)
            dt = pd.to_datetime(self.df["Datetime"], errors="coerce")
            self.df = self.df.loc[dt >= requested_start].reset_index(drop=True)

        if self.df.empty:
            raise ValueError("No market data available after applying requested start_date/end_date.")

        # Total number of bars available
        self.total_bars = len(self.df)

        # ── Simulated price tracking ─────────────────────────────────
        # Initialise simulated price to the first historical close.
        # self.df["Close"] is NEVER mutated; the simulated series lives
        # entirely in self.current_price / self.price_history_simulated.
        initial_close = float(self.df.iloc[0]["Close"])
        self.current_price: float = initial_close
        self.price_history_simulated: list[float] = [initial_close]

    # ------------------------------------------------------------------ #
    # Reset (rewind without re-downloading)
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset the market to step 0 without re-downloading data."""
        self.current_step = 0
        initial_close = float(self.df.iloc[0]["Close"])
        self.current_price = initial_close
        self.price_history_simulated = [initial_close]

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _bar_to_dict(self, idx: int) -> dict:
        """
        Convert a single DataFrame row to a plain dict.

        * ``Close`` remains the **raw historical** close.
        * ``SimulatedPrice`` is the endogenous impact-adjusted price
          that agents should treat as the actual trade price.
        """
        row = self.df.iloc[idx]
        d = row.to_dict()
        # Convert Timestamp / datetime objects to ISO strings for JSON
        if "Datetime" in d and hasattr(d["Datetime"], "isoformat"):
            d["Datetime"] = d["Datetime"].isoformat()

        # Inject the simulated price alongside the historical Close.
        # For the current step use self.current_price; for past bars
        # look up the simulated history (falls back to historical).
        if idx < len(self.price_history_simulated):
            d["SimulatedPrice"] = round(self.price_history_simulated[idx], 4)
        else:
            d["SimulatedPrice"] = round(float(row["Close"]), 4)
        return d

    def _build_step_state(self, idx: int) -> dict:
        """
        Build the compact state dict returned by ``step()``.

        Contains both the historical and simulated prices plus the
        technical indicators agents need.  Agents should use
        ``simulated_price`` as the actual trade price.
        """
        row = self.df.iloc[idx]
        return {
            "t": self.current_step,
            "historical_price": float(row["Close"]),
            "simulated_price": float(self.current_price),
            "sma20": float(row["SMA20"]),
            "sma50": float(row["SMA50"]),
            "bb_up": float(row["BB_UP"]),
            "bb_low": float(row["BB_LOW"]),
            "volatility": float(row["Volatility"]),
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_state(self) -> dict:
        """
        Return the current market state for dashboard / orchestrator.

        Returns dict with both historical **and** simulated prices:
            current_bar             – row dict with Close (hist) + SimulatedPrice
            recent_window           – list of bar dicts (last 50 bars)
            step                    – current step index
            total_bars              – total available bars
            simulated_price         – current endogenous price (convenience)
            price_history_simulated – full simulated price series so far
        """
        current_bar = self._bar_to_dict(self.current_step)

        start = max(0, self.current_step - 49)
        recent = [
            self._bar_to_dict(i)
            for i in range(start, self.current_step + 1)
        ]

        return {
            "current_bar": current_bar,
            "recent_window": recent,
            "step": self.current_step,
            "total_bars": self.total_bars,
            # Endogenous price data for the dashboard / agents
            "simulated_price": round(self.current_price, 4),
            "price_history_simulated": [
                round(p, 4) for p in self.price_history_simulated
            ],
        }

    def step(self, net_volume: float = 0.0) -> dict:
        """
        Advance the market by one bar — Endogenous Price Impact Model.

        Prices respond to net buy/sell volume from agents:
            impact         = sensitivity_factor × net_volume  (capped ±20 %)
            simulated_next = historical_close_next × (1 + impact)

        If called without arguments (net_volume defaults to 0.0) the
        behaviour is identical to pure historical replay — backward
        compatible with any caller that does not supply volume info.

        Args:
            net_volume: total_buy_qty − total_sell_qty across all agents
                        for the current tick.

        Returns:
            dict – step state with keys:
                t, historical_price, simulated_price,
                sma20, sma50, bb_up, bb_low, volatility,
                finished   (bool — True when all bars exhausted)
        """
        # ── End-of-data guard ────────────────────────────────────────
        if self.current_step >= self.total_bars - 1:
            state = self._build_step_state(self.current_step)
            state["finished"] = True
            return state

        self.current_step += 1

        # ── Historical baseline for the new step ─────────────────────
        # self.df["Close"] is never mutated — always the raw yfinance data.
        hist_price_next = float(self.df.iloc[self.current_step]["Close"])

        # ── Endogenous price impact ──────────────────────────────────
        # impact = sensitivity × net_volume, capped at ±20 % per step
        impact = self.sensitivity_factor * net_volume
        impact = max(min(impact, 0.20), -0.20)

        simulated_price = hist_price_next * (1.0 + impact)

        # Floor: simulated price must stay positive
        simulated_price = max(simulated_price, 0.01)

        # ── Update simulated price state ─────────────────────────────
        self.current_price = simulated_price
        self.price_history_simulated.append(simulated_price)

        # NOTE: we intentionally do NOT patch self.df["Close"].
        # Historical data stays pristine; the simulated series is
        # tracked entirely in self.current_price / price_history_simulated.

        state = self._build_step_state(self.current_step)
        state["finished"] = False
        return state

    def is_done(self) -> bool:
        return self.current_step >= self.total_bars - 1
