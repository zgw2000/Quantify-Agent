import argparse
import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order


# =============================
# Strategy Configuration (aligned with main.py principles)
# =============================

# Exposure and rebalancing constraints
MAX_EXPOSURE: float = 1.0
REBALANCE_TOLERANCE: float = 0.03
MAX_TRADE_PORTFOLIO_PCT: float = 0.25

# ML and feature parameters
FUTURE_RET_DAYS: int = 5
MOMENTUM_LOOKBACK: int = 5
RSI_PERIOD: int = 14
VOL_LOOKBACK: int = 20

# Signal mapping
SIGMOID_TEMP: float = 0.75
TREND_MA_SHORT: int = 3
TREND_MA_LONG: int = 10


# =============================
# IB API App Wrapper
# =============================

class IBApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        # Request and order id management
        self._req_id_lock = threading.Lock()
        self._next_req_id = 1

        self._order_id_ready = threading.Event()
        self._next_order_id = None

        # Data stores
        self._hist_data: Dict[int, List[Tuple]] = {}
        self._hist_done: Dict[int, threading.Event] = {}

        self._positions: Dict[str, float] = {}
        self._positions_done = threading.Event()

        self._acct_values: Dict[str, float] = {}
        self._acct_done = threading.Event()

        self._ticks_last: Dict[str, float] = {}

        # Open orders
        self._open_orders: List[Tuple[int, str, str, int, str]] = []  # (orderId, symbol, side, qty, status)
        self._open_orders_done = threading.Event()

    # ----- Utility -----
    def next_req_id(self) -> int:
        with self._req_id_lock:
            rid = self._next_req_id
            self._next_req_id += 1
            return rid

    # ----- EWrapper overrides -----
    def nextValidId(self, orderId: int):
        self._next_order_id = orderId
        self._order_id_ready.set()

    def error(self, reqId: int, errorCode: int, errorString: str):
        # Common benign codes include 2104/2106/2158 (market data connection OK etc.)
        print(f"IB ERROR: reqId={reqId}, code={errorCode}, msg={errorString}")

    # Historical data
    def historicalData(self, reqId, bar):
        self._hist_data.setdefault(reqId, []).append(
            (bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume)
        )

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        if reqId in self._hist_done:
            self._hist_done[reqId].set()

    # Positions
    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        symbol = contract.symbol
        self._positions[symbol] = position

    def positionEnd(self):
        self._positions_done.set()

    # Account values
    def updateAccountValue(self, key: str, val: str, currency: str, accountName: str):
        # Key examples: NetLiquidation, AvailableFunds, BuyingPower, etc.
        try:
            self._acct_values[key] = float(val)
        except ValueError:
            pass

    def accountDownloadEnd(self, account: str):
        self._acct_done.set()

    # Tick snapshot (optional)
    def tickPrice(self, reqId: int, tickType: int, price: float, attrib):
        # tickType: 1=Bid, 2=Ask, 4=Last, 6=High, 7=Low, 9=Close, etc.
        # We record by symbol via a mapping stored in a side dict when requesting
        pass

    # Open orders
    def openOrder(self, orderId, contract, order, orderState):
        status = getattr(orderState, "status", "") or ""
        side = getattr(order, "action", "") or ""
        qty = int(getattr(order, "totalQuantity", 0) or 0)
        symbol = getattr(contract, "symbol", "") or ""
        # Record all open/pre-submitted/submitted/pending
        self._open_orders.append((orderId, symbol, side.upper(), qty, status.upper()))

    def openOrderEnd(self):
        self._open_orders_done.set()


# =============================
# Contracts and Orders
# =============================

def stock_contract(symbol: str, currency: str = "USD", exchange: str = "SMART") -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.exchange = exchange
    c.currency = currency
    return c


def market_order(action: str, quantity: int) -> Order:
    o = Order()
    o.action = action  # "BUY" or "SELL"
    o.orderType = "MKT"
    o.totalQuantity = quantity
    o.tif = "DAY"
    # Avoid venue attributes that may not be supported
    # Explicitly set to False to prevent 10268 (EtradeOnly/FirmQuoteOnly not supported)
    o.eTradeOnly = False
    o.firmQuoteOnly = False
    return o


# =============================
# Feature Engineering and Model
# =============================

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=series.index).rolling(period).mean()
    roll_down = pd.Series(loss, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Assumes df index is datetime and has columns: Open, High, Low, Close, Volume
    out = pd.DataFrame(index=df.index)
    close = df["Close"].copy()
    volume = df["Volume"].copy()

    out["ret1"] = close.pct_change()
    out["mom_%d" % MOMENTUM_LOOKBACK] = close.pct_change(MOMENTUM_LOOKBACK)
    out["ma_3"] = close.rolling(3).mean()
    out["ma_10"] = close.rolling(10).mean()
    out["ma_50"] = close.rolling(50).mean()
    out["ma_ratio_3_10"] = out["ma_3"] / (out["ma_10"] + 1e-9)
    out["ma_ratio_10_50"] = out["ma_10"] / (out["ma_50"] + 1e-9)
    out["rsi_%d" % RSI_PERIOD] = compute_rsi(close, RSI_PERIOD)
    out["vol_%d" % VOL_LOOKBACK] = close.pct_change().rolling(VOL_LOOKBACK).std()

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_upper"] = bb_mid + 2.0 * bb_std
    out["bb_lower"] = bb_mid - 2.0 * bb_std
    out["bb_pos"] = (close - bb_mid) / (bb_std + 1e-9)

    out["volume_z"] = (volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1e-9)

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def make_target(close: pd.Series, horizon_days: int) -> pd.Series:
    future_price = close.shift(-horizon_days)
    target = (future_price / close) - 1.0
    return target


def sigmoid(x: np.ndarray, temperature: float) -> np.ndarray:
    # map to (0,1)
    return 1.0 / (1.0 + np.exp(-x / max(1e-9, temperature)))


def compute_target_exposure_from_model(
    features: pd.DataFrame,
    close: pd.Series,
    model: RandomForestRegressor,
    temperature: float = SIGMOID_TEMP,
) -> Tuple[float, float]:
    # Predict future 5d returns for last row
    X_last = features.iloc[[-1]].values
    pred_next_5d = float(model.predict(X_last)[0])
    # Scale by realized vol (annualized) to adjust aggressiveness
    realized_vol_daily = close.pct_change().rolling(VOL_LOOKBACK).std().iloc[-1]
    realized_vol_ann = realized_vol_daily * math.sqrt(252.0)
    score = pred_next_5d / max(1e-6, realized_vol_ann)
    raw_exposure = float(sigmoid(np.array([score]), temperature=temperature)[0])

    # Trend filter boost/cut
    ma3 = close.rolling(TREND_MA_SHORT).mean().iloc[-1]
    ma10 = close.rolling(TREND_MA_LONG).mean().iloc[-1]
    trend_adj = 0.0
    if ma3 > ma10:
        trend_adj += 0.05
    elif ma3 < ma10:
        trend_adj -= 0.05

    target_exposure = np.clip(raw_exposure + trend_adj, 0.0, MAX_EXPOSURE)
    return target_exposure, pred_next_5d


# =============================
# IB Helpers (sync wrappers)
# =============================

def start_ib_and_wait(host: str, port: int, client_id: int) -> IBApp:
    app = IBApp()
    app.connect(host, port, clientId=client_id)
    t = threading.Thread(target=app.run, daemon=True)
    t.start()
    # Wait for nextValidId
    if not app._order_id_ready.wait(timeout=10.0):
        raise RuntimeError("Timed out waiting for nextValidId from IB")
    return app


def fetch_historical_daily_bars(
    app: IBApp,
    contract: Contract,
    duration: str = "10 Y",
    bar_size: str = "1 day",
    what_to_show: str = "TRADES",
    use_rth: int = 1,
    timeout_sec: float = 30.0,
) -> pd.DataFrame:
    rid = app.next_req_id()
    done = threading.Event()
    app._hist_done[rid] = done
    app._hist_data[rid] = []

    app.reqHistoricalData(
        rid,
        contract,
        endDateTime="",
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow=what_to_show,
        useRTH=use_rth,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[],
    )

    if not done.wait(timeout=timeout_sec):
        raise RuntimeError("Timed out waiting for historical data")

    rows = app._hist_data.get(rid, [])
    if not rows:
        raise RuntimeError("No historical data received")

    df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    # IB returns date as YYYYMMDD for daily
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df.set_index("Date").sort_index()
    return df


def fetch_positions_sync(app: IBApp, timeout_sec: float = 10.0) -> Dict[str, float]:
    app._positions.clear()
    app._positions_done.clear()
    app.reqPositions()
    app._positions_done.wait(timeout=timeout_sec)
    return dict(app._positions)


def fetch_account_values_sync(app: IBApp, timeout_sec: float = 10.0) -> Dict[str, float]:
    app._acct_values.clear()
    app._acct_done.clear()
    app.reqAccountUpdates(True, "")
    app._acct_done.wait(timeout=timeout_sec)
    # Stop updates to avoid spam
    app.reqAccountUpdates(False, "")
    return dict(app._acct_values)


def place_market_order_sync(app: IBApp, contract: Contract, action: str, quantity: int) -> int:
    if quantity <= 0:
        return -1
    # Ensure we have a valid order id
    if app._next_order_id is None:
        raise RuntimeError("Order id not ready")
    order_id = app._next_order_id
    app._next_order_id += 1

    order = market_order(action, quantity)
    print(f"Placing order: id={order_id}, {action} {quantity} {contract.symbol}")
    app.placeOrder(order_id, contract, order)
    # No sync confirm here; user can check in TWS
    return order_id


# =============================
# Rebalance Logic
# =============================

def compute_rebalance_shares(
    net_liq_value: float,
    current_shares: float,
    last_price: float,
    target_exposure: float,
) -> int:
    current_value = current_shares * last_price
    target_value = net_liq_value * target_exposure
    delta_value = target_value - current_value

    # Rebalance tolerance: skip small moves
    if abs(delta_value) < REBALANCE_TOLERANCE * net_liq_value:
        return 0

    # Cap per rebalance trade size
    max_trade_value = MAX_TRADE_PORTFOLIO_PCT * net_liq_value
    adj_value = float(np.sign(delta_value)) * min(abs(delta_value), max_trade_value)
    qty = int(math.floor(abs(adj_value) / max(1e-6, last_price)))
    if qty <= 0:
        return 0
    return int(np.sign(delta_value)) * qty


# =============================
# Main flow
# =============================


@dataclass
class RunConfig:
    host: str = "127.0.0.1"
    port: int = 7497  # ENFORCED: TWS Paper only (7497)
    client_id: int = 101
    ticker: str = "TQQQ"
    duration: str = "10 Y"
    dry_run: bool = True
    outside_rth: bool = False  # allow outside regular trading hours


def run_once_rebalance(cfg: RunConfig) -> None:
    # Enforce paper port at runtime for safety
    if cfg.port != 7497:
        raise RuntimeError("This script is restricted to TWS Paper port 7497. Provided port is not allowed.")
    # Connect IB
    app = start_ib_and_wait(cfg.host, cfg.port, cfg.client_id)
    try:
        # 1) Historical data for features/model
        contract = stock_contract(cfg.ticker)
        hist_df = fetch_historical_daily_bars(app, contract, duration=cfg.duration)

        # 2) Build dataset
        features = build_features(hist_df)
        # Align target
        target = make_target(hist_df["Close"], FUTURE_RET_DAYS).reindex(features.index)
        dataset = features.join(target.rename("target")).dropna()
        X = dataset[features.columns].values
        y = dataset["target"].values
        if len(dataset) < 200:
            raise RuntimeError("Not enough data to train the model")

        # 3) Train model (simple and fast for live use)
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y)

        # 4) Compute target exposure from the latest row
        latest_features = features.iloc[-1:]
        latest_close = hist_df["Close"]
        target_exposure, pred5d = compute_target_exposure_from_model(
            features, latest_close, model, temperature=SIGMOID_TEMP
        )

        last_price = float(hist_df["Close"].iloc[-1])

        # 5) Fetch account and position
        acct = fetch_account_values_sync(app)
        net_liq = acct.get("NetLiquidation")
        if net_liq is None:
            raise RuntimeError("Failed to get NetLiquidation from account values")
        net_liq = float(net_liq)

        pos = fetch_positions_sync(app)
        current_shares = float(pos.get(cfg.ticker, 0.0))

        # 6) Compute desired rebalance quantity
        desired_qty = compute_rebalance_shares(
            net_liq_value=net_liq,
            current_shares=current_shares,
            last_price=last_price,
            target_exposure=target_exposure,
        )

        # 6.1) Fetch existing open orders and net out same-direction qty to avoid duplicates
        pending_buy = 0
        pending_sell = 0
        try:
            app._open_orders.clear()
            app._open_orders_done.clear()
            app.reqOpenOrders()
            app._open_orders_done.wait(timeout=5.0)
            for orderId, symbol, side, qty, status in app._open_orders:
                if symbol != cfg.ticker:
                    continue
                # Consider only working states
                if status in ("SUBMITTED", "PRESUBMITTED", "PENDINGSUBMIT") or status == "":
                    if side == "BUY":
                        pending_buy += qty
                    elif side == "SELL":
                        pending_sell += qty
        except Exception as _:
            pass

        trade_qty = desired_qty
        if desired_qty > 0 and pending_buy > 0:
            trade_qty = max(0, desired_qty - pending_buy)
        elif desired_qty < 0 and pending_sell > 0:
            trade_qty = min(0, desired_qty + pending_sell)

        print("=== Live Rebalance Preview ===")
        print(f"Ticker: {cfg.ticker}")
        print(f"NetLiq: ${net_liq:,.2f}")
        print(f"Last Price: ${last_price:,.2f}")
        print(f"Current Shares: {current_shares:,.2f}")
        print(f"Model 5D Pred: {pred5d*100:.2f}%")
        print(f"Target Exposure: {target_exposure:.3f}")
        if pending_buy or pending_sell:
            print(f"Open Orders (same symbol) - BUY:{pending_buy} SELL:{pending_sell}")
        if trade_qty == 0:
            print("No trade needed (within tolerance or too small).")
        else:
            side = "BUY" if trade_qty > 0 else "SELL"
            print(f"Planned Trade: {side} {abs(trade_qty)} shares")

        # 7) Place order if not dry-run
        if not cfg.dry_run and trade_qty != 0:
            action = "BUY" if trade_qty > 0 else "SELL"
            order = market_order(action, abs(trade_qty))
            # Allow outside RTH if configured
            order.outsideRth = bool(cfg.outside_rth)
            # Ensure order id and send
            if app._next_order_id is None:
                raise RuntimeError("Order id not ready")
            order_id = app._next_order_id
            app._next_order_id += 1
            print(f"Placing order: id={order_id}, {action} {abs(trade_qty)} {contract.symbol} (outsideRth={order.outsideRth})")
            app.placeOrder(order_id, contract, order)
            print("Order sent to IB. Check TWS for status.")
        else:
            print("Dry-run mode: no order was sent.")
    finally:
        # Give IB a moment to flush, then disconnect
        time.sleep(1.0)
        app.disconnect()


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="IB TQQQ Strategy Trader (RandomForest + Exposure Rebalance)")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=7497, help="TWS Paper only. Port enforced to 7497.")
    p.add_argument("--client-id", type=int, default=101)
    p.add_argument("--ticker", type=str, default="TQQQ")
    p.add_argument("--duration", type=str, default="10 Y", help="IB durationStr, e.g., '5 Y', '3 Y', '2 Y'")
    p.add_argument("--live", action="store_true", help="If set, actually place orders (not a dry-run)")
    p.add_argument("--outside-rth", action="store_true", help="Allow orders outside regular trading hours")
    args = p.parse_args()
    if args.port != 7497:
        raise SystemExit("This script is restricted to TWS Paper port 7497. Please use --port 7497.")
    return RunConfig(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        ticker=args.ticker.upper(),
        duration=args.duration,
        dry_run=(not args.live),
        outside_rth=bool(args.outside_rth),
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_once_rebalance(cfg)


