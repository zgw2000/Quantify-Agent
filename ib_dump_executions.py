import argparse
import datetime as dt
import os
import threading
from typing import List, Tuple, Set

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.execution import ExecutionFilter


DAEMON_LOG_PATH = "/Users/zengguowang/Quantify-Agent/daemon.log"
EXEC_CSV_PATH = "/Users/zengguowang/Quantify-Agent/executions.csv"


class ExecApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self._ready = threading.Event()
        self._exec_end = threading.Event()
        self._rows: List[Tuple[str, str, str, int, float, int, str]] = []
        self._lock = threading.Lock()

    def nextValidId(self, orderId: int):  # noqa: N802
        self._ready.set()

    def execDetails(self, reqId, contract: Contract, execution):  # noqa: N802
        # execution: ExecDetails
        ts = execution.time  # format e.g. YYYYMMDD  HH:MM:SS US/Eastern
        symbol = contract.symbol
        side = execution.side  # BUY/SELL
        qty = int(execution.shares)
        price = float(execution.price)
        order_id = int(execution.orderId)
        exec_id = str(execution.execId)
        with self._lock:
            self._rows.append((ts, symbol, side, qty, price, order_id, exec_id))

    def execDetailsEnd(self, reqId):  # noqa: N802
        self._exec_end.set()

    def error(self, reqId, code, msg, advancedOrderRejectJson=""):  # noqa: N802
        # Keep running; just print to stdout which will be captured by caller if needed
        print(f"[exec_dump][error] code={code} msg={msg}")


def yyyymmdd_hhmmss_et_start_of_day(offset_days: int = 0) -> str:
    try:
        from zoneinfo import ZoneInfo  # python 3.9+
    except Exception:
        ZoneInfo = None  # type: ignore
    if ZoneInfo is not None:
        now_utc = dt.datetime.now(dt.timezone.utc)
        et = now_utc.astimezone(ZoneInfo("America/New_York"))
        sod = (et + dt.timedelta(days=offset_days)).replace(hour=0, minute=0, second=0, microsecond=0)
        return sod.strftime("%Y%m%d-%H:%M:%S")
    # Fallback to local time
    sod_local = dt.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    sod_local = sod_local + dt.timedelta(days=offset_days)
    return sod_local.strftime("%Y%m%d-%H:%M:%S")


def _load_existing_exec_ids() -> Set[str]:
    if not os.path.exists(EXEC_CSV_PATH) or os.path.getsize(EXEC_CSV_PATH) == 0:
        return set()
    ids: Set[str] = set()
    try:
        with open(EXEC_CSV_PATH, "r", encoding="utf-8") as f:
            # skip header
            next(f, None)
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 7:
                    ids.add(parts[6])
    except Exception:
        # If CSV is malformed, fall back to empty set to avoid data loss
        return set()
    return ids


def append_to_log_and_csv(rows: List[Tuple[str, str, str, int, float, int, str]]):
    if not rows:
        return 0
    existing_ids = _load_existing_exec_ids()
    new_rows = [r for r in rows if r[6] not in existing_ids]
    if not new_rows:
        return 0
    # Append to daemon.log as plain text lines
    with open(DAEMON_LOG_PATH, "a", encoding="utf-8") as f:
        for ts, symbol, side, qty, price, order_id, exec_id in new_rows:
            f.write(
                f"[exec] time={ts} symbol={symbol} side={side} qty={qty} price={price:.4f} "
                f"orderId={order_id} execId={exec_id}\n"
            )
            f.flush()
    # Append to executions.csv (create header if not exists)
    header_needed = not os.path.exists(EXEC_CSV_PATH) or os.path.getsize(EXEC_CSV_PATH) == 0
    with open(EXEC_CSV_PATH, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("time,symbol,side,qty,price,orderId,execId\n")
        for ts, symbol, side, qty, price, order_id, exec_id in new_rows:
            f.write(f"{ts},{symbol},{side},{qty},{price:.6f},{order_id},{exec_id}\n")
    return len(new_rows)


def dump_executions_and_append(host: str = "127.0.0.1", port: int = 7497, client_id: int = 199, since: str | None = None) -> int:
    if port != 7497:
        raise RuntimeError("This tool is restricted to TWS Paper port 7497.")
    app = ExecApp()
    app.connect(host, port, clientId=client_id)

    t = threading.Thread(target=app.run, daemon=True)
    t.start()
    if not app._ready.wait(timeout=10):
        raise RuntimeError("IB connection not ready (nextValidId timeout)")

    filt = ExecutionFilter()
    if since is None:
        since = yyyymmdd_hhmmss_et_start_of_day(0)
    filt.time = str(since)
    app.reqExecutions(1, filt)
    app._exec_end.wait(timeout=10)

    with app._lock:
        rows = list(app._rows)
    appended = append_to_log_and_csv(rows)
    app.disconnect()
    return appended


def main():
    p = argparse.ArgumentParser(description="Dump IB executions and append to daemon.log and executions.csv")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7497, help="TWS Paper port only (enforced)")
    p.add_argument("--client-id", type=int, default=199)
    p.add_argument(
        "--since",
        default=yyyymmdd_hhmmss_et_start_of_day(0),
        help="Filter time in format YYYYMMDD-HH:MM:SS (ET). Default: today 00:00 ET",
    )
    args = p.parse_args()
    if args.port != 7497:
        raise RuntimeError("This tool is restricted to TWS Paper port 7497.")

    appended = dump_executions_and_append(args.host, args.port, args.client_id, args.since)
    print(f"Appended {appended} executions to {DAEMON_LOG_PATH} and {EXEC_CSV_PATH}")


if __name__ == "__main__":
    main()


