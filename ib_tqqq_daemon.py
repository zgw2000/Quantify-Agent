import argparse
import datetime as dt
import time
from typing import Optional

from ib_tqqq_trader import RunConfig, run_once_rebalance
from ib_dump_executions import dump_executions_and_append, yyyymmdd_hhmmss_et_start_of_day

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def is_us_market_hours(now_utc: dt.datetime, include_extended: bool = False) -> bool:
    if ZoneInfo is None:
        return True  # fallback: don't block if tz is unavailable
    et = now_utc.astimezone(ZoneInfo("America/New_York"))
    # Monday-Friday only
    if et.weekday() > 4:
        return False
    # RTH: 09:30-16:00 ET; Extended: 04:00-20:00 ET (approx)
    if include_extended:
        start = et.replace(hour=4, minute=0, second=0, microsecond=0)
        end = et.replace(hour=20, minute=0, second=0, microsecond=0)
    else:
        start = et.replace(hour=9, minute=30, second=0, microsecond=0)
        end = et.replace(hour=16, minute=0, second=0, microsecond=0)
    return start <= et <= end


def main():
    parser = argparse.ArgumentParser(description="IB TQQQ Strategy Daemon - periodic rebalance runner")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497, help="TWS Paper only. Port enforced to 7497.")
    parser.add_argument("--client-id", type=int, default=201)
    parser.add_argument("--ticker", type=str, default="TQQQ")
    parser.add_argument("--duration", type=str, default="10 Y")
    parser.add_argument("--interval-sec", type=int, default=600, help="Loop interval seconds (default 10 min)")
    parser.add_argument("--live", action="store_true", help="Place orders; default is dry-run")
    parser.add_argument("--outside-rth", action="store_true", help="Allow orders outside regular trading hours")
    parser.add_argument("--append-executions", action="store_true", help="After each cycle, fetch executions since SOD and append to log/CSV")
    parser.add_argument(
        "--market-hours-only",
        action="store_true",
        help="Run only during US regular trading hours (09:30-16:00 ET)",
    )
    parser.add_argument(
        "--include-extended",
        action="store_true",
        help="If set with --market-hours-only, include extended hours (04:00-20:00 ET)",
    )
    parser.add_argument(
        "--cooldown-sec",
        type=int,
        default=3600,
        help="Minimum seconds between live orders (default 1 hour); only applies with --live",
    )
    args = parser.parse_args()

    # Enforce paper port
    if args.port != 7497:
        raise SystemExit("This daemon is restricted to TWS Paper port 7497. Please use --port 7497.")

    last_live_order_time: Optional[float] = None

    while True:
        loop_start = time.time()
        try:
            # Heartbeat & market-hours gate
            now_utc = dt.datetime.now(dt.timezone.utc)
            if args.market_hours_only:
                allowed = is_us_market_hours(now_utc, include_extended=args.include_extended)
                print(
                    f"[daemon] {now_utc.isoformat()} market_hours_only=True include_extended={args.include_extended} allowed={allowed}"
                )
                if not allowed:
                    print("[daemon] Outside allowed trading hours; sleeping...")
                    time.sleep(args.interval_sec)
                    continue

            cfg = RunConfig(
                host=args.host,
                port=args.port,
                client_id=args.client_id,
                ticker=args.ticker.upper(),
                duration=args.duration,
                dry_run=(not args.live),
                outside_rth=bool(args.outside_rth),
            )

            # Run one rebalance pass
            print(f"[daemon] Running rebalance pass for {args.ticker.upper()} (live={args.live})")
            run_once_rebalance(cfg)

            # Append executions to log/CSV if requested
            if args.append_executions:
                try:
                    since = yyyymmdd_hhmmss_et_start_of_day(0)
                    appended = dump_executions_and_append(host=args.host, port=args.port, client_id=args.client_id, since=since)
                    print(f"[daemon] appended {appended} executions to log/CSV")
                except Exception as e:
                    print(f"[daemon] append executions failed: {e}")

            # Simple cooldown guard for live mode
            if args.live:
                now = time.time()
                if last_live_order_time is None or (now - last_live_order_time) >= args.cooldown_sec:
                    last_live_order_time = now
                else:
                    print(
                        f"[daemon] Live cooldown active ({int(now - last_live_order_time)}s elapsed); "
                        f"no additional live orders will be placed until cooldown passes."
                    )

        except Exception as e:
            print(f"[daemon] Error: {e}")
            # Backoff on error
            time.sleep(min(60, args.interval_sec))

        # Sleep until next interval (accounting for work time)
        elapsed = time.time() - loop_start
        to_sleep = max(1, args.interval_sec - int(elapsed))
        time.sleep(to_sleep)


if __name__ == "__main__":
    main()


