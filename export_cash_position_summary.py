import os
import pandas as pd


def add_pct_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    total = out.get("portfolio_value")
    if total is None:
        total = out.get("cash", 0) + out.get("position_value", 0)
        out["portfolio_value"] = total
    total_safe = out["portfolio_value"].replace(0, pd.NA)
    out["cash_pct"] = (out["cash"] / total_safe) * 100
    out["position_pct"] = (out["position_value"] / total_safe) * 100
    return out


def main():
    results_dir = os.path.expanduser("~/strategy_results")
    equity_csv = os.path.join(results_dir, "hybrid_adaptive_ml_tqqq_equity.csv")
    if not os.path.exists(equity_csv):
        raise FileNotFoundError(f"Equity CSV not found: {equity_csv}")

    df = pd.read_csv(equity_csv, parse_dates=["date"])  # expects: date, cash, position_value, portfolio_value
    required = {"date", "cash", "position_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in equity csv: {missing}")
    if "portfolio_value" not in df.columns:
        df["portfolio_value"] = df["cash"] + df["position_value"]

    dfi = df.sort_values("date").set_index("date")

    # Monthly end-of-month summary
    monthly = dfi.resample("ME").last().dropna(subset=["cash", "position_value", "portfolio_value"]).copy()
    monthly = add_pct_columns(monthly)
    # MoM changes
    monthly["cash_change"] = monthly["cash"].diff()
    monthly["position_change"] = monthly["position_value"].diff()
    monthly["portfolio_change"] = monthly["portfolio_value"].diff()
    monthly["cash_mom_pct"] = monthly["cash"].pct_change() * 100
    monthly["position_mom_pct"] = monthly["position_value"].pct_change() * 100
    monthly["portfolio_mom_pct"] = monthly["portfolio_value"].pct_change() * 100
    monthly.index.name = "month_end"
    monthly_out = os.path.join(results_dir, "cash_position_monthly_summary.csv")
    monthly.reset_index().to_csv(monthly_out, index=False, date_format="%Y-%m-%d")
    print(f"Saved: {monthly_out}")

    # Yearly end-of-year summary
    yearly = dfi.resample("YE").last().dropna(subset=["cash", "position_value", "portfolio_value"]).copy()
    yearly = add_pct_columns(yearly)
    # YoY changes
    yearly["cash_change"] = yearly["cash"].diff()
    yearly["position_change"] = yearly["position_value"].diff()
    yearly["portfolio_change"] = yearly["portfolio_value"].diff()
    yearly["cash_yoy_pct"] = yearly["cash"].pct_change() * 100
    yearly["position_yoy_pct"] = yearly["position_value"].pct_change() * 100
    yearly["portfolio_yoy_pct"] = yearly["portfolio_value"].pct_change() * 100
    yearly.index.name = "year_end"
    yearly_out = os.path.join(results_dir, "cash_position_yearly_summary.csv")
    yearly.reset_index().to_csv(yearly_out, index=False, date_format="%Y-%m-%d")
    print(f"Saved: {yearly_out}")


if __name__ == "__main__":
    main()


