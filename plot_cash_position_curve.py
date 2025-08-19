import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    results_dir = os.path.expanduser("~/strategy_results")
    equity_csv = os.path.join(results_dir, "hybrid_adaptive_ml_tqqq_equity.csv")

    if not os.path.exists(equity_csv):
        raise FileNotFoundError(f"Equity CSV not found: {equity_csv}")

    df = pd.read_csv(equity_csv, parse_dates=["date"])  # expects: date, cash, position_value, portfolio_value

    required_cols = {"date", "cash", "position_value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in equity csv: {missing}")

    df = df.sort_values("date")
    if "portfolio_value" not in df.columns:
        df["portfolio_value"] = df["cash"] + df["position_value"]

    # --- Figure 1: Daily curves (cash / position / portfolio) ---
    plt.figure(figsize=(14, 6))
    plt.plot(df["date"], df["cash"], label="Cash ($)", color="#1f77b4", linewidth=1.6)
    plt.plot(df["date"], df["position_value"], label="Position Value ($)", color="#ff7f0e", linewidth=1.6)
    plt.plot(df["date"], df["portfolio_value"], label="Portfolio Value ($)", color="#2ca02c", linewidth=1.6)

    plt.title("Cash / Position / Portfolio Over Time")
    plt.xlabel("Date")
    plt.ylabel("Amount ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path1 = os.path.join(results_dir, "cash_position_portfolio_curve.png")
    plt.savefig(out_path1, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path1}")

    # Set index for resampling
    dfi = df.set_index("date")

    # --- Figure 2: Monthly (end-of-month) curves ---
    monthly = dfi.resample("M").last().dropna(subset=["cash", "position_value", "portfolio_value"]).copy()
    plt.figure(figsize=(14, 6))
    plt.stackplot(
        monthly.index,
        monthly["cash"],
        monthly["position_value"],
        labels=["Cash", "Position"],
        colors=["#1f77b4", "#ff7f0e"],
        alpha=0.7,
    )
    plt.plot(monthly.index, monthly["portfolio_value"], color="#2ca02c", linewidth=1.6, label="Portfolio")
    plt.title("Monthly EOM Levels: Cash / Position (stacked) and Portfolio")
    plt.xlabel("Month")
    plt.ylabel("Amount ($)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    out_path2 = os.path.join(results_dir, "cash_position_monthly_curve.png")
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path2}")

    # --- Figure 3: Yearly (end-of-year) bars ---
    yearly = dfi.resample("Y").last().dropna(subset=["cash", "position_value", "portfolio_value"]).copy()
    if not yearly.empty:
        plt.figure(figsize=(12, 6))
        x = yearly.index.year.astype(str)
        width = 0.35
        import numpy as np
        idx = np.arange(len(x))
        plt.bar(idx - width/2, yearly["cash"].values, width=width, label="Cash", color="#1f77b4")
        plt.bar(idx + width/2, yearly["position_value"].values, width=width, label="Position", color="#ff7f0e")
        plt.plot(idx, yearly["portfolio_value"].values, color="#2ca02c", linewidth=1.8, marker="o", label="Portfolio")
        plt.xticks(idx, x, rotation=0)
        plt.title("Yearly EOY Levels: Cash / Position (bars) and Portfolio (line)")
        plt.xlabel("Year")
        plt.ylabel("Amount ($)")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_path3 = os.path.join(results_dir, "cash_position_yearly_bars.png")
        plt.savefig(out_path3, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path3}")


if __name__ == "__main__":
    main()


