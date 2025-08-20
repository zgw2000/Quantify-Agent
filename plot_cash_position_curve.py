import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="TQQQ", help="Ticker used for backtest, e.g., TQQQ, SOXL")
    args = parser.parse_args()
    ticker = args.ticker.upper()

    results_dir = os.path.expanduser("~/strategy_results")
    equity_csv = os.path.join(results_dir, f"hybrid_adaptive_ml_{ticker.lower()}_equity.csv")

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

    plt.title(f"Cash / Position / Portfolio Over Time - {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Amount ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path1 = os.path.join(results_dir, f"cash_position_portfolio_curve_{ticker.lower()}.png")
    plt.savefig(out_path1, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path1}")

    # Set index for resampling
    dfi = df.set_index("date")

    # --- Figure 2: Monthly (end-of-month) curves ---
    monthly = dfi.resample("ME").last().dropna(subset=["cash", "position_value", "portfolio_value"]).copy()
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
    plt.title(f"Monthly EOM Levels: Cash / Position (stacked) and Portfolio - {ticker}")
    plt.xlabel("Month")
    plt.ylabel("Amount ($)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    out_path2 = os.path.join(results_dir, f"cash_position_monthly_curve_{ticker.lower()}.png")
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path2}")

    # --- Figure 3: Yearly (end-of-year) bars ---
    yearly = dfi.resample("YE").last().dropna(subset=["cash", "position_value", "portfolio_value"]).copy()
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
        plt.title(f"Yearly EOY Levels: Cash / Position (bars) and Portfolio (line) - {ticker}")
        plt.xlabel("Year")
        plt.ylabel("Amount ($)")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_path3 = os.path.join(results_dir, f"cash_position_yearly_bars_{ticker.lower()}.png")
        plt.savefig(out_path3, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path3}")

    # --- Figure 4: Exposure (actual vs target) ---
    if {"exposure", "target_exposure"}.issubset(df.columns):
        plt.figure(figsize=(14, 5))
        plt.plot(df["date"], df["exposure"], label="Exposure (actual)", color="#2ca02c", linewidth=1.6)
        plt.plot(df["date"], df["target_exposure"], label="Target Exposure", color="#d62728", linewidth=1.2, alpha=0.8)
        plt.ylim(0, 1.05)
        plt.title(f"Exposure Over Time - {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Exposure (0-1)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left")
        plt.tight_layout()
        out_path4 = os.path.join(results_dir, f"exposure_curve_{ticker.lower()}.png")
        plt.savefig(out_path4, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path4}")

    # --- Figure 5: Price vs Exposure overlay ---
    price_col = f"{ticker.lower()}_price"
    if price_col in df.columns and "exposure" in df.columns:
        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax1.plot(df["date"], df[price_col], color="#1f77b4", linewidth=1.5, label=f"{ticker} Price")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price ($)", color="#1f77b4")
        ax1.tick_params(axis='y', labelcolor="#1f77b4")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(df["date"], df["exposure"], color="#2ca02c", linewidth=1.2, alpha=0.9, label="Exposure")
        if "target_exposure" in df.columns:
            ax2.plot(df["date"], df["target_exposure"], color="#d62728", linewidth=1.0, alpha=0.7, label="Target Exposure")
        ax2.set_ylabel("Exposure (0-1)", color="#2ca02c")
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis='y', labelcolor="#2ca02c")

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        fig.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
        plt.title(f"{ticker} Price vs Exposure")
        fig.tight_layout()
        out_path5 = os.path.join(results_dir, f"price_exposure_overlay_{ticker.lower()}.png")
        plt.savefig(out_path5, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path5}")


if __name__ == "__main__":
    main()


