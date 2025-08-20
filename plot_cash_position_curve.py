import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="TQQQ", help="Ticker used for backtest, e.g., TQQQ, SOXL")
    parser.add_argument("--resample", type=str, choices=["D", "W", "M"], default="W", help="Resample frequency for ratio/overlay charts: D=Daily, W=Weekly, M=Monthly")
    parser.add_argument("--smooth", type=int, default=10, help="Rolling window for smoothing (0 disables)")
    parser.add_argument("--log-portfolio", action="store_true", help="Use log scale for portfolio axis in overlay")
    args = parser.parse_args()
    ticker = args.ticker.upper()
    resample_rule = args.resample
    smooth_n = max(0, int(args.smooth))
    use_log_port = bool(args.log_portfolio)

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

    # --- Figure 6: Returns curves (cumulative and daily) ---
    equity_series = dfi["portfolio_value"].dropna()
    if len(equity_series) > 1:
        daily_returns = equity_series.pct_change().dropna()
        cum_return = (equity_series / equity_series.iloc[0]) - 1.0

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        ax1.plot(equity_series.index, cum_return, color="#2ca02c", linewidth=1.8, label="Cumulative Return")
        ax1.set_title(f"Cumulative Return - {ticker}")
        ax1.set_ylabel("Return (fraction)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left")

        ax2.plot(daily_returns.index, daily_returns, color="#1f77b4", alpha=0.55, linewidth=1.0, label="Daily Return")
        if len(daily_returns) > 20:
            ax2.plot(daily_returns.index, daily_returns.rolling(20).mean(), color="#ff7f0e", linewidth=1.2, label="20D Avg")
        ax2.set_title("Daily Returns")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Return (fraction)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper left")

        fig.tight_layout()
        out_path6 = os.path.join(results_dir, f"returns_curve_{ticker.lower()}.png")
        plt.savefig(out_path6, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path6}")

    # --- Figure 7: Cash and Position ratios over time ---
    if {"cash", "position_value", "portfolio_value"}.issubset(dfi.columns):
        ratio_df = dfi.copy()
        # 重采样以降噪
        if resample_rule != "D":
            rule = "W" if resample_rule == "W" else "M"
            ratio_df = ratio_df.resample(rule).last()
        # 安全除法，避免除以零
        pv = ratio_df["portfolio_value"].replace(0, pd.NA)
        cash_ratio = (ratio_df["cash"] / pv).astype(float)
        pos_ratio = (ratio_df["position_value"] / pv).astype(float)
        # 若已有 exposure，用其填补持仓比例缺口
        if "exposure" in ratio_df.columns:
            pos_ratio = pos_ratio.fillna(ratio_df["exposure"]).astype(float)
        # 平滑处理
        if smooth_n > 0:
            cash_ratio = cash_ratio.rolling(smooth_n, min_periods=1).mean()
            pos_ratio = pos_ratio.rolling(smooth_n, min_periods=1).mean()

        # 7. 比例曲线（更简洁）
        plt.figure(figsize=(14, 5))
        plt.plot(ratio_df.index, cash_ratio, label="Cash Ratio", color="#1f77b4", linewidth=1.6)
        plt.plot(ratio_df.index, pos_ratio, label="Position Ratio", color="#ff7f0e", linewidth=1.6)
        plt.ylim(0, 1.05)
        plt.title(f"Cash and Position Ratios - {ticker}  (resample={resample_rule}, smooth={smooth_n})")
        plt.xlabel("Date")
        plt.ylabel("Ratio (0-1)")
        plt.grid(True, alpha=0.3)
        ax = plt.gca()
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        plt.legend(loc="upper left")
        plt.tight_layout()
        out_path7 = os.path.join(results_dir, f"cash_position_ratio_curve_{ticker.lower()}.png")
        plt.savefig(out_path7, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path7}")

        # 8. 比例堆叠 + 组合净值（右轴，归一化/可选对数）
        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax1.stackplot(
            ratio_df.index,
            cash_ratio.fillna(0.0).values,
            pos_ratio.fillna(0.0).values,
            labels=["Cash Ratio", "Position Ratio"],
            colors=["#1f77b4", "#ff7f0e"],
            alpha=0.65,
        )
        ax1.set_ylim(0, 1.05)
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Ratio (0-1)")
        ax1.grid(True, alpha=0.3)
        locator = mdates.AutoDateLocator()
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

        ax2 = ax1.twinx()
        port_series = ratio_df["portfolio_value"].copy().dropna()
        if smooth_n > 0:
            port_series = port_series.rolling(smooth_n, min_periods=1).mean()
        if len(port_series) > 0:
            port_index = port_series / float(port_series.iloc[0])
            l3 = ax2.plot(port_index.index, port_index.values, label="Portfolio Index (×)", color="#2ca02c", linewidth=1.6)
        ax2.set_ylabel("Portfolio Index (×)")
        if use_log_port:
            ax2.set_yscale("log")

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
        fig.suptitle(f"Ratios (stacked) + Portfolio Index - {ticker}  (resample={resample_rule}, smooth={smooth_n})")
        fig.tight_layout()
        out_path8 = os.path.join(results_dir, f"cash_position_ratio_with_portfolio_{ticker.lower()}.png")
        fig.savefig(out_path8, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path8}")


if __name__ == "__main__":
    main()


