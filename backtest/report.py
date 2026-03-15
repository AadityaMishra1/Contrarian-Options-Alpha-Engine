"""HTML report generator for backtest results.

Produces a self-contained HTML page with embedded base64 charts and a
summary metrics table.  No external image files are written.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import BaseLoader, Environment

if TYPE_CHECKING:
    from replay_engine import BacktestResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Jinja2 HTML template (inline — no external template file needed)
# ---------------------------------------------------------------------------

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Contrarian Options Backtest Report</title>
  <style>
    body { font-family: "Segoe UI", Arial, sans-serif; background: #0d1117; color: #c9d1d9; margin: 40px; }
    h1, h2 { color: #58a6ff; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
    th { background: #161b22; color: #58a6ff; padding: 10px 16px; text-align: left; border-bottom: 2px solid #30363d; }
    td { padding: 8px 16px; border-bottom: 1px solid #21262d; }
    tr:hover td { background: #161b22; }
    .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 30px; }
    .chart-box { background: #161b22; border-radius: 8px; padding: 16px; }
    img { width: 100%; border-radius: 4px; }
    .metric-positive { color: #3fb950; }
    .metric-negative { color: #f85149; }
    .footer { color: #8b949e; font-size: 0.85em; margin-top: 40px; }
  </style>
</head>
<body>
  <h1>Contrarian Options Backtest Report</h1>
  <p>Generated {{ generated_at }}</p>

  <h2>Strategy Parameters</h2>
  <table>
    <tr><th>Parameter</th><th>Value</th></tr>
    {% for k, v in params.items() %}
    <tr><td>{{ k }}</td><td>{{ v }}</td></tr>
    {% endfor %}
  </table>

  <h2>Performance Summary</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    {% for k, v in metrics.items() %}
    <tr>
      <td>{{ k }}</td>
      <td class="{{ 'metric-positive' if v is number and v > 0 else ('metric-negative' if v is number and v < 0 else '') }}">
        {{ "%.4f"|format(v) if v is number else v }}
      </td>
    </tr>
    {% endfor %}
  </table>

  <h2>Charts</h2>
  <div class="chart-grid">
    <div class="chart-box">
      <h3>Equity Curve</h3>
      <img src="data:image/png;base64,{{ equity_chart }}" alt="Equity Curve" />
    </div>
    <div class="chart-box">
      <h3>Drawdown</h3>
      <img src="data:image/png;base64,{{ drawdown_chart }}" alt="Drawdown" />
    </div>
    <div class="chart-box">
      <h3>P&L Distribution</h3>
      <img src="data:image/png;base64,{{ pnl_chart }}" alt="P&L Distribution" />
    </div>
    <div class="chart-box">
      <h3>Monthly Returns Heatmap</h3>
      <img src="data:image/png;base64,{{ heatmap_chart }}" alt="Monthly Returns" />
    </div>
  </div>

  <p class="footer">Contrarian Options Backtesting Framework</p>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

_DARK_BG = "#0d1117"
_AXES_BG = "#161b22"
_GRID_COLOR = "#30363d"
_TEXT_COLOR = "#c9d1d9"
_BLUE = "#58a6ff"
_GREEN = "#3fb950"
_RED = "#f85149"
_ORANGE = "#d29922"


def _apply_dark_style(fig: plt.Figure, ax: plt.Axes) -> None:
    fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor(_AXES_BG)
    ax.tick_params(colors=_TEXT_COLOR, labelsize=8)
    ax.xaxis.label.set_color(_TEXT_COLOR)
    ax.yaxis.label.set_color(_TEXT_COLOR)
    ax.title.set_color(_TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID_COLOR)
    ax.grid(color=_GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.7)


def _fig_to_b64(fig: plt.Figure) -> str:
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _equity_chart(equity: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    _apply_dark_style(fig, ax)
    ax.plot(equity.index, equity.values, color=_BLUE, linewidth=1.5)
    ax.fill_between(equity.index, equity.values, equity.iloc[0], alpha=0.15, color=_BLUE)
    ax.set_title("Portfolio Equity Curve")
    ax.set_ylabel("Equity ($)")
    _maybe_format_xaxis(ax, equity.index)
    return _fig_to_b64(fig)


def _drawdown_chart(equity: pd.Series) -> str:
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max * 100

    fig, ax = plt.subplots(figsize=(7, 3.5))
    _apply_dark_style(fig, ax)
    ax.fill_between(drawdown.index, drawdown.values, 0, color=_RED, alpha=0.6)
    ax.plot(drawdown.index, drawdown.values, color=_RED, linewidth=1.0)
    ax.set_title("Drawdown (%)")
    ax.set_ylabel("Drawdown (%)")
    _maybe_format_xaxis(ax, drawdown.index)
    return _fig_to_b64(fig)


def _pnl_distribution_chart(trades: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    _apply_dark_style(fig, ax)

    if trades.empty or "pnl" not in trades.columns:
        ax.text(0.5, 0.5, "No trades", ha="center", va="center",
                transform=ax.transAxes, color=_TEXT_COLOR)
        return _fig_to_b64(fig)

    pnl = trades["pnl"].dropna()
    n_bins = min(50, max(10, len(pnl) // 5))
    ax.hist(pnl[pnl >= 0], bins=n_bins, color=_GREEN, alpha=0.7, label="Winners")
    ax.hist(pnl[pnl < 0], bins=n_bins, color=_RED, alpha=0.7, label="Losers")
    ax.axvline(pnl.mean(), color=_ORANGE, linewidth=1.5, linestyle="--", label=f"Mean ${pnl.mean():.2f}")
    ax.set_title("P&L Distribution")
    ax.set_xlabel("P&L ($)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8, facecolor=_AXES_BG, labelcolor=_TEXT_COLOR)
    return _fig_to_b64(fig)


def _monthly_heatmap_chart(equity: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    _apply_dark_style(fig, ax)

    if equity.empty or len(equity) < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, color=_TEXT_COLOR)
        return _fig_to_b64(fig)

    try:
        monthly = equity.resample("ME").last().pct_change().dropna() * 100
    except (TypeError, ValueError):
        # Index may not be datetime — fall back to numeric index
        ax.text(0.5, 0.5, "Non-datetime index; heatmap unavailable",
                ha="center", va="center", transform=ax.transAxes, color=_TEXT_COLOR)
        return _fig_to_b64(fig)

    if monthly.empty:
        ax.text(0.5, 0.5, "Insufficient monthly data", ha="center", va="center",
                transform=ax.transAxes, color=_TEXT_COLOR)
        return _fig_to_b64(fig)

    years = monthly.index.year.unique()
    months_range = range(1, 13)
    matrix = np.full((len(years), 12), np.nan)

    for yi, yr in enumerate(years):
        for mi, mo in enumerate(months_range):
            mask = (monthly.index.year == yr) & (monthly.index.month == mo)
            if mask.any():
                matrix[yi, mi] = monthly[mask].iloc[-1]

    vmax = np.nanmax(np.abs(matrix)) if not np.all(np.isnan(matrix)) else 1.0
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(12))
    ax.set_xticklabels(
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        fontsize=7, color=_TEXT_COLOR,
    )
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years, fontsize=8, color=_TEXT_COLOR)
    ax.set_title("Monthly Returns (%)")

    for yi in range(len(years)):
        for mi in range(12):
            val = matrix[yi, mi]
            if not np.isnan(val):
                ax.text(mi, yi, f"{val:.1f}", ha="center", va="center",
                        fontsize=6, color="black")

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    return _fig_to_b64(fig)


def _maybe_format_xaxis(ax: plt.Axes, index: pd.Index) -> None:
    """Apply date formatting if the index is datetime-like."""
    if isinstance(index, pd.DatetimeIndex) or (
        len(index) > 0 and isinstance(index[0], pd.Timestamp | np.datetime64)
    ):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    result: BacktestResult,
    output_path: str = "reports/backtest_report.html",
) -> Path:
    """Render a full HTML backtest report with embedded charts.

    Args:
        result: ``BacktestResult`` from ``ReplayEngine.run()``.
        output_path: Relative or absolute path for the output HTML file.
                     The parent directory is created if it does not exist.

    Returns:
        Absolute ``Path`` to the written HTML file.
    """
    out = Path(output_path)
    if not out.is_absolute():
        out = Path(__file__).parent / output_path
    out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating report: %s", out)

    equity = result.equity_curve
    trades = result.trades
    metrics = result.metrics
    params = result.params

    equity_b64 = _equity_chart(equity)
    dd_b64 = _drawdown_chart(equity)
    pnl_b64 = _pnl_distribution_chart(trades)
    hm_b64 = _monthly_heatmap_chart(equity)

    env = Environment(loader=BaseLoader())
    # Custom Jinja2 test: is the value a number?
    env.tests["number"] = lambda v: isinstance(v, int | float) and not isinstance(v, bool)
    template = env.from_string(_TEMPLATE)

    html = template.render(
        generated_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        params=params,
        metrics=metrics,
        equity_chart=equity_b64,
        drawdown_chart=dd_b64,
        pnl_chart=pnl_b64,
        heatmap_chart=hm_b64,
    )

    out.write_text(html, encoding="utf-8")
    logger.info("Report written to %s", out)
    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate backtest HTML report")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-03-01")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--output", default="reports/backtest_report.html")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import PolygonDataLoader
    from replay_engine import ReplayEngine

    loader = PolygonDataLoader()
    df = loader.load_stock_bars(args.symbol, args.start, args.end, "day")
    if df.empty:
        print("No data loaded; exiting.")
        sys.exit(1)

    engine = ReplayEngine()
    result = engine.run(df, initial_capital=args.capital)

    out_path = generate_report(result, output_path=args.output)
    print(f"Report saved to: {out_path}")
