"""
modules/backtester.py
Module mô phỏng giao dịch và tính toán hiệu năng chiến lược AI.

Chức năng:
- Mô phỏng giao dịch nhị phân: Long / Cash dựa trên tín hiệu mô hình
- Tính toán chỉ số hiệu năng: Cumulative Return, CAGR, Sharpe Ratio, Max Drawdown
- Xây dựng đường cong vốn (Equity Curve) so sánh AI vs Buy & Hold
- Hỗ trợ tùy chỉnh ngưỡng tin cậy và phí giao dịch
"""

import logging
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


# ───────────────────────────────────────────────────
# 1. MÔ PHỎNG GIAO DỊCH
# ───────────────────────────────────────────────────

def run_backtest(
    hybrid_df: pd.DataFrame,
    model=None,
    scaler=None,
    threshold: float = cfg.DEFAULT_THRESHOLD,
    fee_rate: float = cfg.DEFAULT_FEE,
    initial_capital: float = 10000.0,
    model_name: str = cfg.DEFAULT_MODEL_NAME
) -> Dict:
    """
    Chạy backtest chiến lược AI trên dữ liệu lịch sử.

    Chiến lược:
    - Nếu P(Up) >= threshold → Long (nắm giữ chỉ số)
    - Nếu P(Up) < threshold → Thoát ra (giữ tiền mặt)
    - Phí giao dịch áp dụng mỗi khi thay đổi vị thế

    Args:
        hybrid_df: DataFrame với đầy đủ features + target
        model: Mô hình ML đã tải
        scaler: Scaler đã fit
        threshold: Ngưỡng xác suất để mua (0.40 - 0.60)
        fee_rate: Phí giao dịch (0.0 - 0.002)
        initial_capital: Vốn ban đầu (USD)
        model_name: Tên mô hình đang thực thi

    Returns:
        Dict gồm equity curve, chỉ số tài chính, metadata
    """
    from modules.model_inference import predict_from_hybrid_df

    if hybrid_df is None or len(hybrid_df) < 10:
        return _empty_backtest_result()

    df = hybrid_df.copy().sort_values("date").reset_index(drop=True)

    # Tính tín hiệu từ mô hình
    df = predict_from_hybrid_df(df, model=model, scaler=scaler, threshold=threshold, model_name=model_name)

    # Lợi suất thực từ dữ liệu
    if "ret_1d" in df.columns:
        daily_ret = df["ret_1d"].values
    elif "close" in df.columns:
        daily_ret = df["close"].pct_change().fillna(0).values
    else:
        daily_ret = np.zeros(len(df))

    signal = df["signal"].values  # 1 = Long, 0 = Cash

    # ── Tính lợi suất chiến lược AI (Cơ chế khớp lệnh t+1 loại bỏ Lookahead Bias) ──
    # Tín hiệu tạo ra tại phiên t được thực thi từ đầu phiên t+1 (shift 1)
    # Phí giao dịch được trừ chính xác vào mỗi lượt thay đổi vị thế (Rebalancing)
    pos = pd.Series(signal).shift(1).fillna(0).values
    trade_shifts = np.abs(np.diff(np.concatenate([[0], pos])))
    strat_returns = pos * daily_ret - trade_shifts * fee_rate

    # ── Equity curves & Underwater Drawdown ───────────
    # Chiến lược AI
    cum_ai = np.cumprod(1 + strat_returns)
    equity_ai = initial_capital * cum_ai
    equity_ai = np.concatenate([[initial_capital], equity_ai])

    peak_ai = np.maximum.accumulate(np.concatenate([[1.0], cum_ai]))
    drawdown_ai = (np.concatenate([[1.0], cum_ai]) - peak_ai) / peak_ai

    # Buy & Hold S&P 500 (Điểm chuẩn Benchmark)
    cum_bh = np.cumprod(1 + daily_ret)
    equity_bh = initial_capital * cum_bh
    equity_bh = np.concatenate([[initial_capital], equity_bh])

    peak_bh = np.maximum.accumulate(np.concatenate([[1.0], cum_bh]))
    drawdown_bh = (np.concatenate([[1.0], cum_bh]) - peak_bh) / peak_bh

    # Ngày (bao gồm ngày khởi tạo)
    dates = pd.to_datetime(df["date"].values)
    dates_full = pd.DatetimeIndex([dates[0] - pd.Timedelta(days=1)] + list(dates))

    # ── Chỉ số tài chính chuẩn tắc học thuật ───────────
    metrics_ai = _compute_performance_metrics(strat_returns, label="AI Strategy")
    metrics_bh = _compute_performance_metrics(daily_ret, label="Buy & Hold")

    # Số lượt giao dịch và Win Rate
    n_trades = int(np.sum(trade_shifts))
    win_rate = _compute_win_rate(strat_returns, pos)

    return {
        "dates":        dates_full,
        "equity_ai":    equity_ai,
        "equity_bh":    equity_bh,
        "drawdown_ai":  drawdown_ai,
        "drawdown_bh":  drawdown_bh,
        "signals":      signal,
        "positions":    pos,
        "daily_ret_ai": strat_returns,
        "daily_ret_bh": daily_ret,
        "metrics_ai":   metrics_ai,
        "metrics_bh":   metrics_bh,
        "n_trades":     n_trades,
        "win_rate":     win_rate,
        "threshold":    threshold,
        "fee_rate":     fee_rate,
        "n_days":       len(df),
    }


def _compute_performance_metrics(
    daily_returns: np.ndarray,
    label: str = "Strategy",
    annual_factor: int = 252
) -> Dict:
    """
    Tính toán bộ chỉ số hiệu năng tài chính chuẩn.

    Returns:
        Dict gồm:
        - cumulative_return: Lợi nhuận tích lũy (%)
        - cagr: Tỷ suất tăng trưởng kép hàng năm (%)
        - annual_vol: Biến động hàng năm (%)
        - sharpe: Tỷ số Sharpe (risk-free rate = 0.0%)
        - max_drawdown: Sụt giảm tối đa từ đỉnh (%)
        - win_days: Tỷ lệ ngày dương (%)
    """
    rets = np.array(daily_returns, dtype=float)
    n    = len(rets)

    if n == 0:
        return _zero_metrics(label)

    # Cumulative Return
    cum_ret = np.prod(1 + rets) - 1

    # CAGR
    n_years = n / annual_factor
    if n_years > 0 and (1 + cum_ret) > 0:
        cagr = (1 + cum_ret) ** (1 / n_years) - 1
    else:
        cagr = 0.0

    # Annual Volatility
    vol = float(np.std(rets) * np.sqrt(annual_factor))

    # Sharpe Ratio (risk-free = 0 để đơn giản)
    mean_daily = float(np.mean(rets))
    std_daily  = float(np.std(rets))
    sharpe = (mean_daily / std_daily * np.sqrt(annual_factor)) if std_daily > 0 else 0.0

    # Max Drawdown
    cum_curve = np.cumprod(1 + rets)
    running_max = np.maximum.accumulate(cum_curve)
    drawdowns = (cum_curve - running_max) / running_max
    max_dd = float(np.min(drawdowns))

    # Win Rate (ngày dương)
    win_days = float(np.sum(rets > 0) / n * 100)

    return {
        "label":              label,
        "cumulative_return":  float(cum_ret * 100),
        "cagr":               float(cagr * 100),
        "annual_vol":         float(vol * 100),
        "sharpe":             float(sharpe),
        "max_drawdown":       float(max_dd * 100),
        "win_days":           float(win_days),
        "n_days":             int(n),
    }


def _compute_win_rate(returns: np.ndarray, signals: np.ndarray) -> float:
    """Tính win rate: % ngày dương trong các ngày có vị thế Long."""
    active = returns[signals == 1]
    if len(active) == 0:
        return 0.0
    return float(np.sum(active > 0) / len(active) * 100)


def _zero_metrics(label: str) -> Dict:
    """Trả về chỉ số bằng 0 khi không có dữ liệu."""
    return {
        "label": label, "cumulative_return": 0.0, "cagr": 0.0,
        "annual_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
        "win_days": 0.0, "n_days": 0,
    }


def _empty_backtest_result() -> Dict:
    """Trả về kết quả rỗng khi không có đủ dữ liệu."""
    return {
        "dates":      [], "equity_ai": [], "equity_bh": [], "signals": [],
        "daily_ret_ai": [], "daily_ret_bh": [],
        "metrics_ai": _zero_metrics("AI"), "metrics_bh": _zero_metrics("BH"),
        "n_trades":   0, "win_rate":   0.0,
        "threshold":  cfg.DEFAULT_THRESHOLD, "fee_rate": cfg.DEFAULT_FEE,
        "n_days":     0,
    }


# Section 2: Sensitivity Analysis - Rolling Metrics

def compute_rolling_sharpe(
    returns: np.ndarray,
    window: int = 60,
    annual_factor: int = 252
) -> np.ndarray:
    """
    Tính Sharpe ratio cuộn theo cửa sổ time window.
    Dùng để vẽ biểu đồ biến động Sharpe theo thời gian.
    """
    n = len(returns)
    rolling = np.full(n, np.nan)

    for i in range(window, n):
        rets_window = returns[i - window:i]
        mean = np.mean(rets_window)
        std  = np.std(rets_window)
        if std > 0:
            rolling[i] = mean / std * np.sqrt(annual_factor)

    return rolling


def compute_monthly_returns(
    dates: pd.DatetimeIndex,
    daily_returns: np.ndarray
) -> pd.DataFrame:
    """
    Tổng hợp lợi suất theo tháng để vẽ heatmap.

    Returns:
        DataFrame pivot: Index=Year, Columns=Month, Values=Monthly Return (%)
    """
    df = pd.DataFrame({
        "date": dates[:len(daily_returns)],
        "ret":  daily_returns
    })
    df["date"] = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Tổng hợp lợi suất tháng (log-return)
    monthly = df.groupby(["year", "month"])["ret"].apply(
        lambda x: (np.prod(1 + x) - 1) * 100
    ).reset_index()
    monthly.columns = ["year", "month", "monthly_return"]

    pivot = monthly.pivot(index="year", columns="month", values="monthly_return")
    month_names = {
        1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
        7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"
    }
    pivot.columns = [month_names.get(c, c) for c in pivot.columns]
    return pivot


def format_metric_display(metrics: Dict) -> Dict:
    """
    Format các chỉ số tài chính thành chuỗi hiển thị.
    """
    def fmt_pct(v: float, decimals: int = 2) -> str:
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.{decimals}f}%"

    def fmt_num(v: float, decimals: int = 2) -> str:
        return f"{v:.{decimals}f}x"

    return {
        "cumulative_return": fmt_pct(metrics["cumulative_return"]),
        "cagr":              fmt_pct(metrics["cagr"]),
        "annual_vol":        fmt_pct(metrics["annual_vol"]),
        "sharpe":            f"{metrics['sharpe']:.3f}",
        "max_drawdown":      fmt_pct(metrics["max_drawdown"]),
        "win_days":          fmt_pct(metrics["win_days"], 1),
    }
