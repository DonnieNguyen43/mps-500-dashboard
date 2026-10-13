"""
modules/feature_engine.py
Module tính toán đặc trưng kỹ thuật và tổng hợp đặc trưng cảm xúc.

Chức năng:
- Tính toán 12 chỉ báo kỹ thuật từ dữ liệu OHLCV:
  ret_1d, ret_2d, ret_5d, SMA20, SMA50, dist_sma20, dist_sma50,
  RSI(14), MACD, Bollinger Bands, vol_change_1d, vol_ratio_sma20
- Tổng hợp đặc trưng cảm xúc FinBERT từ tin tức
- Xây dựng vector đặc trưng lai (hybrid) để đưa vào mô hình
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
# 1. TÍNH CHỈ BÁO KỸ THUẬT
# ───────────────────────────────────────────────────

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính toán toàn bộ chỉ báo kỹ thuật từ DataFrame OHLCV.

    Input:  DataFrame với cột [date, open, high, low, close, volume]
    Output: DataFrame gốc + thêm các cột kỹ thuật đã tính
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # ── Lợi suất giá ──────────────────────────────
    df["ret_1d"] = df["close"].pct_change(1)
    df["ret_2d"] = df["close"].pct_change(2)
    df["ret_5d"] = df["close"].pct_change(5)

    # ── Đường trung bình động đơn giản (SMA) ──────
    df["sma_20"] = df["close"].rolling(window=20, min_periods=1).mean()
    df["sma_50"] = df["close"].rolling(window=50, min_periods=1).mean()

    # Khoảng cách tương đối từ SMA (dist = (price - sma) / sma)
    df["dist_sma20"] = (df["close"] - df["sma_20"]) / df["sma_20"]
    df["dist_sma50"] = (df["close"] - df["sma_50"]) / df["sma_50"]

    # RSI(14): Chỉ số sức mạnh tương đối
    df["rsi_14"] = _compute_rsi(df["close"], period=14)

    # MACD (12, 26, 9)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_diff"]   = df["macd"] - df["macd_signal"]

    # Bollinger Bands (20, 2σ): %B
    bb_mid = df["sma_20"]
    bb_std = df["close"].rolling(window=20, min_periods=1).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    # %B = (Price - Lower) / (Upper - Lower), clip [0,1]
    denom = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_pct_b"] = ((df["close"] - bb_lower) / denom).clip(0, 1)
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower

    # ── Chỉ báo thanh khoản / volume ──────────────
    df["vol_change_1d"]  = df["volume"].pct_change(1)
    vol_sma20 = df["volume"].rolling(window=20, min_periods=1).mean()
    df["vol_ratio_sma20"] = df["volume"] / vol_sma20.replace(0, np.nan)

    # Điền NaN bằng 0 (các phiên đầu chưa đủ dữ liệu)
    tech_cols = [
        "ret_1d", "ret_2d", "ret_5d",
        "dist_sma20", "dist_sma50",
        "rsi_14", "macd", "macd_signal", "macd_diff",
        "bb_pct_b", "vol_change_1d", "vol_ratio_sma20"
    ]
    df[tech_cols] = df[tech_cols].fillna(0)

    return df


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Tính RSI(period) theo phương pháp Wilder's smoothing.
    """
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # Giá trị mặc định trung tính


# ───────────────────────────────────────────────────
# 2. XÂY DỰNG VECTOR ĐẶC TRƯNG HYBRID
# ───────────────────────────────────────────────────

def build_hybrid_feature_vector(
    tech_row: pd.Series,
    sentiment_summary: Dict
) -> pd.DataFrame:
    """
    Ghép các đặc trưng kỹ thuật (từ hàng DataFrame cuối cùng)
    với đặc trưng cảm xúc (từ dict sentiment_summary).

    Returns:
        DataFrame 1 hàng với đúng thứ tự cột ALL_FEATURES
    """
    # ── Đặc trưng kỹ thuật ─────────────────────────
    tech_vals = {}
    for feat in cfg.TECH_FEATURES:
        if feat in tech_row.index:
            tech_vals[feat] = float(tech_row[feat])
        else:
            tech_vals[feat] = 0.0

    # ── Đặc trưng cảm xúc ─────────────────────────
    # Tính sent_momentum_1d và sent_momentum_3d (ước lượng từ dữ liệu hiện tại)
    sent_mean   = sentiment_summary.get("sent_mean",  0.0)
    sent_std    = sentiment_summary.get("sent_std",   0.5)
    p_pos       = sentiment_summary.get("p_pos",      0.333)
    p_neg       = sentiment_summary.get("p_neg",      0.333)
    p_neu       = sentiment_summary.get("p_neu",      0.333)
    news_vol    = sentiment_summary.get("news_vol",   5)
    pos_ratio   = sentiment_summary.get("pos_ratio",  0.333)
    neg_ratio   = sentiment_summary.get("neg_ratio",  0.333)
    # Momentum ước lượng: sử dụng sent_mean làm proxy (không đủ lịch sử rolling)
    sent_mom_1d = sentiment_summary.get("sent_momentum_1d", sent_mean * 0.5)
    sent_mom_3d = sentiment_summary.get("sent_momentum_3d", sent_mean * 0.3)
    news_vol_ratio_5d = sentiment_summary.get("news_vol_ratio_5d", 1.0)

    sent_vals = {
        "sent_mean":          sent_mean,
        "sent_std":           sent_std,
        "p_pos":              p_pos,
        "p_neg":              p_neg,
        "p_neu":              p_neu,
        "news_vol":           float(news_vol),
        "pos_ratio":          pos_ratio,
        "neg_ratio":          neg_ratio,
        "sent_momentum_1d":   sent_mom_1d,
        "sent_momentum_3d":   sent_mom_3d,
        "news_vol_ratio_5d":  news_vol_ratio_5d,
    }

    # Ghép lại theo thứ tự chuẩn
    combined = {**tech_vals, **sent_vals}
    feature_vector = pd.DataFrame([combined])[cfg.ALL_FEATURES]
    return feature_vector


def build_feature_matrix_from_history(
    hybrid_df: pd.DataFrame,
    n_rows: int = None
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Trích xuất ma trận X, nhãn target_t1 và target_t5
    từ hybrid_dataset đã có đầy đủ cột đặc trưng.

    Dùng để backtesting và huấn luyện lại.

    Returns:
        X: DataFrame features (ALL_FEATURES)
        y_t1: Series nhãn t+1
        y_t5: Series nhãn t+5
    """
    # Kiểm tra có đủ cột không
    missing = [c for c in cfg.ALL_FEATURES if c not in hybrid_df.columns]
    if missing:
        logger.warning(f"Thiếu các cột feature: {missing}")

    available = [c for c in cfg.ALL_FEATURES if c in hybrid_df.columns]
    X = hybrid_df[available].copy()

    # Điền NaN
    X = X.fillna(0)

    # Lấy nhãn
    y_t1 = hybrid_df.get("target_t1", pd.Series(np.zeros(len(hybrid_df))))
    y_t5 = hybrid_df.get("target_t5", pd.Series(np.zeros(len(hybrid_df))))

    if n_rows:
        X    = X.tail(n_rows)
        y_t1 = y_t1.tail(n_rows)
        y_t5 = y_t5.tail(n_rows)

    return X, y_t1, y_t5


# ───────────────────────────────────────────────────
# 3. TÍNH TOÁN THÊM CHỈ BÁO CHO BIỂU ĐỒ
# ───────────────────────────────────────────────────

def add_chart_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bổ sung thêm cột cần thiết cho biểu đồ Plotly:
    SMA20, SMA50, RSI, MACD, Bollinger Bands.
    Đảm bảo luôn có dữ liệu sạch để vẽ.
    """
    df = compute_technical_features(df)

    # Đổi tên để tương thích với biểu đồ
    if "sma_20" not in df.columns and "close" in df.columns:
        df["sma_20"] = df["close"].rolling(20, min_periods=1).mean()
    if "sma_50" not in df.columns and "close" in df.columns:
        df["sma_50"] = df["close"].rolling(50, min_periods=1).mean()

    return df
