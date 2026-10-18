# -*- coding: utf-8 -*-
"""
config.py: Cấu hình trung tâm cho MPS-500 | Multimodal Market Prediction System
Đồng bộ 100% với kết quả thực nghiệm từ Gr9_ML.ipynb và Báo cáo Nhóm 9 (Lớp 50K33).
"""

import os

# Section 1: System Info & Project Identity
APP_NAME = "MPS-500"
APP_TITLE = "MPS-500 | Multimodal Market Prediction System"
APP_TAGLINE = "Hệ thống Dự báo Thị trường Đa phương thức (FinBERT × Machine Learning)"
AUTHOR_INFO = "Đồ án Nghiên cứu Machine Learning: Nhóm 9 (Lớp 50K33, Khoa Tài chính, Trường ĐH Kinh tế, ĐH Đà Nẵng)"
SUPERVISOR = "Giảng viên hướng dẫn: PGS.TS. Nguyễn Văn Chức"

# Section 2: Directory Paths & Files (Dynamic Cloud & Local Resolution)
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent if (APP_DIR.parent / "models").exists() else APP_DIR

# Thư mục models & data linh hoạt (hỗ trợ cả khi chạy độc lập hoặc trong thư mục dự án lớn)
if (APP_DIR / "models").exists():
    MODELS_DIR = str(APP_DIR / "models")
elif (PROJECT_ROOT / "models").exists():
    MODELS_DIR = str(PROJECT_ROOT / "models")
else:
    MODELS_DIR = str(APP_DIR / "models")

if (APP_DIR / "data_processed").exists():
    DATA_DIR = str(APP_DIR / "data_processed")
elif (PROJECT_ROOT / "data_processed").exists():
    DATA_DIR = str(PROJECT_ROOT / "data_processed")
else:
    DATA_DIR = str(APP_DIR / "data_processed")

ASSETS_DIR = str(APP_DIR / "assets")
LOGO_PATH = str(Path(ASSETS_DIR) / "logo_icon.svg")
LOGO_FULL_PATH = str(Path(ASSETS_DIR) / "logo_full.svg")

# Thư mục dữ liệu kết quả thực nghiệm
HYBRID_DATASET_PATH = str(Path(DATA_DIR) / "hybrid_dataset.csv")
DAILY_SENTIMENT_PATH = str(Path(DATA_DIR) / "daily_sentiment_features.csv")
MARKET_RAW_PATH = str(Path(DATA_DIR) / "sp500_market_raw.csv")
CLEANED_NEWS_PATH = str(Path(DATA_DIR) / "cleaned_news_raw.csv")

# Các tệp kết quả thực nghiệm chính thức từ Gr9_ML.ipynb
ABLATION_CSV_PATH = str(Path(DATA_DIR) / "ablation_study_results.csv")
FEATURE_IMPORTANCE_CSV_PATH = str(Path(DATA_DIR) / "feature_importance_consensus.csv")
BACKTEST_1D_CSV_PATH = str(Path(DATA_DIR) / "backtest_results_1d.csv")
TIMEFRAME_CSV_PATH = str(Path(DATA_DIR) / "timeframe_comparison_results.csv")
CONFIDENCE_TUNING_CSV_PATH = str(Path(DATA_DIR) / "confidence_band_tuning_results.csv")

# Thư mục mô hình & Scaler
MODEL_RF_PATH = str(Path(MODELS_DIR) / "hybrid_random_forest.pkl")
MODEL_XGB_PATH = str(Path(MODELS_DIR) / "hybrid_xgboost.pkl")
MODEL_LGB_PATH = str(Path(MODELS_DIR) / "hybrid_lightgbm.pkl")
MODEL_LR_PATH = str(Path(MODELS_DIR) / "hybrid_logistic_regression.pkl")
SCALER_PATH = str(Path(MODELS_DIR) / "scaler_hybrid.pkl")

# Mô hình kiểm thử cho khung 5 ngày (T+5)
MODEL_RF_T5_PATH = str(Path(MODELS_DIR) / "model_rf_t5.joblib")
MODEL_XGB_T5_PATH = str(Path(MODELS_DIR) / "model_xgb_t5.joblib")
MODEL_LGB_T5_PATH = str(Path(MODELS_DIR) / "model_lgb_t5.joblib")
MODEL_LR_T5_PATH = str(Path(MODELS_DIR) / "model_lr_t5.joblib")
MODEL_PATH = MODEL_XGB_PATH

# Danh mục mô hình học máy hỗ trợ kiểm định và suy luận
AVAILABLE_MODELS = {
    "XGBoost (T+1)": MODEL_XGB_PATH,
    "Random Forest (T+1)": MODEL_RF_PATH,
    "LightGBM (T+1)": MODEL_LGB_PATH,
    "Logistic Regression (T+1)": MODEL_LR_PATH,
    "Random Forest (T+5)": MODEL_RF_T5_PATH,
    "XGBoost (T+5)": MODEL_XGB_T5_PATH,
    "LightGBM (T+5)": MODEL_LGB_T5_PATH,
    "Logistic Regression (T+5)": MODEL_LR_T5_PATH
}

# Thư mục hình ảnh đồ thị 300 DPI
FIGURES_DIR = str(PROJECT_ROOT / "figures") if (PROJECT_ROOT / "figures").exists() else str(APP_DIR / "figures")

# Thư mục cache Hugging Face cho FinBERT
HF_CACHE_DIR = str(PROJECT_ROOT / "hf_cache") if (PROJECT_ROOT / "hf_cache").exists() else str(APP_DIR / "hf_cache")
FINBERT_MODEL_NAME = "ProsusAI/finbert"

# Section 3: Trading Parameters & Quantitative Thresholds
TICKER = "^GSPC"
TICKER_NAME = "S&P 500 Index"
BENCHMARK_NAME = "Buy & Hold S&P 500"

# Dải ngưỡng tự tin tối ưu được chứng minh trong Bảng 7 Báo cáo Nhóm 9
THRESHOLD_OPTIMAL = 0.42   # Ngưỡng tối ưu trên Random Forest khung 5 ngày
THRESHOLD_BASELINE = 0.50  # Ngưỡng nhị phân tiêu chuẩn
DEFAULT_THRESHOLD = 0.42
DEFAULT_FEE = 0.0005
DEFAULT_MODEL_NAME = "Random Forest (T+5)"

# Chi phí giao dịch thực tế
TRADING_FEE = 0.0005       # 0.05% cho mỗi lần đảo vị thế

# Cửa sổ thời gian phân tích
YFINANCE_PERIOD = "1y"
CACHE_TTL_MARKET = 300     # 5 phút (300 giây) cache API Yahoo Finance

# Section 4: 22 Hybrid Features
TECH_FEATURES = [
    "ret_1d", "ret_2d", "ret_5d",
    "dist_sma20", "dist_sma50",
    "rsi_14", "macd", "macd_diff",
    "bb_pct_b", "vol_change_1d", "vol_ratio_sma20"
]

SENTIMENT_FEATURES = [
    "sent_mean", "sent_std",
    "p_pos", "p_neg", "p_neu",
    "news_vol", "pos_ratio", "neg_ratio",
    "sent_momentum_1d", "sent_momentum_3d",
    "news_vol_ratio_5d"
]

ALL_22_FEATURES = TECH_FEATURES + SENTIMENT_FEATURES
ALL_FEATURES = ALL_22_FEATURES

# Section 5: Design System Palette (Linear / Vercel / Bloomberg Luxury Dark Theme)
COLOR_BG_DARK = "#080c14"                        # Canvas base
COLOR_CARD_BG = "rgba(16, 24, 40, 0.65)"         # Frosted glass card
COLOR_CARD_BORDER = "rgba(255, 255, 255, 0.08)" # Crisp subtle border
COLOR_CARD_HOVER = "rgba(0, 229, 153, 0.35)"     # Emerald highlight

COLOR_BULLISH = "#00E599"                        # Emerald Bullish / Alpha
COLOR_BULLISH_BG = "rgba(0, 229, 153, 0.12)"

COLOR_BEARISH = "#FF4D4F"                        # Crimson Bearish / Risk
COLOR_BEARISH_BG = "rgba(255, 77, 79, 0.12)"

COLOR_NEUTRAL = "#F5A623"                        # Amber Neutral / Macro Uncertainty
COLOR_NEUTRAL_BG = "rgba(245, 166, 35, 0.12)"

COLOR_CYAN = "#00B2FF"                           # Cyan Macro / Technology
COLOR_TEXT_PRIMARY = "#F0F4F8"                   # Crisp snow white text
COLOR_TEXT_SECONDARY = "#8E9CAE"                 # Muted slate gray (WCAG AA compliant >= 4.5:1)
