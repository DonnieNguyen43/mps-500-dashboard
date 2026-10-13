"""
train_models.py: Kịch bản huấn luyện và xuất bản mô hình tự động cho MPS-500
Tương ứng trực tiếp với các ô lệnh 5, 6, 7 trong Gr9_ML.ipynb và Báo cáo Nhóm 9.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainQuantPulse")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PARENT_DIR, "data_processed", "hybrid_dataset.csv")
MODELS_DIR = os.path.join(PARENT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

TECH_FEATURES = [
    'ret_1d', 'ret_2d', 'ret_5d', 'dist_sma20', 'dist_sma50', 
    'rsi_14', 'macd', 'macd_diff', 'bb_pct_b', 'vol_change_1d', 'vol_ratio_sma20'
]

SENT_FEATURES = [
    'sent_mean', 'sent_std', 'p_pos', 'p_neg', 'p_neu', 
    'news_vol', 'pos_ratio', 'neg_ratio', 'sent_momentum_1d', 
    'sent_momentum_3d', 'news_vol_ratio_5d'
]

HYBRID_FEATURES = TECH_FEATURES + SENT_FEATURES


def main():
    logger.info("=== BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN QUANTPULSE-500 THEO GR9_ML ===")
    
    if not os.path.exists(DATA_PATH):
        logger.error(f"Không tìm thấy tập dữ liệu tại: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    logger.info(f"Đã nạp {len(df):,} phiên giao dịch hợp lệ ({df['date'].min().strftime('%Y-%m-%d')} -> {df['date'].max().strftime('%Y-%m-%d')}).")

    # Xử lý số thực vô cực và ngoại lai
    df[HYBRID_FEATURES] = df[HYBRID_FEATURES].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    for col in HYBRID_FEATURES:
        df[col] = df[col].clip(df[col].quantile(0.001), df[col].quantile(0.999))

    # Phân chia chuỗi thời gian không xáo trộn: 80% train, 20% test
    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    logger.info(f"Tập Train (80%): {len(train_df):,} phiên | Tập Test ngoài mẫu (20%): {len(test_df):,} phiên.")

    # Chuẩn hóa StandardScaler
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        import joblib
    except ImportError as e:
        logger.error(f"Môi trường thiếu thư viện học máy: {e}. Vui lòng chạy trong môi trường có scikit-learn, joblib.")
        return

    try:
        import lightgbm as lgb
        HAS_LGB = True
    except ImportError:
        HAS_LGB = False
        logger.warning("Không tìm thấy LightGBM. Sẽ sử dụng GradientBoosting thay thế nếu cần.")

    try:
        import xgboost as xgb
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False
        logger.warning("Không tìm thấy XGBoost. Sẽ sử dụng Random Forest làm Champion t+1.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[HYBRID_FEATURES])
    X_test = scaler.transform(test_df[HYBRID_FEATURES])

    y_train_t1 = train_df['target_t1'].values
    y_test_t1 = test_df['target_t1'].values
    y_train_t5 = train_df['target_t5'].values
    y_test_t5 = test_df['target_t5'].values

    # Lưu Scaler
    scaler_path = os.path.join(MODELS_DIR, "hybrid_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    logger.info(f"Đã lưu Scaler tại: {scaler_path}")

    # 1. Huấn luyện Logistic Regression
    logger.info("Huấn luyện Logistic Regression...")
    model_lr = LogisticRegression(max_iter=1000, random_state=42)
    model_lr.fit(X_train, y_train_t1)
    joblib.dump(model_lr, os.path.join(MODELS_DIR, "model_lr.joblib"))

    # 2. Huấn luyện Random Forest
    logger.info("Huấn luyện Random Forest (t+1 và t+5)...")
    model_rf_t1 = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_split=10, random_state=42, n_jobs=-1)
    model_rf_t1.fit(X_train, y_train_t1)

    model_rf_t5 = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_split=10, random_state=42, n_jobs=-1)
    model_rf_t5.fit(X_train, y_train_t5)
    joblib.dump(model_rf_t5, os.path.join(MODELS_DIR, "model_rf_t5.joblib"))

    # 3. Huấn luyện LightGBM (nếu có)
    if HAS_LGB:
        logger.info("Huấn luyện LightGBM...")
        model_lgb = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.03, num_leaves=15, random_state=42, verbose=-1, importance_type='gain')
        model_lgb.fit(X_train, y_train_t1)
        joblib.dump(model_lgb, os.path.join(MODELS_DIR, "model_lgb.joblib"))
    else:
        model_lgb = None

    # 4. Huấn luyện XGBoost (Champion t+1 theo Báo cáo Nhóm 9)
    if HAS_XGB:
        logger.info("Huấn luyện XGBoost Classifier (Champion t+1)...")
        model_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.03, eval_metric='logloss', random_state=42, n_jobs=-1, importance_type='gain')
        model_xgb.fit(X_train, y_train_t1)
        joblib.dump(model_xgb, os.path.join(MODELS_DIR, "best_hybrid_model.joblib"))
        champion_model = model_xgb
        champion_name = "XGBoost Hybrid"
    else:
        joblib.dump(model_rf_t1, os.path.join(MODELS_DIR, "best_hybrid_model.joblib"))
        champion_model = model_rf_t1
        champion_name = "Random Forest Hybrid"

    # Đánh giá trên tập test t+1
    preds_t1 = champion_model.predict(X_test)
    probs_t1 = champion_model.predict_proba(X_test)[:, 1]
    acc_t1 = accuracy_score(y_test_t1, preds_t1)
    f1_t1 = f1_score(y_test_t1, preds_t1, average='macro')
    auc_t1 = roc_auc_score(y_test_t1, probs_t1)

    logger.info(f"=== KẾT QUẢ CHAMPION t+1 [{champion_name}] ===")
    logger.info(f"Accuracy: {acc_t1:.2%} | Macro F1: {f1_t1:.3f} | ROC-AUC: {auc_t1:.3f}")

    # Đánh giá Random Forest t+5 với Threshold 0.42
    rf_probs_t5 = model_rf_t5.predict_proba(X_test)[:, 1]
    rf_preds_042 = (rf_probs_t5 >= 0.42).astype(int)
    acc_t5_042 = accuracy_score(y_test_t5, rf_preds_042)
    logger.info(f"=== KẾT QUẢ TỐI ƯU t+5 [Random Forest (P >= 0.42)] ===")
    logger.info(f"Accuracy (t+5, Threshold 0.42): {acc_t5_042:.2%}")

    # Lưu metadata mô hình
    meta = {
        "project": "MPS-500",
        "team": "Nhóm 9 - Lớp 50K33",
        "champion_t1": champion_name,
        "champion_t1_accuracy": float(acc_t1),
        "champion_t1_f1": float(f1_t1),
        "champion_t1_auc": float(auc_t1),
        "optimal_threshold_t5": 0.42,
        "features_count": len(HYBRID_FEATURES),
        "features_list": HYBRID_FEATURES,
        "train_samples": len(train_df),
        "test_samples": len(test_df)
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    logger.info("=== HOÀN TẤT HUẤN LUYỆN VÀ XUẤT BẢN MÔ HÌNH THÀNH CÔNG ===")


if __name__ == "__main__":
    main()
