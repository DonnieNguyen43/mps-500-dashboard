# DỰ ÁN HỌC MÁY: DỰ BÁO HƯỚNG BIẾN ĐỘNG GIÁ CHỈ SỐ S&P 500
### Multimodal Machine Learning for S&P 500 Market Direction Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-FinBERT-yellow.svg)](https://huggingface.co/ProsusAI/finbert)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App%20Live-brightgreen.svg)](https://mps-500-quant.streamlit.app)

* **Giảng viên hướng dẫn:** PGS.TS. Nguyễn Văn Chức (Khoa Thống kê - Tin học, Trường Đại học Kinh tế – Đại học Đà Nẵng)
* **Nhóm thực hiện:** Nhóm 9 (Lớp 50K33, Khoa Tài chính - Ngân hàng, Trường Đại học Kinh tế – Đại học Đà Nẵng)
* **Ứng dụng trực tuyến (Web Application):** [https://mps-500-quant.streamlit.app](https://mps-500-quant.streamlit.app/)

---

## TỔNG QUAN ĐỀ TÀI (QUY TRÌNH 6 BƯỚC TRIỂN KHAI)

### Bước 1: Xác định bài toán kinh doanh (Business Understanding)
* **Phạm vi bài toán:** Phân loại nhị phân chuỗi thời gian nhằm dự báo hướng biến động giá đóng cửa chỉ số S&P 500 ($y_{t+k} \in \{0, 1\}$ với $k \in \{1, 5\}$ phiên giao dịch).
* **Mục tiêu định lượng:** Xây dựng chiến lược phân loại tích hợp cơ chế phòng vệ rủi ro giá giảm. Dựa trên xu thế tăng trưởng dài hạn của thị trường chứng khoán Mỹ, hệ thống mặc định duy trì vị thế mua (*Long Position*) để tối ưu hóa tỷ suất sinh lời, và chỉ đóng vị thế chuyển sang tiền mặt (*Cash Position*) khi mô hình xác nhận xác suất sụt giảm với độ tin cậy cao. Phương pháp này giúp triệt tiêu các lệnh đảo chiều không cần thiết và hạn chế tối đa chi phí ma sát giao dịch.
* **Phương pháp tiếp cận:** Tích hợp mô hình lai đa phương thức (*Multimodal Hybrid Model*) giữa xử lý ngôn ngữ tự nhiên tài chính chuyên biệt (FinBERT) và 11 chỉ báo kỹ thuật số học định lượng.

---

### Bước 2: Thấu hiểu dữ liệu (Data Understanding)
* **Quy mô tập dữ liệu:** 139.919 bản tin tài chính vĩ mô (giai đoạn 2016 – 2026, 0 giá trị thiếu, 0 bản ghi trùng lặp) và 2.584 phiên giao dịch chỉ số S&P 500 (`^GSPC` thu thập từ Yahoo Finance).
* **Ma trận đặc trưng lai (22 biến độc lập):**
  * **11 chỉ báo kỹ thuật định lượng:** `ret_1d`, `ret_2d`, `ret_5d`, `dist_sma20`, `dist_sma50`, `rsi_14`, `macd`, `macd_diff`, `bb_pct_b`, `vol_change_1d`, `vol_ratio_sma20`.
  * **11 biến cảm xúc FinBERT:** `sent_mean`, `sent_std`, `p_pos`, `p_neg`, `p_neu`, `news_vol`, `pos_ratio`, `neg_ratio`, `sent_momentum_1d`, `sent_momentum_3d`, `news_vol_ratio_5d`.
* **Không gian mẫu hợp lệ:** 2.169 phiên giao dịch hoàn chỉnh sau tiền xử lý (từ ngày 16/03/2016 đến ngày 07/04/2026).

---

### Bước 3: Phân tích khám phá dữ liệu (Exploratory Data Analysis)
* **Chuỗi lợi suất:** Giá trị trung bình $\mu = 0{,}00048$ (+12,11%/năm), độ lệch chuẩn $\sigma = 0{,}01140$ (18,09%/năm). Hệ số xiên Skewness = -0,6720 (phân phối lệch trái), hệ số nhọn dư Excess Kurtosis = 16,42640. Kiểm định Jarque-Bera đạt 29.234,38 ($p < 0{,}001$), bác bỏ hoàn toàn giả thuyết phân phối chuẩn, chứng minh sự hiện diện rõ nét của hiện tượng đuôi cực dày (*Fat Tails*).
* **Dữ liệu văn bản:** Độ dài tiêu đề có trung vị 11–12 từ (tối đa 57 từ), đảm bảo nằm trọn trong giới hạn mã hóa 64 token của FinBERT mà không làm mất mát thông tin.
* **Hiện tượng phân tán thời gian:** Ghi nhận 889 ngày cuối tuần phát sinh tin tức khi thị trường đóng cửa, đòi hỏi cơ chế xử lý căn chỉnh thời gian phù hợp để phản ánh đúng tác động dòng thông tin.

---

### Bước 4: Tiền xử lý dữ liệu (Data Preprocessing)
* **Đồng bộ hóa thời gian:** Áp dụng thuật toán tích lũy tin tức cuối tuần (*Weekend News Accumulation*), dồn toàn bộ dữ liệu tin tức phát sinh trong kỳ nghỉ vào phiên sáng thứ Hai trước giờ mở cửa sàn giao dịch nhằm loại bỏ triệt để rủi ro sử dụng trước dữ liệu tương lai (*Look-ahead Bias*).
* **Làm sạch dữ liệu:** Loại bỏ các giá trị khuyết phát sinh từ độ trễ tính toán chỉ báo kỹ thuật (cửa sổ trượt), xác lập bộ dữ liệu sạch đồng nhất gồm đúng 2.169 quan sát.
* **Chuẩn hóa & Chống rò rỉ dữ liệu:** Bộ chuẩn hóa Z-Score (`StandardScaler`) chỉ khớp tham số trên tập huấn luyện (*Fit on Train Only*). Mô hình FinBERT được cố định toàn bộ trọng số ở chế độ suy luận (`model.eval()`).
* **Phân chia tập dữ liệu tuần tự (80/20):**
  * Tập huấn luyện (*Train Set*): 1.735 mẫu (từ 16/03/2016 đến 14/12/2023).
  * Tập kiểm thử ngoài mẫu (*Out-of-Sample Test Set*): 434 mẫu (từ 15/12/2023 đến 07/04/2026).

---

### Bước 5: Huấn luyện, đánh giá và triển khai mô hình (Model Training & Evaluation)
* **Thuật toán phân loại:** Hồi quy Logistic, Rừng ngẫu nhiên (150 cây, độ sâu tối đa 6), LightGBM (100 cây, độ sâu tối đa 4), XGBoost (100 cây, độ sâu tối đa 3).
* **Kết quả thực nghiệm khung 1 ngày ($t+1$):** Mô hình lai XGBoost đạt độ chính xác 53,00%, Macro F1 0,458, lợi nhuận tích lũy 29,93%, Sharpe 1,09 ở ngưỡng mặc định $P \ge 0{,}50$. Lợi nhuận thấp hơn chiến lược Mua và Nắm giữ thụ động (51,63%, Sharpe 1,59) do chi phí ma sát giao dịch phát sinh khi đảo vị thế liên tục quanh vùng xác suất có độ nhiễu cao.
* **Kết quả thực nghiệm khung tuần ($t+5$):** Độ chính xác của cả 4 mô hình tăng lên mức 56,91% – 61,52% nhờ độ trễ hấp thụ thông tin vĩ mô của dòng tiền thể chế. Rừng ngẫu nhiên đạt Macro F1 (0,449) và ROC-AUC (0,488) cao nhất.
* **Tối ưu hóa ngưỡng xác suất quyết định ($P \ge 0{,}42$):** Trên tập kiểm thử có 64,52% phiên tăng giá, việc điều chỉnh ngưỡng kích hoạt mua về $P \ge 0{,}42$ giúp hệ thống duy trì vị thế mua theo xu hướng chính và chỉ bán phòng vệ khi xác suất suy giảm đạt độ tin cậy cao ($P < 0{,}42$). Cơ chế này cắt giảm 71,7% số lệnh giao dịch (từ 53 xuống 15 lệnh), đạt lợi nhuận tích lũy 51,91%, hệ số Sharpe 1,60, mức sụt giảm tối đa (MDD) -15,54%, chính thức vượt qua điểm chuẩn Mua và Nắm giữ (51,70%, Sharpe 1,59, MDD -15,54%).
* **Triển khai ứng dụng:** Đóng gói toàn bộ mô hình và quy trình xử lý dữ liệu lên nền tảng đám mây Streamlit tại [https://mps-500-quant.streamlit.app](https://mps-500-quant.streamlit.app/).

---

### Bước 6: Trích xuất tri thức và đề xuất (Insights & Proposals)
* **Tri thức từ đặc trưng đồng thuận:** Xác suất trung tính `p_neu` đạt thứ hạng đồng thuận số 1 (thứ hạng trung bình 2,25) trên cả 4 mô hình. Trạng thái hoài nghi hoặc chờ đợi vĩ mô của thị trường phản ánh sự biến động dòng tiền có ý nghĩa dự báo cao hơn các tin tức phân cực rõ nét.
* **Khuyến nghị thực tiễn:** Không áp dụng chiến lược giao dịch hàng ngày ở khung 1 ngày với ngưỡng phân loại đối xứng 0,50. Khuyến nghị vận hành mô hình Rừng ngẫu nhiên khung 5 ngày theo cơ chế bộ lọc phòng vệ rủi ro đuôi với ngưỡng quyết định $P < 0{,}42$.

---

## 📁 CẤU TRÚC THƯ MỤC DỰ ÁN

```text
ML_Dự báo hướng biến động giá chỉ số S&P 500/
├── README.md                      # Tài liệu tổng quan dự án chuẩn học thuật
├── requirements.txt               # Danh mục thư viện Python yêu cầu
├── data/
│   ├── raw/                       # Dữ liệu gốc ban đầu
│   │   ├── sp500_market_raw.csv   # Dữ liệu giá giao dịch S&P 500 (2.584 phiên)
│   │   └── stock_news_2016 to 2026.csv # 139.919 bản tin vĩ mô
│   └── processed/                 # Dữ liệu sau tiền xử lý
│       ├── cleaned_news_raw.csv   # Dữ liệu tin tức sau làm sạch
│       ├── daily_sentiment_features.csv # 11 chỉ báo cảm xúc FinBERT theo ngày
│       ├── hybrid_dataset.csv     # Bộ dữ liệu lai 22 đặc trưng (2.169 dòng)
│       └── train_test_info.pkl    # Metadata phân chia tập Train / Test
├── notebooks/
│   └── Gr9_ML.ipynb               # Sổ tay thực nghiệm Jupyter Notebook
├── src/                           # Mã nguồn thực thi chuẩn hóa mô-đun
│   ├── __init__.py
│   ├── 1_data_collection.py       # Thu thập dữ liệu giá S&P 500 và tin tức vĩ mô
│   ├── 2_eda_statistical_test.py  # Phân tích thống kê mô tả và kiểm định Jarque-Bera
│   ├── 3_finbert_sentiment.py     # Trích xuất cảm xúc văn bản chuyên biệt FinBERT
│   ├── 4_feature_engineering.py   # Dồn tin cuối tuần và kiến tạo 22 biến lai
│   ├── 5_train_models.py          # Huấn luyện mô hình và thực nghiệm bóc tách
│   ├── 6_feature_importance.py    # Phân tích tầm quan trọng đặc trưng đồng thuận
│   └── 7_backtest_strategy.py     # Kiểm nghiệm chiến lược và tối ưu hóa ngưỡng cắt lỗ
├── models/                        # Trọng số mô hình và bộ chuẩn hóa đã lưu trữ
│   ├── scaler_hybrid.pkl          # Bộ chuẩn hóa StandardScaler khớp trên tập Train
│   ├── hybrid_logistic_regression.pkl
│   ├── hybrid_random_forest.pkl
│   ├── hybrid_lightgbm.pkl
│   └── hybrid_xgboost.pkl
└── reports/                       # Báo cáo học thuật và đồ thị trực quan hóa
    ├── Bao_Cao_Machine_Learning_Nhom_9.docx # Báo cáo Word chuẩn mực đề tài
    └── figures/                   # 14 đồ thị trực quan hóa kết quả phân tích
        ├── fig1_sp500_price_and_volume.png
        ├── fig2_news_distribution_and_tiers.png
        ├── fig3_news_density_violin.png
        ├── fig4_sp500_fat_tails_distribution.png
        ├── fig5_nlp_headline_eda_and_bigrams.png
        ├── fig6_sp500_volatility_vs_news_volume.png
        ├── fig7_sentiment_dynamics.png
        ├── fig8_correlation_matrix.png
        ├── fig9_ablation_benchmark.png
        ├── fig10_roc_curves.png
        ├── fig11_feature_importance.png
        ├── fig12_backtest_equity_drawdown.png
        ├── fig13_timeframe_comparison.png
        └── fig14_optimized_confidence_band.png
```

---

## 🚀 HƯỚNG DẪN CÀI ĐẶT VÀ VẬN HÀNH

### 1. Cài đặt môi trường
Yêu cầu Python phiên bản 3.10 trở lên. Cài đặt toàn bộ các thư viện phụ thuộc bằng lệnh:
```bash
pip install -r requirements.txt
```

### 2. Thực thi qua Jupyter Notebook
Mở và thực thi tuần tự các bước trong sổ tay thực nghiệm:
```bash
jupyter notebook notebooks/Gr9_ML.ipynb
```

### 3. Thực thi qua các kịch bản dòng lệnh độc lập
Có thể thực thi tuần tự từng giai đoạn nghiên cứu thông qua thư mục `src/`:
```bash
python src/1_data_collection.py
python src/2_eda_statistical_test.py
python src/3_finbert_sentiment.py
python src/4_feature_engineering.py
python src/5_train_models.py
python src/6_feature_importance.py
python src/7_backtest_strategy.py
```

---

*Dự án thực hiện bởi Nhóm 9 (Lớp 50K33 / 49K33, Khoa Tài chính, Trường Đại học Kinh tế – Đại học Đà Nẵng) dưới sự hướng dẫn khoa học của PGS.TS. Nguyễn Văn Chức.*
