# MPS-500: Multimodal Market Prediction System

> **Báo cáo Đồ án Nghiên cứu Học máy & Trí tuệ Nhân tạo trong Tài chính – Nhóm 9 (Lớp 50K33)**  
> **Khoa Tài chính, Trường Đại học Kinh tế – Đại học Đà Nẵng**  
> **Giảng viên hướng dẫn:** PGS.TS. Nguyễn Văn Chức  
> **Đề tài:** Dự báo hướng biến động giá chỉ số S&P 500 dựa trên phân tích cảm xúc tin tức vĩ mô và các chỉ báo kỹ thuật định lượng  
> **Kiến trúc mô hình:** FinBERT (`ProsusAI/finbert`) × Random Forest / XGBoost / LightGBM / Logistic Regression (22 đặc trưng lai)  
> **Tập dữ liệu chuẩn:** 139.919 bản tin tài chính • 2.584 phiên S&P 500 • 2.169 mẫu sạch sau Weekend Accumulation và khởi tạo SMA50  

---

## 🏆 Đột Phá Định Lượng & Tri Thức Thực Nghiệm Mới Nhất

1. **Vị trí số 1 đồng thuận tuyệt đối của biến `p_neu` (Hạng 2.25):**
   - Biến `p_neu` (xác suất tin tức mang tính trung tính hoặc tâm lý chờ đợi chính sách) vượt lên giữ vị trí số 1 qua cả 4 mô hình học máy (Hạng 1 ở Random Forest, Hạng 2 ở Logistic Regression và XGBoost).
   - Phản ánh chính xác tâm lý thận trọng, trì hoãn giải ngân của các định chế tài chính trước các kỳ họp lãi suất FOMC hoặc công bố CPI.
2. **Nghịch lý ROC-AUC < 0.50 ở khung 1 ngày (24h):**
   - Ở khung 1D với ngưỡng nhị phân 0.50, chỉ số ROC-AUC của cả 12 cấu hình đều < 0.50 do độ nhiễu vi mô và sự mất cân bằng tự nhiên của thị trường tăng điểm (54.7% phiên tăng). Điều này đặt ra yêu cầu phải mở rộng khung thời gian tuần ($t+5$).
3. **Bùng nổ độ chính xác ở khung tuần ($t+5$):**
   - Nhờ hiện tượng *Độ trễ thẩm thấu thông tin (Information Absorption Delay)*, độ chính xác tăng vọt lên **61.52% (Logistic Regression)** và **59.68% (Random Forest)** khi dòng tiền lớn cần vài ngày để phản ánh đầy đủ tin tức vĩ mô.
4. **Chiến lược Tối ưu Hóa Ngưỡng Tự Tin $P \ge 0.42$ trên Random Forest Khung 5 Ngày:**
   - Cơ chế Bagging 150 cây cho xác suất đầu ra trơn mượt (smooth calibration).
   - Đặt vị thế mặc định là Mua (Long) nhằm tận dụng xu thế tăng dài hạn, chỉ thoát sang tiền mặt khi mô hình cực kỳ tự tin thị trường sẽ sụt giảm mạnh ($P < 0.42$).
   - Cắt giảm **71.7% số lệnh thừa** (từ 53 xuống 15 lệnh), triệt tiêu chi phí ma sát (Turnover Drag), và **chính thức vượt qua Buy & Hold S&P 500 (Lợi nhuận 51.91% vs 51.63%, Hệ số Sharpe 1.60 vs 1.59)**.

---

## 📋 Cấu Trúc Mã Nguồn Module Hóa Của Dashboard

```text
sp500_quant_dashboard/
├── app.py                      # 🚀 File chạy chính (Main Streamlit App - 4 Phân hệ cốt lõi)
├── config.py                   # ⚙️ Cấu hình đường dẫn, màu sắc, ngưỡng 0.42, 22 biến lai
├── run.bat                     # ⚡ Khởi chạy 1-click trên Windows Command Prompt
├── run.ps1                     # ⚡ Khởi chạy tự động trên Windows PowerShell
├── requirements.txt            # 📦 Thư viện Python phụ thuộc
├── utils/
│   ├── market_data.py          # 📊 Kết nối Yahoo Finance (cache 300s) & tính 11 chỉ báo kỹ thuật
│   ├── finbert_inference.py    # 🤖 Pipeline FinBERT chấm điểm cảm xúc (cache RAM) & kho tin tức
│   └── model_loader.py         # 🧠 Nạp mô hình pkl & suy luận dự báo kép (t+1, t+5)
├── components/
│   ├── ui_cards.py             # 🎴 Thẻ metric, badge Bullish/Bearish, khuyến nghị đầu tư
│   └── charts.py               # 📈 Đồ thị tương tác Plotly (Nến + MACD, Gauge 0.42, Equity Curve)
└── styles/
    └── custom.css              # 🎨 Giao diện Dark Theme chuẩn Bloomberg/TradingView
```

---

## 🚀 Hướng Dẫn Khởi Chạy

### Cách 1: Khởi chạy nhanh bằng file script (Khuyến nghị)
Nhấp đúp chuột vào file `run.bat` hoặc mở PowerShell và chạy:
```powershell
.\run.ps1
```

### Cách 2: Khởi chạy trực tiếp bằng dòng lệnh Streamlit
Mở terminal tại thư mục `sp500_quant_dashboard/` và thực thi:
```bash
streamlit run app.py
```
Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ mặc định: `http://localhost:8501`.
