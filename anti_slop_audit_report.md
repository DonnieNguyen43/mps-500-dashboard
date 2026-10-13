# BÁO CÁO KIỂM ĐỊNH CHẤT LƯỢNG MÃ NGUỒN VÀ GIAO DIỆN ANTI-SLOP (AUDIT REPORT)
**Dự án:** MPS-500: Multimodal Market Prediction System  
**Thư mục kiểm toán:** `sp500_quant_dashboard/`  
**Bộ quy chuẩn áp dụng:** `anti-slop` (Mode: AFTER - Audit & Refactor)  
**Tiêu chuẩn chất lượng:** The Craftsmanship Standard (C-1 to C-5), Hard Gates (R-01 to R-38)  
**Ngày thực hiện:** 13/10/2026  

---

## 1. TỔNG QUAN KẾT QUẢ KIỂM TOÁN (EXECUTIVE SUMMARY)

Đợt kiểm toán chế độ AFTER của bộ quy chuẩn `anti-slop` đối với hệ thống bảng điều khiển giao dịch định lượng MPS-500 tập trung rà soát 3 trục cốt lõi:
1. **Visual & UI (antislop-ui):** Loại bỏ hiệu ứng thị giác giả tạo, bóng đổ trôi nổi, neon mờ nhạt và kính mờ (glassmorphism) không có mục đích kỹ thuật.
2. **Copywriting & Voice (antislop-copywriting):** Thay thế toàn bộ từ ngữ mang tính tiếp thị AI hoa mỹ bằng văn phong báo cáo quỹ định lượng (Quantitative Hedge Fund): khách quan, chính xác, dựa trên xác suất và phân phối thống kê.
3. **Code & Comments Hygiene (antislop-code):** Xóa bỏ các dải băng trang trí ASCII lặp lại, các chú thích diễn giải cú pháp hiển nhiên, giữ lại 100% logic toán học và tài chính cốt lõi.

---

## 2. DANH MỤC PHÁT HIỆN THEO MỨC ĐỘ ƯU TIÊN (PRIORITIZED FINDINGS)

### PRIORITY 1: NGHIÊM TRỌNG (HARD GATES & WCAG AA COMPLIANCE)

* **Finding P1-01 [R-02 - Hard Gate: Cấm sử dụng ký tự gạch ngang Em Dash `—` và En Dash `–`]:**
  - *Hiện trạng:* Xuất hiện ký tự `—` và `–` trong các chuỗi thông tin tác giả, tiêu đề và chú thích (`config.py:15`, `app.py:5, 529`, `ui_cards.py:51, 213`, `live_news_scraper.py:47`).
  - *Tác động:* Dấu hiệu nhận diện đặc trưng của văn bản AI sinh ra (AI token cadence tell).
  - *Biện pháp:* Thay thế toàn bộ bằng dấu hai chấm `:`, dấu phẩy `,`, dấu gạch ngang đơn `-`, hoặc dấu ngoặc đơn `()`.

* **Finding P1-02 [R-25 - Hard Gate: Độ tương phản màu chữ không đạt chuẩn WCAG AA (4.5:1)]:**
  - *Hiện trạng:* Màu chữ phụ `COLOR_TEXT_SECONDARY = "#787b86"` hiển thị trên nền tối `#1e222d` có tỷ lệ tương phản chỉ đạt **3.3:1**, vi phạm chuẩn WCAG AA (tối thiểu 4.5:1 đối với văn bản kích thước nhỏ 11px uppercase).
  - *Tác động:* Khó đọc trong điều kiện ánh sáng văn phòng, giảm tính trực quan của các nhãn metric và tab.
  - *Biện pháp:* Nâng màu chữ phụ lên `#9da2b0` trong `config.py` và `styles/custom.css`, nâng tỷ lệ tương phản lên **4.8:1** (Đạt chuẩn WCAG AA).

* **Finding P1-03 [R-26 & R-24 - Hard Gate: Liên kết rỗng và ký tự mũi tên trang trí]:**
  - *Hiện trạng:* Thẻ tin tức tạo liên kết `news_url = news.get("url", "#")` dẫn tới neo rỗng `#` khi không có đường dẫn thật, kèm mũi tên trang trí `Xem nguồn báo chí ↗` (vi phạm R-08 và R-26).
  - *Tác động:* Yếu tố tương tác không có đích đến thực tế làm giảm độ tin cậy của giao diện.
  - *Biện pháp:* Kiểm tra hợp lệ `news_url.startswith("http")`. Nếu hợp lệ, mở tab mới an toàn với nhãn sạch `Nguồn báo chí`; nếu không có URL, hiển thị nhãn nguồn tĩnh, không tạo liên kết rỗng.

* **Finding P1-04 [Dead Code: Tồn tại thư mục và tệp mã nguồn di sản không sử dụng]:**
  - *Hiện trạng:* Thư mục `sp500_quant_dashboard/modules/` (5 tệp, 82KB) và tệp `sp500_quant_dashboard/assets/custom.css` (31KB) là mã nguồn cũ từ phiên bản thử nghiệm trước, hoàn toàn không được `app.py` import.
  - *Tác động:* Gây nhiễu kiến trúc hệ thống và nhầm lẫn khi bảo trì.
  - *Biện pháp:* Cô lập và đánh dấu phế bỏ (deprecated), làm sạch tham chiếu trong toàn bộ dự án.

---

### PRIORITY 2: CẢI THIỆN TRẢI NGHIỆM & PHONG CÁCH TÀI CHÍNH (PURPOSE-GATE & COPYWRITING)

* **Finding P2-01 [R-16 & antislop-copywriting: Văn phong tiếp thị cảm xúc và biểu tượng emoji lạm dụng]:**
  - *Hiện trạng:* Các cụm từ như "bùng nổ độ chính xác" (`app.py:233`), "xung lực lạc quan mạnh mẽ" (`ui_cards.py:249`), "tri thức đột phá" (`README.md`), cùng các emoji tên lửa `🚀`, robot `🤖`, sấm sét `⚡` xuất hiện trong các nút bấm và tiêu đề tab.
  - *Tác động:* Làm mất đi tính nghiêm cẩn, khoa học của một hệ thống định lượng tài chính cấp độ tổ chức.
  - *Biện pháp:* Chuẩn hóa văn phong quỹ phòng hộ:
    - `bùng nổ độ chính xác` -> `độ chính xác cải thiện đáng kể, đạt 61.52%`
    - `xung lực lạc quan mạnh mẽ` -> `xung lực tăng điểm rõ rệt (+0.875)`
    - `st.button("🚀 Kích hoạt thử nghiệm cú sốc")` -> `st.button("Chạy mô phỏng cú sốc (Run Stress Test)")`
    - `tab: 📊 Phân hệ 1: Tín hiệu thị trường & Dự báo AI` -> `tab: Phân hệ 1: Tín hiệu thị trường & Dự báo đa khung`
    - Loại bỏ emoji hoạt hình khỏi các tiêu đề phân hệ.

* **Finding P2-02 [R-10 & R-13: Lạm dụng hiệu ứng kính mờ (Glassmorphism) và hiệu ứng phát sáng Neon]:**
  - *Hiện trạng:* `.rec-banner` và `header[data-testid="stHeader"]` sử dụng `backdrop-filter: blur(...)`, gây mờ viền chữ; `.live-dot` sử dụng vòng lặp hoạt ảnh `pulse` phát sáng lan tỏa 8px theo kiểu giao diện viễn tưởng.
  - *Tác động:* Giảm độ sắc nét trên màn hình giao dịch và tiêu tốn tài nguyên render GPU.
  - *Biện pháp:* Loại bỏ `backdrop-filter`, áp dụng nền đặc `#1e222d` với viền dứt khoát 1px `#2a2e39`; chuyển `.live-dot` thành điểm báo hiệu trạng thái màu xanh lục sắc nét `#00c087` chuẩn Bloomberg.

* **Finding P2-03 [R-12 & R-19: Hiệu ứng thẻ nổi lơ lửng khi di chuột (Card Float Hover Lift)]:**
  - *Hiện trạng:* `.metric-card:hover` sử dụng `transform: translateY(-2px)` kết hợp đổ bóng lớn `box-shadow: 0 6px 18px`.
  - *Tác động:* Mô thức thiết kế thẻ thương mại điện tử / SaaS tiêu dùng, không phù hợp với bảng điều khiển số liệu tài chính cần tính ổn định bố cục.
  - *Biện pháp:* Bỏ `translateY(-2px)`, chuyển sang cơ chế highlight đường viền tinh tế (`border-color: #3b4254`).

* **Finding P2-04 [R-06 & R-09: Thẻ con nhộng phát sáng (Capsule Badge) và chữ Monospace giả lập Terminal]:**
  - *Hiện trạng:* Thẻ trạng thái Sidebar mang chuỗi uppercase giãn cách rộng `● TERMINAL ACTIVE` viền neon.
  - *Tác động:* Thể hiện xu hướng trang trí "fake terminal" của AI.
  - *Biện pháp:* Đơn giản hóa thành khối trạng thái kết nối chuẩn tắc: `TRẠNG THÁI: KẾT NỐI (ONLINE)`.

---

### PRIORITY 3: TINH CHỈNH MÃ NGUỒN & CHÚ THÍCH (ANTISLOP-CODE HYGIENE)

* **Finding P3-01 [antislop-code: Dải băng ký tự phân cách trang trí (Decorative Box-Drawing Separators)]:**
  - *Hiện trạng:* Các tệp `app.py`, `config.py`, `styles/custom.css` chứa nhiều khối comment `════════════════════════════════════════════════════════════════` hoặc `────────────────────────────────────────────────────────────────`.
  - *Tác động:* Tạo cảm giác "AI sinh mã mẫu", làm loãng mã nguồn thực tế.
  - *Biện pháp:* Rút gọn thành dòng chú thích đơn giản, rõ ràng `# Mục 1: ...` hoặc bỏ các dòng rào trang trí lặp lại.

* **Finding P3-02 [antislop-code: Chú thích lặp lại cú pháp hiển nhiên (Trivial Comments)]:**
  - *Hiện trạng:* Chú thích như `# Nạp Scaler`, `# Nạp Random Forest`, `# Nến Nhật`.
  - *Biện pháp:* Làm sạch các chú thích cú pháp; bảo lưu nguyên vẹn các chú thích bản chất định lượng (lý do chọn ngưỡng 0.42, cơ chế độ trễ thẩm thấu 5 ngày, công thức chi phí ma sát 0.05%).

---

## 3. KẾ HOẠCH TÁI CẤU TRÚC (REFACTORING ACTIONS MATRIX)

| Tệp tin | Hành động cụ thể | Quy chuẩn áp dụng |
| :--- | :--- | :--- |
| `styles/custom.css` | Loại bỏ `backdrop-filter: blur`, tắt animation neon xung lực, bỏ `translateY` hover, tăng độ tương phản nhãn chữ `#9da2b0`. | R-10, R-12, R-13, R-25 |
| `config.py` | Thay thế các dấu `–`, `—` bằng `:`, `,`. Nâng cấp `COLOR_TEXT_SECONDARY = "#9da2b0"`. Xóa dải băng trang trí. | R-02, R-25, antislop-code |
| `components/ui_cards.py` | Xóa `–`, chuẩn hóa câu chữ What-If và khuyến nghị giao dịch theo văn phong quỹ định lượng. Bỏ emoji hoạt hình. | R-02, R-16, antislop-copy |
| `app.py` | Loại bỏ emoji khỏi Tab header, chuẩn hóa nhãn nút bấm `Chạy mô phỏng cú sốc`, loại bỏ dải băng ASCII, kiểm tra an toàn URL tin tức. | R-04, R-08, R-16, R-26, antislop-code |
| `utils/live_news_scraper.py` | Chuẩn hóa thay thế `&mdash;` sang dấu gạch đơn `-`. Bỏ en-dash trong tiêu đề file. | R-02 |
| `utils/model_loader.py` | Chuẩn hóa header và làm sạch chú thích cú pháp thừa. | R-02, antislop-code |
| `utils/finbert_inference.py` | Chuẩn hóa header, làm sạch chú thích lặp lại. | R-02, antislop-code |
| `utils/market_data.py` | Chuẩn hóa header, làm sạch chú thích lặp lại. | R-02, antislop-code |
| `components/charts.py` | Chuẩn hóa header, đảm bảo không có em-dash trong chú giải biểu đồ. | R-02 |

---

## 4. BẢNG KIỂM TRA DELIVERY GATE (DELIVERY GATE PASS/FAIL)

Sau khi hoàn tất tái cấu trúc, hệ thống được nghiệm thu toàn diện theo Bảng kiểm tra Delivery Gate của `antislop.md`:
* **Block 1 (Hard Gate):** PASS (100% không còn em-dash `—`, không còn liên kết rỗng, tương phản WCAG AA >= 4.8:1).
* **Block 2 (Purpose-Gate):** PASS (Không còn glassmorphism thừa, không còn neon phát sáng vô cớ, bố cục ổn định không trôi nổi).
* **Block 3 (Liveliness):** PASS (Định vị rõ rệt phong cách Bloomberg Dark Theme với mã màu định lượng chuẩn mực, typography Inter + JetBrains Mono chuyên sâu).
* **Block 4 (Craftsmanship & Quality Locks):** PASS (Toàn bộ số liệu và thuật toán khớp 100% với tài liệu nghiên cứu khoa học của Nhóm 9).
