# -*- coding: utf-8 -*-
"""
MPS-500: Multimodal Market Prediction System
=============================================================================
Đồ án Nghiên cứu Học máy và Trí tuệ Nhân tạo trong Tài chính: Nhóm 9 (Lớp 50K33)
Khoa Tài chính, Trường Đại học Kinh tế, Đại học Đà Nẵng
Giảng viên hướng dẫn: PGS.TS. Nguyễn Văn Chức

Đề tài: "Dự báo hướng biến động giá chỉ số S&P 500 dựa trên phân tích cảm xúc
         tin tức vĩ mô và các chỉ báo kỹ thuật định lượng"

Hệ thống được module hóa thành 4 Phân hệ chức năng cốt lõi:
1. Phân hệ 1: Tín hiệu thị trường thời gian thực & Dự báo đa khung (Market Pulse & Signals)
2. Phân hệ 2: Luồng tin tức vĩ mô & Chấm điểm cảm xúc FinBERT (News Sentiment Feed)
3. Phân hệ 3: Kiểm định định lượng & Hiệu năng mô hình (Quant Performance & Evaluation)
4. Phân hệ 4: Kịch bản giả định & Thử nghiệm cú sốc vĩ mô (What-If Stress Testing)
"""

import os
import sys
import re
import logging
import warnings
from datetime import datetime, date
import pandas as pd
import numpy as np
import streamlit as st

# Đường dẫn module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import config as cfg
from utils.market_data import (
    fetch_sp500_history, compute_technical_indicators,
    get_latest_price_summary, get_latest_tech_features_vector
)
from utils.finbert_inference import (
    load_finbert_pipeline, score_text_finbert, score_news_batch,
    load_latest_daily_sentiment, fetch_filtered_news, fetch_live_news_cached,
    get_historical_sentiment_by_horizon
)
from utils.model_loader import (
    load_all_models_and_scaler, predict_dual_horizon,
    predict_whatif_shift
)
import streamlit.components.v1 as components

from components.ui_cards import (
    render_hero_intro,
    render_terminal_header, render_market_metric_cards,
    render_recommendation_banner, render_whatif_alert_card,
    render_color_coded_badges, render_whatif_explainability_card,
    render_news_feed_card, render_glass_table,
    render_dominant_sentiment_badge, render_section_spine_header,
    render_floating_nav_bar
)
from components.charts import (
    plot_sp500_candlestick_macd, plot_confidence_gauge,
    plot_feature_importance_consensus, plot_equity_curves,
    plot_sentiment_donut_chart, plot_whatif_feature_attribution
)

# Section 1: Streamlit Configuration
st.set_page_config(
    page_title=cfg.APP_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Nạp file CSS giao diện Bloomberg Dark Theme
css_path = os.path.join(os.path.dirname(__file__), "styles", "custom.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Section 2: Sidebar Controls
with st.sidebar:
    # 1. Hiển thị Logo Brand Card độc bản
    if os.path.exists(cfg.LOGO_FULL_PATH):
        st.image(cfg.LOGO_FULL_PATH, use_container_width=True)
    elif os.path.exists(cfg.LOGO_PATH):
        st.image(cfg.LOGO_PATH, use_container_width=True)
    
    st.markdown(
        """
        <div style="text-align: center; margin-top: -6px; margin-bottom: 14px;">
            <span style="background: rgba(0, 192, 135, 0.12); color: #00c087; font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 4px; border: 1px solid rgba(0,192,135,0.3);">
                TRẠNG THÁI: KẾT NỐI (ONLINE)
            </span>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown("### BẢNG ĐIỀU KHIỂN HỆ THỐNG")
    
    st.markdown(f"**Hệ thống:** `{cfg.APP_NAME}`")
    st.markdown(f"**Chỉ số mục tiêu:** `{cfg.TICKER}` ({cfg.TICKER_NAME})")
    
    selected_horizon = st.radio(
        "Khung thời gian mục tiêu:",
        options=["Khung tuần (t+5 - Trọng tâm)", "Khung 1 ngày (t+1)"],
        index=0,
        help="Khung tuần (5 phiên) giúp lọc nhiễu vi mô và phát huy tối đa hiệu quả hấp thụ thông tin vĩ mô."
    )

    st.markdown("---")
    st.markdown("### THIẾT LẬP CHIẾN LƯỢC")
    
    threshold_input = st.slider(
        "Ngưỡng xác suất Long (P):",
        min_value=0.35, max_value=0.60,
        value=cfg.THRESHOLD_OPTIMAL, step=0.01,
        help="Ngưỡng tối ưu 0.42 được chứng minh trên Random Forest giúp đạt Sharpe 1.60 và lợi nhuận 51.91%."
    )
    if threshold_input == cfg.THRESHOLD_OPTIMAL:
        st.success("Đang kích hoạt: Ngưỡng Tối Ưu (0.42)")
    elif threshold_input == cfg.THRESHOLD_BASELINE:
        st.info("Đang áp dụng: Ngưỡng Tiêu Chuẩn (0.50)")

    st.markdown(f"**Phí giao dịch giả định:** `{cfg.TRADING_FEE*100:.2f}% / lượt`")

    st.markdown("---")
    st.markdown("### CẤU TRÚC MÔ HÌNH ĐỊNH LƯỢNG")
    st.markdown("• **FinBERT:** `ProsusAI/finbert` (110M tham số)\n"
                "• **Phân loại t+5:** `Random Forest Hybrid` (150 cây)\n"
                "• **Phân loại t+1:** `XGBoost Hybrid` (100 cây)\n"
                "• **Số đặc trưng:** `22 biến lai` (11 KT + 11 CX)")

    if st.button("Làm mới dữ liệu thị trường"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(f"<div style='font-size: 11px; color: {cfg.COLOR_TEXT_SECONDARY}; line-height: 1.5;'>"
                f"{cfg.AUTHOR_INFO}<br>{cfg.SUPERVISOR}</div>", unsafe_allow_html=True)

# Section 3: Data Ingestion & Signal Inference
# 1. Dữ liệu giá & kỹ thuật
raw_market_df = fetch_sp500_history(period=cfg.YFINANCE_PERIOD)
calc_market_df = compute_technical_indicators(raw_market_df)
price_summary = get_latest_price_summary(calc_market_df)
latest_tech_row = get_latest_tech_features_vector(calc_market_df)

# 2. Dữ liệu cảm xúc FinBERT (chuẩn hóa p_neu)
latest_sent_dict = load_latest_daily_sentiment()

# 3. Dự báo kép từ mô hình Machine Learning
forecast_result = predict_dual_horizon(latest_tech_row, latest_sent_dict)

# Ghi đè ngưỡng nếu người dùng điều chỉnh trên Sidebar
if threshold_input != cfg.THRESHOLD_OPTIMAL:
    forecast_result["optimal_threshold"] = threshold_input
    is_long = forecast_result["prob_t5"] >= threshold_input
    forecast_result["is_long_5d"] = is_long
    if is_long:
        forecast_result["recommendation"] = {
            "action": "LONG",
            "title": "MỞ VỊ THẾ MUA / TIẾP TỤC NẮM GIỮ (LONG)",
            "badge": "BULLISH (LONG)",
            "color": cfg.COLOR_BULLISH,
            "bg_color": cfg.COLOR_BULLISH_BG,
            "rationale": f"Xác suất khung tuần đạt {forecast_result['prob_t5']*100:.1f}%, vượt ngưỡng người dùng chọn ({threshold_input*100:.1f}%)."
        }
    else:
        forecast_result["recommendation"] = {
            "action": "DEFENSIVE CASH",
            "title": "HẠ TỶ TRỌNG / NẮM GIỮ TIỀN MẶT (DEFENSIVE CASH)",
            "badge": "CẢNH BÁO RỦI RO (CASH)",
            "color": cfg.COLOR_BEARISH,
            "bg_color": cfg.COLOR_BEARISH_BG,
            "rationale": f"Xác suất khung tuần đạt {forecast_result['prob_t5']*100:.1f}%, thấp hơn ngưỡng ({threshold_input*100:.1f}%). Khuyến nghị phòng vệ."
        }

# Section 4: Main Layout & Single-Page Vertical Narrative
render_floating_nav_bar()
render_hero_intro()
render_terminal_header()
render_market_metric_cards(price_summary, forecast_result, latest_sent_dict)
render_recommendation_banner(forecast_result)

# =============================================================================
# PHÂN HỆ 1: TÍN HIỆU THỊ TRƯỜNG & ĐỘNG LỰC DỰ BÁO AI ĐA KHUNG THỜI GIAN
# =============================================================================
render_section_spine_header(
    section_id="section-1",
    badge_icon="⚡",
    badge_glow_color="rgba(0, 229, 153, 0.4)",
    headline="Phân hệ 1: Tín hiệu Thị trường & Động lực Dự báo Đa khung",
    subtitle="Động cơ học máy lượng tử kết hợp 11 chỉ báo kỹ thuật và dự báo xác suất kép (1D / 5D)."
)

col_chart, col_gauge = st.columns([0.68, 0.32])

with col_chart:
    st.markdown("##### Diễn biến Giá & Xu hướng Đường trung bình SMA")
    fig_candle = plot_sp500_candlestick_macd(calc_market_df)
    st.plotly_chart(fig_candle, use_container_width=True, config={'displayModeBar': False})

with col_gauge:
    st.markdown(f"""
    <div class="metric-card" style="margin-bottom: 12px;">
        <div class="metric-label">ĐỒ THỊ ĐO LƯỜNG NGƯỠNG TỰ TIN (t+5)</div>
        <div style="font-size: 12px; color: {cfg.COLOR_TEXT_SECONDARY}; margin-bottom: 6px;">
            Vùng màu xanh thể hiện vùng Mua/Nắm giữ an toàn (≥ 42.0%). Vùng đỏ cảnh báo hạ tỷ trọng về tiền mặt.
        </div>
    </div>
    """, unsafe_allow_html=True)
    fig_gauge = plot_confidence_gauge(forecast_result["prob_t5"], threshold=threshold_input)
    st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

    st.markdown(f"""
    <div style="background: rgba(16, 24, 40, 0.65); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 14px 18px; font-size: 12.5px; box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
        <strong style="color: #F0F4F8; font-size: 12px; text-transform: uppercase; letter-spacing: 0.8px;">Thông số kỹ thuật phiên hiện hành:</strong><br><br>
        • RSI (14 phiên): <strong style="color: #F0F4F8; font-family: 'JetBrains Mono';">{calc_market_df['rsi_14'].iloc[-1]:.2f}</strong><br>
        • MACD Histogram: <strong style="color: #F0F4F8; font-family: 'JetBrains Mono';">{calc_market_df['macd_diff'].iloc[-1]:.4f}</strong><br>
        • Bollinger %B: <strong style="color: #F0F4F8; font-family: 'JetBrains Mono';">{calc_market_df['bb_pct_b'].iloc[-1]:.3f}</strong><br>
        • Độ lệch SMA50: <strong style="color: #F0F4F8; font-family: 'JetBrains Mono';">{calc_market_df['dist_sma50'].iloc[-1]*100:+.2f}%</strong>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="github-card-ambient" style="margin-top: 18px;">
    <div style="font-size: 13.5px; font-weight: 700; color: #00E599; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.8px;">
        Luận giải định lượng về cấu trúc dự báo kép:
    </div>
    <div style="font-size: 13px; color: #C2CBD6; line-height: 1.6;">
        • <b>Khung 1 ngày (t+1):</b> Đạt độ chính xác khoảng 53% - 55% nhưng chỉ số ROC-AUC dưới 0.50. Biến động 24 giờ chịu chi phối mạnh bởi giao dịch nhiễu (microstructure noise) và tâm lý đám đông ngắn hạn.<br>
        • <b>Khung tuần 5 ngày (t+5):</b> Độ chính xác cải thiện đáng kể, đạt <b>61.52% (Logistic)</b> và <b>59.68% (Random Forest)</b>. Hiện tượng <i>Độ trễ thẩm thấu thông tin (Information Absorption Delay)</i> cho phép dòng tiền lớn của các định chế tài chính phản ánh đầy đủ các cú sốc tin tức vĩ mô vào xu thế giá.
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# PHÂN HỆ 2: LUỒNG TIN TỨC VĨ MÔ THỜI GIAN THỰC & GIẢI MÃ FINBERT
# =============================================================================
render_section_spine_header(
    section_id="section-2",
    badge_icon="🌐",
    badge_glow_color="rgba(0, 163, 255, 0.4)",
    headline="Phân hệ 2: Luồng Tin tức Vĩ mô & Chấm điểm FinBERT",
    subtitle="Giải mã cảm xúc ngôn ngữ tài chính tự động với mô hình ngôn ngữ lớn ProsusAI/finbert (110M tham số)."
)

# 1. BỘ CHỌN DUAL NEWS SOURCE (2 NGUỒN TIN LINH HOẠT)
news_source_mode = st.radio(
    "Chọn phương thức nạp dữ liệu tin tức:",
    options=[
        "Luồng tin tức trực tuyến Real-time (Live RSS Feeds)",
        "Kho dữ liệu kiểm định nghiên cứu (Dataset 2016 - 2026)"
    ],
    index=0,
    horizontal=True,
    help="Chuyển đổi linh hoạt giữa tin tức thị trường thực tế đang diễn ra hôm nay và tập dữ liệu nghiên cứu 10 năm của đồ án."
)

is_live_mode = "Real-time" in news_source_mode

if is_live_mode:
    col_live_ctrl, col_live_info = st.columns([0.35, 0.65])
    with col_live_ctrl:
        if st.button("Quét tin tức mới ngay (Refresh Live News)", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col_live_info:
        st.markdown(
            f"<div style='margin-top: 6px; font-size: 12px; color: {cfg.COLOR_TEXT_SECONDARY};'>"
            f"Nguồn cấp: <strong style='color: #00c087;'>Google News Financial, Yahoo Finance & CNBC Markets</strong> "
            f"(Tự động làm mới mỗi 10 phút)</div>",
            unsafe_allow_html=True
        )
    
    with st.spinner("Đang kết nối cổng RSS tài chính và quét tin tức thực tế hôm nay..."):
        news_items, is_fallback = fetch_live_news_cached(limit=25)
        if is_fallback:
            st.toast("Mất kết nối mạng RSS quốc tế, hệ thống tự động kích hoạt tập dữ liệu dự phòng an toàn.", icon="ℹ️")

else:
    col_filter1, col_filter2 = st.columns([2, 3])
    with col_filter1:
        time_preset = st.selectbox(
            "Chọn mốc thời gian quan sát lịch sử:",
            ["Phiên cuối tập dữ liệu (04/2026)", "3 ngày gần nhất", "7 ngày gần nhất", "30 ngày gần nhất", "Tùy chỉnh khoảng ngày"],
            index=0
        )
    with col_filter2:
        start_d, end_d = None, None
        if time_preset == "Tùy chỉnh khoảng ngày":
            date_range = st.date_input(
                "Khoảng ngày quan sát:",
                value=(date(2026, 4, 1), date(2026, 4, 15))
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_d, end_d = date_range[0], date_range[1]
        else:
            st.markdown(f"<div style='margin-top: 28px; font-size: 12px; color: {cfg.COLOR_TEXT_SECONDARY};'>"
                        f"Đang hiển thị luồng dữ liệu kiểm thử theo: <strong style='color: #ffffff;'>{time_preset}</strong></div>", 
                        unsafe_allow_html=True)
    
    news_items = fetch_filtered_news(time_preset, start_date=start_d, end_date=end_d)

st.markdown("---")

col_news_list, col_sent_pie = st.columns([0.65, 0.35])

with col_news_list:
    sub_title = "Luồng tin tức tài chính thực tế hôm nay (Live Feeds):" if is_live_mode else "Danh sách bản tin kinh tế vĩ mô lịch sử:"
    st.markdown(f"**{sub_title}**")
    
    headlines = [n["title"] for n in news_items]
    scored_news = score_news_batch(headlines)

    # Quản lý số lượng bản tin hiển thị qua session_state để tối ưu cuộn trang
    if "news_display_limit" not in st.session_state:
        st.session_state.news_display_limit = 5

    total_news = len(news_items)
    display_limit = st.session_state.news_display_limit

    # Chỉ hiển thị số lượng bản tin trong phạm vi giới hạn (mặc định 5 bản tin đầu)
    visible_pairs = list(zip(news_items, scored_news))[:display_limit]
    for idx, (news, score) in enumerate(visible_pairs):
        render_news_feed_card(news, score, is_open=(idx == 0))

    # Bộ điều khiển: Xem thêm (+5 tin), Xem toàn bộ (Xem full), và Thu gọn
    if total_news > 5:
        st.markdown(
            f"<div style='font-size: 12px; color: #8E9CAE; margin: 10px 0 8px 0;'>"
            f"Đang hiển thị <strong style='color: #00E599;'>{min(display_limit, total_news)}</strong> "
            f"trên tổng số <strong style='color: #F0F4F8;'>{total_news}</strong> bản tin thị trường</div>",
            unsafe_allow_html=True
        )

        col_more, col_full, col_collapse = st.columns([1, 1, 1])

        with col_more:
            if display_limit < total_news:
                if st.button("➕ Xem thêm (+5 tin)", key="btn_more_news", use_container_width=True):
                    st.session_state.news_display_limit = min(display_limit + 5, total_news)
                    st.rerun()

        with col_full:
            if display_limit < total_news:
                if st.button(f"📜 Xem toàn bộ ({total_news} tin)", key="btn_full_news", use_container_width=True):
                    st.session_state.news_display_limit = total_news
                    st.rerun()

        with col_collapse:
            if display_limit > 5:
                if st.button("⬆️ Thu gọn về 5 tin", key="btn_collapse_news", use_container_width=True):
                    st.session_state.news_display_limit = 5
                    st.rerun()

with col_sent_pie:
    st.markdown("**Tâm lý thị trường tổng hợp:**")
    if is_live_mode:
        p_pos_avg = float(np.mean([s["p_pos"] for s in scored_news])) if scored_news else 0.35
        p_neg_avg = float(np.mean([s["p_neg"] for s in scored_news])) if scored_news else 0.25
        p_neu_avg = float(np.mean([s["p_neu"] for s in scored_news])) if scored_news else 0.40
        net_avg = float(np.mean([s["net_score"] for s in scored_news])) if scored_news else 0.10
        active_sent_dict = {
            "p_pos": p_pos_avg,
            "p_neg": p_neg_avg,
            "p_neu": p_neu_avg,
            "sent_mean": net_avg,
            "news_vol": len(news_items)
        }
        donut_title = "CƠ CẤU CẢM XÚC THỊ TRƯỜNG THỰC TẾ"
        
        # Badge trạng thái tổng quan tâm lý thị trường chủ đạo
        render_dominant_sentiment_badge(p_pos_avg, p_neu_avg, p_neg_avg)

        fig_donut = plot_sentiment_donut_chart(active_sent_dict, title=donut_title)
        st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})

        st.markdown(f"""
        <div class="pneu-analysis-card">
            <div style="font-size: 13px; font-weight: 800; color: #00E599; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.8px;">
                GIÁM SÁT TÂM LÝ THỜI GIAN THỰC (LIVE MONITOR):
            </div>
            <div style="font-size: 12.5px; color: #F0F4F8; line-height: 1.6;">
                • Số bản tin hôm nay: <strong style="color: #F0F4F8; font-family: 'JetBrains Mono';">{len(news_items)} bản tin</strong><br>
                • Điểm cảm xúc ròng: <strong style="color: #00E599; font-family: 'JetBrains Mono';">{net_avg:+.3f}</strong><br>
                • Tâm lý chờ đợi chính sách P(Neu): <strong style="color: #F5A623; font-family: 'JetBrains Mono'; font-weight: 700;">{p_neu_avg*100:.1f}%</strong>
            </div>
            <div style="margin-top: 10px; font-size: 11.5px; color: #8E9CAE; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px; line-height: 1.5;">
                <strong style="color: #F5A623;">Ý nghĩa định lượng:</strong> Biến <code>p_neu</code> là đặc trưng quan trọng số 1 (Đồng thuận 2.25) phản ánh sự lưỡng lự của các định chế tài chính trước khi đón nhận thông tin vĩ mô.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        active_sent_dict = get_historical_sentiment_by_horizon(time_preset, start_date=start_d, end_date=end_d)
        donut_title = f"CƠ CẤU CẢM XÚC - {active_sent_dict['timeframe_label'].upper()}"
        
        # Badge trạng thái tổng quan tâm lý thị trường chủ đạo
        render_dominant_sentiment_badge(active_sent_dict["p_pos"], active_sent_dict["p_neu"], active_sent_dict["p_neg"])

        fig_donut = plot_sentiment_donut_chart(active_sent_dict, title=donut_title)
        st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})

        sent_val = active_sent_dict["sent_mean"]
        sent_color = cfg.COLOR_BULLISH if sent_val >= 0 else cfg.COLOR_BEARISH

        html_hist_card = f"""<div class="pneu-analysis-card">
<div style="font-size: 13px; font-weight: 800; color: #00B2FF; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.8px;">
GIÁM SÁT TÂM LÝ TẬP NGHIÊN CỨU ({active_sent_dict['timeframe_label']}):
</div>
<div style="font-size: 12.5px; color: #F0F4F8; line-height: 1.6;">
• Quy mô quan sát: <strong style="color: #F0F4F8; font-family: 'JetBrains Mono';">{active_sent_dict['n_days']} phiên</strong> ({active_sent_dict['news_vol']:,} bản tin)<br>
• Điểm cảm xúc ròng (sent_mean): <strong style="color: {sent_color}; font-family: 'JetBrains Mono'; font-weight: 700;">{sent_val:+.4f}</strong><br>
• Tích cực P(Pos): <strong style="color: {cfg.COLOR_BULLISH}; font-family: 'JetBrains Mono';">{active_sent_dict['p_pos']*100:.1f}%</strong> | Tiêu cực P(Neg): <strong style="color: {cfg.COLOR_BEARISH}; font-family: 'JetBrains Mono';">{active_sent_dict['p_neg']*100:.1f}%</strong><br>
• Chờ đợi chính sách P(Neu: Top 1): <strong style="color: #F5A623; font-family: 'JetBrains Mono'; font-weight: 700;">{active_sent_dict['p_neu']*100:.1f}%</strong>
</div>
<div style="margin-top: 10px; font-size: 11.5px; color: #8E9CAE; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px; line-height: 1.5;">
<strong style="color: #F5A623;">Cơ cấu động:</strong> Tỷ lệ <code>p_neu</code> biến thiên theo các chu kỳ vĩ mô 10 năm (2016 - 2026), giúp mô hình không bị thiên lệch trong các giai đoạn biến động lớn.
</div>
</div>"""
        st.markdown(html_hist_card, unsafe_allow_html=True)

# =============================================================================
# PHÂN HỆ 3: KIỂM ĐỊNH CHIẾN LƯỢC NGOÀI MẪU & ĐỐI ĐẦU TÀI CHÍNH
# =============================================================================
render_section_spine_header(
    section_id="section-3",
    badge_icon="🛡️",
    badge_glow_color="rgba(136, 65, 157, 0.4)",
    headline="Phân hệ 3: Kiểm định Chiến lược Ngoài mẫu & Đối đầu Tài chính",
    subtitle="Đối đầu thực nghiệm 434 phiên ngoài mẫu giữa mô hình lai tối ưu P >= 0.42 và chuẩn tham chiếu S&P 500."
)

st.markdown("##### 1. ĐỐI ĐẦU ĐƯỜNG CONG TĂNG TRƯỞNG VỐN TRÊN TẬP KIỂM THỬ NGOÀI MẪU (434 PHIÊN)")
fig_equity = plot_equity_curves()
st.plotly_chart(fig_equity, use_container_width=True, config={'displayModeBar': False})

# Bảng chỉ số tài chính Bảng 5 dạng Glass Table tương phản cao
st.markdown("##### 2. HIỆU NĂNG CÁC CHIẾN LƯỢC TRÊN TẬP KIỂM THỬ NGOÀI MẪU (BẢNG 5)")
df_b5 = pd.DataFrame([
    {"Chiến lược giao dịch": "Mua & Nắm giữ S&P 500 (Benchmark)", "Lợi nhuận tích lũy": "51.63%", "CAGR": "27.34%", "Biến động năm": "16.02%", "Hệ số Sharpe": "1.59", "Max Drawdown": "-15.54%", "Tỷ lệ thắng": "58.53%"},
    {"Chiến lược giao dịch": "Hồi quy Logistic 1D (0.50)", "Lợi nhuận tích lũy": "14.03%", "CAGR": "7.92%", "Biến động năm": "14.04%", "Hệ số Sharpe": "0.61", "Max Drawdown": "-14.26%", "Tỷ lệ thắng": "56.23%"},
    {"Chiến lược giao dịch": "Rừng ngẫu nhiên 1D (0.50)", "Lợi nhuận tích lũy": "24.49%", "CAGR": "13.57%", "Biến động năm": "14.66%", "Hệ số Sharpe": "0.94", "Max Drawdown": "-15.54%", "Tỷ lệ thắng": "57.23%"},
    {"Chiến lược giao dịch": "LightGBM 1D (0.50)", "Lợi nhuận tích lũy": "17.64%", "CAGR": "9.89%", "Biến động năm": "14.59%", "Hệ số Sharpe": "0.72", "Max Drawdown": "-15.64%", "Tỷ lệ thắng": "55.84%"},
    {"Chiến lược giao dịch": "XGBoost 1D (0.50)", "Lợi nhuận tích lũy": "29.93%", "CAGR": "16.42%", "Biến động năm": "14.94%", "Hệ số Sharpe": "1.09", "Max Drawdown": "-14.98%", "Tỷ lệ thắng": "57.40%"},
    {"Chiến lược giao dịch": "Rừng ngẫu nhiên 5D TỐI ƯU (P ≥ 0.42)", "Lợi nhuận tích lũy": "51.91%", "CAGR": "27.48%", "Biến động năm": "14.85%", "Hệ số Sharpe": "1.60", "Max Drawdown": "-15.54%", "Tỷ lệ thắng": "60.00%"}
])
render_glass_table(df_b5, highlight_keyword="TỐI ƯU")

col_ablation, col_feat = st.columns([0.48, 0.52])

with col_ablation:
    st.markdown("##### 3. THỰC NGHIỆM BÓC TÁCH ĐẶC TRƯNG (BẢNG 3)")
    df_b3 = pd.DataFrame([
        {"Kịch bản": "Chỉ kỹ thuật (11)", "Mô hình": "Random Forest", "Acc": "55.07%", "F1": "0.459", "ROC-AUC": "0.469"},
        {"Kịch bản": "Chỉ kỹ thuật (11)", "Mô hình": "XGBoost", "Acc": "54.38%", "F1": "0.454", "ROC-AUC": "0.480"},
        {"Kịch bản": "Chỉ cảm xúc (11)", "Mô hình": "XGBoost", "Acc": "55.07%", "F1": "0.483", "ROC-AUC": "0.449"},
        {"Kịch bản": "Chỉ cảm xúc (11)", "Mô hình": "Random Forest", "Acc": "51.84%", "F1": "0.485", "ROC-AUC": "0.472"},
        {"Kịch bản": "Mô hình lai (22)", "Mô hình": "XGBoost", "Acc": "53.00%", "F1": "0.458", "ROC-AUC": "0.452"},
        {"Kịch bản": "Mô hình lai (22)", "Mô hình": "Random Forest", "Acc": "51.84%", "F1": "0.470", "ROC-AUC": "0.463"}
    ])
    render_glass_table(df_b3, highlight_keyword="Mô hình lai")

with col_feat:
    st.markdown("##### 4. TẦM QUAN TRỌNG ĐẶC TRƯNG ĐỒNG THUẬN (BẢNG 4)")
    fig_feat = plot_feature_importance_consensus()
    st.plotly_chart(fig_feat, use_container_width=True, config={'displayModeBar': False})

st.markdown("---")
st.markdown("##### 5. TỐI ƯU HÓA NGƯỠNG TỰ TIN TRÊN KHUNG 5 NGÀY (BẢNG 7)")
df_b7 = pd.DataFrame([
    {"Ngưỡng xác suất P": "0.42 (TỐI ƯU)", "Số lệnh giao dịch": "15 lệnh", "Lợi nhuận tích lũy": "51.91%", "Hệ số Sharpe": "1.60", "Max Drawdown": "-15.54%"},
    {"Ngưỡng xác suất P": "0.45", "Số lệnh giao dịch": "25 lệnh", "Lợi nhuận tích lũy": "46.46%", "Hệ số Sharpe": "1.48", "Max Drawdown": "-15.54%"},
    {"Ngưỡng xác suất P": "0.48", "Số lệnh giao dịch": "41 lệnh", "Lợi nhuận tích lũy": "45.08%", "Hệ số Sharpe": "1.47", "Max Drawdown": "-15.54%"},
    {"Ngưỡng xác suất P": "0.50 (Gốc)", "Số lệnh giao dịch": "53 lệnh", "Lợi nhuận tích lũy": "39.99%", "Hệ số Sharpe": "1.36", "Max Drawdown": "-15.54%"},
    {"Ngưỡng xác suất P": "0.52", "Số lệnh giao dịch": "67 lệnh", "Lợi nhuận tích lũy": "37.20%", "Hệ số Sharpe": "1.33", "Max Drawdown": "-15.54%"}
])
render_glass_table(df_b7, highlight_keyword="TỐI ƯU")
st.markdown(
    "**Kết luận định lượng:** Khi hạ ngưỡng thoát vị thế xuống **0.42**, số lệnh giao dịch giảm **71.7%** (từ 53 xuống 15 lệnh). "
    "Việc loại bỏ hiện tượng đảo vị thế liên tục (Whipsaw) triệt tiêu phần lớn chi phí ma sát (Turnover Drag), "
    "giúp chiến lược chính thức vượt qua Buy & Hold S&P 500 cả về lợi nhuận (51.91% vs 51.63%) và Sharpe (1.60 vs 1.59)."
)

# =============================================================================
# PHÂN HỆ 4: KỊCH BẢN GIẢ ĐỊNH & THỬ NGHIỆM CÚ SỐC VĨ MÔ
# =============================================================================
render_section_spine_header(
    section_id="section-4",
    badge_icon="🔮",
    badge_glow_color="rgba(245, 166, 35, 0.4)",
    headline="Phân hệ 4: Kịch bản Giả định & Thử nghiệm Cú sốc Vĩ mô",
    subtitle="Hệ thống mô phỏng What-If Simulator phản ứng tức thì với các tin tức sự kiện đột biến và giải trình 3 kênh truyền dẫn AI."
)

sample_scenarios = [
    "Chọn một kịch bản mẫu...",
    "Federal Reserve unexpectedly cuts interest rates by 50 bps to stimulate economic momentum",
    "Middle East geopolitical conflict escalates sharply, oil benchmarks surge 8%",
    "US Core CPI inflation accelerates unexpectedly to 3.9%, renewing stagflation fears",
    "Tech sector reports blowout artificial intelligence cloud revenue, raising full-year guidance",
    "Major commercial bank discloses unexpected credit defaults, prompting emergency liquidity facility"
]

selected_sample = st.selectbox("Chọn kịch bản mẫu sẵn có:", options=sample_scenarios)
default_text = "" if selected_sample == sample_scenarios[0] else selected_sample

user_headline = st.text_area(
    "Nhập tiêu đề bản tin giả định (tiếng Anh):",
    value=default_text,
    placeholder="Ví dụ: Federal Reserve cuts rates by 50 bps amid economic slowdown...",
    height=90
)

col_btn, _ = st.columns([0.35, 0.65])
with col_btn:
    run_sim = st.button("Chạy mô phỏng cú sốc (Run Stress Test)", use_container_width=True)

if run_sim and user_headline.strip():
    with st.spinner("Đang chạy suy luận FinBERT và tính toán ma trận đặc trưng lai..."):
        whatif_output = predict_whatif_shift(latest_tech_row, latest_sent_dict, user_headline.strip())
        render_whatif_alert_card(whatif_output)

        st.markdown("---")
        st.markdown("#### Giải trình Cơ chế Tác động Vĩ mô & Đóng góp Đặc trưng (AI Decision Explainability)")
        fig_whatif_attr = plot_whatif_feature_attribution(whatif_output)
        st.plotly_chart(fig_whatif_attr, use_container_width=True, config={'displayModeBar': False})
        render_whatif_explainability_card(whatif_output)
elif run_sim:
    st.warning("Vui lòng nhập hoặc chọn một tiêu đề tin tức để tiến hành thử nghiệm.")

# Section 5: Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; font-size: 12px; color: {cfg.COLOR_TEXT_SECONDARY}; padding: 10px 0 20px 0;">
    <strong>{cfg.APP_TITLE}</strong> • {cfg.AUTHOR_INFO} • {cfg.SUPERVISOR}<br>
    Bản quyền nghiên cứu © 2026 Khoa Tài chính, Trường Đại học Kinh tế, Đại học Đà Nẵng.
</div>
""", unsafe_allow_html=True)

# Section 6: GitHub-Style Slow Cinematic Motion Observer
components.html(
    """
    <script>
    (function() {
        const parentDoc = window.parent.document;
        if (!parentDoc) return;
        
        const options = {
            root: null,
            rootMargin: "-10% 0px -18% 0px",
            threshold: 0.2
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('in-view');
                } else {
                    if (entry.boundingClientRect.top > window.innerHeight * 0.75) {
                        entry.target.classList.remove('in-view');
                    }
                }
            });
        }, options);

        const initAnimation = () => {
            const targets = parentDoc.querySelectorAll(
                '.github-bounce-badge, .github-section-header, .github-card-ambient'
            );
            targets.forEach(el => observer.observe(el));
        };

        setInterval(initAnimation, 600);
    })();
    </script>
    """,
    height=0,
    width=0
)
