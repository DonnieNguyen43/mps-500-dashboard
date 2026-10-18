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
    render_confidence_gauge_card,
    render_hero_intro,
    render_terminal_header, render_market_metric_cards,
    render_recommendation_banner, render_whatif_alert_card,
    render_color_coded_badges, render_whatif_explainability_card,
    render_news_feed_card, render_glass_table,
    render_dominant_sentiment_badge, get_dominant_sentiment_badge_html,
    render_section_spine_header, render_floating_nav_bar, get_section_icon_svg
)
from components.charts import (
    plot_sp500_candlestick_macd, plot_confidence_gauge,
    plot_feature_importance_consensus, plot_equity_curves,
    plot_sentiment_donut_chart, plot_whatif_feature_attribution,
    plot_real_equity_curve, plot_real_drawdown_curve,
    plot_real_dual_chart, plot_feature_importance_for_model
)
from modules.backtester import run_out_of_sample_backtest

# Section 1: Streamlit Configuration
st.set_page_config(
    page_title=cfg.APP_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Nạp Google Fonts (Inter & JetBrains Mono)
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700;800&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Nạp file CSS giao diện Bloomberg Dark Theme
css_path = os.path.join(os.path.dirname(__file__), "styles", "custom.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Section 2: Cấu hình tham số mặc định (Ẩn Sidebar theo yêu cầu UI)
threshold_input = cfg.THRESHOLD_OPTIMAL

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
            "title": "Mở vị thế mua / Tiếp tục nắm giữ (Long)",
            "badge": "Lạc quan (Long)",
            "color": cfg.COLOR_BULLISH,
            "bg_color": cfg.COLOR_BULLISH_BG,
            "rationale": f"Xác suất khung tuần đạt {forecast_result['prob_t5']*100:.1f}%, vượt ngưỡng người dùng chọn ({threshold_input*100:.1f}%)."
        }
    else:
        forecast_result["recommendation"] = {
            "action": "DEFENSIVE CASH",
            "title": "Hạ tỷ trọng / Nắm giữ tiền mặt (Defensive Cash)",
            "badge": "Cảnh báo rủi ro (Cash)",
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
# PHÂN HỆ 1: TÍN HIỆU THỊ TRƯỜNG VÀ ĐỘNG LỰC DỰ BÁO ĐA KHUNG
# =============================================================================
with st.container(key="quant_section_1"):
    render_section_spine_header(
        section_id="section-1",
        badge_icon=get_section_icon_svg("icon_sec1_pulse.svg"),
        badge_glow_color="rgba(0, 229, 153, 0.45)",
        headline="Phân hệ 1: Tín hiệu thị trường và động lực dự báo đa khung",
        subtitle="Động cơ học máy lượng tử kết hợp 11 chỉ báo kỹ thuật và dự báo xác suất kép (1D / 5D).",
        spine_class="spine-to-sec1"
    )

    col_chart, col_gauge = st.columns([0.68, 0.32])

    with col_chart:
        st.markdown("##### Diễn biến giá và xu hướng đường trung bình SMA")
        fig_candle = plot_sp500_candlestick_macd(calc_market_df)
        st.plotly_chart(fig_candle, use_container_width=True, config={'displayModeBar': False})

    with col_gauge:
        st.markdown("##### Đo lường xung lực và tham số kỹ thuật")
        fig_gauge = plot_confidence_gauge(forecast_result["prob_t5"], threshold=threshold_input)
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

        rsi_val = float(calc_market_df['rsi_14'].iloc[-1])
        macd_val = float(calc_market_df['macd_diff'].iloc[-1])
        bb_val = float(calc_market_df['bb_pct_b'].iloc[-1])
        dist_val = float(calc_market_df['dist_sma50'].iloc[-1] * 100)

        macd_color = cfg.COLOR_BULLISH if macd_val >= 0 else cfg.COLOR_BEARISH
        dist_color = cfg.COLOR_BULLISH if dist_val >= 0 else cfg.COLOR_BEARISH

        st.markdown(f"""
        <div class="quant-card" style="margin-top: 12px; padding: 16px 18px; border-radius: 14px; background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.08);">
            <div style="font-size: 11px; font-weight: 600; color: #94A3B8; letter-spacing: 0.6px; text-transform: uppercase; margin-bottom: 12px; font-family: 'Inter', sans-serif;">
                Thông số kỹ thuật phiên hiện hành
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 10px 14px;">
                    <div style="font-size: 11px; color: #64748B; margin-bottom: 3px; font-family: 'Inter', sans-serif;">RSI (14 phiên)</div>
                    <div style="font-size: 17px; font-weight: 700; font-family: 'Inter', sans-serif; font-variant-numeric: tabular-nums; color: #F8FAFC;">{rsi_val:.2f}</div>
                </div>
                <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 10px 14px;">
                    <div style="font-size: 11px; color: #64748B; margin-bottom: 3px; font-family: 'Inter', sans-serif;">MACD Histogram</div>
                    <div style="font-size: 17px; font-weight: 700; font-family: 'Inter', sans-serif; font-variant-numeric: tabular-nums; color: {'#10B981' if macd_val >= 0 else '#F43F5E'};">{macd_val:+.4f}</div>
                </div>
                <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 10px 14px;">
                    <div style="font-size: 11px; color: #64748B; margin-bottom: 3px; font-family: 'Inter', sans-serif;">Bollinger %B</div>
                    <div style="font-size: 17px; font-weight: 700; font-family: 'Inter', sans-serif; font-variant-numeric: tabular-nums; color: #F8FAFC;">{bb_val:.3f}</div>
                </div>
                <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 10px 14px;">
                    <div style="font-size: 11px; color: #64748B; margin-bottom: 3px; font-family: 'Inter', sans-serif;">Độ lệch SMA50</div>
                    <div style="font-size: 17px; font-weight: 700; font-family: 'Inter', sans-serif; font-variant-numeric: tabular-nums; color: {'#10B981' if dist_val >= 0 else '#F43F5E'};">{dist_val:+.2f}%</div>
                </div>
            </div>
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
# PHÂN HỆ 2: LUỒNG TIN TỨC VĨ MÔ VÀ GIẢI MÃ CẢM XÚC FINBERT
# =============================================================================
with st.container(key="quant_section_2"):
    render_section_spine_header(
        section_id="section-2",
        badge_icon=get_section_icon_svg("icon_sec2_nlp.svg"),
        badge_glow_color="rgba(0, 178, 255, 0.45)",
        headline="Phân hệ 2: Luồng tin tức vĩ mô và giải mã cảm xúc FinBERT",
        subtitle="Giải mã tâm lý tài chính thời gian thực qua mô hình ngôn ngữ lớn ProsusAI/finbert với 110 triệu tham số.",
        spine_class="spine-to-sec2"
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

    col_news_list, col_sent_pie = st.columns([0.56, 0.44], gap="medium")

    with col_news_list:
        sub_title = "Luồng tin tức tài chính thực tế hôm nay (Live Feeds):" if is_live_mode else "Danh sách bản tin kinh tế vĩ mô lịch sử:"
        st.markdown(f"<div style='min-height: 28px; margin-bottom: 8px; font-weight: 700; font-size: 14px; color: #F0F4F8;'>{sub_title}</div>", unsafe_allow_html=True)
        
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
            donut_title = "Cơ cấu cảm xúc thị trường thực tế"
            badge_html = get_dominant_sentiment_badge_html(p_pos_avg, p_neu_avg, p_neg_avg)

            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; min-height: 32px; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.08);">
                <span style="font-weight: 700; font-size: 14px; color: #F0F4F8; display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 14px; font-weight: 700; color: #F0F4F8;">Tâm lý thị trường tổng hợp:</span>
                </span>
                {badge_html}
            </div>
            """, unsafe_allow_html=True)

            fig_donut = plot_sentiment_donut_chart(active_sent_dict, title=donut_title)
            st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})

            st.markdown(f"""
            <div class="pneu-analysis-card">
                <div style="font-size: 13px; font-weight: 800; color: #00E599; margin-bottom: 8px; letter-spacing: 0.5px;">
                    Giám sát tâm lý thời gian thực (Live monitor):
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
            donut_title = f"Cơ cấu cảm xúc - {active_sent_dict['timeframe_label']}"
            badge_html = get_dominant_sentiment_badge_html(active_sent_dict["p_pos"], active_sent_dict["p_neu"], active_sent_dict["p_neg"])

            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; min-height: 32px; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.08);">
                <span style="font-weight: 700; font-size: 14px; color: #F0F4F8; display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 14px; font-weight: 700; color: #F0F4F8;">Tâm lý thị trường tổng hợp:</span>
                </span>
                {badge_html}
            </div>
            """, unsafe_allow_html=True)

            fig_donut = plot_sentiment_donut_chart(active_sent_dict, title=donut_title)
            st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})

            sent_val = active_sent_dict["sent_mean"]
            sent_color = cfg.COLOR_BULLISH if sent_val >= 0 else cfg.COLOR_BEARISH

            html_hist_card = f"""<div class="pneu-analysis-card">
<div style="font-size: 13px; font-weight: 800; color: #00B2FF; margin-bottom: 8px; letter-spacing: 0.5px;">
Giám sát tâm lý tập nghiên cứu ({active_sent_dict['timeframe_label']}):
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
# PHÂN HỆ 3: KIỂM ĐỊNH CHIẾN LƯỢC NGOÀI MẪU VÀ ĐỐI ĐẦU TÀI CHÍNH
# =============================================================================
with st.container(key="quant_section_3"):
    render_section_spine_header(
        section_id="section-3",
        badge_icon=get_section_icon_svg("icon_sec3_alpha.svg"),
        badge_glow_color="rgba(168, 85, 247, 0.45)",
        headline="Phân hệ 3: Kiểm định chiến lược ngoài mẫu & Đối đầu tài chính",
        subtitle="Động cơ kiểm định định lượng trên tập ngoài mẫu 20% (Test-set 434 phiên) và toàn bộ chuỗi lịch sử (2.169 phiên). Tùy chọn mô hình, khung T+1 & T+5 và tối ưu hóa ngưỡng xác suất P.",
        spine_class="spine-to-sec3"
    )

    # ── 1. BẢNG ĐIỀU KHIỂN THỰC NGHIỆM ĐỊNH LƯỢNG (INTERACTIVE QUANT CONTROLS) ──
    st.markdown("""
    <div style="background: rgba(16, 24, 40, 0.70); backdrop-filter: blur(16px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 18px 22px 14px 22px; margin-bottom: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 13.5px; font-weight: 700; color: #F0F4F8; letter-spacing: 0.8px; text-transform: uppercase;">Bảng điều khiển kiểm định thực nghiệm định lượng</span>
            </div>
            <span style="font-size: 11px; font-weight: 600; font-family: \'Inter\', sans-serif; color: #38BDF8; background: rgba(56, 189, 248, 0.10); padding: 4px 12px; border-radius: 20px; border: 1px solid rgba(56, 189, 248, 0.25); letter-spacing: 0.5px;">DATASET QUANT ENGINE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hàng 1: Chọn tập dữ liệu kiểm định & Chọn mô hình học máy
    col_ds, col_model = st.columns([0.54, 0.46])

    with col_ds:
        dataset_options = [
            "Tập kiểm thử ngoài mẫu (Test-set 20% | 434 phiên, 15/12/2023 - 07/04/2026)",
            "Toàn bộ chuỗi dữ liệu thực nghiệm (2016 - 2026 | 2,169 phiên)"
        ]
        selected_dataset_raw = st.selectbox(
            "Tập dữ liệu kiểm định:",
            options=dataset_options,
            index=0,
            key="sec3_select_dataset"
        )
        if "test" in selected_dataset_raw.lower() or "ngoài mẫu" in selected_dataset_raw.lower():
            dataset_code = "test_set_20"
            dataset_badge_text = "TEST-SET NGOÀI MẪU 20% (434 PHIÊN)"
        else:
            dataset_code = "full"
            dataset_badge_text = "TOÀN BỘ CHUỖI DỮ LIỆU (2,169 PHIÊN)"

    with col_model:
        model_options = [
            "Random Forest (Rừng ngẫu nhiên)",
            "XGBoost (Gradient Boosting)",
            "LightGBM",
            "Logistic Regression (Hồi quy Logistic)"
        ]
        selected_model_raw = st.selectbox(
            "Mô hình định lượng:",
            options=model_options,
            index=0,
            key="sec3_select_model"
        )
        if "random forest" in selected_model_raw.lower():
            model_clean = "Random Forest"
        elif "xgboost" in selected_model_raw.lower():
            model_clean = "XGBoost"
        elif "lightgbm" in selected_model_raw.lower():
            model_clean = "LightGBM"
        else:
            model_clean = "Logistic Regression"

    # Hàng 2: Khung thời gian, Ngưỡng xác suất P và Chế độ biểu đồ
    col_ctrl2, col_ctrl3, col_mode = st.columns([0.28, 0.32, 0.40])

    with col_ctrl2:
        horizon_options = ["Khung 5 ngày (T+5)", "Khung 1 ngày (T+1)"]
        selected_horizon_raw = st.radio(
            "Khung thời gian dự báo:",
            options=horizon_options,
            horizontal=True,
            index=0,
            key="sec3_select_horizon"
        )
        horizon_clean = "T+5" if "5" in selected_horizon_raw else "T+1"

    with col_ctrl3:
        # Dynamic default threshold
        default_p = 0.42 if horizon_clean == "T+5" else 0.50
        selected_threshold = st.slider(
            f"Ngưỡng xác suất P (Tối ưu: {default_p:.2f}):",
            min_value=0.38,
            max_value=0.58,
            value=default_p,
            step=0.01,
            format="%.2f",
            key=f"sec3_threshold_slider_{horizon_clean}_{dataset_code}"
        )

    with col_mode:
        chart_mode = st.radio(
            "Chế độ hiển thị biểu đồ:",
            options=[
                "Tăng trưởng vốn (Equity)",
                "Sụt giảm vốn (Drawdown)",
                "Song song (Dual View)"
            ],
            horizontal=True,
            index=0,
            key="sec3_chart_mode"
        )

    # Dynamic status info banner
    if dataset_code == "test_set_20":
        ds_desc = "Tập kiểm thử ngoài mẫu 20% (434 phiên out-of-sample) bảo đảm tính khách quan khoa học, triệt tiêu hoàn toàn rủi ro overfitting và rò rỉ dữ liệu (khớp 100% Báo cáo Bảng 5 & Bảng 7)."
    else:
        ds_desc = "Toàn bộ chuỗi dữ liệu thực nghiệm (2,169 phiên, 2016 - 2026) kiểm tra sức bền của chiến lược qua các chu kỳ kinh tế, cú sốc Covid-19 và chu kỳ thắt chặt tiền tệ."

    extra_note = ""
    if horizon_clean == "T+5" and abs(selected_threshold - 0.42) < 0.005:
        extra_note = " | <span style='color: #94A3B8;'>Ngưỡng P=0.42 trên T+5 triệt tiêu 71.7% whipsaw đảo vị thế, tối ưu chi phí ma sát.</span>"

    st.markdown(f"""<div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.08); border-left: 3px solid #38BDF8; border-radius: 8px; padding: 11px 16px; margin-bottom: 16px; font-size: 12.5px; color: #CBD5E1; display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;">
<div style="line-height: 1.5;">
<strong style="color: #F8FAFC;">Tập dữ liệu:</strong> {ds_desc}{extra_note}
</div>
<span style="font-size: 11px; font-weight: 600; font-family: 'Inter', sans-serif; color: #38BDF8; background: rgba(56, 189, 248, 0.10); padding: 3px 10px; border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.25); white-space: nowrap;">{dataset_badge_text}</span>
</div>""", unsafe_allow_html=True)

    # ── 2. THỰC THI KIỂM ĐỊNH THỰC TẾ (LIVE ENGINE RUN) ──
    with st.spinner(f"Đang tính toán động cơ kiểm định trên {dataset_badge_text}..."):
        backtest_res = run_out_of_sample_backtest(
            model_name=model_clean,
            horizon=horizon_clean,
            threshold=selected_threshold,
            fee_rate=cfg.TRADING_FEE,
            dataset_source=dataset_code
        )

    m_ai = backtest_res["metrics_ai"]
    m_bh = backtest_res["metrics_bh"]
    m_base = backtest_res["metrics_base"]
    trades_red = backtest_res["n_trades_reduction_pct"]
    alpha_cum = m_ai["cum_ret"] - m_bh["cum_ret"]

    # ── 3. KPI STATS CARDS (INSTITUTIONAL QUANT STANDARDS) ──
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    c_ret = "#10B981" if m_ai['cum_ret'] >= 0 else "#F43F5E"
    c_alpha = "#10B981" if alpha_cum >= 0 else "#F43F5E"
    c_sharpe = "#38BDF8" if m_ai['sharpe'] >= m_bh['sharpe'] else "#94A3B8"

    with col_kpi1:
        st.markdown(f"""
        <div class="kpi-mini-card">
            <div class="kpi-mini-label">Lợi nhuận tích lũy</div>
            <div class="kpi-mini-value" style="color: {c_ret};">{m_ai['cum_ret']:+.2f}%</div>
            <div class="kpi-mini-sub">S&P 500 Benchmark: <strong>{m_bh['cum_ret']:+.2f}%</strong></div>
            <div class="kpi-mini-alpha" style="color: {c_alpha};">Alpha vượt trội: {alpha_cum:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi2:
        st.markdown(f"""
        <div class="kpi-mini-card">
            <div class="kpi-mini-label">Hệ số Sharpe (Annualized)</div>
            <div class="kpi-mini-value" style="color: {c_sharpe};">{m_ai['sharpe']:.2f}</div>
            <div class="kpi-mini-sub">S&P 500: <strong>{m_bh['sharpe']:.2f}</strong> | Gốc P=0.50: <strong>{m_base['sharpe']:.2f}</strong></div>
            <div class="kpi-mini-alpha" style="color: #38BDF8;">CAGR mô hình: {m_ai['cagr']:+.2f}%/năm</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi3:
        st.markdown(f"""
        <div class="kpi-mini-card">
            <div class="kpi-mini-label">Mức sụt giảm tối đa (MDD)</div>
            <div class="kpi-mini-value" style="color: #F43F5E;">{m_ai['mdd']:.2f}%</div>
            <div class="kpi-mini-sub">S&P 500 Benchmark: <strong>{m_bh['mdd']:.2f}%</strong></div>
            <div class="kpi-mini-alpha" style="color: #94A3B8;">Biến động năm: {m_ai['vol']:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi4:
        st.markdown(f"""
        <div class="kpi-mini-card">
            <div class="kpi-mini-label">Số lệnh thực thi</div>
            <div class="kpi-mini-value" style="color: #F59E0B;">{m_ai['n_trades']} lệnh</div>
            <div class="kpi-mini-sub">Chiến lược gốc P=0.50: <strong>{m_base['n_trades']} lệnh</strong></div>
            <div class="kpi-mini-alpha" style="color: {'#10B981' if trades_red >= 0 else '#F43F5E'};">
                {f"Giảm {trades_red:.1f}% số lệnh (tiết kiệm phí)" if trades_red >= 0 else f"Tăng {abs(trades_red):.1f}% số lệnh"}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # ── 4. BIỂU ĐỒ ĐỐI ĐẦU ĐƯỜNG CONG TƯƠNG TÁC (CHARTS) ──
    st.markdown(f"##### 1. Đối đầu đường cong kiểm định: {model_clean} ({horizon_clean} | P ≥ {selected_threshold:.2f})")
    ds_exec_label = backtest_res['dataset_label']
    st.markdown(f"<div style='font-size: 12.5px; color: #94A3B8; margin-top: -4px; margin-bottom: 12px;'><strong style='color: #CBD5E1;'>Tập dữ liệu thực thi:</strong> {ds_exec_label}</div>", unsafe_allow_html=True)
    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
    
    if "tăng trưởng" in chart_mode.lower():
        fig_real = plot_real_equity_curve(backtest_res, model_clean, horizon_clean, selected_threshold)
    elif "sụt giảm" in chart_mode.lower():
        fig_real = plot_real_drawdown_curve(backtest_res, model_clean, horizon_clean, selected_threshold)
    else:
        fig_real = plot_real_dual_chart(backtest_res, model_clean, horizon_clean, selected_threshold)

    st.plotly_chart(fig_real, use_container_width=True, config={'displayModeBar': False})

    # ── 5. BẢNG CÁC CHỈ SỐ ĐỐI ĐẦU ĐỐI VỚI RIÊNG MÔ HÌNH ĐANG CHỌN ──
    st.markdown(f"##### 2. Bảng các chỉ số đối đầu tài chính cho mô hình {model_clean} ({horizon_clean})")

    highlight_tag = "TỐI ƯU" if (selected_threshold == 0.42 and horizon_clean == "T+5") else "HIỆN TẠI"
    df_single_model = pd.DataFrame([
        {
            "Chiến lược giao dịch": f"{model_clean} {horizon_clean} ({selected_threshold:.2f} - {highlight_tag})",
            "Lợi nhuận tích lũy": f"{m_ai['cum_ret']:+.2f}%",
            "CAGR": f"{m_ai['cagr']:+.2f}%",
            "Biến động năm": f"{m_ai['vol']:.2f}%",
            "Hệ số Sharpe": f"{m_ai['sharpe']:.2f}",
            "Max Drawdown": f"{m_ai['mdd']:.2f}%",
            "Tỷ lệ thắng": f"{m_ai['win_rate']:.2f}%",
            "Số lệnh": f"{m_ai['n_trades']} lệnh"
        },
        {
            "Chiến lược giao dịch": f"{model_clean} {horizon_clean} (Gốc P ≥ 0.50)",
            "Lợi nhuận tích lũy": f"{m_base['cum_ret']:+.2f}%",
            "CAGR": f"{m_base['cagr']:+.2f}%",
            "Biến động năm": f"{m_base['vol']:.2f}%",
            "Hệ số Sharpe": f"{m_base['sharpe']:.2f}",
            "Max Drawdown": f"{m_base['mdd']:.2f}%",
            "Tỷ lệ thắng": f"{m_base['win_rate']:.2f}%",
            "Số lệnh": f"{m_base['n_trades']} lệnh"
        },
        {
            "Chiến lược giao dịch": "Mua & Nắm giữ S&P 500 (Benchmark)",
            "Lợi nhuận tích lũy": f"{m_bh['cum_ret']:+.2f}%",
            "CAGR": f"{m_bh['cagr']:+.2f}%",
            "Biến động năm": f"{m_bh['vol']:.2f}%",
            "Hệ số Sharpe": f"{m_bh['sharpe']:.2f}",
            "Max Drawdown": f"{m_bh['mdd']:.2f}%",
            "Tỷ lệ thắng": f"{m_bh['win_rate']:.2f}%",
            "Số lệnh": "1 lệnh (Nắm giữ)"
        }
    ])
    render_glass_table(df_single_model, highlight_keyword=highlight_tag)

    # ── 6. TOP ĐẶC TRƯNG QUAN TRỌNG VÀ BÓC TÁCH ĐỘ NHẠY ──
    col_feat, col_sens = st.columns([0.52, 0.48])

    with col_feat:
        st.markdown(f"##### 3. Top đặc trưng quan trọng qua {model_clean} ({horizon_clean})")
        fig_feat_model = plot_feature_importance_for_model(backtest_res["df_feats"], model_clean, horizon_clean)
        st.plotly_chart(fig_feat_model, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("""
        <div style="font-size: 11.5px; color: #8E9CAE; line-height: 1.6; margin-top: -6px;">
            • <span style="color: #D97706; font-weight: 700;">■ Vàng ánh kim:</span> Đặc trưng đóng góp lớn nhất (Top 1 Model)<br>
            • <span style="color: #2563EB; font-weight: 700;">■ Xanh thép:</span> Chỉ báo kỹ thuật định lượng (Technical: Momentum / Trend / Volatility)<br>
            • <span style="color: #059669; font-weight: 700;">■ Xanh ngọc lục bảo:</span> Cảm xúc FinBERT & Xung lượng tâm lý vĩ mô
        </div>
        """, unsafe_allow_html=True)

    with col_sens:
        st.markdown(f"##### 4. Thử nghiệm độ nhạy ngưỡng P trên {model_clean} ({horizon_clean})")
        th_list = [0.42, 0.45, 0.48, 0.50, 0.52]
        sens_rows = []
        for th_val in th_list:
            th_res = run_out_of_sample_backtest(
                model_name=model_clean,
                horizon=horizon_clean,
                threshold=th_val,
                fee_rate=cfg.TRADING_FEE,
                dataset_source=dataset_code
            )
            m_th = th_res["metrics_ai"]
            is_cur = abs(th_val - selected_threshold) < 0.005
            is_opt = (th_val == 0.42 and horizon_clean == "T+5")
            label_th = f"{th_val:.2f} (ĐANG CHỌN)" if is_cur else (f"{th_val:.2f} (TỐI ƯU)" if is_opt else f"{th_val:.2f}")
            sens_rows.append({
                "Ngưỡng P": label_th,
                "Số lệnh": f"{m_th['n_trades']} lệnh",
                "Lợi nhuận": f"{m_th['cum_ret']:+.2f}%",
                "Sharpe": f"{m_th['sharpe']:.2f}",
                "Max DD": f"{m_th['mdd']:.2f}%"
            })
        df_sens = pd.DataFrame(sens_rows)
        render_glass_table(df_sens, highlight_keyword="ĐANG CHỌN" if any("ĐANG CHỌN" in r["Ngưỡng P"] for r in sens_rows) else "TỐI ƯU")
        
        st.markdown(f"""
        <div style="font-size: 11.5px; color: #8E9CAE; line-height: 1.5; margin-top: 8px;">
            <strong style="color: #00E599;">Cơ chế triệt tiêu Whipsaw:</strong> Khi hạ ngưỡng thoát vị thế xuống 0.42 trên khung 5 ngày, mô hình tránh được các đợt cắt lỗ non trong sóng điều chỉnh ngắn hạn, giảm mạnh chi phí ma sát và tăng hệ số Sharpe thực tế.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 7. EXPANDABLE DỮ LIỆU ĐỐI CHIẾU BÁO CÁO NGHIÊN CỨU GỐC (BẢNG 3, BẢNG 5, BẢNG 7) ──
    with st.expander("  📚 Dữ liệu đối chiếu báo cáo nghiên cứu gốc Nhóm 9 (Bảng 3, Bảng 5, Bảng 7)", expanded=False):
        st.markdown("###### Hiệu năng các chiến lược trên tập kiểm thử ngoài mẫu (Báo cáo Bảng 5)")
        df_b5 = pd.DataFrame([
            {"Chiến lược giao dịch": "Mua & Nắm giữ S&P 500 (Benchmark)", "Lợi nhuận tích lũy": "51.63%", "CAGR": "27.34%", "Biến động năm": "16.02%", "Hệ số Sharpe": "1.59", "Max Drawdown": "-15.54%", "Tỷ lệ thắng": "58.53%"},
            {"Chiến lược giao dịch": "Hồi quy Logistic 1D (0.50)", "Lợi nhuận tích lũy": "14.03%", "CAGR": "7.92%", "Biến động năm": "14.04%", "Hệ số Sharpe": "0.61", "Max Drawdown": "-14.26%", "Tỷ lệ thắng": "56.23%"},
            {"Chiến lược giao dịch": "Rừng ngẫu nhiên 1D (0.50)", "Lợi nhuận tích lũy": "24.49%", "CAGR": "13.57%", "Biến động năm": "14.66%", "Hệ số Sharpe": "0.94", "Max Drawdown": "-15.54%", "Tỷ lệ thắng": "57.23%"},
            {"Chiến lược giao dịch": "LightGBM 1D (0.50)", "Lợi nhuận tích lũy": "17.64%", "CAGR": "9.89%", "Biến động năm": "14.59%", "Hệ số Sharpe": "0.72", "Max Drawdown": "-15.64%", "Tỷ lệ thắng": "55.84%"},
            {"Chiến lược giao dịch": "XGBoost 1D (0.50)", "Lợi nhuận tích lũy": "29.93%", "CAGR": "16.42%", "Biến động năm": "14.94%", "Hệ số Sharpe": "1.09", "Max Drawdown": "-14.98%", "Tỷ lệ thắng": "57.40%"},
            {"Chiến lược giao dịch": "Rừng ngẫu nhiên 5D TỐI ƯU (P ≥ 0.42)", "Lợi nhuận tích lũy": "51.91%", "CAGR": "27.48%", "Biến động năm": "14.85%", "Hệ số Sharpe": "1.60", "Max Drawdown": "-15.54%", "Tỷ lệ thắng": "60.00%"}
        ])
        render_glass_table(df_b5, highlight_keyword="TỐI ƯU")

        col_b3, col_b7 = st.columns(2)
        with col_b3:
            st.markdown("###### Thực nghiệm bóc tách đặc trưng (Bảng 3)")
            df_b3 = pd.DataFrame([
                {"Kịch bản": "Chỉ kỹ thuật (11)", "Mô hình": "Random Forest", "Acc": "55.07%", "F1": "0.459", "ROC-AUC": "0.469"},
                {"Kịch bản": "Chỉ kỹ thuật (11)", "Mô hình": "XGBoost", "Acc": "54.38%", "F1": "0.454", "ROC-AUC": "0.480"},
                {"Kịch bản": "Chỉ cảm xúc (11)", "Mô hình": "XGBoost", "Acc": "55.07%", "F1": "0.483", "ROC-AUC": "0.449"},
                {"Kịch bản": "Chỉ cảm xúc (11)", "Mô hình": "Random Forest", "Acc": "51.84%", "F1": "0.485", "ROC-AUC": "0.472"},
                {"Kịch bản": "Mô hình lai (22)", "Mô hình": "XGBoost", "Acc": "53.00%", "F1": "0.458", "ROC-AUC": "0.452"},
                {"Kịch bản": "Mô hình lai (22)", "Mô hình": "Random Forest", "Acc": "51.84%", "F1": "0.470", "ROC-AUC": "0.463"}
            ])
            render_glass_table(df_b3, highlight_keyword="Mô hình lai")

        with col_b7:
            st.markdown("###### Tối ưu hóa ngưỡng tự tin trên khung 5 ngày (Bảng 7)")
            df_b7 = pd.DataFrame([
                {"Ngưỡng xác suất P": "0.42 (TỐI ƯU)", "Số lệnh giao dịch": "15 lệnh", "Lợi nhuận tích lũy": "51.91%", "Hệ số Sharpe": "1.60", "Max Drawdown": "-15.54%"},
                {"Ngưỡng xác suất P": "0.45", "Số lệnh giao dịch": "25 lệnh", "Lợi nhuận tích lũy": "46.46%", "Hệ số Sharpe": "1.48", "Max Drawdown": "-15.54%"},
                {"Ngưỡng xác suất P": "0.48", "Số lệnh giao dịch": "41 lệnh", "Lợi nhuận tích lũy": "45.08%", "Hệ số Sharpe": "1.47", "Max Drawdown": "-15.54%"},
                {"Ngưỡng xác suất P": "0.50 (Gốc)", "Số lệnh giao dịch": "53 lệnh", "Lợi nhuận tích lũy": "39.99%", "Hệ số Sharpe": "1.36", "Max Drawdown": "-15.54%"},
                {"Ngưỡng xác suất P": "0.52", "Số lệnh giao dịch": "67 lệnh", "Lợi nhuận tích lũy": "37.20%", "Hệ số Sharpe": "1.33", "Max Drawdown": "-15.54%"}
            ])
            render_glass_table(df_b7, highlight_keyword="TỐI ƯU")

# =============================================================================
# PHÂN HỆ 4: KỊCH BẢN GIẢ ĐỊNH VÀ THỬ NGHIỆM CÚ SỐC VĨ MÔ
# =============================================================================
with st.container(key="quant_section_4"):
    render_section_spine_header(
        section_id="section-4",
        badge_icon=get_section_icon_svg("icon_sec4_shock.svg"),
        badge_glow_color="rgba(245, 158, 11, 0.45)",
        headline="Phân hệ 4: Kịch bản giả định và thử nghiệm cú sốc vĩ mô",
        subtitle="Mô phỏng phản ứng nhạy cảm của xác suất thị trường trước các cú sốc tin tức bất ngờ qua 3 kênh truyền dẫn AI.",
        spine_class="spine-to-sec4"
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
            st.markdown("#### Giải trình cơ chế tác động vĩ mô và đóng góp đặc trưng (XAI)")
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

# Section 6: GitHub-Style Slow Cinematic Motion, 3D Sculptures & Full-Section Ambient Morphing Engine
components.html(
    """
    <script>
    (function() {
        const parentDoc = window.parent.document;
        if (!parentDoc) return;
        
        const themeColors = {
            'section-1': { r: 0, g: 229, b: 153 },    // Emerald Pulse
            'section-2': { r: 0, g: 178, b: 255 },    // Electric Cyan NLP
            'section-3': { r: 168, g: 85, b: 247 },   // Royal Violet Alpha
            'section-4': { r: 245, g: 158, b: 11 }    // Warm Amber Shock
        };

        const sectionMapping = [
            { id: 'section-1', tag: 'sec1', key: 'st-key-quant_section_1', c: themeColors['section-1'] },
            { id: 'section-2', tag: 'sec2', key: 'st-key-quant_section_2', c: themeColors['section-2'] },
            { id: 'section-3', tag: 'sec3', key: 'st-key-quant_section_3', c: themeColors['section-3'] },
            { id: 'section-4', tag: 'sec4', key: 'st-key-quant_section_4', c: themeColors['section-4'] }
        ];

        // 1. Ham tao 3 bien the dieu khac 3D (monolith, hexring, prism)
        function create3DSculptureVariant(variant, uid, cPrimary, cSecondary, cDark) {
            // Bien the 1: Khoi Khien M-Alpha 3D Duc Noi (Full Emblem Monolith)
            if (variant === 'monolith') {
                return `
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 260 260" width="100%" height="100%" style="overflow:visible;">
                  <defs>
                    <linearGradient id="ext_${uid}" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stop-color="${cSecondary}" stop-opacity="0.65"/>
                      <stop offset="100%" stop-color="#020408" stop-opacity="0.98"/>
                    </linearGradient>
                    <linearGradient id="face_${uid}" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stop-color="${cPrimary}" stop-opacity="0.45"/>
                      <stop offset="50%" stop-color="${cDark}" stop-opacity="0.9"/>
                      <stop offset="100%" stop-color="#050811" stop-opacity="0.96"/>
                    </linearGradient>
                    <linearGradient id="rim_${uid}" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.85"/>
                      <stop offset="40%" stop-color="${cPrimary}" stop-opacity="0.85"/>
                      <stop offset="100%" stop-color="${cSecondary}" stop-opacity="0.2"/>
                    </linearGradient>
                    <filter id="sh_${uid}" x="-30%" y="-30%" width="160%" height="160%">
                      <feDropShadow dx="12" dy="18" stdDeviation="10" flood-color="#000000" flood-opacity="0.8"/>
                    </filter>
                  </defs>
                  <polygon points="130,30 220,78 220,186 130,234 40,186 40,78" transform="translate(18, 22)" fill="url(#ext_${uid})"/>
                  <polygon points="130,30 220,78 220,186 130,234 40,186 40,78" fill="url(#face_${uid})" stroke="url(#rim_${uid})" stroke-width="5" filter="url(#sh_${uid})"/>
                  <polygon points="130,46 205,86 205,178 130,218 55,178 55,86" fill="none" stroke="${cPrimary}" stroke-width="1.5" stroke-opacity="0.4"/>
                  <g filter="url(#sh_${uid})">
                    <rect x="74" y="138" width="12" height="36" rx="4" fill="#F43F5E"/>
                    <rect x="92" y="112" width="12" height="62" rx="4" fill="#F59E0B"/>
                    <rect x="110" y="82" width="12" height="92" rx="4" fill="${cPrimary}"/>
                    <path d="M98,174 L98,78 C98,73 103,70 107,73 L130,102 L153,73 C157,70 162,73 162,78 L162,130 L148,130 L148,96 L134,114 C132,117 128,117 126,114 L112,96 L112,174 Z" fill="#FFFFFF"/>
                    <path d="M162,120 L186,88 L186,174 L172,174 L172,136 L158,154 C154,158 148,156 148,150 L148,134 L162,120 Z" fill="${cPrimary}"/>
                    <circle cx="186" cy="88" r="6" fill="#FFFFFF"/>
                  </g>
                </svg>`;
            }

            // Bien the 2: Khoi Vong Dai Lang Kinh 3D Rong Tam (3D Isometric Interlocking Hex-Link)
            if (variant === 'hexring') {
                return `
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 260 260" width="100%" height="100%" style="overflow:visible;">
                  <defs>
                    <linearGradient id="ringA_${uid}" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.9"/>
                      <stop offset="30%" stop-color="${cPrimary}" stop-opacity="0.85"/>
                      <stop offset="100%" stop-color="${cDark}" stop-opacity="0.95"/>
                    </linearGradient>
                    <linearGradient id="ringB_${uid}" x1="100%" y1="0%" x2="0%" y2="100%">
                      <stop offset="0%" stop-color="${cSecondary}" stop-opacity="0.85"/>
                      <stop offset="60%" stop-color="${cDark}" stop-opacity="0.95"/>
                      <stop offset="100%" stop-color="#03060C" stop-opacity="0.98"/>
                    </linearGradient>
                    <filter id="rsh_${uid}" x="-30%" y="-30%" width="160%" height="160%">
                      <feDropShadow dx="14" dy="20" stdDeviation="12" flood-color="#000000" flood-opacity="0.75"/>
                    </filter>
                  </defs>
                  <path d="M130,24 L42,74 L42,186 L115,228 L115,190 L74,166 L74,94 L130,62 Z"
                        transform="translate(12, 16)" fill="${cDark}" opacity="0.85"/>
                  <path d="M130,24 L42,74 L42,186 L115,228 L115,190 L74,166 L74,94 L130,62 Z"
                        fill="url(#ringA_${uid})" stroke="#FFFFFF" stroke-width="1.5" stroke-opacity="0.45" filter="url(#rsh_${uid})"/>
                  <path d="M145,32 L218,74 L218,186 L130,236 L130,198 L186,166 L186,94 L145,70 Z"
                        transform="translate(14, 18)" fill="#020408" opacity="0.9"/>
                  <path d="M145,32 L218,74 L218,186 L130,236 L130,198 L186,166 L186,94 L145,70 Z"
                        fill="url(#ringB_${uid})" stroke="${cPrimary}" stroke-width="2" stroke-opacity="0.6" filter="url(#rsh_${uid})"/>
                </svg>`;
            }

            // Bien the 3: Cum Tinh The Lang Kinh Alpha Tach Lop 3D (Deconstructed 3D Prism Cluster)
            return `
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 240" width="100%" height="100%" style="overflow:visible;">
              <defs>
                <linearGradient id="pr_${uid}" x1="0%" y1="100%" x2="100%" y2="0%">
                  <stop offset="0%" stop-color="${cDark}"/>
                  <stop offset="50%" stop-color="${cSecondary}"/>
                  <stop offset="100%" stop-color="${cPrimary}"/>
                </linearGradient>
                <filter id="prsh_${uid}" x="-30%" y="-30%" width="160%" height="160%">
                  <feDropShadow dx="10" dy="16" stdDeviation="10" flood-color="#000000" flood-opacity="0.75"/>
                </filter>
              </defs>
              <g transform="rotate(-22 120 120)" filter="url(#prsh_${uid})">
                <rect x="45" y="95" width="34" height="95" rx="10" transform="translate(10,12)" fill="#02050A" opacity="0.8"/>
                <rect x="45" y="95" width="34" height="95" rx="10" fill="url(#pr_${uid})" stroke="#FFFFFF" stroke-width="1.5" stroke-opacity="0.5"/>
                <rect x="100" y="55" width="38" height="135" rx="12" transform="translate(12,15)" fill="#02050A" opacity="0.85"/>
                <rect x="100" y="55" width="38" height="135" rx="12" fill="url(#pr_${uid})" stroke="${cPrimary}" stroke-width="2"/>
                <rect x="158" y="30" width="34" height="110" rx="10" transform="translate(10,12)" fill="#02050A" opacity="0.8"/>
                <rect x="158" y="30" width="34" height="110" rx="10" fill="url(#pr_${uid})" stroke="#FFFFFF" stroke-width="1.5" stroke-opacity="0.6"/>
              </g>
            </svg>`;
        }

        const stations3D = [
            {
                id: 'hero',
                selector: '#hero-intro',
                colors: { primary: '#00E599', secondary: '#00B2FF', dark: '#061C18' },
                items: [
                    {
                        variant: 'monolith',
                        side: 'right',
                        offset: '-35px',
                        top: '18px',
                        width: '320px',
                        transform: 'perspective(1000px) rotateX(14deg) rotateY(-24deg) rotateZ(10deg)',
                        parallax: 0.08,
                        blur: 0,
                        opacity: 0.86
                    },
                    {
                        variant: 'hexring',
                        side: 'left',
                        offset: '-45px',
                        top: '45px',
                        width: '270px',
                        transform: 'perspective(1000px) rotate3d(1, 1, 0, 22deg) rotateZ(-30deg)',
                        parallax: -0.06,
                        blur: 0,
                        opacity: 0.78
                    },
                    {
                        variant: 'prism',
                        side: 'left',
                        offset: '20px',
                        top: '360px',
                        width: '185px',
                        transform: 'perspective(1000px) rotateZ(35deg)',
                        parallax: 0.04,
                        blur: 5,
                        opacity: 0.52
                    },
                    {
                        variant: 'hexring',
                        side: 'right',
                        offset: '45px',
                        top: '320px',
                        width: '150px',
                        transform: 'perspective(1000px) rotateZ(-45deg)',
                        parallax: -0.05,
                        blur: 10,
                        opacity: 0.36
                    }
                ]
            },
            {
                id: 'sec1',
                selector: 'div[class*="st-key-quant_section_1"]',
                colors: { primary: '#00E599', secondary: '#00B2FF', dark: '#061C18' },
                items: [
                    {
                        variant: 'hexring',
                        side: 'left',
                        offset: '-55px',
                        top: '15px',
                        width: '310px',
                        transform: 'perspective(1000px) rotate3d(1, 1, 0, 28deg) rotateZ(-25deg)',
                        parallax: -0.07,
                        blur: 0,
                        opacity: 0.82
                    },
                    {
                        variant: 'monolith',
                        side: 'right',
                        offset: '-45px',
                        top: '85px',
                        width: '340px',
                        transform: 'perspective(1000px) rotateX(16deg) rotateY(-28deg) rotateZ(14deg)',
                        parallax: 0.08,
                        blur: 0,
                        opacity: 0.88
                    },
                    {
                        variant: 'prism',
                        side: 'left',
                        offset: '15px',
                        top: '480px',
                        width: '190px',
                        transform: 'perspective(1000px) rotateZ(35deg)',
                        parallax: 0.04,
                        blur: 5,
                        opacity: 0.55
                    },
                    {
                        variant: 'hexring',
                        side: 'right',
                        offset: '80px',
                        top: '560px',
                        width: '140px',
                        transform: 'perspective(1000px) rotateZ(-50deg)',
                        parallax: -0.05,
                        blur: 11,
                        opacity: 0.38
                    }
                ]
            },
            {
                id: 'sec2',
                selector: 'div[class*="st-key-quant_section_2"]',
                colors: { primary: '#00B2FF', secondary: '#38BDF8', dark: '#071626' },
                items: [
                    {
                        variant: 'monolith',
                        side: 'left',
                        offset: '-40px',
                        top: '30px',
                        width: '350px',
                        transform: 'perspective(1000px) rotateX(18deg) rotateY(30deg) rotateZ(-18deg)',
                        parallax: -0.07,
                        blur: 0,
                        opacity: 0.85
                    },
                    {
                        variant: 'hexring',
                        side: 'right',
                        offset: '-65px',
                        top: '40px',
                        width: '380px',
                        transform: 'perspective(1000px) rotateZ(42deg)',
                        parallax: 0.08,
                        blur: 0,
                        opacity: 0.8
                    },
                    {
                        variant: 'prism',
                        side: 'left',
                        offset: '-30px',
                        top: '460px',
                        width: '210px',
                        transform: 'perspective(1000px) rotateZ(-28deg)',
                        parallax: 0.05,
                        blur: 6,
                        opacity: 0.55
                    },
                    {
                        variant: 'monolith',
                        side: 'right',
                        offset: '10px',
                        top: '620px',
                        width: '165px',
                        transform: 'perspective(1000px) rotateZ(135deg)',
                        parallax: -0.05,
                        blur: 9,
                        opacity: 0.4
                    }
                ]
            },
            {
                id: 'sec3',
                selector: 'div[class*="st-key-quant_section_3"]',
                colors: { primary: '#A855F7', secondary: '#C084FC', dark: '#180C26' },
                items: [
                    {
                        variant: 'monolith',
                        side: 'right',
                        offset: '-50px',
                        top: '20px',
                        width: '360px',
                        transform: 'perspective(1000px) rotateX(-16deg) rotateY(-26deg) rotateZ(16deg)',
                        parallax: 0.08,
                        blur: 0,
                        opacity: 0.88
                    },
                    {
                        variant: 'hexring',
                        side: 'left',
                        offset: '-60px',
                        top: '90px',
                        width: '320px',
                        transform: 'perspective(1000px) rotateX(24deg) rotateY(22deg) rotateZ(-35deg)',
                        parallax: -0.07,
                        blur: 0,
                        opacity: 0.82
                    },
                    {
                        variant: 'prism',
                        side: 'right',
                        offset: '-45px',
                        top: '580px',
                        width: '240px',
                        transform: 'perspective(1000px) rotateZ(22deg)',
                        parallax: 0.05,
                        blur: 4,
                        opacity: 0.55
                    },
                    {
                        variant: 'hexring',
                        side: 'left',
                        offset: '-35px',
                        top: '920px',
                        width: '280px',
                        transform: 'perspective(1000px) rotateZ(65deg)',
                        parallax: -0.06,
                        blur: 6,
                        opacity: 0.45
                    }
                ]
            },
            {
                id: 'sec4',
                selector: 'div[class*="st-key-quant_section_4"]',
                colors: { primary: '#F59E0B', secondary: '#FBBF24', dark: '#201409' },
                items: [
                    {
                        variant: 'hexring',
                        side: 'left',
                        offset: '-45px',
                        top: '25px',
                        width: '340px',
                        transform: 'perspective(1000px) rotateZ(-32deg)',
                        parallax: -0.07,
                        blur: 0,
                        opacity: 0.82
                    },
                    {
                        variant: 'monolith',
                        side: 'right',
                        offset: '-45px',
                        top: '55px',
                        width: '330px',
                        transform: 'perspective(1000px) rotateX(15deg) rotateY(-25deg) rotateZ(-15deg)',
                        parallax: 0.08,
                        blur: 0,
                        opacity: 0.88
                    },
                    {
                        variant: 'monolith',
                        side: 'left',
                        offset: '-25px',
                        top: '440px',
                        width: '210px',
                        transform: 'perspective(1000px) rotateZ(40deg)',
                        parallax: 0.04,
                        blur: 7,
                        opacity: 0.48
                    },
                    {
                        variant: 'prism',
                        side: 'right',
                        offset: '25px',
                        top: '520px',
                        width: '170px',
                        transform: 'perspective(1000px) rotateZ(-45deg)',
                        parallax: -0.05,
                        blur: 10,
                        opacity: 0.38
                    }
                ]
            }
        ];

        // 2. Gan he thong da cum dieu khac 3D hai ben cho tung phan he
        const mount3DLogoDecorations = () => {
            stations3D.forEach(st => {
                const container = parentDoc.querySelector(st.selector);
                if (!container) return;
                
                const existing = container.querySelector('.mps-3d-stage');
                if (existing) {
                    if (existing.dataset.version === 'v3') return;
                    existing.remove();
                }

                container.style.position = 'relative';
                container.style.overflow = 'visible';

                const stage = parentDoc.createElement('div');
                stage.className = 'mps-3d-stage';
                stage.dataset.version = 'v3';

                let itemsHtml = '';
                st.items.forEach((item, idx) => {
                    const svg = create3DSculptureVariant(item.variant, `${st.id}_${idx}`, st.colors.primary, st.colors.secondary, st.colors.dark);
                    const filterStyle = item.blur > 0 
                        ? `filter: blur(${item.blur}px) drop-shadow(0 16px 28px rgba(0,0,0,0.75));` 
                        : 'filter: drop-shadow(0 24px 38px rgba(0, 0, 0, 0.85));';
                    
                    itemsHtml += `
                        <div class="mps-3d-sculpture mps-3d-item" data-side="${item.side}" data-parallax="${item.parallax}" data-base-transform="${item.transform}"
                             style="top: ${item.top}; ${item.side}: ${item.offset}; width: ${item.width}; height: ${item.width}; transform: ${item.transform}; ${filterStyle} opacity: ${item.opacity};">
                            ${svg}
                        </div>
                    `;
                });

                stage.innerHTML = itemsHtml;
                container.insertBefore(stage, container.firstChild);
            });
        };

        const update3DParallax = () => {
            const vh = parentDoc.defaultView ? parentDoc.defaultView.innerHeight : 800;
            const stages = parentDoc.querySelectorAll('.mps-3d-stage');
            stages.forEach(stage => {
                const rect = stage.getBoundingClientRect();
                if (rect.bottom < -350 || rect.top > vh + 350) return;
                const centerDiff = (rect.top + rect.height * 0.5) - vh * 0.5;

                const items = stage.querySelectorAll('.mps-3d-item');
                items.forEach(el => {
                    const factor = parseFloat(el.dataset.parallax || '0.05');
                    const baseTransform = el.dataset.baseTransform || '';
                    const py = centerDiff * factor * -1;
                    el.style.transform = `${baseTransform} translateY(${py.toFixed(1)}px)`;
                });
            });
        };

        const applyAmbientTheme = (targetId) => {
            const c = themeColors[targetId];
            if (c) {
                parentDoc.documentElement.style.setProperty('--ambient-r', c.r);
                parentDoc.documentElement.style.setProperty('--ambient-g', c.g);
                parentDoc.documentElement.style.setProperty('--ambient-b', c.b);
                parentDoc.body.setAttribute('data-active-section', targetId);
            }
        };

        const options = {
            root: null,
            rootMargin: "-5% 0px -20% 0px",
            threshold: 0.12
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('in-view');
                    const targetId = entry.target.id || (entry.target.closest && entry.target.closest('.github-spine-wrapper') ? entry.target.closest('.github-spine-wrapper').id : '');
                    if (targetId && targetId.startsWith('section-')) {
                        applyAmbientTheme(targetId);
                    }
                } else {
                    if (entry.boundingClientRect.top > window.innerHeight * 0.75) {
                        entry.target.classList.remove('in-view');
                    }
                }
            });
        }, options);

        // Continuous scroll watcher
        const onScroll = () => {
            let activeId = 'section-1';
            const vh = parentDoc.defaultView ? parentDoc.defaultView.innerHeight : 800;
            sectionMapping.forEach(sec => {
                const el = parentDoc.getElementById(sec.id);
                if (el) {
                    const rect = el.getBoundingClientRect();
                    if (rect.top <= vh * 0.65) {
                        activeId = sec.id;
                    }
                }
            });
            if (parentDoc.body.getAttribute('data-active-section') !== activeId) {
                applyAmbientTheme(activeId);
            }
            update3DParallax();
        };

        if (parentDoc.defaultView) {
            parentDoc.defaultView.addEventListener('scroll', onScroll, { passive: true });
        }

        // DOM Grouping Engine: Tags all containers and sibling elements per section
        const groupDomSections = () => {
            const targets = parentDoc.querySelectorAll(
                '.github-bounce-badge, .github-section-header, .github-card-ambient, .github-spine-wrapper, div[class*="st-key-quant_section_"]'
            );
            targets.forEach(el => observer.observe(el));

            sectionMapping.forEach((sec, idx) => {
                const directContainer = parentDoc.querySelector(`div[class*="${sec.key}"]`);
                if (directContainer && directContainer.getAttribute('data-mps-section') !== sec.tag) {
                    directContainer.setAttribute('data-mps-section', sec.tag);
                }

                const headerEl = parentDoc.getElementById(sec.id);
                if (!headerEl) return;

                let block = headerEl.closest('[data-testid="stVerticalBlock"] > div, .block-container > div');
                const nextSec = sectionMapping[idx + 1];

                while (block) {
                    if (nextSec && block.querySelector(`#${nextSec.id}`)) break;
                    if (!block.hasAttribute('data-mps-section')) {
                        block.setAttribute('data-mps-section', sec.tag);
                    }
                    block = block.nextElementSibling;
                }
            });

            mount3DLogoDecorations();
        };

        setInterval(groupDomSections, 500);
        groupDomSections();
        onScroll();
    })();
    </script>
    """,
    height=0,
    width=0
)
