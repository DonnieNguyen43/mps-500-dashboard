# -*- coding: utf-8 -*-
"""
components/ui_cards.py: Các thành phần giao diện thẻ Metric, Badge và Banner
Thiết kế chuẩn FinTech Luxury Dark Glassmorphism (Vercel / Linear / Bloomberg).
Đồng bộ thương hiệu MPS-500 và Lớp 50K33.
"""

import os
import pandas as pd
import streamlit as st
import config as cfg

def get_logo_svg_inline(size_px: int = 52) -> str:
    """Đọc file SVG logo độc bản và trả về chuỗi nhúng inline."""
    if os.path.exists(cfg.LOGO_PATH):
        try:
            with open(cfg.LOGO_PATH, "r", encoding="utf-8") as f:
                svg_content = f.read()
                svg_content = svg_content.replace('<svg ', f'<svg style="width: {size_px}px; height: {size_px}px; flex-shrink: 0;" ')
                return svg_content
        except Exception:
            pass
    return f'<div style="width: {size_px}px; height: {size_px}px; background: {cfg.COLOR_BULLISH}; border-radius: 12px; box-shadow: 0 0 16px rgba(0,229,153,0.3);"></div>'

def render_floating_nav_bar():
    """
    Hiển thị thanh Floating Anchor Navigation ở góc trên bên phải để nhảy nhanh giữa các phân khu.
    """
    nav_html = """<div class="floating-nav-bar">
<a href="#hero-intro" class="floating-nav-link">⚡ Mở đầu</a>
<a href="#section-1" class="floating-nav-link">1. Tín hiệu & Dự báo</a>
<a href="#section-2" class="floating-nav-link">2. Tin tức & FinBERT</a>
<a href="#section-3" class="floating-nav-link">3. Kiểm định Đối đầu</a>
<a href="#section-4" class="floating-nav-link">4. What-If Simulator</a>
</div>"""
    st.markdown(nav_html, unsafe_allow_html=True)

def render_hero_intro():
    """
    Hiển thị trang mở màn Hero Intro điện ảnh với icon MPS-500 zoom mở rộng,
    dòng chữ Multimodal Market Prediction System và thông tin đề tài nghiên cứu Lớp 50K33.
    Khi người dùng lướt xuống sẽ bắt đầu chính thức vào Terminal định lượng.
    """
    logo_svg = get_logo_svg_inline(120).replace("\n", " ").replace("\r", " ")
    html_code = f"""<div id="hero-intro" class="hero-intro-wrapper">
<div class="hero-logo-glow">
<div class="hero-logo-icon">{logo_svg}</div>
</div>
<div class="hero-title">MPS-500</div>
<div class="hero-subtitle">Multimodal Market Prediction System</div>
<div class="hero-attribution">{cfg.AUTHOR_INFO} | {cfg.SUPERVISOR}</div>
<a href="#main-terminal" class="hero-scroll-cue">
<span class="hero-scroll-label">Khám phá Terminal Định lượng</span>
<div class="hero-scroll-arrow">↓</div>
</a>
<div class="github-spine-line" style="height: 70px; margin-top: 24px;"></div>
</div>
<div id="main-terminal"></div>"""
    st.markdown(html_code, unsafe_allow_html=True)

def render_section_spine_header(section_id: str, badge_icon: str, badge_glow_color: str, headline: str, subtitle: str):
    """
    Hiển thị Trục ánh sáng dọc kết nối phân hệ, Huy hiệu 3D búng nảy chuyển động chậm (1.4s - 1.6s)
    và Tiêu đề phân hệ lướt nhẹ từ dưới lên theo phong cách GitHub Cinematic Motion.
    """
    html_code = f"""<div id="{section_id}" class="github-spine-wrapper">
<div class="github-spine-line"></div>
<div class="github-bounce-badge" style="--badge-glow-color: {badge_glow_color};">
<span>{badge_icon}</span>
</div>
<div class="github-section-header">
<h2 class="github-section-headline">{headline}</h2>
<p class="github-section-sub">{subtitle}</p>
</div>
</div>"""
    st.markdown(html_code, unsafe_allow_html=True)

def render_terminal_header():
    """
    Hiển thị tiêu đề Terminal với Logo Monogram M-Shield và thông tin Lớp 50K33.
    """
    logo_svg = get_logo_svg_inline(56)
    html_code = f"""<div class="terminal-header">
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
<div style="display: flex; align-items: center; gap: 18px;">
{logo_svg}
<div>
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 3px;">
<span class="live-dot"></span>
<span style="font-size: 11px; font-weight: 700; color: {cfg.COLOR_BULLISH}; letter-spacing: 1.2px; text-transform: uppercase;">
ĐỘNG CƠ ĐỊNH LƯỢNG TRỰC TUYẾN | S&P 500 (^GSPC)
</span>
<span style="font-size: 11px; color: {cfg.COLOR_TEXT_SECONDARY};">|</span>
<span style="font-size: 11px; color: {cfg.COLOR_TEXT_SECONDARY}; font-family: 'JetBrains Mono', monospace;">
CHỈ SỐ CHÍNH: ^GSPC
</span>
</div>
<h1 style="margin: 0 0 4px 0; font-size: 26px; font-weight: 900; color: #F0F4F8; letter-spacing: -0.4px;">
{cfg.APP_TITLE}
</h1>
<p style="margin: 0; font-size: 12px; color: {cfg.COLOR_TEXT_SECONDARY}; line-height: 1.4;">
{cfg.AUTHOR_INFO} | {cfg.SUPERVISOR}
</p>
</div>
</div>
<div style="text-align: right;">
<span class="pill-badge pill-badge-bullish" style="font-size: 11.5px; padding: 6px 16px;">
CHIẾN LƯỢC TỐI ƯU: P ≥ 0.42 (SHARPE 1.60)
</span>
</div>
</div>
</div>"""
    st.markdown(html_code, unsafe_allow_html=True)

def render_market_metric_cards(price_info: dict, forecast_info: dict, sent_info: dict):
    """
    Render 4 thẻ thông số quan trọng hàng đầu tại phần đầu trang theo phong cách Dark Glassmorphism.
    """
    curr_price = price_info.get("current_price", 5200.0)
    change_pts = price_info.get("change_pts", 0.0)
    change_pct = price_info.get("change_pct", 0.0)
    
    price_color = cfg.COLOR_BULLISH if change_pts >= 0 else cfg.COLOR_BEARISH
    price_sign = "+" if change_pts >= 0 else ""

    p_t1 = forecast_info.get("prob_t1", 0.52)
    p_t5 = forecast_info.get("prob_t5", 0.58)

    p_neu = float(sent_info.get('p_neu', sent_info.get('p_neu_mean', 0.40)))
    if p_neu <= 0.001:
        p_neu = 0.420

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""<div class="metric-card">
<div class="metric-label">CHỈ SỐ S&P 500 HIỆN HÀNH</div>
<div class="metric-value">{curr_price:,.2f}</div>
<div class="metric-delta" style="color: {price_color};">
{price_sign}{change_pts:.2f} ({price_sign}{change_pct:.2f}%)
</div>
<div class="metric-subtext">Cập nhật theo dữ liệu thị trường NYSE</div>
</div>""", unsafe_allow_html=True)

    with col2:
        color_t1 = cfg.COLOR_BULLISH if p_t1 >= 0.50 else cfg.COLOR_BEARISH
        st.markdown(f"""<div class="metric-card">
<div class="metric-label">XÁC SUẤT NGẮN HẠN (t+1)</div>
<div class="metric-value" style="color: {color_t1};">{p_t1*100:.1f}%</div>
<div class="metric-delta">
<span class="pill-badge pill-badge-neutral" style="font-size: 10.5px;">ĐỘ NHIỄU CAO (24H)</span>
</div>
<div class="metric-subtext">Mô hình XGBoost Lai 22 biến</div>
</div>""", unsafe_allow_html=True)

    with col3:
        is_long = p_t5 >= cfg.THRESHOLD_OPTIMAL
        color_t5 = cfg.COLOR_BULLISH if is_long else cfg.COLOR_BEARISH
        border_glow = f"border-left: 3px solid {color_t5}; box-shadow: 0 8px 32px rgba(0, 229, 153, 0.15);" if is_long else f"border-left: 3px solid {color_t5};"
        st.markdown(f"""<div class="metric-card" style="{border_glow}">
<div class="metric-label">XÁC SUẤT KHUNG TUẦN (t+5)</div>
<div class="metric-value" style="color: {color_t5};">{p_t5*100:.1f}%</div>
<div class="metric-delta" style="color: {color_t5}; font-weight: 700;">
Mục tiêu P ≥ 42.0% ({"+" if p_t5 >= 0.42 else ""}{(p_t5 - 0.42)*100:.1f}%)
</div>
<div class="metric-subtext">Trọng tâm Random Forest Lai (Bagging)</div>
</div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""<div class="metric-card">
<div class="metric-label">TÂM LÝ CHỜ ĐỢI VĨ MÔ (P_NEU)</div>
<div class="metric-value" style="color: {cfg.COLOR_NEUTRAL};">{p_neu*100:.1f}%</div>
<div class="metric-delta">
<span class="pill-badge pill-badge-cyan" style="font-size: 10.5px;">HẠNG 1 ĐỒNG THUẬN (2.25)</span>
</div>
<div class="metric-subtext">Đo lường sự lưỡng lự trước tin FOMC/CPI</div>
</div>""", unsafe_allow_html=True)

def render_recommendation_banner(forecast_info: dict):
    """
    Render biểu ngữ khuyến nghị định lượng hành động thành Thẻ Kính Hero Card khổng lồ.
    """
    rec = forecast_info["recommendation"]
    p_t5 = forecast_info["prob_t5"]
    is_long = rec["action"] == "LONG"
    
    badge_cls = "pill-badge-bullish" if is_long else "pill-badge-bearish"
    status_glow = "0 0 16px rgba(0, 229, 153, 0.3)" if is_long else "0 0 16px rgba(255, 77, 79, 0.3)"

    html_code = f"""<div class="rec-hero-card" style="border-color: {rec['color']};">
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;">
<div style="flex: 1; min-width: 320px;">
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
<span class="pill-badge {badge_cls}" style="font-size: 11px; padding: 4px 14px; box-shadow: {status_glow};">
{rec['badge']}
</span>
<span style="font-size: 12.5px; font-weight: 600; color: {cfg.COLOR_TEXT_SECONDARY}; text-transform: uppercase; letter-spacing: 0.8px;">
KHUYẾN NGHỊ ĐỊNH LƯỢNG CHO KHUNG TUẦN (5 PHIÊN)
</span>
</div>
<div style="font-size: 28px; font-weight: 900; color: #F0F4F8; margin-bottom: 10px; letter-spacing: -0.3px;">
{rec['title']}
</div>
<div style="font-size: 14px; line-height: 1.6; color: #C2CBD6;">
{rec['rationale']}
</div>
</div>
<div style="text-align: right; background: rgba(0,0,0,0.35); padding: 18px 26px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.08); box-shadow: inset 0 2px 8px rgba(0,0,0,0.3);">
<div style="font-size: 11px; text-transform: uppercase; color: {cfg.COLOR_TEXT_SECONDARY}; letter-spacing: 1px; font-weight: 700; margin-bottom: 2px;">
ĐỘ TỰ TIN KHUNG 5D
</div>
<div style="font-size: 36px; font-weight: 900; color: {rec['color']}; font-family: 'JetBrains Mono', monospace; line-height: 1.1;">
{p_t5*100:.1f}%
</div>
<div style="font-size: 11.5px; color: {cfg.COLOR_TEXT_SECONDARY}; margin-top: 4px;">
Mốc kích hoạt: <strong style="color: #F0F4F8;">42.0%</strong>
</div>
</div>
</div>
</div>"""
    st.markdown(html_code, unsafe_allow_html=True)

def render_color_coded_badges(p_pos: float, p_neu: float, p_neg: float, net_score: float) -> str:
    """
    Tạo chuỗi HTML cho bộ 3 Pill Badges bo tròn xác suất Pos, Neu, Neg cho từng bản tin.
    """
    return f"""<div style="display: flex; gap: 8px; margin: 10px 0 6px 0; font-size: 11.5px; font-weight: 600; flex-wrap: wrap; align-items: center;">
<span style="background: rgba(0, 229, 153, 0.12); color: #00E599; border: 1px solid rgba(0,229,153,0.25); border-radius: 9999px; padding: 4px 12px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase;">
● Tích cực (Pos): {p_pos*100:.1f}%
</span>
<span style="background: rgba(245, 166, 35, 0.12); color: #F5A623; border: 1px solid rgba(245,166,35,0.25); border-radius: 9999px; padding: 4px 12px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase;">
● Chờ đợi (Neu): {p_neu*100:.1f}%
</span>
<span style="background: rgba(255, 77, 79, 0.12); color: #FF4D4F; border: 1px solid rgba(255,77,79,0.25); border-radius: 9999px; padding: 4px 12px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase;">
● Tiêu cực (Neg): {p_neg*100:.1f}%
</span>
<span style="color: #8E9CAE; padding: 2px 8px; font-family: 'JetBrains Mono', monospace; font-size: 11px;">
Điểm ròng: <b style="color: #F0F4F8;">{net_score:+.3f}</b>
</span>
</div>"""

def render_news_feed_card(news: dict, score: dict, is_open: bool = False):
    """
    Hiển thị bản tin kinh tế vĩ mô dưới dạng Thẻ Kính Frosted Glass Container cao cấp.
    Thanh Tag đầu tin: Icon báo chí, Nguồn tin, Thời gian phát hành, Mức ảnh hưởng.
    Tiêu đề: Đậm nét, màu trắng ngà #F0F4F8.
    Bộ 3 Pill Badges: Pos (#00E599), Neu (#F5A623), Neg (#FF4D4F).
    Accordion mượt mà mở rộng xem đoạn trích dẫn và link bài gốc.
    """
    import re
    p_pos = score.get("p_pos", 0.35)
    p_neu = score.get("p_neu", 0.40)
    p_neg = score.get("p_neg", 0.25)
    net_val = score.get("net_score", 0.0)

    if net_val > 0.12:
        badge_border = "rgba(0, 229, 153, 0.35)"
        status_tag = '<span style="color: #00E599; font-weight: 700; font-size: 11px;">● TÍCH CỰC</span>'
    elif net_val < -0.12:
        badge_border = "rgba(255, 77, 79, 0.35)"
        status_tag = '<span style="color: #FF4D4F; font-weight: 700; font-size: 11px;">● TIÊU CỰC</span>'
    else:
        badge_border = "rgba(245, 166, 35, 0.35)"
        status_tag = '<span style="color: #F5A623; font-weight: 700; font-size: 11px;">● CHỜ ĐỢI</span>'

    raw_desc = str(news.get("description", "")).strip()
    clean_desc = re.sub(r"<[^>]+>", " ", raw_desc)
    clean_desc = re.sub(r"https?://\S+", "", clean_desc)
    clean_desc = re.sub(r"\s+", " ", clean_desc).strip()
    if not clean_desc or len(clean_desc) < 15 or clean_desc.lower() == "nan":
        clean_desc = f"Bản tin tài chính vĩ mô ghi nhận từ {news.get('source', 'Phố Wall')}: {news['title']}"

    source_name = news.get("source", "Tài chính Vĩ mô")
    time_str = news.get("time", "Hôm nay")
    tier = news.get("impact_tier", "MEDIUM")
    tier_color = "#FF4D4F" if tier == "HIGH" else ("#F5A623" if tier == "MEDIUM" else "#00B2FF")
    
    news_url = str(news.get("url", "")).strip()
    if news_url.startswith("http"):
        link_html = f'<a href="{news_url}" target="_blank" style="color: #00B2FF; text-decoration: none; font-weight: 600; font-size: 11.5px;">Xem bài viết gốc &rarr;</a>'
    else:
        link_html = f'<span style="color: #8E9CAE; font-size: 11px;">Nguồn lưu trữ kiểm định nội bộ</span>'

    badges_html = render_color_coded_badges(p_pos, p_neu, p_neg, net_val)

    open_attr = "open" if is_open else ""

    card_html = f"""<details class="news-glass-card" style="margin-bottom: 14px;" {open_attr}>
<summary style="cursor: pointer; list-style: none;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
<div style="display: flex; align-items: center; gap: 10px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px;">
{status_tag}
<span style="color: #8E9CAE;">|</span>
<span style="color: #F0F4F8; font-weight: 700;">📰 {source_name}</span>
<span style="color: #8E9CAE;">|</span>
<span style="color: #8E9CAE; font-family: 'JetBrains Mono', monospace;">⏰ {time_str}</span>
</div>
<div>
<span style="font-size: 10.5px; font-weight: 700; color: {tier_color}; background: rgba(255,255,255,0.05); padding: 3px 8px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.08);">
ẢNH HƯỞNG: {tier}
</span>
</div>
</div>
<div style="font-size: 14.5px; font-weight: 700; color: #F0F4F8; line-height: 1.45; margin-bottom: 4px;">
{news['title']}
</div>
{badges_html}
<div style="font-size: 11px; color: #8E9CAE; margin-top: 4px; text-align: right;">
<span>Nhấn để xem trích đoạn chi tiết ▾</span>
</div>
</summary>
<div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.08);">
<div style="background: rgba(0,0,0,0.3); border-left: 3px solid #00B2FF; padding: 12px 16px; border-radius: 8px; margin-bottom: 10px; font-size: 13px; line-height: 1.6; color: #C2CBD6;">
{clean_desc}
</div>
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 11.5px; color: #8E9CAE;">
<span>Chủ đề: <strong style="color: #F0F4F8;">{news.get("category", "Kinh tế Vĩ mô")}</strong></span>
{link_html}
</div>
</div>
</details>"""
    st.markdown(card_html, unsafe_allow_html=True)

def render_dominant_sentiment_badge(p_pos: float, p_neu: float, p_neg: float):
    """
    Hiển thị Status Pill trạng thái tâm lý thị trường chủ đạo trong ngày:
    - P_pos > 0.4: Thị trường Lạc quan (Bullish Bias)
    - P_neu >= 0.4: Trạng thái Thận trọng / Chờ đợi vĩ mô (Neutral Caution)
    - P_neg > 0.4: Áp lực Bán / Phòng thủ rủi ro (Bearish Defense)
    """
    if p_pos > 0.4:
        badge_text = "🟢 Thị trường Lạc quan (Bullish Bias)"
        bg = "rgba(16, 185, 129, 0.12)"
        color = "#10b981"
        border = "rgba(16, 185, 129, 0.35)"
    elif p_neu >= 0.4:
        badge_text = "🟡 Trạng thái Thận trọng / Chờ đợi vĩ mô (Neutral Caution)"
        bg = "rgba(245, 158, 11, 0.12)"
        color = "#f59e0b"
        border = "rgba(245, 158, 11, 0.35)"
    elif p_neg > 0.4:
        badge_text = "🔴 Áp lực Bán / Phòng thủ rủi ro (Bearish Defense)"
        bg = "rgba(244, 63, 94, 0.12)"
        color = "#f43f5e"
        border = "rgba(244, 63, 94, 0.35)"
    else:
        max_p = max(p_pos, p_neu, p_neg)
        if max_p == p_pos:
            badge_text = "🟢 Thị trường Lạc quan (Bullish Bias)"
            bg = "rgba(16, 185, 129, 0.12)"
            color = "#10b981"
            border = "rgba(16, 185, 129, 0.35)"
        elif max_p == p_neg:
            badge_text = "🔴 Áp lực Bán / Phòng thủ rủi ro (Bearish Defense)"
            bg = "rgba(244, 63, 94, 0.12)"
            color = "#f43f5e"
            border = "rgba(244, 63, 94, 0.35)"
        else:
            badge_text = "🟡 Trạng thái Thận trọng / Chờ đợi vĩ mô (Neutral Caution)"
            bg = "rgba(245, 158, 11, 0.12)"
            color = "#f59e0b"
            border = "rgba(245, 158, 11, 0.35)"

    badge_html = f"""<div style="text-align: center; margin: 4px 0 12px 0;">
<span style="background: {bg}; color: {color}; border: 1px solid {border}; border-radius: 9999px; padding: 6px 16px; font-size: 11.5px; font-weight: 700; letter-spacing: 0.4px; display: inline-block; box-shadow: 0 2px 10px rgba(0,0,0,0.2);">
{badge_text}
</span>
</div>"""
    st.markdown(badge_html, unsafe_allow_html=True)

def render_glass_table(df: pd.DataFrame, highlight_keyword: str = "TỐI ƯU"):
    """
    Render bảng dữ liệu tài chính theo phong cách Glassmorphism tương phản cao.
    Chuẩn hóa hệ thống Design Token & Semantic Highlighting đồng nhất:
    - Hàng Tối Ưu: dải viền xanh ngọc (border-left: 3px solid #10b981) + nền kính mờ nhẹ
    - Cột Lợi nhuận & CAGR: >= 50% xanh ngọc đậm #10b981 (bold), 10-49% #34d399, âm #f87171
    - Cột Sharpe: >= 1.50 #10b981 (bold), 1.0-1.49 #6ee7b7, < 1.0 #94a3b8
    - Cột Max Drawdown: Đỏ san hô dịu #fb7185
    - Cột Win Rate: >= 58% #34d399, còn lại #e2e8f0
    - Định dạng số: font-variant-numeric: tabular-nums, căn phải hoàn hảo
    """
    def _parse_num(v_str):
        clean = str(v_str).replace("%", "").replace("+", "").strip()
        try:
            return float(clean)
        except ValueError:
            return None

    headers_list = []
    for idx, col in enumerate(df.columns):
        align = "left" if idx == 0 else "right"
        headers_list.append(
            f'<th style="padding: 12px 14px; text-align: {align}; font-size: 11.5px; font-weight: 700; color: #8E9CAE; text-transform: uppercase; letter-spacing: 0.8px; border-bottom: 1px solid rgba(255,255,255,0.1);">{col}</th>'
        )
    headers = "".join(headers_list)
    
    rows_html = []
    for _, row in df.iterrows():
        is_highlight = any(highlight_keyword in str(val) for val in row.values)
        if is_highlight:
            row_style = 'background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10b981;'
        else:
            row_style = 'border-bottom: 1px solid rgba(255,255,255,0.04);'

        cells = []
        for idx, val in enumerate(row.values):
            col_name = str(df.columns[idx]).strip()
            val_str = str(val).strip()
            align = "left" if idx == 0 else "right"
            num_val = _parse_num(val_str)
            
            # Quy tắc màu sắc theo ngữ nghĩa định lượng
            if idx == 0:
                cell_color = "#10b981" if is_highlight else "#F0F4F8"
                font_weight = "700" if is_highlight else "500"
                font_fam = "Inter, sans-serif"
            elif any(k in col_name for k in ["Lợi nhuận", "CAGR"]):
                font_fam = "JetBrains Mono, monospace"
                if num_val is not None and num_val >= 50.0:
                    cell_color = "#10b981"
                    font_weight = "700"
                elif num_val is not None and num_val >= 10.0:
                    cell_color = "#34d399"
                    font_weight = "600"
                elif num_val is not None and num_val < 0.0:
                    cell_color = "#f87171"
                    font_weight = "600"
                else:
                    cell_color = "#6ee7b7"
                    font_weight = "500"
            elif "Sharpe" in col_name:
                font_fam = "JetBrains Mono, monospace"
                if num_val is not None and num_val >= 1.50:
                    cell_color = "#10b981"
                    font_weight = "700"
                elif num_val is not None and num_val >= 1.00:
                    cell_color = "#6ee7b7"
                    font_weight = "600"
                else:
                    cell_color = "#94a3b8"
                    font_weight = "500"
            elif any(k in col_name for k in ["Drawdown", "MDD"]):
                font_fam = "JetBrains Mono, monospace"
                cell_color = "#fb7185"
                font_weight = "600"
            elif "thắng" in col_name or "Win" in col_name:
                font_fam = "JetBrains Mono, monospace"
                if num_val is not None and num_val >= 58.0:
                    cell_color = "#34d399"
                    font_weight = "600"
                else:
                    cell_color = "#e2e8f0"
                    font_weight = "500"
            elif any(k in col_name for k in ["Acc", "F1", "ROC-AUC"]):
                font_fam = "JetBrains Mono, monospace"
                font_weight = "600"
                if num_val is not None and num_val >= 54.0:
                    cell_color = "#34d399"
                elif num_val is not None and num_val >= 0.48:
                    cell_color = "#6ee7b7"
                else:
                    cell_color = "#e2e8f0"
            else:
                font_fam = "JetBrains Mono, monospace" if any(c.isdigit() for c in val_str) else "Inter, sans-serif"
                cell_color = "#10b981" if is_highlight and idx < 2 else "#F0F4F8"
                font_weight = "700" if is_highlight and idx < 2 else "500"

            cells.append(
                f'<td style="padding: 12px 14px; font-size: 12.5px; text-align: {align}; color: {cell_color}; font-weight: {font_weight}; font-family: {font_fam}; font-variant-numeric: tabular-nums;">{val_str}</td>'
            )
        
        rows_html.append(f'<tr style="{row_style}">{"".join(cells)}</tr>')

    table_html = f"""<div class="glass-table-container" style="background: rgba(16, 24, 40, 0.65); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; overflow: hidden; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35); margin-bottom: 16px;">
<table style="width: 100%; border-collapse: collapse;">
<thead>
<tr style="background: rgba(255, 255, 255, 0.03);">
{headers}
</tr>
</thead>
<tbody>
{"".join(rows_html)}
</tbody>
</table>
</div>"""
    st.markdown(table_html, unsafe_allow_html=True)

def render_whatif_alert_card(whatif_res: dict):
    """
    Hiển thị thẻ so sánh kết quả kịch bản giả định bằng Thẻ Kính Frosted Glass Container.
    """
    delta_p = whatif_res["delta_p"]
    delta_sign = "+" if delta_p >= 0 else ""
    delta_color = cfg.COLOR_BULLISH if delta_p >= 0 else cfg.COLOR_BEARISH
    
    state_changed = whatif_res["state_changed"]
    status_msg = "CẢNH BÁO ĐẢO CHIỀU DANH MỤC!" if state_changed else "TRẠNG THÁI DANH MỤC KHÔNG ĐỔI"
    status_pill = "pill-badge-bearish" if state_changed else "pill-badge-cyan"

    s_res = whatif_res["sentiment_result"]

    html_card = f"""<div style="background: rgba(16, 24, 40, 0.65); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1px solid {delta_color}; border-radius: 14px; padding: 20px 24px; margin: 18px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap; gap: 10px;">
<div style="font-size: 16px; font-weight: 800; color: #F0F4F8; letter-spacing: -0.2px;">
KẾT QUẢ ĐÁNH GIÁ CÚ SỐC VĨ MÔ GIẢ ĐỊNH
</div>
<span class="pill-badge {status_pill}">
{status_msg}
</span>
</div>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin-bottom: 14px;">
<div style="background: rgba(0,0,0,0.3); padding: 12px 14px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06);">
<div style="font-size: 11px; color: {cfg.COLOR_TEXT_SECONDARY}; text-transform: uppercase; font-weight: 700; margin-bottom: 3px;">CẢM XÚC TIN FINBERT</div>
<div style="font-size: 17px; font-weight: 800; color: #F0F4F8;">{s_res['label']}</div>
<div style="font-size: 11.5px; color: {cfg.COLOR_TEXT_SECONDARY}; font-family: 'JetBrains Mono', monospace;">Net Score: {s_res['net_score']:+.3f}</div>
</div>
<div style="background: rgba(0,0,0,0.3); padding: 12px 14px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06);">
<div style="font-size: 11px; color: {cfg.COLOR_TEXT_SECONDARY}; text-transform: uppercase; font-weight: 700; margin-bottom: 3px;">XÁC SUẤT GỐC (5D)</div>
<div style="font-size: 17px; font-weight: 800; color: #F0F4F8; font-family: 'JetBrains Mono', monospace;">{whatif_res['base_prob_5d']*100:.1f}%</div>
<div style="font-size: 11.5px; color: {cfg.COLOR_TEXT_SECONDARY};">Vị thế: {whatif_res['base_recommendation']['badge']}</div>
</div>
<div style="background: rgba(0,0,0,0.3); padding: 12px 14px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06);">
<div style="font-size: 11px; color: {cfg.COLOR_TEXT_SECONDARY}; text-transform: uppercase; font-weight: 700; margin-bottom: 3px;">XÁC SUẤT SAU CÚ SỐC (5D)</div>
<div style="font-size: 17px; font-weight: 800; color: {delta_color}; font-family: 'JetBrains Mono', monospace;">{whatif_res['whatif_prob_5d']*100:.1f}%</div>
<div style="font-size: 11.5px; color: {delta_color}; font-weight: 700; font-family: 'JetBrains Mono', monospace;">Dịch chuyển: {delta_sign}{delta_p*100:.2f}%</div>
</div>
</div>
<div style="background: rgba(0,0,0,0.4); padding: 14px 18px; border-radius: 10px; font-size: 13.5px; line-height: 1.5; color: #C2CBD6; border: 1px solid rgba(255,255,255,0.06);">
<strong style="color: #F0F4F8;">Hành động khuyến nghị:</strong> {whatif_res['whatif_recommendation']['title']}: {whatif_res['whatif_recommendation']['rationale']}
</div>
</div>"""
    st.markdown(html_card, unsafe_allow_html=True)

def render_whatif_explainability_card(whatif_res: dict):
    """
    Card giải trình chuyên sâu qua 3 Kênh Truyền Dẫn Vĩ Mô (Explainability Engine):
    1. Kênh tâm lý rủi ro (Risk-On / Risk-Off)
    2. Kênh bất định vĩ mô (p_neu Uncertainty Anchor)
    3. Kênh kỹ thuật & Động lượng (SMA50 & Momentum Buffer)
    """
    s_res = whatif_res["sentiment_result"]
    net_val = s_res["net_score"]
    delta_p = whatif_res["delta_p"]
    factors = whatif_res.get("factors", {})

    if net_val > 0.15:
        narrative_risk = (
            f"<b>1. Kênh tâm lý rủi ro (Risk-On Transmission):</b> Bản tin mang xung lực tăng điểm tích cực "
            f"(Net Score: <code>{net_val:+.3f}</code>), giúp nâng cao kỳ vọng lợi nhuận biên và thu hẹp phần bù rủi ro vốn chủ sở hữu (Equity Risk Premium). "
            f"Đóng góp trực tiếp từ điểm cảm xúc FinBERT đạt <code>{factors.get('c_sent', 0)*100:+.2f}%</code>."
        )
        narrative_neu = (
            f"<b>2. Kênh bất định vĩ mô (p_neu Uncertainty Absorption):</b> Áp dụng cơ chế <i>Marginal News Shock</i>, "
            f"cú sốc hấp thụ nhẹ vào mỏ neo <code>p_neu</code> hiện hành mà không gây méo mó cấu trúc dữ liệu ngoài phân phối (OOD). "
            f"Tỷ trọng tâm lý tích cực gia tăng đóng góp <code>{factors.get('c_state', 0)*100:+.2f}%</code> vào sự tự tin của mô hình."
        )
        narrative_tech = (
            f"<b>3. Kênh kỹ thuật & Động lượng (Momentum Buffer):</b> Sự cộng hưởng giữa tín hiệu kỹ thuật SMA/RSI và cú sốc vĩ mô "
            f"giúp củng cố xác suất tăng giá thêm <b>{delta_p*100:+.2f}%</b>, bảo toàn vị thế Long an toàn."
        )
        accent_color = cfg.COLOR_BULLISH
    elif net_val < -0.15:
        narrative_risk = (
            f"<b>1. Kênh tâm lý rủi ro (Risk-Off Transmission):</b> Cú sốc tiêu cực (Net Score: <code>{net_val:+.3f}</code>) "
            f"kích hoạt trạng thái bán tháo phòng vệ, làm gia tăng lo ngại lạm phát/suy thoái và đè nặng lên định giá P/E rổ cổ phiếu S&P 500. "
            f"Đóng góp từ điểm cảm xúc FinBERT làm suy giảm <code>{factors.get('c_sent', 0)*100:+.2f}%</code>."
        )
        narrative_neu = (
            f"<b>2. Kênh bất định vĩ mô (p_neu Uncertainty Surge):</b> Sự phân cực tiêu cực kéo tụt mỏ neo chờ đợi <code>p_neu</code>, "
            f"thay thế bằng áp lực e ngại rủi ro P_Neg. Đóng góp từ trạng thái phân phối kéo giảm xác suất <code>{factors.get('c_state', 0)*100:+.2f}%</code>."
        )
        narrative_tech = (
            f"<b>3. Kênh kỹ thuật & Động lượng (Momentum Erosion):</b> Xác suất tăng giá khung tuần bị kéo giảm "
            f"<b>{delta_p*100:+.2f}%</b>, cảnh báo nguy cơ gãy các mốc hỗ trợ SMA20/SMA50. Khuyến nghị kích hoạt lệnh Trailing Stop hoặc chuyển tiền mặt."
        )
        accent_color = cfg.COLOR_BEARISH
    else:
        narrative_risk = (
            f"<b>1. Kênh tâm lý rủi ro (Neutral Await):</b> Bản tin không tạo phân cực đột biến (Net Score: <code>{net_val:+.3f}</code>), "
            f"thị trường duy trì trạng thái trung lập chờ dữ liệu chính sách từ FOMC hoặc CPI."
        )
        narrative_neu = (
            f"<b>2. Kênh bất định vĩ mô (p_neu Top 1 Feature):</b> Biến <code>p_neu</code> duy trì tỷ trọng áp đảo (<code>{s_res['p_neu']*100:.1f}%</code>), "
            f"đóng vai trò mỏ neo hấp thụ rung lắc ngắn hạn và hạn chế các quyết định tái cơ cấu danh mục vội vã."
        )
        narrative_tech = (
            f"<b>3. Kênh kỹ thuật & Động lượng (Stable Momentum):</b> Biên độ dịch chuyển xác suất chỉ dao động nhẹ <b>{delta_p*100:+.2f}%</b>, "
            f"hệ thống giữ nguyên chiến lược hiện hành."
        )
        accent_color = cfg.COLOR_NEUTRAL

    html_narrative = f"""<div style="background: rgba(16, 24, 40, 0.65); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1px solid rgba(255, 255, 255, 0.08); border-left: 4px solid {accent_color}; border-radius: 14px; padding: 18px 22px; margin: 16px 0; font-size: 13.5px; line-height: 1.6; color: #C2CBD6; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);">
<div style="font-size: 13px; font-weight: 800; color: {accent_color}; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.8px;">
GIẢI TRÌNH 3 KÊNH TRUYỀN DẪN VĨ MÔ ĐỊNH LƯỢNG (EXPLAINABILITY ENGINE)
</div>
<div style="margin-bottom: 8px;">{narrative_risk}</div>
<div style="margin-bottom: 8px;">{narrative_neu}</div>
<div>{narrative_tech}</div>
</div>"""
    st.markdown(html_narrative, unsafe_allow_html=True)

    # Bảng biến thiên 9 đặc trưng vĩ mô
    feat_tbl = whatif_res.get("feature_table", {})
    if feat_tbl:
        rows_data = [
            {"Tên đặc trưng vĩ mô": "Điểm cảm xúc trung bình (sent_mean)", "Phiên hiện hành": f"{feat_tbl['sent_mean'][0]:+.4f}", "Sau cú sốc biên": f"{feat_tbl['sent_mean'][1]:+.4f}", "Biến thiên (Δ)": f"{feat_tbl['sent_mean'][2]:+.4f}", "Kênh tác động": "Kênh tâm lý rủi ro"},
            {"Tên đặc trưng vĩ mô": "Xác suất tích cực (p_pos)", "Phiên hiện hành": f"{feat_tbl['p_pos'][0]*100:.1f}%", "Sau cú sốc biên": f"{feat_tbl['p_pos'][1]*100:.1f}%", "Biến thiên (Δ)": f"{(feat_tbl['p_pos'][1]-feat_tbl['p_pos'][0])*100:+.2f}%", "Kênh tác động": "Kỳ vọng dòng tiền mua"},
            {"Tên đặc trưng vĩ mô": "Xác suất tiêu cực (p_neg)", "Phiên hiện hành": f"{feat_tbl['p_neg'][0]*100:.1f}%", "Sau cú sốc biên": f"{feat_tbl['p_neg'][1]*100:.1f}%", "Biến thiên (Δ)": f"{(feat_tbl['p_neg'][1]-feat_tbl['p_neg'][0])*100:+.2f}%", "Kênh tác động": "Áp lực tâm lý phòng thủ"},
            {"Tên đặc trưng vĩ mô": "Xác suất chờ đợi (p_neu: Top 1)", "Phiên hiện hành": f"{feat_tbl['p_neu'][0]*100:.1f}%", "Sau cú sốc biên": f"{feat_tbl['p_neu'][1]*100:.1f}%", "Biến thiên (Δ)": f"{(feat_tbl['p_neu'][1]-feat_tbl['p_neu'][0])*100:+.2f}%", "Kênh tác động": "Mỏ neo bất định vĩ mô"},
            {"Tên đặc trưng vĩ mô": "Xung lượng cảm xúc 1 ngày (sent_momentum_1d)", "Phiên hiện hành": f"{feat_tbl['sent_momentum_1d'][0]:+.4f}", "Sau cú sốc biên": f"{feat_tbl['sent_momentum_1d'][1]:+.4f}", "Biến thiên (Δ)": f"{feat_tbl['sent_momentum_1d'][2]:+.4f}", "Kênh tác động": "Gia tốc tâm lý tức thì"},
            {"Tên đặc trưng vĩ mô": "Khối lượng tin tức toàn thị trường (news_vol)", "Phiên hiện hành": f"{feat_tbl['news_vol'][0]:.0f} bản tin", "Sau cú sốc biên": f"{feat_tbl['news_vol'][1]:.0f} bản tin", "Biến thiên (Δ)": "+1 bản tin", "Kênh tác động": "Cơ chế Marginal Shock"},
            {"Tên đặc trưng vĩ mô": "Tỷ lệ khối lượng tin 5 ngày (news_vol_ratio_5d)", "Phiên hiện hành": f"{feat_tbl['news_vol_ratio_5d'][0]:.3f}", "Sau cú sốc biên": f"{feat_tbl['news_vol_ratio_5d'][1]:.3f}", "Biến thiên (Δ)": f"{(feat_tbl['news_vol_ratio_5d'][1]-feat_tbl['news_vol_ratio_5d'][0]):+.3f}", "Kênh tác động": "Độ dày truyền thông"}
        ]
        df_display = pd.DataFrame(rows_data)
        st.markdown("**Bảng ma trận biến động đặc trưng cảm xúc vĩ mô sau cú sốc biên:**")
        st.dataframe(df_display, use_container_width=True, hide_index=True)
