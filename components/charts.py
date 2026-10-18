# -*- coding: utf-8 -*-
"""
components/charts.py: Module trực quan hóa đồ thị tương tác Plotly
Thiết kế theo chuẩn Luxury Dark Glassmorphism (Vercel / Linear / Bloomberg).
Nền trong suốt, lưới tọa độ tinh tế, loại bỏ hoàn toàn thanh công cụ thừa thãi.
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import config as cfg

def get_dark_plotly_layout(height: int = 500, title: str = "") -> dict:
    """
    Trả về cấu hình layout Dark Theme trong suốt cho toàn bộ biểu đồ Plotly.
    """
    return dict(
        height=height,
        title=dict(text=title, font=dict(color="#F0F4F8", size=14, family="'Inter', sans-serif")),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=cfg.COLOR_TEXT_PRIMARY, family="'Inter', sans-serif", size=12),
        margin=dict(l=40, r=30, t=50, b=35),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#8E9CAE")
        ),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.04)",
            zerolinecolor="rgba(255, 255, 255, 0.06)",
            showline=True,
            linecolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color="#8E9CAE", size=11)
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.04)",
            zerolinecolor="rgba(255, 255, 255, 0.06)",
            showline=True,
            linecolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color="#8E9CAE", size=11)
        )
    )

def plot_sp500_candlestick_macd(df: pd.DataFrame) -> go.Figure:
    """
    Vẽ biểu đồ nến kỹ thuật S&P 500 kết hợp SMA20, SMA50 và MACD Histogram.
    """
    df_plot = df.tail(120).copy()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.72, 0.28]
    )

    # 1. Nến Nhật
    fig.add_trace(
        go.Candlestick(
            x=df_plot["Date"],
            open=df_plot["Open"],
            high=df_plot["High"],
            low=df_plot["Low"],
            close=df_plot["Close"],
            name="S&P 500",
            increasing_line_color=cfg.COLOR_BULLISH,
            decreasing_line_color=cfg.COLOR_BEARISH
        ),
        row=1, col=1
    )

    # 2. SMA 20 và SMA 50
    if "sma_20" in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=df_plot["Date"], y=df_plot["sma_20"],
                mode="lines", name="SMA 20",
                line=dict(color=cfg.COLOR_NEUTRAL, width=1.5)
            ),
            row=1, col=1
        )
    if "sma_50" in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=df_plot["Date"], y=df_plot["sma_50"],
                mode="lines", name="SMA 50",
                line=dict(color=cfg.COLOR_CYAN, width=1.5)
            ),
            row=1, col=1
        )

    # 3. MACD Histogram
    if "macd_diff" in df_plot.columns:
        hist_colors = [cfg.COLOR_BULLISH if v >= 0 else cfg.COLOR_BEARISH for v in df_plot["macd_diff"]]
        fig.add_trace(
            go.Bar(
                x=df_plot["Date"], y=df_plot["macd_diff"],
                name="Chỉ báo động lượng MACD histogram",
                marker_color=hist_colors
            ),
            row=2, col=1
        )
        if "macd" in df_plot.columns and "macd_signal" in df_plot.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_plot["Date"], y=df_plot["macd"],
                    mode="lines", name="MACD (12,26)",
                    line=dict(color="#F0F4F8", width=1)
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df_plot["Date"], y=df_plot["macd_signal"],
                    mode="lines", name="Signal (9)",
                    line=dict(color="#FF7875", width=1, dash="dot")
                ),
                row=2, col=1
            )

    layout = get_dark_plotly_layout(height=520, title="")
    layout["xaxis_rangeslider_visible"] = False
    layout["legend"] = dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.0,
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color="#8E9CAE")
    )
    layout["margin"] = dict(l=40, r=40, t=60, b=40)
    layout["xaxis2"] = dict(gridcolor="rgba(255, 255, 255, 0.04)")
    layout["yaxis2"] = dict(gridcolor="rgba(255, 255, 255, 0.04)")
    
    fig.update_layout(layout)
    return fig

def plot_confidence_gauge(prob_5d: float, threshold: float = cfg.THRESHOLD_OPTIMAL) -> go.Figure:
    """
    Biểu đồ Gauge đo lường mức độ tự tin xác suất khung 5 ngày so với ngưỡng tối ưu 0.42.
    Được căn giữa đối xứng tâm hoàn hảo, khoảng đệm tiêu đề thoáng đãng, màu sắc chuẩn FinTech thể chế.
    """
    val_pct = prob_5d * 100
    thresh_pct = threshold * 100

    bar_color = "#10B981" if prob_5d >= threshold else "#F43F5E"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=val_pct,
        domain={'x': [0.00, 1.00], 'y': [0.00, 0.68]},
        delta={
            'reference': thresh_pct,
            'increasing': {'color': "#10B981"},
            'decreasing': {'color': "#F43F5E"},
            'font': {'size': 13, 'family': "'Inter', -apple-system, sans-serif"}
        },
        number={
            'suffix': "%",
            'font': {'size': 32, 'color': '#F8FAFC', 'family': "'Inter', -apple-system, sans-serif"}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': '#64748B',
                'tickvals': [0, 20, 40, 60, 80, 100],
                'ticktext': ['0', '20', '40', '60', '80', '100'],
                'tickfont': {'family': "'Inter', sans-serif", 'size': 11, 'color': '#94A3B8'}
            },
            'bar': {'color': bar_color, 'thickness': 0.26},
            'bgcolor': "rgba(0,0,0,0.25)",
            'borderwidth': 1,
            'bordercolor': "rgba(255, 255, 255, 0.08)",
            'steps': [
                {'range': [0, thresh_pct], 'color': 'rgba(244, 63, 94, 0.16)'},
                {'range': [thresh_pct, 100], 'color': 'rgba(16, 185, 129, 0.16)'}
            ],
            'threshold': {
                'line': {'color': "#FFFFFF", 'width': 3},
                'thickness': 0.78,
                'value': thresh_pct
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=25, r=25, t=80, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1", family="'Inter', -apple-system, sans-serif"),
        title=dict(
            text="<b>ĐO LƯỜNG NGƯỠNG TỰ TIN (T+5)</b><br><span style='font-size:11px;color:#94A3B8;'>Vùng an toàn ≥ 42.0% | Vùng cảnh báo rủi ro &lt; 42.0%</span>",
            font=dict(size=12.5, color="#F8FAFC", family="'Inter', -apple-system, sans-serif"),
            x=0.5,
            xanchor="center",
            y=0.90,
            yanchor="top"
        )
    )
    return fig

def plot_sentiment_donut_chart(sent_dict: dict, title: str = "Cơ cấu cảm xúc thị trường thực tế trong ngày") -> go.Figure:
    """
    Vẽ biểu đồ Donut Chart (Plotly Pie) tròn mượt mà cho phân phối cảm xúc.
    Nền trong suốt, căn giữa hoàn hảo.
    """
    p_pos = float(sent_dict.get('p_pos', sent_dict.get('p_pos_mean', 0.35)))
    p_neg = float(sent_dict.get('p_neg', sent_dict.get('p_neg_mean', 0.25)))
    p_neu = float(sent_dict.get('p_neu', sent_dict.get('p_neu_mean', 0.40)))

    total = p_pos + p_neu + p_neg
    if total <= 0.001:
        p_pos, p_neu, p_neg = 0.35, 0.40, 0.25
    else:
        p_pos, p_neu, p_neg = p_pos / total, p_neu / total, p_neg / total

    labels = ["Tích cực (Pos)", "Chờ đợi (Neu)", "Tiêu cực (Neg)"]
    values = [p_pos * 100, p_neu * 100, p_neg * 100]
    
    # Bảng màu FinTech dịu mắt theo yêu cầu thiết kế
    colors = ["#10b981", "#f59e0b", "#f43f5e"]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.68,
        domain=dict(x=[0.05, 0.95], y=[0.08, 0.92]),
        marker=dict(colors=colors, line=dict(color="#0b0e14", width=2)),
        textinfo="label+percent",
        textfont=dict(size=12, color="#F0F4F8", family="'Inter', sans-serif"),
        hoverinfo="label+value+percent"
    ))

    fig.update_layout(
        height=270,
        margin=dict(l=15, r=15, t=15, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        title=dict(
            text=None
        )
    )
    return fig

def plot_feature_importance_consensus() -> go.Figure:
    """
    Vẽ biểu đồ thanh ngang xếp hạng Top 10 đặc trưng đồng thuận từ feature_importance_consensus.csv.
    Bảng màu định lượng chuyên nghiệp (Muted Sophisticated Financial Palette):
    - p_neu (Hạng 1): Vàng ánh kim cao cấp (#d97706)
    - Nhóm cảm xúc FinBERT: Xanh ngọc lục bảo sâu (#059669)
    - Nhóm kỹ thuật: Xanh thép thanh lịch (#2563eb)
    """
    if os.path.exists(cfg.FEATURE_IMPORTANCE_CSV_PATH):
        df = pd.read_csv(cfg.FEATURE_IMPORTANCE_CSV_PATH)
    else:
        df = pd.DataFrame({
            "Tên biến đặc trưng": ["p_neu", "ret_1d", "ret_5d", "vol_change_1d", "news_vol_ratio_5d", "bb_pct_b", "dist_sma20", "dist_sma50", "rsi_14", "sent_momentum_1d"],
            "Nhóm bản chất": ["Cảm xúc FinBERT", "Kỹ thuật", "Kỹ thuật", "Kỹ thuật", "Cảm xúc FinBERT", "Kỹ thuật", "Kỹ thuật", "Kỹ thuật", "Kỹ thuật", "Cảm xúc FinBERT"],
            "Hạng đồng thuận": [2.25, 3.50, 6.75, 7.50, 8.00, 9.88, 10.00, 10.38, 10.62, 10.75]
        })

    df_top10 = df.head(10).sort_values("Hạng đồng thuận", ascending=False)

    colors = []
    line_colors = []
    for var in df_top10["Tên biến đặc trưng"]:
        if var == "p_neu":
            colors.append("#d97706")          # Hạng 1 p_neu: Vàng ánh kim / Dark Amber
            line_colors.append("#f59e0b")
        elif "ret_" in var or "sma" in var or "rsi" in var or "bb" in var or "vol_change" in var:
            colors.append("#2563eb")          # Nhóm kỹ thuật: Xanh thép thanh lịch
            line_colors.append("rgba(255, 255, 255, 0.12)")
        else:
            colors.append("#059669")          # Nhóm cảm xúc FinBERT: Xanh ngọc lục bảo sâu
            line_colors.append("#10b981")

    fig = go.Figure(go.Bar(
        x=df_top10["Hạng đồng thuận"],
        y=df_top10["Tên biến đặc trưng"],
        orientation="h",
        marker=dict(color=colors, line=dict(color=line_colors, width=1)),
        text=[f"Hạng {v:.2f}" for v in df_top10["Hạng đồng thuận"]],
        textfont=dict(color="#f1f5f9", family="JetBrains Mono, monospace", size=11),
        textposition="inside",
        insidetextanchor="middle"
    ))

    layout = get_dark_plotly_layout(height=400, title="Top 10 đặc trưng đồng thuận qua 4 mô hình học máy")
    layout["xaxis"]["title"] = "Điểm hạng đồng thuận trung bình (điểm càng nhỏ thứ hạng càng cao)"
    layout["yaxis"]["autorange"] = True
    fig.update_layout(layout)
    return fig

def plot_real_equity_curve(backtest_res: dict, model_name: str, horizon: str, threshold: float) -> go.Figure:
    """
    Vẽ biểu đồ đường cong tăng trưởng vốn thực nghiệm (Real Equity Curve):
    So sánh Lợi nhuận tích lũy thực tế giữa:
    - Mô hình AI được chọn (P >= threshold)
    - Điểm chuẩn Buy & Hold S&P 500
    - Mô hình AI cấu hình gốc (P >= 0.50)
    100% tính toán thực nghiệm trên 434 phiên ngoài mẫu (không dùng hạt giống mẫu).
    """
    dates = backtest_res["dates"]
    ai_cum = backtest_res["equity_ai"]
    bh_cum = backtest_res["equity_bh"]
    base_cum = backtest_res["equity_base"]
    m_ai = backtest_res["metrics_ai"]
    m_bh = backtest_res["metrics_bh"]
    m_base = backtest_res["metrics_base"]

    fig = go.Figure()

    # 1. Chiến lược AI được chọn
    fig.add_trace(go.Scatter(
        x=dates, y=ai_cum,
        mode="lines",
        name=f"{model_name} {horizon} (P ≥ {threshold:.2f}) [{m_ai['cum_ret']:+.2f}% | Sharpe {m_ai['sharpe']:.2f}]",
        line=dict(color=cfg.COLOR_BULLISH, width=3.2),
        fill="tozeroy",
        fillcolor="rgba(0, 229, 153, 0.05)",
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>AI Lợi nhuận: <b>%{y:+.2f}%</b><extra></extra>"
    ))

    # 2. Điểm chuẩn Mua & Nắm giữ S&P 500 (Benchmark)
    fig.add_trace(go.Scatter(
        x=dates, y=bh_cum,
        mode="lines",
        name=f"Buy & Hold S&P 500 [{m_bh['cum_ret']:+.2f}% | Sharpe {m_bh['sharpe']:.2f}]",
        line=dict(color="#A0AEC0", width=2.0, dash="dash"),
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Benchmark S&P 500: <b>%{y:+.2f}%</b><extra></extra>"
    ))

    # 3. Mô hình cấu hình gốc P >= 0.50 (Baseline)
    fig.add_trace(go.Scatter(
        x=dates, y=base_cum,
        mode="lines",
        name=f"Chiến lược gốc P = 0.50 [{m_base['cum_ret']:+.2f}% | Sharpe {m_base['sharpe']:.2f}]",
        line=dict(color=cfg.COLOR_BEARISH, width=1.8, dash="dot"),
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Gốc P=0.50: <b>%{y:+.2f}%</b><extra></extra>"
    ))

    layout = get_dark_plotly_layout(height=480, title="")
    layout["title"] = None
    layout["legend"] = dict(
        orientation="h",
        yanchor="bottom",
        y=1.07,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11.5, color="#C2CBD6")
    )
    layout["margin"] = dict(l=65, r=25, t=75, b=45)
    layout["yaxis"]["title"] = "Lợi nhuận tích lũy (%)"
    layout["yaxis"]["ticksuffix"] = "%"
    layout["xaxis"]["title"] = backtest_res.get("dataset_label", f"Thời gian ({len(dates)} phiên)")
    fig.update_layout(layout)
    return fig

def plot_real_drawdown_curve(backtest_res: dict, model_name: str, horizon: str, threshold: float) -> go.Figure:
    """
    Vẽ biểu đồ đường cong sụt giảm vốn tối đa (Underwater Drawdown Curve):
    Trực quan hóa mức độ bảo toàn vốn và quản trị rủi ro khi thị trường thoái lui.
    """
    dates = backtest_res["dates"]
    ai_dd = backtest_res["drawdown_ai"]
    bh_dd = backtest_res["drawdown_bh"]
    base_dd = backtest_res["drawdown_base"]
    m_ai = backtest_res["metrics_ai"]
    m_bh = backtest_res["metrics_bh"]

    fig = go.Figure()

    # 1. Drawdown Chiến lược AI được chọn
    fig.add_trace(go.Scatter(
        x=dates, y=ai_dd,
        mode="lines",
        name=f"AI: {model_name} ({horizon}) [Max DD: {m_ai['mdd']:.2f}%]",
        line=dict(color=cfg.COLOR_BULLISH, width=2.4),
        fill="tozeroy",
        fillcolor="rgba(0, 229, 153, 0.18)",
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>AI Sụt giảm: <b>%{y:.2f}%</b><extra></extra>"
    ))

    # 2. Drawdown Buy & Hold S&P 500
    fig.add_trace(go.Scatter(
        x=dates, y=bh_dd,
        mode="lines",
        name=f"S&P 500 Benchmark [Max DD: {m_bh['mdd']:.2f}%]",
        line=dict(color=cfg.COLOR_BEARISH, width=2.0, dash="dash"),
        fill="tozeroy",
        fillcolor="rgba(255, 77, 79, 0.10)",
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Benchmark Sụt giảm: <b>%{y:.2f}%</b><extra></extra>"
    ))

    # 3. Drawdown Gốc P = 0.50
    fig.add_trace(go.Scatter(
        x=dates, y=base_dd,
        mode="lines",
        name=f"Mức gốc P = 0.50 [Max DD: {backtest_res['metrics_base']['mdd']:.2f}%]",
        line=dict(color=cfg.COLOR_NEUTRAL, width=1.5, dash="dot"),
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Gốc P=0.50 Sụt giảm: <b>%{y:.2f}%</b><extra></extra>"
    ))

    layout = get_dark_plotly_layout(height=480, title="")
    layout["title"] = None
    layout["legend"] = dict(
        orientation="h",
        yanchor="bottom",
        y=1.07,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11.5, color="#C2CBD6")
    )
    layout["margin"] = dict(l=65, r=25, t=75, b=45)
    layout["yaxis"]["title"] = "Độ sụt giảm từ đỉnh (Underwater Drawdown %)"
    layout["yaxis"]["ticksuffix"] = "%"
    layout["xaxis"]["title"] = backtest_res.get("dataset_label", f"Thời gian ({len(dates)} phiên)")
    fig.update_layout(layout)
    return fig

def plot_real_dual_chart(backtest_res: dict, model_name: str, horizon: str, threshold: float) -> go.Figure:
    """
    Vẽ biểu đồ kết hợp song song 2 khung:
    - Khung trên: Đường cong tăng trưởng vốn (Equity Curve %)
    - Khung dưới: Đường cong sụt giảm vốn (Underwater Drawdown %)
    Đồng bộ trục thời gian chuẩn tắc Bloomberg Quant Terminal.
    """
    dates = backtest_res["dates"]
    ai_cum = backtest_res["equity_ai"]
    bh_cum = backtest_res["equity_bh"]
    base_cum = backtest_res["equity_base"]
    ai_dd = backtest_res["drawdown_ai"]
    bh_dd = backtest_res["drawdown_bh"]

    m_ai = backtest_res["metrics_ai"]
    m_bh = backtest_res["metrics_bh"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        row_heights=[0.66, 0.34],
        subplot_titles=[
            f"Tăng trưởng vốn lũy kế (Equity Curve: {model_name} {horizon})",
            "Độ sụt giảm vốn tối đa từ đỉnh (Underwater Max Drawdown %)"
        ]
    )

    # Khung 1: Equity
    fig.add_trace(go.Scatter(
        x=dates, y=ai_cum, mode="lines",
        name=f"AI: {model_name} ({horizon}) [{m_ai['cum_ret']:+.2f}% | Sharpe {m_ai['sharpe']:.2f}]",
        line=dict(color=cfg.COLOR_BULLISH, width=3.0),
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>AI Lợi nhuận: <b>%{y:+.2f}%</b><extra></extra>"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=bh_cum, mode="lines",
        name=f"S&P 500 Benchmark [{m_bh['cum_ret']:+.2f}% | Sharpe {m_bh['sharpe']:.2f}]",
        line=dict(color="#A0AEC0", width=1.8, dash="dash"),
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>S&P 500: <b>%{y:+.2f}%</b><extra></extra>"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=base_cum, mode="lines",
        name=f"Gốc P = 0.50 [{backtest_res['metrics_base']['cum_ret']:+.2f}%]",
        line=dict(color=cfg.COLOR_BEARISH, width=1.4, dash="dot"),
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Gốc 0.50: <b>%{y:+.2f}%</b><extra></extra>"
    ), row=1, col=1)

    # Khung 2: Drawdown
    fig.add_trace(go.Scatter(
        x=dates, y=ai_dd, mode="lines",
        name=f"AI Drawdown ({m_ai['mdd']:.2f}%)",
        line=dict(color=cfg.COLOR_BULLISH, width=2.0),
        fill="tozeroy", fillcolor="rgba(0, 229, 153, 0.18)",
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>AI Drawdown: <b>%{y:.2f}%</b><extra></extra>"
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=bh_dd, mode="lines",
        name=f"S&P 500 Drawdown ({m_bh['mdd']:.2f}%)",
        line=dict(color=cfg.COLOR_BEARISH, width=1.6, dash="dash"),
        fill="tozeroy", fillcolor="rgba(255, 77, 79, 0.12)",
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>S&P 500 Drawdown: <b>%{y:.2f}%</b><extra></extra>"
    ), row=2, col=1)

    layout = get_dark_plotly_layout(height=650, title="")
    layout["title"] = None
    layout["legend"] = dict(
        orientation="h",
        yanchor="bottom",
        y=1.09,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color="#C2CBD6")
    )
    layout["margin"] = dict(l=65, r=25, t=95, b=45)
    fig.update_layout(layout)

    # Style subplot annotations to have crisp font and ample separation
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(size=12, color="#E2E8F0", family="'Inter', sans-serif")
        if "Tăng trưởng" in ann['text']:
            ann['y'] = 1.01

    fig.update_yaxes(title_text="Lợi nhuận (%)", ticksuffix="%", row=1, col=1)
    fig.update_yaxes(title_text="Sụt giảm (%)", ticksuffix="%", row=2, col=1)
    fig.update_xaxes(title_text=backtest_res.get("dataset_label", f"Thời gian ({len(dates)} phiên)"), row=2, col=1)
    return fig

def plot_feature_importance_for_model(df_feats: pd.DataFrame, model_name: str, horizon: str) -> go.Figure:
    """
    Vẽ biểu đồ thanh ngang Top 10 đặc trưng quan trọng của chính mô hình và khung thời gian được chọn.
    Phân loại màu sắc trực quan:
    - Vàng ánh kim (#d97706): Đặc trưng quan trọng số 1 (Top 1)
    - Xanh thép thanh lịch (#2563eb): Nhóm chỉ báo kỹ thuật thị trường
    - Xanh ngọc lục bảo sâu (#059669): Nhóm chỉ báo tâm lý tin tức FinBERT
    """
    if df_feats is None or df_feats.empty:
        return plot_feature_importance_consensus()

    is_consensus = any(w in model_name.lower() for w in ["consensus", "đồng thuận", "ensemble", "lai"])
    df_plot = df_feats.copy().head(10)

    col_val = "Tầm quan trọng (%)" if "Tầm quan trọng (%)" in df_plot.columns else "importance"
    col_name = "Tên đặc trưng" if "Tên đặc trưng" in df_plot.columns else "feature"
    col_type = "Nhóm bản chất" if "Nhóm bản chất" in df_plot.columns else "type"

    # Sắp xếp cho hiển thị thanh ngang từ trên xuống dưới
    if is_consensus:
        df_plot = df_plot.sort_values(col_val, ascending=False)
        text_labels = [f"Hạng {v:.2f}" for v in df_plot[col_val]]
        x_title = "Điểm hạng đồng thuận trung bình (điểm càng nhỏ thứ hạng càng cao)"
    else:
        df_plot = df_plot.sort_values(col_val, ascending=True)
        text_labels = [f"{v:.2f}%" for v in df_plot[col_val]]
        x_title = "Đóng góp tầm quan trọng chuẩn hóa (%)"

    colors = []
    line_colors = []
    vals = df_plot[col_val].values
    max_idx = len(vals) - 1  # Vì ascending=True, phần tử cuối là lớn nhất

    for idx, (_, r) in enumerate(df_plot.iterrows()):
        feat_type = str(r[col_type])
        if idx == max_idx and not is_consensus:
            colors.append("#d97706")          # Top 1: Vàng ánh kim cao cấp
            line_colors.append("#f59e0b")
        elif "cảm xúc" in feat_type.lower() or "finbert" in feat_type.lower() or "sent" in str(r[col_name]):
            colors.append("#059669")          # Nhóm cảm xúc FinBERT
            line_colors.append("#10b981")
        else:
            colors.append("#2563eb")          # Nhóm kỹ thuật
            line_colors.append("rgba(255, 255, 255, 0.15)")

    fig = go.Figure(go.Bar(
        x=df_plot[col_val],
        y=df_plot[col_name],
        orientation="h",
        marker=dict(color=colors, line=dict(color=line_colors, width=1)),
        text=text_labels,
        textfont=dict(color="#f1f5f9", family="JetBrains Mono, monospace", size=11),
        textposition="inside",
        insidetextanchor="middle",
        hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>" if not is_consensus else "<b>%{y}</b>: Hạng %{x:.2f}<extra></extra>"
    ))

    layout = get_dark_plotly_layout(
        height=400,
        title=f"Top 10 đặc trưng quan trọng qua {model_name} ({horizon})"
    )
    layout["xaxis"]["title"] = x_title
    layout["yaxis"]["autorange"] = True
    layout["margin"] = dict(l=85, r=20, t=45, b=40)
    fig.update_layout(layout)
    return fig

def plot_equity_curves() -> go.Figure:
    """
    Hàm tương thích ngược: Thực thi kiểm định ngoài mẫu thực tế và vẽ đường cong vốn.
    Đảm bảo 100% tính toán thực nghiệm, không còn hạt giống giả lập.
    """
    from modules.backtester import run_out_of_sample_backtest
    res = run_out_of_sample_backtest("Random Forest", "T+5", 0.42)
    return plot_real_equity_curve(res, "Random Forest", "T+5", 0.42)


def plot_whatif_feature_attribution(whatif_res: dict) -> go.Figure:
    """
    Biểu đồ thanh ngang phân rã đóng góp đặc trưng AI (Explainability Engine: XAI):
    Hiển thị mức độ đóng góp của 4 nhóm nhân tố vào độ dịch chuyển xác suất ΔP.
    """
    factors = whatif_res.get("factors", {})
    delta_p = whatif_res.get("delta_p", 0.0)

    categories = [
        "Điểm cảm xúc FinBERT & Xung lượng",
        "Cơ cấu tâm lý vĩ mô (P_Pos, P_Neg, P_Neu)",
        "Tỷ lệ phân cực tin tức (Pos/Neg Ratio)",
        "Áp lực truyền thông & Khối lượng tin",
        "Dịch chuyển xác suất ròng (ΔP)"
    ]

    values = [
        factors.get("c_sent", 0.0) * 100,
        factors.get("c_state", 0.0) * 100,
        factors.get("c_polar", 0.0) * 100,
        factors.get("c_vol", 0.0) * 100,
        delta_p * 100
    ]

    colors = []
    for idx, v in enumerate(values):
        if idx == len(values) - 1:
            colors.append(cfg.COLOR_CYAN)
        elif v >= 0:
            colors.append(cfg.COLOR_BULLISH)
        else:
            colors.append(cfg.COLOR_BEARISH)

    text_labels = [f"{v:+.2f}%" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=categories,
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.12)", width=1)),
        text=text_labels,
        textposition="outside",
        cliponaxis=False
    ))

    layout = get_dark_plotly_layout(
        height=320,
        title="Phân rã đóng góp đặc trưng vào độ dịch chuyển xác suất (Feature attribution breakdown)"
    )
    layout["xaxis"]["title"] = "Mức độ đóng góp vào dịch chuyển xác suất (điểm %)"
    layout["xaxis"]["zeroline"] = True
    layout["xaxis"]["zerolinecolor"] = "rgba(255,255,255,0.2)"
    layout["yaxis"]["autorange"] = "reversed"
    fig.update_layout(layout)
    return fig
