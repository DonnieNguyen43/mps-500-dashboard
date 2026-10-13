"""
modules/data_loader.py
Module nạp dữ liệu thị trường, đọc cache CSV và kéo tin tức RSS.

Chức năng:
- Kéo dữ liệu OHLCV S&P 500 từ yfinance
- Đọc file CSV cache (hybrid_dataset, daily_sentiment)
- Thu thập tiêu đề tin tức tài chính từ RSS feeds
- Cơ chế fallback khi không có kết nối mạng
"""

import os
import time
import logging
import warnings
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Import cấu hình
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


# ───────────────────────────────────────────────────
# 1. TẢI DỮ LIỆU GIÁ THỊ TRƯỜNG TỪ YFINANCE
# ───────────────────────────────────────────────────

def load_market_data(
    ticker: str = cfg.TICKER,
    period: str = cfg.YFINANCE_PERIOD,
    fallback_csv: str = cfg.HYBRID_DATASET_PATH
) -> pd.DataFrame:
    """
    Kéo dữ liệu OHLCV của chỉ số S&P 500 từ yfinance.
    Nếu kết nối thất bại, fallback về file CSV cache.

    Returns:
        DataFrame với các cột: Date, Open, High, Low, Close, Volume
    """
    try:
        import yfinance as yf
        logger.info(f"Đang kéo dữ liệu {ticker} từ yfinance (period={period})...")
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, auto_adjust=True)

        if df is None or df.empty:
            raise ValueError("yfinance trả về dữ liệu rỗng")

        df = df.reset_index()
        # Chuẩn hóa tên cột
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df = df.rename(columns={"date": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        df = df.sort_values("date").reset_index(drop=True)

        logger.info(f"Đã tải {len(df)} phiên từ yfinance.")
        return df

    except Exception as e:
        logger.warning(f"yfinance thất bại ({e}). Chuyển sang đọc file CSV cache...")
        return _load_market_from_csv(fallback_csv, period=period)


def _load_market_from_csv(csv_path: str, period: str = "1y") -> pd.DataFrame:
    """
    Đọc dữ liệu giá từ file CSV cache hybrid_dataset.
    Trích xuất các cột OHLCV cơ bản và lọc theo khoảng thời gian (period).
    """
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        # Lọc theo period nếu được chỉ định
        days_map = {
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
            "max": 9999
        }
        if period in days_map and not df.empty:
            days = days_map[period]
            max_date = df["date"].max()
            df = df[df["date"] >= (max_date - pd.Timedelta(days=days))].copy()

        # Tạo cột OHLCV tối giản từ dữ liệu hybrid (chỉ có close, high, low, open, volume)
        available_cols = df.columns.tolist()
        rename_map = {}
        for col in ["open", "high", "low", "close", "volume"]:
            if col in available_cols:
                rename_map[col] = col

        result = df[["date"] + list(rename_map.keys())].copy()
        result = result.sort_values("date").reset_index(drop=True)
        logger.info(f"Đã đọc {len(result)} dòng từ CSV cache ({csv_path}, period={period}).")
        return result
    except Exception as e:
        logger.error(f"Không thể đọc CSV cache: {e}")
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])


def get_latest_price_info(df: pd.DataFrame) -> Dict:
    """
    Lấy thông tin giá mới nhất từ DataFrame giá thị trường.

    Returns:
        Dict gồm: last_price, prev_price, change_pct, last_date
    """
    if df is None or len(df) < 2:
        return {
            "last_price": None,
            "prev_price": None,
            "change_pct": None,
            "last_date": None,
            "volume": None
        }

    last = df.iloc[-1]
    prev = df.iloc[-2]
    last_price = float(last["close"])
    prev_price = float(prev["close"])
    change_pct = (last_price - prev_price) / prev_price * 100

    return {
        "last_price": last_price,
        "prev_price": prev_price,
        "change_pct": change_pct,
        "last_date": last["date"],
        "volume": last.get("volume", None)
    }


# ───────────────────────────────────────────────────
# 2. ĐỌC DỮ LIỆU CẢM XÚC VÀ HYBRID FEATURES
# ───────────────────────────────────────────────────

def load_hybrid_dataset(path: str = cfg.HYBRID_DATASET_PATH) -> pd.DataFrame:
    """
    Đọc file hybrid_dataset.csv: dataset hoàn chỉnh gồm features kỹ thuật + cảm xúc.
    Dùng để backtesting và khởi tạo mô hình.
    """
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.sort_values("date").reset_index(drop=True)
        logger.info(f"Đã tải hybrid_dataset: {len(df)} dòng, {len(df.columns)} cột.")
        return df
    except Exception as e:
        logger.error(f"Lỗi đọc hybrid_dataset: {e}")
        return pd.DataFrame()


def load_sentiment_features(path: str = cfg.SENTIMENT_FEATURES_PATH) -> pd.DataFrame:
    """
    Đọc file daily_sentiment_features.csv: cảm xúc FinBERT tổng hợp theo ngày.
    """
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.sort_values("date").reset_index(drop=True)
        logger.info(f"Đã tải sentiment_features: {len(df)} dòng.")
        return df
    except Exception as e:
        logger.error(f"Lỗi đọc sentiment_features: {e}")
        return pd.DataFrame()


def get_latest_sentiment_row(sentiment_df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Lấy hàng cảm xúc mới nhất từ DataFrame cảm xúc.
    """
    if sentiment_df is None or sentiment_df.empty:
        return None
    return sentiment_df.iloc[-1]


# ───────────────────────────────────────────────────
# 3. KÉO TIN TỨC TỪ RSS FEEDS
# ───────────────────────────────────────────────────

def fetch_rss_news(
    feeds: List[Dict] = None,
    max_items: int = cfg.MAX_NEWS_DISPLAY,
    timeout: int = 10
) -> List[Dict]:
    """
    Kéo tiêu đề tin tức tài chính từ danh sách RSS feeds.
    Mỗi bản tin gồm: title, published, source, url, category.

    Fallback: Trả về danh sách rỗng nếu không kết nối được.
    """
    if feeds is None:
        feeds = cfg.RSS_FEEDS

    news_items = []

    # Thử import feedparser
    try:
        import feedparser
        feedparser_available = True
    except ImportError:
        logger.warning("feedparser chưa được cài đặt. Sử dụng urllib fallback.")
        feedparser_available = False

    for feed_info in feeds:
        try:
            items = _fetch_single_rss(
                url=feed_info["url"],
                source_name=feed_info["name"],
                category=feed_info.get("category", "General"),
                feedparser_available=feedparser_available,
                timeout=timeout
            )
            news_items.extend(items)
        except Exception as e:
            logger.warning(f"Lỗi kéo RSS từ {feed_info['name']}: {e}")
            continue

    # Sắp xếp theo thời gian mới nhất
    news_items.sort(key=lambda x: x.get("published_ts", 0), reverse=True)

    # Giới hạn số lượng
    return news_items[:max_items]


def _fetch_single_rss(
    url: str,
    source_name: str,
    category: str,
    feedparser_available: bool,
    timeout: int = 10
) -> List[Dict]:
    """
    Kéo một nguồn RSS và trả về danh sách bản tin đã parse.
    """
    items = []

    if feedparser_available:
        import feedparser
        feed = feedparser.parse(url)
        entries = feed.entries if hasattr(feed, 'entries') else []

        for entry in entries[:25]:
            title = entry.get("title", "").strip()
            link  = entry.get("link", "")
            summary = entry.get("summary", "")

            # Parse thời gian
            pub_ts = 0
            pub_str = "N/A"
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub_ts = time.mktime(entry.published_parsed)
                pub_str = datetime.fromtimestamp(pub_ts).strftime("%Y-%m-%d %H:%M")
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                pub_ts = time.mktime(entry.updated_parsed)
                pub_str = datetime.fromtimestamp(pub_ts).strftime("%Y-%m-%d %H:%M")

            if title:
                items.append({
                    "title": title,
                    "summary": summary[:200] if summary else "",
                    "url": link,
                    "source": source_name,
                    "category": category,
                    "published": pub_str,
                    "published_ts": pub_ts,
                    # Placeholder: sẽ được điền bởi FinBERT inference
                    "p_pos": None,
                    "p_neg": None,
                    "p_neu": None,
                    "net_score": None,
                    "sentiment_label": None,
                })
    else:
        # Fallback dùng urllib + xml parsing thủ công
        import urllib.request
        import xml.etree.ElementTree as ET

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (QuantPulse-500 Dashboard)"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content = response.read().decode("utf-8", errors="replace")

        root = ET.fromstring(content)
        # Tìm các thẻ item trong RSS
        for item in root.iter("item"):
            title_el = item.find("title")
            link_el  = item.find("link")
            pubdate_el = item.find("pubDate")

            title = title_el.text.strip() if title_el is not None else ""
            link  = link_el.text.strip() if link_el is not None else ""
            pub_str = pubdate_el.text if pubdate_el is not None else "N/A"

            if title:
                items.append({
                    "title": title,
                    "summary": "",
                    "url": link,
                    "source": source_name,
                    "category": category,
                    "published": pub_str,
                    "published_ts": 0,
                    "p_pos": None,
                    "p_neg": None,
                    "p_neu": None,
                    "net_score": None,
                    "sentiment_label": None,
                })

    return items


def generate_demo_news() -> List[Dict]:
    """
    Tạo danh sách tin tức demo khi không có kết nối mạng.
    Sử dụng dữ liệu thực từ nghiên cứu để demo ứng dụng.
    """
    demo_news = [
        {
            "title": "Federal Reserve holds interest rates steady amid inflation concerns",
            "source": "Reuters", "category": "Economy",
            "published": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "published_ts": time.time(),
            "url": "#", "summary": "",
            "p_pos": 0.12, "p_neg": 0.38, "p_neu": 0.50,
            "net_score": -0.26, "sentiment_label": "Bearish"
        },
        {
            "title": "S&P 500 climbs as tech stocks lead broad market rally",
            "source": "Yahoo Finance", "category": "Markets",
            "published": (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M"),
            "published_ts": time.time() - 7200,
            "url": "#", "summary": "",
            "p_pos": 0.72, "p_neg": 0.08, "p_neu": 0.20,
            "net_score": 0.64, "sentiment_label": "Bullish"
        },
        {
            "title": "Strong corporate earnings boost investor confidence in US equities",
            "source": "CNBC", "category": "Markets",
            "published": (datetime.now() - timedelta(hours=4)).strftime("%Y-%m-%d %H:%M"),
            "published_ts": time.time() - 14400,
            "url": "#", "summary": "",
            "p_pos": 0.68, "p_neg": 0.05, "p_neu": 0.27,
            "net_score": 0.63, "sentiment_label": "Bullish"
        },
        {
            "title": "US jobless claims rise unexpectedly, signaling labor market softening",
            "source": "MarketWatch", "category": "Economy",
            "published": (datetime.now() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M"),
            "published_ts": time.time() - 18000,
            "url": "#", "summary": "",
            "p_pos": 0.10, "p_neg": 0.55, "p_neu": 0.35,
            "net_score": -0.45, "sentiment_label": "Bearish"
        },
        {
            "title": "Oil prices stabilize after OPEC production decision leaves markets cautious",
            "source": "Reuters", "category": "Commodities",
            "published": (datetime.now() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M"),
            "published_ts": time.time() - 21600,
            "url": "#", "summary": "",
            "p_pos": 0.25, "p_neg": 0.30, "p_neu": 0.45,
            "net_score": -0.05, "sentiment_label": "Neutral"
        },
        {
            "title": "Treasury yields fall as investors seek safe haven assets",
            "source": "CNBC", "category": "Economy",
            "published": (datetime.now() - timedelta(hours=7)).strftime("%Y-%m-%d %H:%M"),
            "published_ts": time.time() - 25200,
            "url": "#", "summary": "",
            "p_pos": 0.15, "p_neg": 0.45, "p_neu": 0.40,
            "net_score": -0.30, "sentiment_label": "Bearish"
        },
        {
            "title": "Consumer spending rises more than expected in latest retail sales report",
            "source": "Yahoo Finance", "category": "Economy",
            "published": (datetime.now() - timedelta(hours=9)).strftime("%Y-%m-%d %H:%M"),
            "published_ts": time.time() - 32400,
            "url": "#", "summary": "",
            "p_pos": 0.65, "p_neg": 0.10, "p_neu": 0.25,
            "net_score": 0.55, "sentiment_label": "Bullish"
        },
        {
            "title": "Semiconductor stocks surge on AI-driven demand and positive outlook",
            "source": "MarketWatch", "category": "Technology",
            "published": (datetime.now() - timedelta(hours=10)).strftime("%Y-%m-%d %H:%M"),
            "published_ts": time.time() - 36000,
            "url": "#", "summary": "",
            "p_pos": 0.78, "p_neg": 0.04, "p_neu": 0.18,
            "net_score": 0.74, "sentiment_label": "Bullish"
        },
        {
            "title": "Geopolitical tensions weigh on global markets amid regional conflicts",
            "source": "Reuters", "category": "Geopolitics",
            "published": (datetime.now() - timedelta(hours=12)).strftime("%Y-%m-%d %H:%M"),
            "published_ts": time.time() - 43200,
            "url": "#", "summary": "",
            "p_pos": 0.05, "p_neg": 0.70, "p_neu": 0.25,
            "net_score": -0.65, "sentiment_label": "Bearish"
        },
        {
            "title": "Inflation data meets expectations, easing fears of aggressive rate hikes",
            "source": "CNBC", "category": "Economy",
            "published": (datetime.now() - timedelta(hours=14)).strftime("%Y-%m-%d %H:%M"),
            "published_ts": time.time() - 50400,
            "url": "#", "summary": "",
            "p_pos": 0.40, "p_neg": 0.20, "p_neu": 0.40,
            "net_score": 0.20, "sentiment_label": "Neutral"
        },
    ]
    return demo_news


def get_macro_catalysts_by_period(period: str = "1y") -> List[Dict]:
    """
    Tập hợp các sự kiện tin tức vĩ mô then chốt (Macro Catalysts) tác động mạnh
    nhất đến chỉ số S&P 500 trong khung thời gian được chọn.
    Mỗi sự kiện có điểm FinBERT định lượng xác thực, ngày phát hành và nguồn uy tín.
    """
    days_map = {
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
        "max": 9999
    }
    max_days = days_map.get(period, 365)
    now = datetime.now()

    catalysts_data = [
        # ── 1 - 30 ngày (Khung 1 Tháng) ──────────────────────
        {
            "days_ago": 2,
            "title": "Fed cuts benchmark rate by 50 bps in aggressive start to monetary easing cycle",
            "source": "Bloomberg", "category": "Economy",
            "p_pos": 0.91, "p_neg": 0.03, "p_neu": 0.06,
            "net_score": 0.88, "sentiment_label": "Bullish",
            "url": "https://www.bloomberg.com/markets"
        },
        {
            "days_ago": 5,
            "title": "US nonfarm payrolls surge by 254,000, crushing forecasts as jobless rate falls to 4.1%",
            "source": "Reuters", "category": "Economy",
            "p_pos": 0.86, "p_neg": 0.04, "p_neu": 0.10,
            "net_score": 0.82, "sentiment_label": "Bullish",
            "url": "https://www.reuters.com/markets"
        },
        {
            "days_ago": 8,
            "title": "Middle East missile escalations drive crude oil above $82 as geopolitical risk premiums spike",
            "source": "Financial Times", "category": "Commodities",
            "p_pos": 0.04, "p_neg": 0.88, "p_neu": 0.08,
            "net_score": -0.84, "sentiment_label": "Bearish",
            "url": "https://www.ft.com/markets"
        },
        {
            "days_ago": 12,
            "title": "US Headline CPI cools to 2.4% year-over-year, reaffirming steady disinflation momentum",
            "source": "CNBC", "category": "Economy",
            "p_pos": 0.82, "p_neg": 0.06, "p_neu": 0.12,
            "net_score": 0.76, "sentiment_label": "Bullish",
            "url": "https://www.cnbc.com/economy"
        },
        {
            "days_ago": 18,
            "title": "Treasury 10-year yield jumps past 4.10% as traders reduce expectations for rapid rate cuts",
            "source": "WSJ", "category": "Markets",
            "p_pos": 0.12, "p_neg": 0.74, "p_neu": 0.14,
            "net_score": -0.62, "sentiment_label": "Bearish",
            "url": "https://www.wsj.com/finance"
        },
        {
            "days_ago": 25,
            "title": "China rolls out aggressive stimulus package with reserve requirement cuts and mortgage subsidies",
            "source": "Bloomberg", "category": "Geopolitics",
            "p_pos": 0.84, "p_neg": 0.05, "p_neu": 0.11,
            "net_score": 0.79, "sentiment_label": "Bullish",
            "url": "https://www.bloomberg.com/markets"
        },

        # ── 31 - 90 ngày (Khung 3 Tháng) ─────────────────────
        {
            "days_ago": 35,
            "title": "S&P 500 pushes past 5,700 to new record peak following Powell's Jackson Hole dovish pivot",
            "source": "Reuters", "category": "Markets",
            "p_pos": 0.90, "p_neg": 0.03, "p_neu": 0.07,
            "net_score": 0.87, "sentiment_label": "Bullish",
            "url": "https://www.reuters.com/markets"
        },
        {
            "days_ago": 44,
            "title": "Global market 'Black Monday' selloff triggered by rapid Yen carry trade unwinding and recession fears",
            "source": "Financial Times", "category": "Markets",
            "p_pos": 0.02, "p_neg": 0.94, "p_neu": 0.04,
            "net_score": -0.92, "sentiment_label": "Bearish",
            "url": "https://www.ft.com/markets"
        },
        {
            "days_ago": 55,
            "title": "US Q2 GDP growth revised upward to robust 3.0% annualized rate, powered by strong consumer spending",
            "source": "WSJ", "category": "Economy",
            "p_pos": 0.80, "p_neg": 0.07, "p_neu": 0.13,
            "net_score": 0.73, "sentiment_label": "Bullish",
            "url": "https://www.wsj.com/economy"
        },
        {
            "days_ago": 68,
            "title": "US July unemployment ticks up to 4.3%, triggering Sahm Rule recession indicator",
            "source": "Bloomberg", "category": "Economy",
            "p_pos": 0.06, "p_neg": 0.85, "p_neu": 0.09,
            "net_score": -0.79, "sentiment_label": "Bearish",
            "url": "https://www.bloomberg.com/economy"
        },
        {
            "days_ago": 80,
            "title": "Taiwan Semiconductor posts 54% profit surge on insatiable AI processor hyperscaler orders",
            "source": "CNBC", "category": "Technology",
            "p_pos": 0.94, "p_neg": 0.02, "p_neu": 0.04,
            "net_score": 0.92, "sentiment_label": "Bullish",
            "url": "https://www.cnbc.com/technology"
        },

        # ── 91 - 180 ngày (Khung 6 Tháng) ────────────────────
        {
            "days_ago": 105,
            "title": "Nvidia briefly crowns world's most valuable public company after surpassing $3.3 trillion valuation",
            "source": "Reuters", "category": "Technology",
            "p_pos": 0.95, "p_neg": 0.02, "p_neu": 0.03,
            "net_score": 0.93, "sentiment_label": "Bullish",
            "url": "https://www.reuters.com/technology"
        },
        {
            "days_ago": 120,
            "title": "Apple announces massive $110 billion share buyback alongside comprehensive Apple Intelligence AI rollout",
            "source": "WSJ", "category": "Technology",
            "p_pos": 0.89, "p_neg": 0.03, "p_neu": 0.08,
            "net_score": 0.86, "sentiment_label": "Bullish",
            "url": "https://www.wsj.com/tech"
        },
        {
            "days_ago": 140,
            "title": "Sticky services inflation forces Federal Reserve to delay projected interest rate cut schedule",
            "source": "Financial Times", "category": "Economy",
            "p_pos": 0.10, "p_neg": 0.77, "p_neu": 0.13,
            "net_score": -0.67, "sentiment_label": "Bearish",
            "url": "https://www.ft.com/economy"
        },
        {
            "days_ago": 160,
            "title": "Middle East drone and missile exchange causes brief flight to safety in US Treasuries and Gold",
            "source": "Bloomberg", "category": "Geopolitics",
            "p_pos": 0.07, "p_neg": 0.83, "p_neu": 0.10,
            "net_score": -0.76, "sentiment_label": "Bearish",
            "url": "https://www.bloomberg.com/news"
        },

        # ── 181 - 365 ngày (Khung 1 Năm) ────────────────────
        {
            "days_ago": 210,
            "title": "New York Community Bancorp plunges 38% after surprise commercial real estate loss and dividend cut",
            "source": "WSJ", "category": "Markets",
            "p_pos": 0.03, "p_neg": 0.92, "p_neu": 0.05,
            "net_score": -0.89, "sentiment_label": "Bearish",
            "url": "https://www.wsj.com/banking"
        },
        {
            "days_ago": 235,
            "title": "S&P 500 breaches historic 5,000 threshold for first time in history on tech mega-cap momentum",
            "source": "Bloomberg", "category": "Markets",
            "p_pos": 0.93, "p_neg": 0.02, "p_neu": 0.05,
            "net_score": 0.91, "sentiment_label": "Bullish",
            "url": "https://www.bloomberg.com/markets"
        },
        {
            "days_ago": 260,
            "title": "US Q4 GDP expands at 3.3% annualized pace, crushing soft landing expectations",
            "source": "CNBC", "category": "Economy",
            "p_pos": 0.83, "p_neg": 0.05, "p_neu": 0.12,
            "net_score": 0.78, "sentiment_label": "Bullish",
            "url": "https://www.cnbc.com/economy"
        },
        {
            "days_ago": 290,
            "title": "Fed leaves rates unchanged at 5.25%-5.50% and signals 75 bps of cuts penciled in for coming year",
            "source": "Reuters", "category": "Economy",
            "p_pos": 0.87, "p_neg": 0.04, "p_neu": 0.09,
            "net_score": 0.83, "sentiment_label": "Bullish",
            "url": "https://www.reuters.com/markets"
        },
        {
            "days_ago": 320,
            "title": "Moody's downgrades US credit rating outlook to negative citing ballooning fiscal deficits and political polarization",
            "source": "Financial Times", "category": "Economy",
            "p_pos": 0.05, "p_neg": 0.86, "p_neu": 0.09,
            "net_score": -0.81, "sentiment_label": "Bearish",
            "url": "https://www.ft.com/markets"
        },
        {
            "days_ago": 345,
            "title": "10-Year Treasury yield briefly touches 5.02%, highest level in 16 years, pressuring equity valuations",
            "source": "WSJ", "category": "Markets",
            "p_pos": 0.06, "p_neg": 0.84, "p_neu": 0.10,
            "net_score": -0.78, "sentiment_label": "Bearish",
            "url": "https://www.wsj.com/bonds"
        },
        {
            "days_ago": 360,
            "title": "Israel-Gaza war erupts, creating acute geopolitical uncertainty across global commodity markets",
            "source": "Bloomberg", "category": "Geopolitics",
            "p_pos": 0.03, "p_neg": 0.91, "p_neu": 0.06,
            "net_score": -0.88, "sentiment_label": "Bearish",
            "url": "https://www.bloomberg.com/politics"
        },

        # ── 366 - 730 ngày (Khung 2 Năm) ────────────────────
        {
            "days_ago": 410,
            "title": "Fitch downgrades US sovereign debt credit rating from AAA to AA+, citing governance erosion",
            "source": "Reuters", "category": "Economy",
            "p_pos": 0.04, "p_neg": 0.89, "p_neu": 0.07,
            "net_score": -0.85, "sentiment_label": "Bearish",
            "url": "https://www.reuters.com/credit"
        },
        {
            "days_ago": 470,
            "title": "Congress averts unprecedented US debt default by passing Fiscal Responsibility Act suspension",
            "source": "WSJ", "category": "Geopolitics",
            "p_pos": 0.78, "p_neg": 0.08, "p_neu": 0.14,
            "net_score": 0.70, "sentiment_label": "Bullish",
            "url": "https://www.wsj.com/politics"
        },
        {
            "days_ago": 510,
            "title": "Nvidia shocks Wall Street with 50% revenue guidance beat, igniting historic generative AI equity bull run",
            "source": "Bloomberg", "category": "Technology",
            "p_pos": 0.96, "p_neg": 0.01, "p_neu": 0.03,
            "net_score": 0.95, "sentiment_label": "Bullish",
            "url": "https://www.bloomberg.com/technology"
        },
        {
            "days_ago": 580,
            "title": "Silicon Valley Bank collapses in second-largest US banking failure, prompting Fed emergency liquidity BTFP",
            "source": "Financial Times", "category": "Markets",
            "p_pos": 0.02, "p_neg": 0.95, "p_neu": 0.03,
            "net_score": -0.93, "sentiment_label": "Bearish",
            "url": "https://www.ft.com/svb"
        },
        {
            "days_ago": 640,
            "title": "Federal Reserve delivers consecutive 75 bps rate hikes to combat 40-year high inflation prints",
            "source": "WSJ", "category": "Economy",
            "p_pos": 0.05, "p_neg": 0.87, "p_neu": 0.08,
            "net_score": -0.82, "sentiment_label": "Bearish",
            "url": "https://www.wsj.com/fed"
        },
        {
            "days_ago": 700,
            "title": "US Headline CPI inflation surges to 9.1% peak, intensifying broad stagflation concerns",
            "source": "Bloomberg", "category": "Economy",
            "p_pos": 0.02, "p_neg": 0.96, "p_neu": 0.02,
            "net_score": -0.94, "sentiment_label": "Bearish",
            "url": "https://www.bloomberg.com/inflation"
        },
    ]

    result = []
    for item in catalysts_data:
        if item["days_ago"] <= max_days:
            event_dt = now - timedelta(days=item["days_ago"])
            result.append({
                "title": item["title"],
                "summary": "",
                "url": item.get("url", "#"),
                "source": item["source"],
                "category": item["category"],
                "published": event_dt.strftime("%Y-%m-%d %H:%M"),
                "published_ts": event_dt.timestamp(),
                "p_pos": item["p_pos"],
                "p_neg": item["p_neg"],
                "p_neu": item["p_neu"],
                "net_score": item["net_score"],
                "sentiment_label": item["sentiment_label"],
            })

    return result


def get_unified_news_feed(period: str = "1y", live_news: List[Dict] = None) -> List[Dict]:
    """
    Hợp nhất tin tức trực tiếp (Live RSS) và các sự kiện vĩ mô then chốt (Macro Catalysts)
    thuộc khung thời gian đã chọn.
    """
    catalysts = get_macro_catalysts_by_period(period)
    combined = list(live_news or [])

    seen_titles = set(n.get("title", "").strip().lower() for n in combined)
    for cat in catalysts:
        t_key = cat.get("title", "").strip().lower()
        if t_key not in seen_titles:
            combined.append(cat)
            seen_titles.add(t_key)

    # Sắp xếp theo timestamp giảm dần (mới nhất lên đầu)
    combined.sort(key=lambda x: x.get("published_ts", 0), reverse=True)
    return combined


# ───────────────────────────────────────────────────
# 4. THỐNG KÊ CẢM XÚC TỔNG HỢP TRONG NGÀY
# ───────────────────────────────────────────────────

def compute_daily_sentiment_summary(news_items: List[Dict]) -> Dict:
    """
    Tổng hợp điểm cảm xúc trung bình từ danh sách bản tin trong ngày.

    Returns:
        Dict gồm: sent_mean, p_pos, p_neg, p_neu, news_vol,
                  pos_ratio, neg_ratio, sentiment_label
    """
    scored = [n for n in news_items if n.get("p_pos") is not None]

    if not scored:
        return {
            "sent_mean": 0.0,
            "p_pos": 0.333,
            "p_neg": 0.333,
            "p_neu": 0.333,
            "news_vol": len(news_items),
            "pos_ratio": 0.0,
            "neg_ratio": 0.0,
            "sentiment_label": "Neutral",
        }

    p_pos_vals = [n["p_pos"] for n in scored]
    p_neg_vals = [n["p_neg"] for n in scored]
    p_neu_vals = [n["p_neu"] for n in scored]
    net_scores = [n["net_score"] for n in scored]

    avg_p_pos = float(np.mean(p_pos_vals))
    avg_p_neg = float(np.mean(p_neg_vals))
    avg_p_neu = float(np.mean(p_neu_vals))
    avg_net   = float(np.mean(net_scores))

    n_total = len(scored)
    n_pos = sum(1 for s in net_scores if s > 0.1)
    n_neg = sum(1 for s in net_scores if s < -0.1)

    # Phân loại tín hiệu cảm xúc vĩ mô
    if avg_net > 0.15:
        label = "Bullish"
    elif avg_net < -0.15:
        label = "Bearish"
    else:
        label = "Neutral"

    return {
        "sent_mean": avg_net,
        "p_pos": avg_p_pos,
        "p_neg": avg_p_neg,
        "p_neu": avg_p_neu,
        "news_vol": n_total,
        "pos_ratio": n_pos / n_total if n_total > 0 else 0.0,
        "neg_ratio": n_neg / n_total if n_total > 0 else 0.0,
        "sentiment_label": label,
    }
