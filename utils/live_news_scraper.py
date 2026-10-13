# -*- coding: utf-8 -*-
"""
utils/live_news_scraper.py: Module cào tin tức tài chính vĩ mô thời gian thực (Live RSS Scraper)
Hỗ trợ cơ chế Dual News Source cho Dashboard MPS-500:
1. Thu thập tin tức trực tiếp từ các nguồn RSS mở: Google News, Yahoo Finance, CNBC.
2. Bóc tách tiêu đề, ngày xuất bản thực tế hôm nay, tóm tắt, nguồn và liên kết gốc.
3. Tự động gắn nhãn danh mục vĩ mô, mức độ ảnh hưởng thị trường (Impact Tier).
4. Cơ chế dự phòng an toàn (Failover Fallback): tự động chuyển sang tập tin tức nội bộ nếu mất mạng.
"""

import os
import re
import urllib.request
import logging
from datetime import datetime
from typing import List, Dict, Tuple
from email.utils import parsedate_to_datetime

import html
import config as cfg

logger = logging.getLogger("LiveNewsScraper")

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Danh sách các luồng RSS tài chính mở quốc tế uy tín (Không cần API Key)
LIVE_RSS_FEEDS = [
    ("Google News S&P 500", "https://news.google.com/rss/search?q=S%26P+500+stock+market&hl=en-US&gl=US&ceid=US:en"),
    ("Google News Business", "https://news.google.com/rss/headlines/section/topic/BUSINESS"),
    ("Yahoo Finance S&P 500", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US"),
    ("CNBC Markets News", "https://www.cnbc.com/id/100003114/device/rss/rss.html")
]

def clean_html_text(raw_text: str) -> str:
    """Làm sạch triệt để các thẻ HTML, giải mã thực thể và loại bỏ URL thừa."""
    if not raw_text:
        return ""
    # Giải mã HTML entities (xử lý cả entity bị encode lồng nhau như &amp;lt;)
    text = html.unescape(str(raw_text))
    text = html.unescape(text)
    # Loại bỏ triệt để mọi thẻ HTML <...> và <!DOCTYPE...>
    text = re.sub(r"<[^>]+>", " ", text)
    # Loại bỏ URL trần nếu lọt vào nội dung mô tả
    text = re.sub(r"https?://\S+", "", text)
    # Chuẩn hóa khoảng trắng và ký tự đặc biệt
    text = text.replace("&quot;", '"').replace("&amp;", "&").replace("&apos;", "'")
    text = text.replace("&#39;", "'").replace("&nbsp;", " ").replace("&mdash;", "-")
    return re.sub(r"\s+", " ", text).strip()

def parse_pub_date(pub_date_str: str) -> str:
    """Chuẩn hóa ngày giờ phát hành tin tức sang định dạng chuẩn YYYY-MM-DD HH:MM EST/GMT."""
    if not pub_date_str:
        return datetime.now().strftime("%Y-%m-%d %H:%M EST")
    try:
        dt = parsedate_to_datetime(pub_date_str)
        return dt.strftime("%Y-%m-%d %H:%M GMT")
    except Exception:
        # Fallback regex nếu chuỗi ngày không chuẩn RFC 2822
        m = re.search(r"(\d{4}-\d{2}-\d{2})", pub_date_str)
        if m:
            return f"{m.group(1)} 09:15 EST"
        return datetime.now().strftime("%Y-%m-%d %H:%M EST")

def classify_macro_category(title: str, desc: str = "") -> str:
    """Phân loại chủ đề vĩ mô tự động dựa trên từ khóa tài chính."""
    text = (title + " " + desc).lower()
    if any(k in text for k in ["fed", "fomc", "rate", "powell", "monetary", "central bank", "yield", "treasury", "bond"]):
        return "Chính sách Tiền tệ & Lãi suất"
    elif any(k in text for k in ["inflation", "cpi", "ppi", "pce", "gdp", "recession", "jobs", "unemployment", "payroll"]):
        return "Lạm phát & Kinh tế Vĩ mô"
    elif any(k in text for k in ["earnings", "revenue", "profit", "tech", "ai", "cloud", "guidance", "apple", "nvidia", "microsoft"]):
        return "Kết quả Kinh doanh & Doanh nghiệp"
    elif any(k in text for k in ["oil", "crude", "energy", "opec", "war", "geopolit", "tariff", "china", "middle east"]):
        return "Hàng hóa & Địa chính trị"
    return "Thị trường Chứng khoán & Phố Wall"

def determine_impact_tier(title: str) -> str:
    """Đánh giá mức độ ảnh hưởng thị trường (HIGH, MEDIUM, LOW)."""
    t = title.lower()
    if any(k in t for k in ["fed", "powell", "hike", "cut", "inflation", "cpi", "jobs report", "plunge", "surge", "crash", "war", "tariff", "recession"]):
        return "HIGH"
    elif any(k in t for k in ["earnings", "revenue", "profit", "stocks", "rally", "drop", "treasury", "yield", "sp500", "nasdaq"]):
        return "MEDIUM"
    return "LOW"

def fetch_live_financial_news(limit: int = 25) -> Tuple[List[Dict], bool]:
    """
    Thu thập tin tức tài chính trực tuyến thời gian thực từ các nguồn RSS mở.
    
    Returns:
        (news_items_list, is_fallback)
        is_fallback = True nếu không thể kết nối mạng và chuyển sang dữ liệu dự phòng.
    """
    live_items = []
    seen_titles = set()

    for feed_name, feed_url in LIVE_RSS_FEEDS:
        try:
            req = urllib.request.Request(feed_url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=5) as resp:
                xml_content = resp.read().decode("utf-8", errors="ignore")
                
                # Bóc tách từng thẻ <item> bằng Regex chịu lỗi cao
                item_matches = re.findall(r"<item>(.*?)</item>", xml_content, re.DOTALL)
                
                for block in item_matches:
                    title_m = re.search(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", block, re.DOTALL)
                    link_m = re.search(r"<link>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</link>", block, re.DOTALL)
                    pub_m = re.search(r"<pubDate>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</pubDate>", block, re.DOTALL)
                    desc_m = re.search(r"<description>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</description>", block, re.DOTALL)
                    source_m = re.search(r"<source[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</source>", block, re.DOTALL)

                    raw_title = clean_html_text(title_m.group(1)) if title_m else ""
                    if not raw_title or len(raw_title) < 15:
                        continue

                    # Tránh trùng lặp tiêu đề
                    norm_key = re.sub(r"[^\w\s]", "", raw_title.lower())[:45]
                    if norm_key in seen_titles:
                        continue
                    seen_titles.add(norm_key)

                    link = link_m.group(1).strip() if link_m else "https://finance.yahoo.com/"
                    pub_date = parse_pub_date(pub_m.group(1).strip() if pub_m else "")
                    desc = clean_html_text(desc_m.group(1)) if desc_m else raw_title
                    
                    # Nguồn tin tức
                    src = clean_html_text(source_m.group(1)) if source_m else feed_name

                    live_items.append({
                        "title": raw_title,
                        "description": desc if len(desc) > 20 else f"Bản tin tài chính vĩ mô ghi nhận từ {src}.",
                        "source": src,
                        "time": pub_date,
                        "category": classify_macro_category(raw_title, desc),
                        "impact_tier": determine_impact_tier(raw_title),
                        "url": link,
                        "is_live": True
                    })

                    if len(live_items) >= limit:
                        break
        except Exception as e:
            logger.warning(f"Lỗi truy vấn nguồn RSS '{feed_name}': {e}")
            continue

        if len(live_items) >= limit:
            break

    # Nếu thành công có ít nhất 5 bản tin thời sự
    if len(live_items) >= 5:
        logger.info(f"Đã thu thập thành công {len(live_items)} bản tin tài chính trực tuyến thời gian thực.")
        return live_items[:limit], False

    # Cơ chế dự phòng (Failover Fallback): lấy từ file mẫu dữ liệu nếu mạng mất kết nối
    logger.warning("Không thể truy cập RSS trực tuyến. Chuyển sang cơ chế dữ liệu mẫu dự phòng.")
    from utils.finbert_inference import fetch_filtered_news
    fallback_news = fetch_filtered_news("Hôm nay (Phiên mới nhất)")
    for n in fallback_news:
        n["is_live"] = False
    return fallback_news, True
