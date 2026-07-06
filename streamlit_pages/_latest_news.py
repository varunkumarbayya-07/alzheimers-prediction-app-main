import streamlit as st
import requests
import xml.etree.ElementTree as ET
import html
import re


# ── Free RSS feeds — no API key needed ───────────────────────────────────────
RSS_FEEDS = [
    {
        "url":    "https://www.nia.nih.gov/news/rss",
        "source": "National Institute on Aging (NIH)"
    },
    {
        "url":    "https://www.alzheimersresearchuk.org/news/feed/",
        "source": "Alzheimer's Research UK"
    },
    {
        "url":    "https://medicalxpress.com/rss-feed/search/?search=alzheimer",
        "source": "Medical Xpress"
    },
]

FALLBACK_ARTICLES = [
    {
        "title":       "Global Alzheimer's Cases Expected to Triple by 2050",
        "description": "The World Health Organization warns that the number of people living with "
                       "Alzheimer's disease could triple to 150 million by 2050 without significant intervention.",
        "url":         "https://www.who.int/",
        "source":      "WHO",
    },
    {
        "title":       "Researchers Discover New Biomarkers for Early Alzheimer's Detection",
        "description": "Scientists have identified a set of blood-based biomarkers that could enable earlier "
                       "and more accurate detection of Alzheimer's disease, potentially years before symptoms appear.",
        "url":         "https://www.alzheimers.gov/",
        "source":      "Alzheimer's.gov",
    },
    {
        "title":       "AI Model Achieves 85% Accuracy in Predicting Alzheimer's from MRI Scans",
        "description": "Researchers at a leading university have developed a deep learning model that analyzes "
                       "MRI brain scans to predict Alzheimer's disease with high accuracy.",
        "url":         "https://www.nih.gov/",
        "source":      "NIH",
    },
    {
        "title":       "Mediterranean Diet Linked to Lower Risk of Alzheimer's Disease",
        "description": "A new study spanning 10 years finds that adherence to a Mediterranean-style diet "
                       "is associated with a significantly lower risk of developing Alzheimer's disease.",
        "url":         "https://www.alz.org/",
        "source":      "Alzheimer's Association",
    },
    {
        "title":       "FDA Approves New Drug to Slow Progression of Early Alzheimer's",
        "description": "The FDA has granted approval to a new drug that has demonstrated the ability to "
                       "slow cognitive decline in patients with early-stage Alzheimer's disease.",
        "url":         "https://www.fda.gov/",
        "source":      "FDA",
    },
]


def _clean(text: str) -> str:
    """Strip HTML tags and decode HTML entities."""
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def _fetch_rss(feed_url: str, source: str, max_items: int = 5):
    headers = {"User-Agent": "Mozilla/5.0 (AlzheimersPredictionSystem/1.0)"}
    resp = requests.get(feed_url, timeout=8, headers=headers)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    channel = root.find("channel") or root
    articles = []
    for item in channel.findall("item")[:max_items]:
        title = _clean(item.findtext("title", ""))
        desc  = _clean(item.findtext("description", ""))
        link  = (item.findtext("link") or "").strip()
        if not link:
            link_el = item.find("link")
            if link_el is not None and link_el.tail:
                link = link_el.tail.strip()
        date  = item.findtext("pubDate", "")
        if title:
            articles.append({
                "title":       title,
                "description": desc[:280] + "…" if len(desc) > 280 else desc,
                "url":         link,
                "source":      source,
                "date":        date,
            })
    return articles


@st.cache_data(ttl=3600)          # cache for 1 hour
def get_live_news():
    all_articles = []
    for feed in RSS_FEEDS:
        try:
            articles = _fetch_rss(feed["url"], feed["source"])
            all_articles.extend(articles)
        except Exception:
            continue
    return all_articles if all_articles else None


# ── Page ──────────────────────────────────────────────────────────────────────
def latest_news():
    st.title("📰 Latest Alzheimer's News")
    st.write("Stay informed with the latest research and updates about Alzheimer's disease.")
    st.write("---")

    with st.spinner("Fetching latest news…"):
        articles = get_live_news()

    if articles:
        st.success(f"✅ Showing {len(articles)} live articles from research organisations.")
    else:
        st.info("⚠️ Could not reach news feeds right now — showing curated articles.")
        articles = FALLBACK_ARTICLES

    for art in articles:
        source = art.get("source", "")
        date   = art.get("date", "")
        date_str = f" · {date[:16]}" if date else ""

        st.markdown(f"""
            <div style="
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(139,92,246,0.25);
                border-left: 4px solid #7c3aed;
                border-radius: 10px;
                padding: 18px 20px;
                margin-bottom: 16px;
                backdrop-filter: blur(4px);
            ">
                <p style="font-size:0.75rem; color:#a78bfa; margin:0 0 6px 0;">
                    📡 {source}{date_str}
                </p>
                <h3 style="font-size:1.05rem; color:#f0e6ff; margin:0 0 8px 0;">
                    {art['title']}
                </h3>
                <p style="font-size:0.88rem; color:#d1d5db; margin:0 0 10px 0; line-height:1.5;">
                    {art['description']}
                </p>
                <a href="{art['url']}" target="_blank" style="
                    font-size:0.82rem;
                    color:#818cf8;
                    text-decoration:none;
                    font-weight:600;
                ">🔗 Read full article →</a>
            </div>
        """, unsafe_allow_html=True)

# Alias so streamlit_app.py import works unchanged
news_page = latest_news
