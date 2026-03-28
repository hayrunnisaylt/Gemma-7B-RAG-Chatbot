import os, re, time, json, hashlib
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

START_URLS = [
    "https://enstitu.btu.edu.tr/tr",
    "https://enstitu.btu.edu.tr/tr/sayfa/basvuru-kriterleri",
    "https://enstitu.btu.edu.tr/tr/sayfa/tez_savunmalari",
    "https://enstitu.btu.edu.tr/tr/sayfa/detay/150/sikca-sorulan-sorular-sss",
    "https://enstitu.btu.edu.tr/tr/sayfa/detay/164/tezli-yuksek-lisans-sureci",
    "https://enstitu.btu.edu.tr/tr/sayfa/detay/155/tezsiz-yuksek-lisans-sureci",
    "https://enstitu.btu.edu.tr/tr/sayfa/doktora",
    "https://enstitu.btu.edu.tr/tr/sayfa/detay/149/ders-acma-kriterleri",
    "https://enstitu.btu.edu.tr/tr/sayfa/detay/4749/akademik-takvim",
]

ALLOWED_DOMAINS = {"enstitu.btu.edu.tr", "btu.edu.tr"}

OUT_DIR = "./data/btu/pages"
INDEX_PATH = "./data/btu/index.jsonl"

KEYWORDS = [
    "lisansüstü", "lisansustu", "yüksek lisans", "yuksek lisans",
    "enstitü", "enstitu", "başvuru", "basvuru", "kontenjan",
    "tez", "savunma", "jüri", "juri", "akademik takvim", "yönetmelik", "yonetmelik",
    "akts", "ects", "kredi", "ders yükü", "ders kaydı", "azami süre", "mezuniyet",
    "ales", "agno", "yabancı dil", "doktora", "sss", "sıkça sorulan",
]

# ✅ EKLENDİ: Bu URL'ler kesinlikle kaydedilsin (keyword filtresi olmadan)
PRIORITY_URLS = {
    "sikca-sorulan-sorular",
    "basvuru-kriterleri",
    "tezli-yuksek-lisans",
    "tezsiz-yuksek-lisans",
    "doktora",
    "akademik-takvim",
    "ders-acma-kriterleri",
}

session = requests.Session()
session.headers.update({
    "User-Agent": "BTU-RAG-Crawler/1.0 (+offline-ingest)"
})


def ok_domain(url: str) -> bool:
    d = urlparse(url).netloc.lower()
    return any(d == dom or d.endswith("." + dom) for dom in ALLOWED_DOMAINS)


def normalize(url: str) -> str:
    u = urlparse(url)
    return u._replace(fragment="", query="").geturl()


def slugify(url: str) -> str:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    path = urlparse(url).path.strip("/").replace("/", "_")
    if not path:
        path = "home"
    return f"{path}_{h}.md"


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # ✅ main/article/content bloğunu tercih et
    main = (
        soup.find("main") or
        soup.find("article") or
        soup.find("div", class_=re.compile(r"content|main|icerik", re.I))
    )
    if main:
        text = main.get_text("\n")
    else:
        body = soup.find("body")
        text = (body.get_text("\n") if body else soup.get_text("\n"))

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def is_relevant(text: str, url: str) -> bool:
    # Öncelikli URL'ler her zaman kaydedilir
    if any(p in url for p in PRIORITY_URLS):
        return True
    hay = (url + " " + text).lower()
    return any(k in hay for k in KEYWORDS)


def save_page(url: str, title: str, text: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    fname = slugify(url)
    fpath = os.path.join(OUT_DIR, fname)

    content = (
        f"# {title or 'BTU Sayfa'}\n\n"
        f"- URL: {url}\n\n"
        f"---\n\n"
        f"{text}\n"
    )
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)

    with open(INDEX_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"url": url, "file": fpath, "title": title}, ensure_ascii=False) + "\n")


def crawl(max_pages=500, delay=0.6):
    seen = set()
    q = [normalize(u) for u in START_URLS]

    saved = 0
    while q and len(seen) < max_pages:
        url = q.pop(0)
        url = normalize(url)
        if url in seen:
            continue
        if not ok_domain(url):
            continue

        seen.add(url)

        try:
            r = session.get(url, timeout=20)
            if r.status_code != 200 or "text/html" not in r.headers.get("Content-Type", ""):
                continue

            soup = BeautifulSoup(r.text, "html.parser")
            title = (soup.title.get_text(strip=True) if soup.title else "").strip()
            text = html_to_text(r.text)

            if is_relevant(text, url):
                save_page(url, title, text)
                saved += 1
                print(f"✅ Kaydedildi [{saved}]: {url}")

            for a in soup.select("a[href]"):
                href = a.get("href")
                if not href:
                    continue
                nxt = normalize(urljoin(url, href))
                if ok_domain(nxt) and nxt not in seen:
                    q.append(nxt)

            time.sleep(delay)

        except Exception as e:
            print(f"⚠ Hata: {url} → {e}")

    print(f"\n✅ Bitti. Gezilen: {len(seen)} | Kaydedilen: {saved}")
    print(f"   Çıktı dizini: {OUT_DIR}")


if __name__ == "__main__":
    crawl()
