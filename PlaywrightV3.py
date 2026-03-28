import time
import json
import psutil
import os
import threading
import requests
from urllib.parse import urlparse, urljoin
from playwright.sync_api import sync_playwright

class PerformanceMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.keep_running = True
        self.cpu_measurements = []
        self.mem_measurements = []
        
    def monitor(self):
        self.process.cpu_percent(interval=None)
        while self.keep_running:
            try:
                total_mem = self.process.memory_info().rss
                total_cpu = self.process.cpu_percent(interval=None)
                for child in self.process.children(recursive=True):
                    try:
                        total_mem += child.memory_info().rss
                        total_cpu += child.cpu_percent(interval=None)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                self.mem_measurements.append(total_mem / (1024 * 1024))
                self.cpu_measurements.append(total_cpu)
            except Exception:
                pass
            time.sleep(0.5)

def download_pdf(url, folder):
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
        # URL'den dosya adını güvenli bir şekilde al
        filename = os.path.join(folder, url.split("/")[-1].split("?")[0])
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
    except Exception as e:
        print(f"PDF indirme hatası ({url}): {e}")

def run_playwright_btu():
    print("Playwright BTÜ Akıllı Tarama Başlatılıyor...")
    
    # --- YAPILANDIRMA ---
    START_URL = "https://btu.edu.tr/" # Öğrenci işleri hedeflenecekse https://oidb.btu.edu.tr/ yapılabilir
    BASE_DOMAIN = "btu.edu.tr"
    MAX_PAGES = 500 # Daha hedefli taradığımız için sayıyı optimize edebiliriz
    
    # RAG için Değerli/Değersiz Kelime Filtreleri
    USEFUL_PDF_KEYWORDS = ["yonetmelik", "mevzuat", "kilavuz", "ogrenci", "akademik", "sss", "rehber", "belge", "yonerge", "ders-plani",]
    JUNK_URL_KEYWORDS = ["haber", "etkinlik", "galeri", "foto", "video", "arsiv", "duyuru", "takvim/gun", "uye-ol"]

    def is_useful_url(url_path):
        """Gereksiz sayfalara (haber, galeri vb.) girmeyi engeller."""
        url_lower = url_path.lower()
        return not any(junk in url_lower for junk in JUNK_URL_KEYWORDS)

    def is_useful_pdf(pdf_url):
        """Sadece bilgi içeren, yönetmelik tarzı PDF'leri seçer."""
        url_lower = pdf_url.lower()
        # İsimde yararlı kelime var mı VE gereksiz kelime yok mu?
        return any(useful in url_lower for useful in USEFUL_PDF_KEYWORDS) and not any(junk in url_lower for junk in JUNK_URL_KEYWORDS)

    metrics = {
        "tool": "Playwright_BTU",
        "start_url": START_URL,
        "execution_time_seconds": 0,
        "cpu_percent_max": 0,
        "memory_usage_mb_max": 0,
        "pages_crawled": 0,
        "pdfs_found": 0,
        "success": False,
        "error": None
    }
    
    monitor = PerformanceMonitor()
    monitor_thread = threading.Thread(target=monitor.monitor)
    monitor_thread.start()
    
    start_time = time.time()
    extracted_data = []
    pdf_links = set()
    visited = set()
    queue = [metrics["start_url"]]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = os.path.join(script_dir, "BTU_PDF_Belgeleri")
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            
            # CSS ve gereksiz kaynakları yüklemeyi reddet (Hızlandırır)
            page.route("**/*", lambda route: route.abort() if route.request.resource_type in ["image", "media", "font", "stylesheet"] else route.continue_())

            while queue and len(visited) < MAX_PAGES:
                current_url = queue.pop(0)
                if current_url in visited:
                    continue
                    
                print(f"[BTÜ Tarama: {len(visited)+1}/{MAX_PAGES}] Taranıyor: {current_url}")
                try:
                    page.goto(current_url, wait_until="domcontentloaded", timeout=15000)
                    visited.add(current_url)
                    
                    page.wait_for_timeout(500) 
                    
                    # --- RAG İÇİN DOM TEMİZLİĞİ ---
                    # Menüleri, alt bilgileri ve javascriptleri silerek sadece asıl metni bırakıyoruz
                    page.evaluate("""() => {
                        const elementsToRemove = document.querySelectorAll('header, footer, nav, aside, script, style, noscript, .menu, .sidebar');
                        elementsToRemove.forEach(el => el.remove());
                    }""")
                    
                    page_text = page.locator("body").inner_text().strip()
                    
                    # Çok kısa sayfaları atla (Boş veya sadece resim olan sayfalar RAG'i bozar)
                    if len(page_text) > 150:
                        extracted_data.append({
                            "url": current_url,
                            "content_length": len(page_text),
                            "content": page_text # Tüm temizlenmiş metni saklıyoruz
                        })
                    
                    # Sayfadaki linkleri topla
                    hrefs = page.evaluate("""() => {
                        return Array.from(document.querySelectorAll('a')).map(a => a.href);
                    }""")
                    
                    for href in hrefs:
                        if not href: continue
                        
                        # Göreceli (relative) linkleri tam URL'ye çevir
                        full_url = urljoin(current_url, href)
                        parsed_href = urlparse(full_url)
                        
                        if full_url.lower().endswith(".pdf"):
                            if is_useful_pdf(full_url): # PDF Filtresi uygulanıyor
                                pdf_links.add(full_url)
                        else:
                            # Sadece BTÜ domainindeyse ve daha önce listeye eklenmediyse
                            if parsed_href.netloc.endswith(BASE_DOMAIN) and full_url not in visited and full_url not in queue:
                                if is_useful_url(parsed_href.path): # URL Filtresi uygulanıyor
                                    queue.append(full_url)
                                
                except Exception as e:
                    print(f"Sayfa atlandı: {current_url} -> {str(e)[:50]}")
            
            browser.close()
            
        print(f"\nTarama bitti. RAG için uygun {len(pdf_links)} eşsiz PDF bulundu. İndiriliyor...")
        for pdf_url in pdf_links:
            download_pdf(pdf_url, pdf_dir)
            
        metrics["pages_crawled"] = len(visited)
        metrics["pdfs_found"] = len(pdf_links)
        metrics["success"] = True
        
    except Exception as e:
        metrics["error"] = str(e)
        print(f"Kritik hata: {e}")
    finally:
        end_time = time.time()
        metrics["execution_time_seconds"] = round(end_time - start_time, 2)
        
        monitor.keep_running = False
        monitor_thread.join()
        
        if monitor.cpu_measurements:
            metrics["cpu_percent_max"] = round(max(monitor.cpu_measurements), 2)
        if monitor.mem_measurements:
            metrics["memory_usage_mb_max"] = round(max(monitor.mem_measurements), 2)
            
        metrics_file = os.path.join(script_dir, "btu_metrics.json")
        data_file = os.path.join(script_dir, "btu_rag_data.json")
        
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
            
        if extracted_data:
            with open(data_file, "w", encoding="utf-8") as f:
                json.dump(extracted_data, f, indent=4, ensure_ascii=False)
            
        print("\n=== BTÜ VERİ ÇEKME İŞLEMİ TAMAMLANDI ===")
        print(f"Metrikler: {metrics_file}")
        print(f"RAG Metin Verileri: {data_file}")
        print(f"Hedefli PDF'ler: {pdf_dir}")

if __name__ == "__main__":
    run_playwright_btu()