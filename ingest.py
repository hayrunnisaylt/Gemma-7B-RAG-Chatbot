import os
import re
import json
import shutil
from datetime import datetime
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

# -----------------------------
# AYARLAR 
# -----------------------------
PDF_DIR = "./BTU_PDF_Belgeleri"
JSON_PATH = "./btu_rag_data.json"
CHROMA_PATH = "./chroma_db"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

CHUNK_SIZE = 1000 
CHUNK_OVERLAP = 300

INGEST_LOG_PATH = "./logs/ingest_log.jsonl"

# -----------------------------
# TEMİZLEME & ÖN İŞLEME
# -----------------------------

NOISE_LINES = {
    "LİSANSÜSTÜ EĞİTİM ENSTİTÜSÜ", "ENSTİTÜ", "Search", "EN", "TR",
    "Hakkımızda", "Misyon ve Vizyon", "Mevzuat", "Yönetim", "Personel",
    "Akademik", "İdari", "İLETİŞİM", "Anasayfa", "Ana Sayfa",
    "Öğrenciyim", "Personelim", "Dış Paydaşım", "Aday Öğrenciyim",
    "/", "...", "Devamını Oku", "Tümünü Gör", "Bursa Teknik Üniversitesi"
}

def clean_text(text: str) -> str:
    """Metni gereksiz boşluklardan ve menü artıklarından temizler."""
    if not text:
        return ""

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in NOISE_LINES:
            continue

        if len(stripped) <= 2:
            continue

        if re.match(r"^\d+$", stripped):
            continue
        lines.append(stripped)


    cleaned_text = "\n".join(lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)

    return cleaned_text.strip()

# -----------------------------
# DÖKÜMAN YÜKLEME (DATA LOADERS)
# -----------------------------

def load_pdf_documents(pdf_dir: str) -> List[Document]:
    """Klasördeki tüm PDF'leri LangChain Document formatında yükler."""
    docs = []
    if not os.path.exists(pdf_dir):
        print(f"Uyarı: {pdf_dir} klasörü bulunamadı.")
        return docs
        
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(pdf_dir, filename)
            try:
                loader = PyPDFLoader(filepath)
                pdf_docs = loader.load()
                # Metadata zenginleştirme (Cevaplarda Kaynak Göstermek İçin Kritik)
                for d in pdf_docs:
                    d.metadata["source_type"] = "pdf"
                    d.metadata["file_name"] = filename
                    d.page_content = clean_text(d.page_content)
                docs.extend(pdf_docs)
            except Exception as e:
                print(f"  PDF Yüklenemedi: {filename} → {e}")
                
    print(f"PDF Klasöründen {len(docs)} sayfa yüklendi.")
    return docs

def load_json_documents(json_path: str) -> List[Document]:
    """Playwright JSON çıktısını LangChain Document formatına çevirir."""
    docs = []
    if not os.path.exists(json_path):
        print(f" Uyarı: {json_path} dosyası bulunamadı.")
        return docs

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for item in data:
            url = item.get("url", "")
            content = item.get("content", "")
            
            content = clean_text(content)
            
            if not content or len(content) < 50:
                continue

            docs.append(Document(
                page_content=content,
                metadata={
                    "source": url, 
                    "source_type": "web_page",
                    "file_name": "btu_website"
                }
            ))
            
    print(f"JSON'dan (Web) {len(docs)} sayfa/bölüm yüklendi.")
    return docs

# -----------------------------
# YARDIMCI DOSYA YAZICI
# -----------------------------
def write_jsonl(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# -----------------------------
# ANA İŞ AKIŞI (PIPELINE)
# -----------------------------
def ingest():
    print("Dokümanlar toplanıyor...")
    
    # 1. Verileri Çek
    pdf_docs = load_pdf_documents(PDF_DIR)
    json_docs = load_json_documents(JSON_PATH)
    all_docs = pdf_docs + json_docs

    if not all_docs:
        print("Hiç doküman bulunamadı. Lütfen Playwright scriptini çalıştırdığınızdan emin olun.")
        return

    print(f"Yüklenen toplam ham belge/sayfa sayısı: {len(all_docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    
    chunks = splitter.split_documents(all_docs)
    print(f"Toplam chunk (anlam parçacığı) sayısı: {len(chunks)}")


    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i

    # 3. Embedding Modeli Hazırlığı
    print(f"\nEmbedding modeli yükleniyor: {EMBEDDING_MODEL} (Local CPU/GPU)")
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}, 
        encode_kwargs={"normalize_embeddings": True},
    )

    # 4. ChromaDB Vektör Veritabanına Yazma
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Eski Chroma DB veritabanı temizlendi (Baştan yazılıyor).")

    print("Vektörler hesaplanıyor ve Chroma DB'ye kaydediliyor... (Bu işlem biraz sürebilir)")

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=CHROMA_PATH,
    )

    # 5. Loglama
    write_jsonl(INGEST_LOG_PATH, {
        "ts": datetime.utcnow().isoformat() + "Z",
        "event": "ingest_complete",
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "docs_loaded": len(all_docs),
        "chunks_created": len(chunks),
    })

    print("\nINGEST İŞLEMİ TAMAMLANDI! Vektör Veritabanı hazır.")
    print(f"   Toplam {len(chunks)} adet metin parçası ChromaDB'ye yerleştirildi.")

if __name__ == "__main__":
    ingest()