from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Ayarlar
CHROMA_PATH = "./chroma_db"
TEST_QUERY = "tez savunma"  # İçinde bu geçen belgeleri getir

def veritabanini_oku():
    print("Veritabanı kontrol ediliyor...")
    
    # 1. Bağlantı
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # 2. Arama Yap
    sonuclar = db.similarity_search(TEST_QUERY, k=3)

    # 3. Sonuçları Yazdır
    if not sonuclar:
        print(" HİÇBİR ŞEY BULUNAMADI! Veritabanı boş veya kelime eşleşmedi.")
    else:
        print(f"✅ {len(sonuclar)} adet parça bulundu:\n")
        for i, doc in enumerate(sonuclar):
            source = doc.metadata.get("source", "Bilinmiyor")
            content = doc.page_content.replace("\n", " ")[:300] # İlk 300 karakter
            print(f"--- SONUÇ {i+1} ---")
            print(f"Kaynak: {source}")
            print(f" İçerik: {content}...") # Devamı var demek
            print("-" * 30)

if __name__ == "__main__":
    veritabanini_oku()