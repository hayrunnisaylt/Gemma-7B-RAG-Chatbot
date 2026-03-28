import os
import json
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

# -----------------------------
# AYARLAR
# -----------------------------
PERSIST_DIRECTORY = "./chroma_db"
MODEL_NAME = "gemma:7b" # Staj hedefine uygun olarak Gemma 7B veya elindeki sürümü kullanabilirsin

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

TEMPERATURE = 0.3   # 0.0'dan 0.3'e çıkardık. Samimiyet ve doğallık için biraz esneklik şart.
NUM_PREDICT = 350   # Detaylı açıklama yapabilmesi için token sınırını biraz artırdık.
NUM_CTX = 2048      # Bağlam ve uzun cevaplar için pencereyi biraz genişlettik.

DEFAULT_K = 3       
GENERAL_K = 3      

CHAT_LOG_PATH = "./logs/chat_log.jsonl"

# -----------------------------
# APP BAŞLATMA
# -----------------------------
app = Flask(__name__)
CORS(app)
os.makedirs(os.path.dirname(CHAT_LOG_PATH), exist_ok=True)

print("Sistem başlatılıyor.")

embedding_function = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vector_db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_function,
)

llm = Ollama(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    num_predict=NUM_PREDICT,
    num_ctx=NUM_CTX,
)

# SAMİMİ VE DETAYLI PROMPT
prompt_template = """Sen Bursa Teknik Üniversitesi (BTÜ) öğrencilerine ve adaylarına yardımcı olan profesyonel bir asistansın.
Görevlerin:
- Yalnızca aşağıdaki <baglam> etiketleri arasındaki bilgileri kullanarak cevap ver.
- Cevabını doğrudan ver. Asla "bağlama göre", "verilen metine göre", "kurallara göre" veya "kaynaklardan aldığım bilgiye göre" gibi cümleler kurma.
- Eğer sorunun cevabı <baglam> içinde kesinlikle yoksa, sadece "Elimdeki güncel belgelerde bu bilgiye ulaşamadım." de ve asla bilgi uydurma.
- Bilgileri okunabilir, akıcı paragraflar veya gerektiğinde normal maddeler halinde sun.

<baglam>
{context}
</baglam>

Soru: {question}
Cevap:"""

print(f"Sistem Hazır, Model: {MODEL_NAME}")

# -----------------------------
# YARDIMCI FONKSİYONLAR
# -----------------------------

def write_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def format_sources(source_documents):
    sources = []
    seen = set()
    for d in source_documents:
        md = d.metadata or {}
        source = md.get("source", "Bilinmiyor")
        page = md.get("page", None)
        chunk_id = md.get("chunk_id", None)
        key = (source, page, chunk_id)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source": source, "page": page, "chunk_id": chunk_id})
    return sources

def is_general_question(q: str) -> bool:
    keywords = [
        "başvuru", "kriter", "şart", "koşul",
        "gereksinim", "lazım", "gerekir",
        "bilgi", "özet", "neler", "hangileri",
        "adım", "süreç", "nasıl",
        "kim", "kimdir", "nerede", "rektör", "dekan"
    ]
    return any(kw in q.lower() for kw in keywords)

def rewrite_query(question: str) -> str:
    """RAG Keyword Booster: Sistemin bilmediği kısaltmaları ve eksik kelimeleri sorguya ekler."""
    q = question.lower().strip()
    
    greetings = ["merhaba", "selam", "nasılsın", "günaydın"]
    if any(g in q for g in greetings) and len(q.split()) < 4:
        return question

    expanded_query = question
    
    if "çap" in q or "cap" in q:
        expanded_query += " Çift Anadal Programı"
    if "yandal" in q:
        expanded_query += " Yandal Programı"
    if "erasmus" in q:
        expanded_query += " Erasmus değişim programı avrupa"
        
    return expanded_query

# -----------------------------
# ENDPOINTS
# -----------------------------

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(".", "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    user_question = (data.get("message") or "").strip()

    if not user_question:
        return jsonify({"error": "Soru boş olamaz"}), 400

    try:
        k_used = GENERAL_K if is_general_question(user_question) else DEFAULT_K
        query = rewrite_query(user_question)

        retriever = vector_db.as_retriever(
            search_type="mmr", 
            search_kwargs={
                "k": 5,             # Modele gidecek son belge sayısı
                "fetch_k": 20,      # Arka planda taranacak havuz (çeşitliliği artırır)
                "lambda_mult": 0.7  # Benzerlik ve çeşitlilik dengesi
            }
        )
        src_docs = retriever.invoke(query)

        context_parts = []
        for doc in src_docs:

            context_parts.append(doc.page_content)

        unique_urls = list(dict.fromkeys(
            (doc.metadata or {}).get("source", "")
            for doc in src_docs
            if (doc.metadata or {}).get("source")
        ))
        
        url_block = ""
        if unique_urls:
            url_block = "\n\nKAYNAKLAR:\n" + "\n".join(f"- {u}" for u in unique_urls)
            
        full_context = "\n\n---\n\n".join(context_parts) + url_block

        filled_prompt = prompt_template.replace("{context}", full_context).replace("{question}", user_question)
        
        answer = (llm.invoke(filled_prompt) or "").strip()

        sources = format_sources(src_docs)

        write_jsonl(CHAT_LOG_PATH, {
            "ts": datetime.utcnow().isoformat() + "Z",
            "question": user_question,
            "answer": answer,
            "model": MODEL_NAME
        })

        return jsonify({"reply": answer, "sources": sources})

    except Exception as e:
        print(f"HATA: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)