# On-Premise RAG-Based Intelligent Assistant
Bu proje, kurumsal verileri kullanarak yüksek doğrulukla yanıt veren, tamamen yerel (on-premise) altyapıda çalışan bir RAG (Retrieval-Augmented Generation) asistanıdır.

Sistem, halüsinasyonları en aza indirmek ve karmaşık sorguları doğru yanıtlamak için optimize edilmiş bir veri işleme ve getirme hattına sahiptir.

# Öne Çıkan Özellikler

Yüksek Doğruluk: 100 soruluk kapsamlı değerlendirme testlerinde yaklaşık %92 doğruluk oranı.  


Çok Dilli Destek: paraphrase-multilingual-mpnet-base-v2 modeli ile çok dilli embedding pipeline desteği.  


Dinamik Veri Toplama: Playwright entegrasyonu ile kurumsal web kaynaklarından gerçek zamanlı veri kazıma.  


Gelişmiş Query Logic: Sorgu yeniden yazma (query rewriting) ve özel prompt şablonları ile halüsinasyon ve prompt sızıntısı koruması.  


Yönetici Paneli: Proje yönetimi ve izleme süreçleri için sıfırdan tasarlanmış özel kontrol paneli.

# Teknik Yığın
Dil: Python   

Frameworkler: Flask (Backend), LangChain (Orkestrasyon)   

Vektör Veritabanı: ChromaDB   

Veri Kazıma: Playwright   

Embedding Modelleri: paraphrase-multilingual-mpnet-base-v2 
