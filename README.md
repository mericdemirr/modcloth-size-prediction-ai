# End-to-End Clothing Fit Prediction AI (ModCloth)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-orange)

Bu proje, e-ticaret (ModCloth) verilerini kullanarak müşterilere **beden tavsiyesi veren** ve işletme için **iade maliyetlerini simüle eden** uçtan uca bir Yapay Zeka uygulamasıdır.

Sadece bir tahminleme modeli değil, aynı zamanda **İş Zekası (Business Intelligence)** ve **Hibrit Karar Mekanizması** içeren bir karar destek sistemidir.

---

## Projenin Amacı ve İş Değeri (Business Value)

Online alışverişte en büyük problem **iadelerdir**. Müşteriler bedenlerinden emin olamadıkları için yanlış ürün alır ve iade ederler. Bu durum şirkete ciddi lojistik ve operasyonel maliyet yaratır.

**Bu projenin çözümü:**
1.  **Müşteri İçin:** Vücut ölçülerine, ürün özelliklerine ve yorumlarına (NLP) göre en doğru bedeni önerir.
2.  **Satıcı İçin:** Yapay zekanın kaç iadeyi önlediğini ve şirkete ne kadar **para kazandırdığını (ROI)** hesaplayan bir simülasyon paneli sunar.

---

## Öne Çıkan Özellikler

### 1. Hibrit Zeka (Hybrid AI Architecture) 
Proje sadece makine öğrenmesine güvenmez. **Yapay Zeka (LightGBM)** ile **Fiziksel İş Kuralları (Business Rules)** birlikte çalışır.
* *Örnek:* Model hata yapsa bile, fiziksel olarak imkansız durumlarda (Örn: Çok geniş basen - Çok küçük beden) **Guardrail** sistemi devreye girer ve müşteriyi uyarır.

### 2. Satıcı Paneli (Business Dashboard) 
Uygulamanın sol panelinde mağaza yöneticileri için bir simülasyon aracı bulunur.
* Aylık satış adedi ve iade maliyeti girilerek, yapay zekanın şirkete sağladığı **Net Tasarruf ($)** canlı olarak hesaplanır.

### 3. Çok Dilli NLP ve Duygu Analizi 
Kullanıcı yorumlarını analiz ederek beden uyumunu tahmin eder.
* **Türkçe ve İngilizce** desteği vardır. Türkçe girilen yorumlar arka planda İngilizceye çevrilir (`deep-translator`) ve duygu analizi (`TextBlob`) yapılır.

---

## Model Performansı

* **Algoritma:** LightGBM (Custom Class Weights & Regularization)
* **Accuracy:** ~%71
* **Small/Large Recall:** İade riski taşıyan ürünleri yakalama başarısı optimize edilmiştir.
* **Güven Skoru:** Kullanıcıya tahminin ne kadar güvenilir olduğu (% Olasılık) gösterilir.

---

## 🛠️ Kurulum ve Çalıştırma

Projeyi yerel makinenizde çalıştırmak için adımları takip edin:

### 1. Depoyu İndirin
```bash
git clone https://github.com/mericdemirr/modcloth-size-prediction-ai.git
cd modcloth-size-prediction-ai
```
### 2. Gerekli Kütüphaneleri Yükleyin
```bash
pip install -r requirements.txt
```
### 3. Veri Setini İndirin (⚠️ Önemli)
Dosya boyutu nedeniyle veri seti bu depoya yüklenmemiştir.

 ---1.ModCloth Dataset'ini Kaggle'dan İndirin.
 
 ---2.İndirdiğiniz modcloth_final_data.json dosyasını projenin içindeki data/ klasörüne atın.

### 4. Modeli Eğitin
Eğitilmiş model dosyaları (.pkl) boyut sınırı nedeniyle yüklenmemiştir. Modeli oluşturmak için:

```bash
python src/train.py
```
Bu komut; veriyi temizler, özellikleri (feature engineering) üretir ve models/ klasörüne yapay zeka modelini kaydeder.

### 5. Uygulamayı Başlatın
Web arayüzünü ve satıcı panelini açmak için:
```bash

streamlit run app.py
```
## Proje Yapısı
```bash

modcloth-ai/
├── data/                  # Ham veri setinin (json) konulacağı klasör
├── models/                # Eğitilen model (.pkl) dosyalarının kaydedildiği yer
├── src/                   # Kaynak kodlar
│   ├── data_prep.py       # Veri temizleme & Feature Engineering işlemleri
│   └── train.py           # LightGBM model eğitimi ve validasyonu
├── app.py                 # Streamlit Web Arayüzü (Frontend & Dashboard)
├── requirements.txt       # Proje bağımlılıkları
└── README.md              # Proje dokümantasyonu
```
## Geliştirici
Bu proje ML Bootcamp bitirme projesi olarak geliştirilmiştir.
