## 📊 Recurrent Neural Network (RNN) Karşılaştırması: NumPy vs PyTorch GRU

Bu proje, doğal dil işleme (NLP) bağlamında duygu analizi yapmak için iki farklı RNN yaklaşımını karşılaştırır:

* `Rnn.py`: NumPy ile sıfırdan yazılmış bir Vanilla RNN (Many-to-One mimarisiyle).
* `Rnn3.py`: PyTorch framework'ü ile geliştirilmiş, embedding ve dropout destekli bir GRU modeli.

Her iki model de ikili sınıflandırma (pozitif/negatif duygu) görevini yerine getirmek üzere eğitilmiş ve değerlendirilmiştir.

---

## 🧠 Kullanılan Modeller

### 1. `Rnn.py` – NumPy ile Vanilla RNN

#### 🔧 Özellikler:

* Tamamen NumPy ile yazılmıştır.
* Xavier (Glorot) ağırlık başlatma.
* Truncated Backpropagation Through Time (BPTT) ile eğitim.
* AdaGrad optimizasyonu.
* One-hot vektörlerle giriş.
* Binary cross-entropy loss + sigmoid aktivasyon.
* Eğitim, doğruluk ve kayıp grafiklerinin görselleştirilmesi.
* Precision, Recall ve F1-Score dahil detaylı metrikler.

#### 📌 Öne Çıkan Fonksiyonlar:

* `RNN`: Model sınıfı.
* `forward`: Girdi dizisi için ileri geçiş (forward pass).
* `backprop`: Truncated BPTT ile geri yayılım (backpropagation).
* `train_rnn`: Eğitim döngüsü (mini-batch destekli).
* `evaluate_rnn`: Test verisi üzerinde doğruluk, loss ve karışıklık matrisi (confusion matrix) hesaplama.
* `plot_results`: Kayıp ve doğruluk grafiklerini kaydeder (`improved_rnn_results.png`).

#### 🔎 Girdi-Çıktı Formatı:

* **Girdi**: one-hot vektör dizisi
* **Çıktı**: Sigmoid aktivasyon sonucu olasılık değeri (0–1 arası)

---

### 2. `Rnn3.py` – PyTorch ile GRU Modeli

#### 🔧 Özellikler:

* PyTorch framework kullanılarak geliştirilmiştir.
* `nn.Embedding`, `nn.GRU`, `nn.Dropout` katmanları.
* GPU desteği ile daha hızlı eğitim.
* `CrossEntropyLoss` ve `Adam` optimizasyonu.
* `Dataset` ve `DataLoader` ile veri yükleme.
* Eğitim ve test süreci boyunca epoch bazlı metrik takibi.
* Eğitim grafikleri ve metrik görselleştirilmesi (`pytorch_gru_results.png`).

#### 📌 Öne Çıkan Yapılar:

* `SentimentDataset`: PyTorch veri kümesi sınıfı (padding içerir).
* `GRUModel`: Embedding + GRU + Dropout + Linear çıkış katmanı.
* `train_model`, `evaluate_model`: Eğitim ve değerlendirme döngüleri.
* `train_and_evaluate_pytorch_gru`: Tüm süreci başlatan ana fonksiyon.
* `confusion_matrix` ve F1 gibi metrikler dahil değerlendirme.

#### 🔎 Girdi-Çıktı Formatı:

* **Girdi**: Integer indeks dizileri (embedding katmanına verilir)
* **Çıktı**: Softmax uygulanmış iki sınıflı logit (CrossEntropyLoss için)

---

## ⚙️ Kurulum ve Kullanım

### 1. Gerekli Paketler

```bash
pip install numpy matplotlib torch scikit-learn
```

### 2. Veri Dosyası

Her iki script de `data.py` adlı bir modül bekler. Aşağıda örnek bir yapı yer almaktadır:

```python
# data.py
train_data = {
    "I love this movie": True,
    "Terrible acting and story": False,
    # ...
}

test_data = {
    "Amazing experience": True,
    "Not worth watching": False,
    # ...
}

# PyTorch için
def prepare_data():
    # ...
    return X_train, y_train, X_test, y_test, vocab_size, word_to_index
```

### 3. Modelleri Çalıştırma

#### NumPy RNN

```bash
python Rnn.py
```

#### PyTorch GRU

```bash
python Rnn3.py
```

---

## 📈 Eğitim ve Test Sonuçları

### ✅ NumPy Vanilla RNN

* **Vocabulary size**: 18
* **Training data size**: 58
* **Test data size**: 20
* **Test Accuracy**: 1.0000
* **Test Loss**: 0.0676
* **Precision**: 1.0000
* **Recall**: 1.0000
* **F1 Score**: 1.0000
* **Confusion Matrix**:

```
                 Predicted
                 Negative  Positive
Actual Negative    10        0
      Positive     0        10
```

### ✅ PyTorch GRU

* **Test Accuracy**: 0.9000
* **Test Loss**: 0.1755
* **Precision**: 1.0000
* **Recall**: 0.8000
* **F1 Score**: 0.8889
* **Confusion Matrix**:

```
              Predicted
              Positive  Negative
Actual Positive     8         2
Actual Negative     0        10
```

---


## 🏁 Sonuç

Bu proje, klasik bir RNN mimarisinin temel prensiplerini öğrenmek isteyenler ve modern, güçlü bir RNN olan GRU'yu uygulamalı şekilde deneyimlemek isteyenler için ideal bir kaynaktır.

Her iki model de kısa metin sınıflandırma ve duygu analizi görevlerinde başarıyla uygulanabilir.
