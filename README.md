## 📊 Hiperparametre ve Mimari Karşılaştırma Deneyleri

Ödev yönergeleri doğrultusunda, modelin farklı hiperparametreler ve mimariler altındaki performansını (Distribution/Generalization) ölçmek için 3 farklı deney gerçekleştirilmiştir. Tüm eğitim süreçleri CUDA donanım hızlandırması ile 10 Epoch üzerinden yapılmıştır.

| Deney                          | Model Mimarisi                    | Optimizer           | Öğrenme Oranı | Eğitim Süresi | Test Başarısı |
| :----------------------------- | :-------------------------------- | :------------------ | :------------ | :------------ | :------------ |
| **Deney 1 (Temel Model)**      | 2 Gizli Katman (256 -> 128)       | Adam                | 0.001         | ~222.00 sn    | ~%96.00       |
| **Deney 2 (Derin Mimari)**     | 3 Gizli Katman (256 -> 128 -> 64) | Adam                | 0.001         | 214.37 sn     | %94.82        |
| **Deney 3 (Farklı Optimizer)** | 2 Gizli Katman (256 -> 128)       | SGD (Momentum: 0.9) | 0.01          | 208.65 sn     | %95.71        |

_(Not: Geliştirme sürecine ait doğruluk ve kayıp grafikleri proje dizininde yer almaktadır.)_

### Sonuç Analizi ve Mühendislik Değerlendirmesi

1. **Model Derinliğinin Etkisi (Deney 1 vs Deney 2):**
   Gizli katman sayısını 2'den 3'e çıkarmak (araya 64 nöronluk yeni bir katman eklemek) test başarısını artırmak yerine %94.82'ye düşürmüştür. Bu durum, MNIST gibi 28x28 boyutlarında nispeten basit özelliklere sahip bir veri setinde model kapasitesini (parametre sayısını) gereğinden fazla artırmanın ezberlemeye (overfitting) yol açabileceğini kanıtlamaktadır. Daha derin ağlar, her zaman daha iyi genelleme (generalization) anlamına gelmemektedir.

2. **Optimizasyon Algoritmalarının Etkisi (Adam vs SGD):**
   Deney 3'te Adam yerine Klasik SGD (Stochastic Gradient Descent) kullanılmıştır. SGD'nin yavaş yakınsama dezavantajını kırmak için `0.9` oranında Momentum parametresi eklenmiştir. Sonuç olarak SGD, Adam'a oldukça yakın (%95.71) bir genelleme başarısı sergilemiş ve matris çarpım operasyonlarındaki sadeliği sayesinde donanımsal olarak en kısa eğitim süresine (208.65 sn) imza atmıştır.
