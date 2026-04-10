# 🏠 House Rent Prediction

Bu proje, **makine öğrenmesi** ve **derin öğrenme** algoritmaları kullanarak ev kira fiyatlarını tahmin eden bir Python uygulamasıdır. Hindistan'daki beş büyük şehirdeki emlak verilerine dayanarak kira tahmini yapmaktadır.

## 📁 Proje Yapısı

```
house-rent-prediction/
├── app.py                      # Streamlit web uygulaması
├── HouseRentPrediction.ipynb   # Model eğitimi ve analiz
├── House_Rent_Dataset.csv      # Ev kira veri seti
├── houserent.joblib            # Eğitilmiş ML modeli (ana model)
├── houserent_bundle.pkl        # Model bundle (scaler + model)
├── house_rent.joblib           # Alternatif ML modeli
├── house_rent.pkl              # Alternatif model (pickle)
├── rent_model.h5               # Derin öğrenme modeli (Keras)
├── requirements.txt            # Python bağımlılıkları
└── README.md                   # Proje dokümantasyonu
```

## 🚀 Kurulum

### 1. Repoyu klonlayın
```bash
git clone https://github.com/tugcesi/house-rent-prediction.git
cd house-rent-prediction
```

### 2. Bağımlılıkları yükleyin
```bash
pip install -r requirements.txt
```

## ▶️ Kullanım

### Streamlit Uygulaması
```bash
streamlit run app.py
```
Tarayıcınızda `http://localhost:8501` adresine gidin.

### Jupyter Notebook
```bash
jupyter notebook HouseRentPrediction.ipynb
```

## 🛠️ Özellikler

- 🏠 Ev özelliklerini girerek anlık kira tahmini
- 🌆 5 farklı şehir desteği: Chennai, Delhi, Hyderabad, Kolkata, Mumbai
- 🎛️ BHK, alan, banyo, mobilya durumu ve kiracı türü parametreleri
- 📊 Gauge chart ile görsel kira seviyesi gösterimi
- 💡 Akıllı öneriler ve kira analizi (aylık / yıllık / ft² başına)

## 📊 Veri Seti

`House_Rent_Dataset.csv` veri seti, Hindistan'daki ev kiralama ilanlarını içermekte olup aşağıdaki özellikleri kapsamaktadır:

| Özellik | Açıklama |
|---------|----------|
| `BHK` | Yatak odası, salon ve mutfak sayısı |
| `Rent` | Aylık kira bedeli (₹) - Hedef değişken |
| `Size` | Ev alanı (sq ft) |
| `Floor` | Kat bilgisi |
| `Area Type` | Alan türü (Residential/Commercial) |
| `Area Locality` | Mahalle / Bölge |
| `City` | Şehir (Chennai, Delhi, Hyderabad, Kolkata, Mumbai) |
| `Furnishing Status` | Mobilya durumu (Furnished / Semi / Unfurnished) |
| `Tenant Preferred` | Tercih edilen kiracı türü |
| `Bathroom` | Banyo sayısı |
| `Point of Contact` | İletişim yöntemi |

## 🤖 Kullanılan Modeller

- ✅ **Random Forest Regressor** (Ana model - `houserent.joblib`)
- ✅ **Gradient Boosting / XGBoost**
- ✅ **Linear Regression**
- ✅ **Deep Learning - Neural Network** (`rent_model.h5`)

## 🏦 Şehirlere Göre Kira Kategorileri

| Kategori | Aylık Kira |
|----------|------------|
| 💰 Ekonomik | < ₹20.000 |
| 💵 Uygun Fiyat | ₹20.000 - ₹50.000 |
| 💳 Premium | ₹50.000 - ₹100.000 |
| 👑 Lüks | > ₹100.000 |

## 📦 Gereksinimler

- Python 3.8+
- scikit-learn
- TensorFlow / Keras
- Streamlit
- Pandas
- NumPy
- Plotly
- Joblib

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.