import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# PAGE CONFIGURATION
st.set_page_config(
    page_title="🏠 House Rent Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .result-box {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        padding: 25px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
    }
    
    .result-rent {
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .result-label {
        font-size: 1.1em;
        opacity: 0.9;
    }
    
    .sidebar-title {
        font-size: 1.2em;
        font-weight: bold;
        color: #2ecc71;
        margin-bottom: 15px;
        text-align: center;
        padding: 15px;
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        border-radius: 10px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        font-weight: bold;
        font-size: 1.1em;
        width: 100%;
        height: 50px;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(46, 204, 113, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    """Model'i yükle"""
    try:
        model = joblib.load('houserent.joblib')
        expected_features = list(model.feature_names_in_)
        return model, expected_features
    except FileNotFoundError:
        st.error("❌ houserent.joblib bulunamadı!")
        return None, None
    except Exception as e:
        st.error(f"❌ Model yükleme hatası: {str(e)}")
        return None, None

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_rent_range_interpretation(predicted_rent):
    """Kira aralığı yorumlama"""
    if predicted_rent < 20000:
        return "💰 Ekonomik", "Düşük kira - Bütçe dostu"
    elif predicted_rent < 50000:
        return "💵 Uygun Fiyat", "Orta kira - İyi konumlar"
    elif predicted_rent < 100000:
        return "💳 Premium", "Yüksek kira - Merkez bölgeler"
    else:
        return "👑 Lüks", "Çok yüksek kira - VIP lokasyonlar"

def predict_rent(model, model_features, input_dict):
    """Kira tahmini yap"""
    try:
        # Feature vector oluştur (model'in beklediği sırada)
        feature_vector = []
        
        for feature in model_features:
            if feature in input_dict:
                feature_vector.append(input_dict[feature])
            else:
                feature_vector.append(0)
        
        X = np.array(feature_vector).reshape(1, -1)
        
        # Tahmin yap
        prediction = model.predict(X)
        
        # Array'den scalar çıkar
        if isinstance(prediction, np.ndarray):
            prediction = prediction.item()
        else:
            prediction = float(prediction)
        
        prediction = max(prediction, 0)
        
        return prediction
        
    except Exception as e:
        st.error(f"❌ Tahmin hatası: {str(e)}")
        return None

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🏠 House Rent Predictor</h1>
        <p>🔮 Evin özelliklerini girerek kira tahmini yapın</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model yükle
    model, model_features = load_model()
    if model is None or model_features is None:
        st.stop()
    
    st.success(f"✓ Model başarıyla yüklendi! ({len(model_features)} features)")
    
    # SIDEBAR
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-title">📊 Evin Özelliklerini Girin</div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        input_dict = {}
        
        # ============================================================
        # SAYISAL ÖZELLİKLER
        # ============================================================
        st.markdown("**📊 Temel Özellikler**")
        
        col1, col2 = st.columns(2)
        with col1:
            bhk = st.slider("🛏️ BHK", 1, 6, 2)
            input_dict['BHK'] = bhk
        
        with col2:
            bathroom = st.slider("🚿 Banyo", 1, 4, 1)
            input_dict['Bathroom'] = bathroom
        
        col1, col2 = st.columns(2)
        with col1:
            size = st.number_input("📐 Alan (Sq Ft)", 300.0, 10000.0, 1000.0, 100.0)
            input_dict['Size'] = size
        
        with col2:
            price_per_sqft = st.number_input("💹 Fiyat/Sq Ft", 100.0, 10000.0, 1000.0, 100.0)
            input_dict['Price_Per_Sqft'] = price_per_sqft
        
        col1, col2 = st.columns(2)
        with col1:
            floor_number = st.slider("🏢 Kat Numarası", 0, 30, 2)
            input_dict['Floor_Number'] = floor_number
        
        with col2:
            is_ground_floor = st.checkbox("🌍 Zemin Kat mı?", value=False)
            input_dict['Is_Ground_Floor'] = int(is_ground_floor)
        
        st.divider()
        
        # ============================================================
        # İLAN ÖZELLİKLERİ
        # ============================================================
        st.markdown("**📅 İlan Özellikleri**")
        
        col1, col2 = st.columns(2)
        with col1:
            days_since_posted = st.slider("📆 Günler (İlan yayınlandığından beri)", 0, 365, 30)
            input_dict['Days_Since_Posted'] = days_since_posted
        
        with col2:
            posted_month = st.slider("🗓️ Ay (1-12)", 1, 12, 6)
            input_dict['Posted_Month'] = posted_month
        
        st.divider()
        
        # ============================================================
        # KATEGORİKAL ÖZELLİKLER
        # ============================================================
        st.markdown("**📂 Kategoriler**")
        
        # Şehir Seçimi
        st.write("🌆 **Şehir Seç:**")
        cities = ['Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai']
        selected_city = st.radio("", cities, label_visibility="collapsed")
        
        for city in cities:
            input_dict[f'City_{city}'] = 1 if city == selected_city else 0
        
        st.divider()
        
        # Alan Türü
        st.write("🏘️ **Alan Türü:**")
        area_types = ['Residential', 'Commercial']
        selected_area_type = st.radio("", area_types, key="area_type", label_visibility="collapsed")
        input_dict['Area_Type'] = 1 if selected_area_type == 'Residential' else 0
        
        st.divider()
        
        # Mobilya Durumu
        st.write("🛋️ **Mobilya Durumu:**")
        furnishing_types = ['Unfurnished', 'Semi-Furnished', 'Furnished']
        selected_furnishing = st.radio("", furnishing_types, key="furnishing", label_visibility="collapsed")
        
        # Furnishing 0, 1, 2 değerleri alabilir
        furnishing_mapping = {
            'Unfurnished': 0,
            'Semi-Furnished': 1,
            'Furnished': 2
        }
        input_dict['Furnishing'] = furnishing_mapping[selected_furnishing]
        
        st.divider()
        
        # Kiracı Türü
        st.write("👥 **Kiracı Türü:**")
        tenant_types = ['Bachelors', 'Family', 'Company']
        selected_tenant = st.radio("", tenant_types, key="tenant", label_visibility="collapsed")
        
        # Tenant 0, 1, 2 değerleri alabilir
        tenant_mapping = {
            'Bachelors': 0,
            'Family': 1,
            'Company': 2
        }
        input_dict['Tenant'] = tenant_mapping[selected_tenant]
        
        st.divider()
        
        # İletişim Türü
        st.write("📞 **İletişim Türü:**")
        contact_types = ['Contact Owner', 'Contact Agent']
        selected_contact = st.radio("", contact_types, key="contact", label_visibility="collapsed")
        input_dict['Point of Contact_Contact Owner'] = 1 if selected_contact == 'Contact Owner' else 0
        
        st.divider()
        
        # Bölge Seçimi
        st.write("📍 **Bölge Seç:**")
        localities = [
            'Chembur',
            'Electronic City',
            'Gachibowli',
            'K R Puram',
            'Kondapur',
            'Laxmi Nagar',
            'Madipakkam',
            'Miyapur, NH 9',
            'Velachery',
            'Other'
        ]
        selected_locality = st.selectbox("Bölge", localities, label_visibility="collapsed")
        
        # Bölge encoding
        for locality in localities:
            feature_name = f'Area_Locality_Top10_{locality}'
            input_dict[feature_name] = 1 if locality == selected_locality else 0
        
        st.divider()
        predict_button = st.button("🔮 Kira Tahmini Yap", use_container_width=True)
    
    # ============================================================
    # MAIN AREA - RESULTS
    # ============================================================
    if predict_button:
        with st.spinner('⏳ Kira tahmini yapılıyor...'):
            # Tahmin yap
            predicted_rent = predict_rent(model, model_features, input_dict)
            
            if predicted_rent is not None and predicted_rent > 0:
                rent_category, rent_desc = get_rent_range_interpretation(predicted_rent)
                
                # Sonuç kutusu
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-label">💵 Tahmini Aylık Kira</div>
                    <div class="result-rent">₹{predicted_rent:,.0f}/ay</div>
                    <div class="result-label">{rent_category} - {rent_desc}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.success("✓ Tahmin başarıyla tamamlandı!")
                
                st.divider()
                
                # Metrikleri göster
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🛏️ BHK", f"{bhk}")
                with col2:
                    st.metric("📐 Alan", f"{size:,.0f} ft²")
                with col3:
                    st.metric("🚿 Banyo", f"{bathroom}")
                with col4:
                    st.metric("💹 Fiyat/ft²", f"₹{price_per_sqft:,.0f}")
                
                st.divider()
                
                # Detaylı Bilgiler
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🏠 Emlak Bilgileri")
                    property_df = pd.DataFrame({
                        'Özellik': ['BHK', 'Alan', 'Banyo', 'Zemin Kat', 'Kat No.'],
                        'Değer': [f"{bhk}", f"{size:,.0f} ft²", 
                                 f"{bathroom}", 
                                 "✅ Evet" if is_ground_floor else "❌ Hayır",
                                 f"{floor_number}"]
                    })
                    st.table(property_df)
                
                with col2:
                    st.subheader("📍 Konum ve Tür")
                    location_df = pd.DataFrame({
                        'Özellik': ['Şehir', 'Bölge', 'Alan Türü', 'Mobilya', 'Kiracı Türü'],
                        'Değer': [selected_city, selected_locality, selected_area_type,
                                 selected_furnishing, selected_tenant]
                    })
                    st.table(location_df)
                
                st.divider()
                
                # Kira Analizi
                st.subheader("💰 Kira Analizi")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    monthly_rent = predicted_rent
                    st.metric("📊 Aylık Kira", f"₹{monthly_rent:,.0f}")
                
                with col2:
                    annual_rent = monthly_rent * 12
                    st.metric("📈 Yıllık Kira", f"₹{annual_rent:,.0f}")
                
                with col3:
                    rent_per_sqft = monthly_rent / size if size > 0 else 0
                    st.metric("📐 Ft² Başına", f"₹{rent_per_sqft:.2f}")
                
                st.divider()
                
                # Gauge Chart
                st.subheader("📊 Kira Seviyesi")
                
                fig = go.Figure(data=[go.Indicator(
                    mode="gauge+number",
                    value=predicted_rent,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Tahmini Aylık Kira (₹)"},
                    gauge={
                        'axis': {'range': [0, max(150000, predicted_rent * 1.5)]},
                        'bar': {'color': "#2ecc71"},
                        'steps': [
                            {'range': [0, 20000], 'color': "#d5f4e6"},
                            {'range': [20000, 50000], 'color': "#abebc6"},
                            {'range': [50000, 100000], 'color': "#82e0aa"},
                            {'range': [100000, max(150000, predicted_rent * 1.5)], 'color': "#58d68d"}
                        ]
                    }
                )])
                
                fig.update_layout(height=400, margin=dict(l=20, r=20, t=60, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                # Öneriler
                st.subheader("💡 Öneriler")
                
                tips = []
                
                if bhk == 1:
                    tips.append("🛏️ 1 BHK - Bekar profesyoneller için ideal")
                elif bhk == 2:
                    tips.append("🛏️ 2 BHK - Genç çift veya küçük aile")
                else:
                    tips.append(f"🛏️ {bhk} BHK - Geniş aile için uygun")
                
                if size < 500:
                    tips.append("📐 Kompakt - Ekonomik seçim")
                elif size < 1000:
                    tips.append("📐 Standart - Standart konut")
                else:
                    tips.append("📐 Geniş - Konforlu yaşam")
                
                if selected_furnishing == 'Furnished':
                    tips.append("🛋️ Mobilyalı - Hemen taşınabilir")
                elif selected_furnishing == 'Semi-Furnished':
                    tips.append("🛋️ Yarı Mobilyalı - Kısmi mobilya")
                else:
                    tips.append("🛋️ Boş - Kişiselleştirme imkanı")
                
                if selected_city in ['Mumbai', 'Delhi']:
                    tips.append(f"🌆 {selected_city} - Yüksek kiralık beklentisi")
                else:
                    tips.append(f"🌆 {selected_city} - Uygun kira")
                
                if is_ground_floor:
                    tips.append("🌍 Zemin Kat - Doğrudan giriş avantajı")
                
                for tip in tips:
                    st.info(tip)

if __name__ == "__main__":
    main()