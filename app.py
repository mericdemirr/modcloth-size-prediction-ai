import streamlit as st
import pandas as pd
import joblib
from textblob import TextBlob
from deep_translator import GoogleTranslator 


st.set_page_config(page_title="ModCloth AI & Business", layout="wide")


try:
    model = joblib.load('models/clothing_fit_model.pkl')
    model_columns = joblib.load('models/model_columns.pkl')
except:
    st.error("Model bulunamadı! Lütfen terminalde 'src/train.py' çalıştırın.")
    st.stop()


with st.sidebar:
    st.header("💼 Satıcı Paneli (ROI Analizi)")
    st.markdown("---")
    
    st.subheader("Aylık Satış Simülasyonu")
    monthly_sales = st.slider("Aylık Satış Adedi", 1000, 50000, 10000, step=1000)
    cost_per_return = st.number_input("İade Başına Maliyet ($)", 5.0, 50.0, 15.0)
    
    
    avg_return_rate = 0.35 
    total_potential_returns = monthly_sales * avg_return_rate
    loss_without_ai = total_potential_returns * cost_per_return
    
    model_success_rate = 0.45 
    prevented_returns = total_potential_returns * model_success_rate
    savings = prevented_returns * cost_per_return
    
    st.markdown("### Finansal Etki")
    st.metric(label="Yapay Zekasız Zarar", value=f"${loss_without_ai:,.0f}", delta="-Risk", delta_color="inverse")
    st.metric(label="Yapay Zeka ile TASARRUF", value=f"${savings:,.0f}", delta=f"+%{model_success_rate*100:.0f} Başarı", delta_color="normal")
    st.markdown("---")
    st.info(f"**Sonuç:** Model kullanımı sayesinde aylık net **${savings:,.0f}** kargo ve operasyon maliyeti kurtarılıyor.")


col_main, col_gap = st.columns([2, 1])

with col_main:
    
    st.title("Beden Tahmin Asistanı")
    st.markdown("Yapay Zeka + **Business Rules** + **Türkçe NLP** ile güçlendirilmiş analiz.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ölçüler")
        height = st.number_input("Boy (cm)", 140, 220, 165)
        hips = st.number_input("Basen (inch)", 20.0, 60.0, 38.0)
        bra_num = st.selectbox("Sütyen Sırtı", [30, 32, 34, 36, 38, 40, 42, 44, 46, 48])
        cup_input = st.selectbox("Cup Size", ['a', 'b', 'c', 'd', 'dd', 'ddd', 'e', 'f', 'g', 'k'])

    with col2:
        st.subheader("Ürün")
        size_input = st.number_input("Alacağınız Beden (US Size)", 0, 30, 8) 
        category = st.selectbox("Kategori", ["Dresses", "Tops", "Outerwear", "Bottoms"])
        length = st.selectbox("Ürün Boyu", ["Just Right", "Slightly Long", "Slightly Short"])
        # ARTIK TÜRKÇE YAZABİLİRSİN!
        review = st.text_area("Yorum (Türkçe Yazabilirsiniz)", "Bu elbiseyi çok sevdim!")

    if st.button("BEDENİM UYGUN MU?", type="primary"):
        # Veri Hazırlığı
        input_data = pd.DataFrame(columns=model_columns)
        input_data.loc[0] = 0
        
        
        try:
            # Türkçe girilen yorumu İngilizceye çevir (Model anlasın diye)
            translated_review = GoogleTranslator(source='auto', target='en').translate(review)
            sentiment = TextBlob(translated_review).sentiment.polarity
            st.toast(f"Algılanan Yorum (EN): {translated_review}") 
        except:
            sentiment = 0 
            
        cup_map = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'dd': 5, 'ddd': 6, 'e': 5, 'f': 6, 'g': 7, 'k': 9}
        cup_score = cup_map.get(cup_input, 2)
        safe_size = 1 if size_input == 0 else size_input
        
        input_data['height_cm'] = height
        input_data['hips'] = hips
        input_data['bra_size_num'] = bra_num
        input_data['quality'] = 4.0
        input_data['size'] = size_input
        input_data['cup_score'] = cup_score
        input_data['bust_estimate'] = bra_num * cup_score
        input_data['hips_to_size_ratio'] = hips / safe_size
        input_data['sentiment_score'] = sentiment
        input_data['review_length'] = len(review)

        
        cat_map = {"Dresses": "category_dresses", "Tops": "category_tops", "Outerwear": "category_outerwear"}
        if category in cat_map and cat_map[category] in input_data.columns: input_data[cat_map[category]] = 1
        len_map = {"Just Right": "length_just right", "Slightly Long": "length_slightly long"}
        if length in len_map and len_map[length] in input_data.columns: input_data[len_map[length]] = 1

        
        ml_pred = model.predict(input_data)[0]
        ml_probs = model.predict_proba(input_data)[0]
        confidence = max(ml_probs) * 100 

        final_pred = ml_pred
        override_msg = ""
        low_confidence_msg = ""

        
        if hips >= 40 and size_input <= 6:
            final_pred = 0
            override_msg = "FİZİKSEL UYARI: Basen/Beden uyumsuzluğu (Çok Dar)."
        elif hips <= 35 and size_input >= 14:
            final_pred = 2
            override_msg = "FİZİKSEL UYARI: Vücut/Beden uyumsuzluğu (Çok Bol)."
        elif confidence < 60:
            low_confidence_msg = f"Yapay zeka kararsız kaldı. Lütfen ölçüleri kontrol edin."

        
        st.divider()
        if override_msg: st.error(override_msg)
        if low_confidence_msg: st.warning(low_confidence_msg)

        if final_pred == 1:
            st.success("HARİKA! Bu beden size TAM (FIT) olacak.")
        elif final_pred == 0:
            st.error("DİKKAT: Bu beden KÜÇÜK (SMALL) gelebilir!")
        else:
            st.warning("DİKKAT: Bu beden BÜYÜK (LARGE) gelebilir!")
            
        if not override_msg:
            st.info(f"Yapay Zeka Güven Oranı: **%{confidence:.1f}**")