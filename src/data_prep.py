import pandas as pd
import numpy as np
import json
import os
from textblob import TextBlob

def load_and_clean_data(filepath):
    """
    VERİ HAZIRLIĞI (HATA DÜZELTİLMİŞ VERSİYON)
    """
    print(f"   [DataPrep] Dosya okunuyor: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"HATA: Veri dosyası bulunamadı: {filepath}")

    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)

    
    def parse_height(h):
        if pd.isna(h): return np.nan
        try:
            h = str(h)
            parts = h.split(' ')
            if len(parts) > 1:
                ft = int(parts[0].replace('ft', ''))
                inch = int(parts[1].replace('in', ''))
                return (ft * 30.48) + (inch * 2.54)
            return np.nan
        except:
            return np.nan

    df['height_cm'] = df['height'].apply(parse_height)
    df['bra_size_num'] = df['bra size'].astype(str).str.extract(r'(\d+)').astype(float)
    df['hips'] = pd.to_numeric(df['hips'], errors='coerce')
    df['quality'] = pd.to_numeric(df['quality'], errors='coerce')
    
    
    for col in ['height_cm', 'hips', 'bra_size_num', 'quality']:
        df[col] = df[col].fillna(df[col].median())
        
    df['length'] = df['length'].fillna('Unknown')
    df['cup_size_clean'] = df['bra size'].astype(str).str.extract(r'([a-zA-Z]+)')

    
    print("   [DataPrep] Yeni özellikler üretiliyor...")

   
    cup_map = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'dd': 5, 'ddd': 6, 'e': 5, 'f': 6, 'g': 7, 'k': 9}
    df['cup_score'] = df['cup_size_clean'].map(cup_map).fillna(2)

    
    df['bust_estimate'] = df['bra_size_num'] * df['cup_score']

    
    safe_size = df['size'].replace(0, 1) 
    df['hips_to_size_ratio'] = df['hips'] / safe_size

    
    def get_sentiment(text):
        if pd.isna(text) or text == "": return 0
        return TextBlob(str(text)).sentiment.polarity

    df['sentiment_score'] = df['review_text'].apply(get_sentiment)
    df['review_length'] = df['review_text'].astype(str).apply(len)

    
    fit_map = {'small': 0, 'fit': 1, 'large': 2}
    df['fit_target'] = df['fit'].map(fit_map)

    df_model = pd.get_dummies(df, columns=['category', 'length'], dummy_na=True, drop_first=True)

    cols_to_drop = ['item_id', 'user_id', 'user_name', 'review_text', 'review_summary', 
                    'height', 'bra size', 'shoe size', 'shoe width', 'waist', 'bust', 'fit', 
                    'cup_size_clean', 'cup size']
    
    existing_drop = [c for c in cols_to_drop if c in df_model.columns]
    df_model = df_model.drop(columns=existing_drop)
    
    
    df_model = df_model.replace([np.inf, -np.inf], 0)
    
    return df_model.select_dtypes(exclude=['object'])
