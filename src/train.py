import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier
from data_prep import load_and_clean_data

def train_model():
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'modcloth_final_data.json')
    model_path = os.path.join(base_dir, 'models', 'clothing_fit_model.pkl')
    columns_path = os.path.join(base_dir, 'models', 'model_columns.pkl')

    print("Model Eğitimi (FİNAL VERSİYON) Başlıyor...")
    
    
    df = load_and_clean_data(data_path)
    
    
    import re
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    X = df.drop('fit_target', axis=1)
    y = df['fit_target']

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    
    print("\n1: BASELINE MODEL")
    baseline_model = DummyClassifier(strategy='most_frequent')
    baseline_model.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline_model.predict(X_test))
    print(f"Baseline Accuracy: %{baseline_acc*100:.2f}")

   
    print("\n2: LIGHTGBM")
    
    
    custom_weights = {0: 1.2, 1: 1, 2: 1.2} 

    model = LGBMClassifier(
        n_estimators=500,       
        learning_rate=0.02,     
        max_depth=8,            
        num_leaves=25,
        class_weight=custom_weights, 
        reg_alpha=0.1,          
        reg_lambda=0.1,         
        random_state=42,
        verbosity=-1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    
    model_acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print(f"LightGBM Accuracy: %{model_acc*100:.2f}")
    print("="*40)

    
    fark = model_acc - baseline_acc
    print(f",GELİŞİM: Baseline modele göre +%{fark*100:.2f} fark!")

    print("\nDetaylı Rapor:")
    print(classification_report(y_test, y_pred, target_names=['Small', 'Fit', 'Large']))
    
    
    joblib.dump(model, model_path)
    joblib.dump(X.columns.tolist(), columns_path)
    print("\n Final modeli kaydedildi.")

if __name__ == "__main__":
    train_model()
