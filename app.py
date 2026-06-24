# app.py
import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.title("🏃 Human Activity Recognition")
st.markdown("Predict physical activity from smartphone sensor data using ML.")

# Load your saved best model (export it from the notebook with joblib)
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
label_classes = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
                 "SITTING", "STANDING", "LAYING"]

st.subheader("Upload sensor data (CSV with 561 features)")
uploaded = st.file_uploader("Choose a CSV file", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    
    META_COLS = ['subject', 'Activity']
    feature_cols = [c for c in df.columns if c not in META_COLS]
    
    X = scaler.transform(df[feature_cols].values)
    preds = model.predict(X)
    
    st.success(f"✅ Processed **{len(preds)}** samples")
    
    # Show prediction distribution across all rows
    pred_labels = [label_classes[p] for p in preds]
    pred_series = pd.Series(pred_labels, name="Predicted Activity")
    counts = pred_series.value_counts().reset_index()
    counts.columns = ["Activity", "Count"]
    
    st.subheader("Prediction Distribution")
    st.bar_chart(counts.set_index("Activity"))
    
    # If ground truth exists, show accuracy too
    if 'Activity' in df.columns:
        from sklearn.metrics import accuracy_score, f1_score
        le = joblib.load("label_encoder.pkl")
        y_true = le.transform(df['Activity'])
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, average='macro')
        
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{acc:.2%}")
        col2.metric("F1-Macro", f"{f1:.4f}")
    
    # Show sample of predictions
    st.subheader("Sample Predictions (first 10 rows)")
    sample_df = df.head(10)[['Activity'] if 'Activity' in df.columns else []].copy()
    sample_df['Predicted'] = pred_labels[:10]
    st.dataframe(sample_df)