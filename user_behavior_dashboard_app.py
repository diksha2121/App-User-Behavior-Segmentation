import streamlit as st
import joblib 
import numpy as np

scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans_model.pkl')

st.title('App User Behavior Segmentation Dashboard')
st.write('Enter Customer Details to predict the segment')

sessions = st.slider("Sessions per Week", min_value=1, max_value=20, value=5)
daily_minutes = st.slider("Daily Active Minutes", min_value=1, max_value=120, value=30)
avg_duration = st.slider("Average Session Duration(Minutes)", min_value=1.0, max_value=60.0, value=10.0)

if st.button("Predict User Segment"):
    input_data = np.array([[sessions, daily_minutes, avg_duration]])
    scaled_input = scaler.transform(input_data)
    cluster_id = kmeans.predict(scaled_input)[0]

    st.success(f"Predicted Cluster ID: {cluster_id}")