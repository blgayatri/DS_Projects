import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load saved models and data
similarity = joblib.load("product_similarity.pkl")
rfm_model = joblib.load("rfm_cluster_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("🛍️ Shopper Spectrum: Product Recommendation & Customer Segmentation")

st.markdown("""
Welcome to the **Shopper Spectrum** app!  
Use the sidebar to choose between:
- 📦 **Product Recommendation**
- 👤 **Customer Segmentation**
""")

# Sidebar menu
option = st.sidebar.selectbox("Select Module", ["Product Recommendation", "Customer Segmentation"])

# Module 1: Product Recommendation
if option == "Product Recommendation":
    st.header("📦 Product Recommendation Engine")

    product_name = st.text_input("Enter a Product Name (exactly as in the dataset):")

    if st.button("Get Recommendations"):
        if product_name in similarity.columns:
            # Get similarity scores
            sim_scores = similarity[product_name].sort_values(ascending=False)[1:6]
            st.success("Here are 5 similar products:")
            for i, prod in enumerate(sim_scores.index, 1):
                st.markdown(f"**{i}.** {prod}")
        else:
            st.error("Product not found. Please check the spelling or use an existing product name.")

# Module 2: Customer Segmentation
elif option == "Customer Segmentation":
    st.header("👤 Customer Segmentation Predictor")

    recency = st.number_input("Recency (days since last purchase)", min_value=1, value=30)
    frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=5)
    monetary = st.number_input("Monetary (total spend)", min_value=1.0, value=100.0)

    if st.button("Predict Segment"):
        user_data = pd.DataFrame([[recency, frequency, monetary]], columns=["Recency", "Frequency", "Monetary"])
        user_scaled = scaler.transform(user_data)
        cluster = rfm_model.predict(user_scaled)[0]

        # Label based on cluster (you can adjust this according to your actual cluster labeling logic)
        segment_map = {
            0: "Occasional",
            1: "High-Value",
            2: "Regular",
            3: "At-Risk"
        }

        segment = segment_map.get(cluster, "Unknown")
        st.success(f"Predicted Customer Segment: **{segment}**")
