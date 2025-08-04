import joblib

# Save clean and compatible versions of your models and similarity matrix
joblib.dump(similarity_df, "product_similarity.pkl", protocol=4)
joblib.dump(kmeans, "rfm_cluster_model.pkl", protocol=4)
joblib.dump(scaler, "scaler.pkl", protocol=4)
