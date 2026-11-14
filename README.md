An end-to-end unsupervised machine learning project that analyzes app user behavior data to identify distinct customer segments, evaluate churn risk, and deploy an interactive Streamlit web application for real-time segment prediction.

## 🚀 Project Overview
Understanding user engagement is critical for mobile application growth. This project processes 50,000 user behavioral records, compares initial multi-feature clustering against refined high-contrast behavioral features, applies Principal Component Analysis (PCA) for visualization, and surfaces actionable business segments.

### Key Milestones:
1. **Data Preprocessing & Cleaning:** Handled missing values, standardized data types, and validated feature independence using correlation heatmaps.
2. **Iterative Clustering Experiments:** 
   - *Attempt 1:* Clustered across all 10 behavioral features (resulted in overlapping profiles).
   - *Attempt 2 (Refined):* Clustered using 3 key contrast features (`sessions_per_week`, `daily_active_minutes`, `avg_session_duration_min`) with $K=4$ to achieve clear behavioral separation.
3. **Model Evaluation:** Validated cluster separation using the **Silhouette Score**.
4. **Dimensionality Reduction:** Used **PCA** to visualize high-dimensional user clusters in a 2D plane.
5. **Deployment:** Saved model artifacts (`scaler.pkl`, `kmeans_model.pkl`) and built an interactive **Streamlit dashboard** for live user segment predictions.

## 📁 Repository Structure
```text
📦 UserSegmentation
│
├── app_user_behavior_app.py        # Main data processing, EDA, and model training script
├── dashboard.py                    # Streamlit web application script
├── app_user_behavior_dataset.csv   # Raw app user behavior dataset
├── customer_segment_assignments.csv# Exported user segment mappings
├── scaler.pkl                      # Saved StandardScaler artifact
├── kmeans_model.pkl                # Trained K-Means clustering model artifact
└── README.md                       # Project documentation
