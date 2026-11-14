import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ---------------------------------------------------------
# 1. LOAD & INITIAL INSPECTION
# ---------------------------------------------------------
df = pd.read_csv("app_user_behavior_dataset.csv")
df.head()

df_clean = df.copy()
df_clean.shape
df_clean.info()
df_clean.describe()
df_clean.duplicated().sum()
df_clean.isnull().sum()

# ---------------------------------------------------------
# 2. MISSING VALUE HANDLING
# ---------------------------------------------------------
df_clean['rating_given'].describe()

sns.histplot(df_clean['rating_given'], kde=True)
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.show()

df_clean['rating_given'].median()
df_clean['rating_given'] = df_clean['rating_given'].fillna(4.0)
df_clean.isnull().sum()
df_clean.dtypes

# ---------------------------------------------------------
# 3. OUTLIER & DATA TYPE CHECKS
# ---------------------------------------------------------
num_cols = df_clean.select_dtypes(exclude='object').columns.to_list()
num_cols.remove('user_id')

for col in num_cols:
    print(f"{col}: Min value is {df_clean[col].min()}, Max value is {df_clean[col].max()}")

df_clean['marketing_source'].value_counts()

cat_cols = df_clean.select_dtypes(include='object').columns.to_list()
for col in cat_cols:
    df_clean[col] = df_clean[col].astype(str).str.strip().str.title()

df_clean['user_id'] = df_clean['user_id'].astype(str).str.strip().str.title()
df_clean['app_version'] = df_clean['app_version'].astype(str).str.strip().str.title()
df_clean.dtypes

# ---------------------------------------------------------
# 4. EDA (NUMERICAL DISTRIBUTIONS & CORRELATION)
# ---------------------------------------------------------
n_cols = len(num_cols)
n_rows = (n_cols + 1) // 2

fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(14, n_rows * 3.5))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(df_clean[col], kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution: {col}", fontsize=11, fontweight="bold")
    axes[i].set_xlabel("")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Multicollinearity Check
corr_matrix = df_clean[num_cols].corr()
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(data=corr_matrix, annot=True, cmap='viridis', fmt='.2f', vmin=-1, vmax=1, linewidth=0.5, ax=ax)
ax.set_title('Correlation Matrix of Numerical Features', fontsize=14, pad=15, weight='bold')
plt.show()

# ---------------------------------------------------------
# 5. ATTEMPT 1: CLUSTERING WITH ALL 10 FEATURES
# ---------------------------------------------------------
main_features = [
    'sessions_per_week', 'avg_session_duration_min', 'daily_active_minutes',
    'feature_clicks_per_session', 'notifications_opened_per_week', 'in_app_search_count',
    'pages_viewed_per_session', 'ads_clicked_last_30_days', 'content_downloads', 'social_shares'
]

X = df_clean[main_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Graph for Attempt 1
wcss = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    model.fit(X_scaled)
    wcss.append(model.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method (All Features)')
plt.xlabel('Number of Clusters(K)')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
df_clean['Cluster'] = kmeans.fit_predict(X_scaled)

profile_cols = main_features + ['engagement_score', 'churn_risk_score']
cluster_profile = df_clean.groupby('Cluster')[profile_cols].mean()

print("--- Attempt 1 Profile Summary (All Features) ---")
print(cluster_profile[['sessions_per_week', 'engagement_score', 'churn_risk_score']])

# PCA for Attempt 1
pca = PCA(n_components=2, random_state=42)
principal_components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = df_clean['Cluster']

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=50, alpha=0.8)
plt.title('PCA Visualization (All Features)', fontsize=14, fontweight='bold')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster ID')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# ---------------------------------------------------------
# 6. ATTEMPT 2 (FINAL): REFINED CLUSTERING WITH KEY FEATURES
# ---------------------------------------------------------
contrast_features = ['sessions_per_week', 'daily_active_minutes', 'avg_session_duration_min']

X_new = df_clean[contrast_features]
scaler_new = StandardScaler()
X_new_scaled = scaler_new.fit_transform(X_new)

# Run K-Means again with refined features
kmeans_new = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
df_clean['Cluster_Refined'] = kmeans_new.fit_predict(X_new_scaled)

# Calculate Silhouette Score
sil_score = silhouette_score(X_new_scaled, df_clean['Cluster_Refined'])
print(f"Silhouette Score (Refined Features): {sil_score:.4f}")

# Check the new profile summary
cluster_profile_refined = df_clean.groupby('Cluster_Refined')[contrast_features + ['engagement_score', 'churn_risk_score']].mean()
print("--- Attempt 2 Profile Summary (Refined Features) ---")
print(cluster_profile_refined)

# Map cluster IDs based on profile insights
cluster_mapping = {
    2: 'High Frequency Users (Cluster 2)',
    1: 'Power Duration Users (Cluster 1)',
    3: 'Deep-Dive Users (Cluster 3)',
    0: 'Low Activity Users (Cluster 0)'
}
df_clean['Cluster_Name'] = df_clean['Cluster_Refined'].map(cluster_mapping)
print(df_clean['Cluster_Name'].value_counts())

# PCA Visualization for Refined Clusters
pca_new = PCA(n_components=2, random_state=42)
principal_components_new = pca_new.fit_transform(X_new_scaled)

pca_df_new = pd.DataFrame(data=principal_components_new, columns=['PC1', 'PC2'])
pca_df_new['Cluster_Name'] = df_clean['Cluster_Name']

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster_Name', data=pca_df_new, palette='viridis', s=50, alpha=0.8)
plt.title('PCA Visualization of Distinct App User Clusters', fontsize=14, fontweight='bold')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='User Segments', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 7. FINAL METRICS & STREAMLIT ARTIFACT EXPORT
# ---------------------------------------------------------
df_clean['true_engagement_score'] = (
    df_clean['sessions_per_week'] / df_clean['sessions_per_week'].max() +
    df_clean['daily_active_minutes'] / df_clean['daily_active_minutes'].max()
) * 50

customer_segments = df_clean[['user_id', 'Cluster_Name', 'true_engagement_score', 'churn_risk_score']]
customer_segments.to_csv('customer_segment_assignments.csv', index=False)
print(customer_segments.head())

# Save artifacts for VS Code / Streamlit app deployment
joblib.dump(scaler_new, 'scaler.pkl')
joblib.dump(kmeans_new, 'kmeans_model.pkl')
print("Saved customer_segment_assignments.csv, scaler.pkl, and kmeans_model.pkl successfully!")