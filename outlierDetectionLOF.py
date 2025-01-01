import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Load the dataset
df = pd.read_csv('sample_data/california_housing_train.csv')

# Select latitude and longitude for clustering
lat_long = df[['latitude', 'longitude']]

# Normalize the data (this helps with scaling differences between features)
scaler = StandardScaler()
lat_long_scaled = scaler.fit_transform(lat_long)

# Apply DBSCAN for density-based clustering (for reachability)
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(lat_long_scaled)

# Apply LOF (Local Outlier Factor) for detecting outliers
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)  # You can adjust n_neighbors and contamination
lof_labels = lof.fit_predict(lat_long_scaled)
lof_scores = -lof.negative_outlier_factor_

# Create a DataFrame for visualization
df['Cluster'] = dbscan_labels
df['LOF_Score'] = lof_scores
df['Outlier'] = lof_labels

# Plot the results
plt.figure(figsize=(10, 6))

# Plot normal points
normal = df[df['Outlier'] == 1]
plt.scatter(normal['longitude'], normal['latitude'], c='blue', label='Normal Data', alpha=0.5)

# Plot outliers
outliers = df[df['Outlier'] == -1]
plt.scatter(outliers['longitude'], outliers['latitude'], c='red', label='Outliers', alpha=0.7)

# Labels and title
plt.title("Outlier Detection using LOF and DBSCAN (California Housing Data)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()

# Show the plot
plt.show()

# Optional: Print summary of clusters and outliers
print("Number of outliers detected: ", len(outliers))
print("Number of normal data points: ", len(normal))
