import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('sample_data/california_housing_train.csv')

# Select latitude and longitude for clustering
lat_long = df[['latitude', 'longitude']]

# Normalize the data (this helps with scaling differences between features)
scaler = StandardScaler()
lat_long_scaled = scaler.fit_transform(lat_long)

# Calculate the average pairwise distance
n_samples = lat_long_scaled.shape[0]
pairwise_dist = pairwise_distances(lat_long_scaled)

# Step 1: Compute global initial radius `r`
epsilon = 0.01  # Small fraction to control sensitivity
r = (epsilon / (n_samples * (n_samples - 1))) * np.sum(pairwise_dist)

# Step 2: Compute Adaptive Radius for each sample
adaptive_radius = []

for i in range(n_samples):
    distances = pairwise_dist[i]  # Distance of point i to all others
    neighbors = distances[distances <= r]  # Points within the radius `r`

    if len(neighbors) > 1:
        # Calculate the adaptive radius as the mean pairwise distance among neighbors
        neighbor_pairwise_distances = pairwise_distances(lat_long_scaled[distances <= r])
        ar_i = np.sum(neighbor_pairwise_distances) / (len(neighbors) * (len(neighbors) - 1))
        adaptive_radius.append(ar_i)
    else:
        # Isolated points are considered outliers
        adaptive_radius.append(0)

# Step 3: Detect outliers based on the adaptive radius
outlier_threshold = np.percentile(adaptive_radius, 5)  # Define a threshold, e.g., 5th percentile
outliers = np.array(adaptive_radius) <= outlier_threshold

# Step 4: Create a DataFrame for visualization
df['Adaptive_Radius'] = adaptive_radius
df['Outlier_ARDOD'] = outliers

# Prepare for plotting
normal = df[df['Outlier_ARDOD'] == False]
outliers_df = df[df['Outlier_ARDOD'] == True]

# Plot 1: Original Data vs Outliers
plt.figure(figsize=(10, 6))
plt.scatter(normal['longitude'], normal['latitude'], c='blue', label='Normal Data', alpha=0.5)
plt.scatter(outliers_df['longitude'], outliers_df['latitude'], c='red', label='Outliers', alpha=0.7)
plt.title("Plot 1: Original Data vs Outliers (ARDOD)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()

# Plot 2: Heatmap-like Scatter Plot Based on Adaptive Radius
plt.figure(figsize=(10, 6))
plt.scatter(df['longitude'], df['latitude'], c=df['Adaptive_Radius'], cmap='viridis', alpha=0.7)
plt.colorbar(label='Adaptive Radius')
plt.title("Plot 2: Adaptive Radius Heatmap (ARDOD)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Plot 3: Clustered Data vs Outliers with Transparency
plt.figure(figsize=(10, 6))
plt.scatter(normal['longitude'], normal['latitude'], c='green', alpha=0.2, label='Clustered Points')
plt.scatter(outliers_df['longitude'], outliers_df['latitude'], c='red', alpha=1.0, label='Outliers')
plt.title("Plot 3: Clustered Points vs Outliers (ARDOD)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()

# Optional: Print summary of outliers and normal data
print("Number of outliers detected (ARDOD): ", len(outliers_df))
print("Number of normal data points: ", len(normal))
