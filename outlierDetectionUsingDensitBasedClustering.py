#Code to view dataset from Kaggle

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('sample_data/california_housing_train.csv')
df




#OUTLIER DETECTION USING LOF AND DBSCAN(dataset sample California housing data)
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
#RESULT: Number of outliers detected (ARDOD):  1180, Number of normal data points:  15820




#ADAPTIVE RADIUS DENSITY-BASED OUTLIER DETECTION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

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

# Plot 1: Basic scatter plot showing outliers vs. non-outliers
plt.figure(figsize=(10, 6))
normal = df[df['Outlier_ARDOD'] == False]
outliers = df[df['Outlier_ARDOD'] == True]

plt.scatter(normal['longitude'], normal['latitude'], c='blue', label='Normal Data', alpha=0.5)
plt.scatter(outliers['longitude'], outliers['latitude'], c='red', label='Outliers', alpha=0.7)

plt.title("Basic ARDOD Outlier Detection")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()


# Plot 2: Scatter plot with point size based on Adaptive Radius
plt.figure(figsize=(10, 6))
sizes = df['Adaptive_Radius'] * 1000  # Scaling sizes for visualization

plt.scatter(normal['longitude'], normal['latitude'], c='blue', s=sizes[df['Outlier_ARDOD'] == False], label='Normal Data', alpha=0.5)
plt.scatter(outliers['longitude'], outliers['latitude'], c='red', s=sizes[df['Outlier_ARDOD'] == True], label='Outliers', alpha=0.7)

plt.title("ARDOD with Point Size Varying by Adaptive Radius")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()


# Plot 3: 3D scatter plot of Latitude, Longitude, and Adaptive Radius
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# 3D scatter plot
ax.scatter(normal['longitude'], normal['latitude'], normal['Adaptive_Radius'], c='blue', label='Normal Data', alpha=0.5)
ax.scatter(outliers['longitude'], outliers['latitude'], outliers['Adaptive_Radius'], c='red', label='Outliers', alpha=0.7)

ax.set_title("3D Plot: Latitude, Longitude and Adaptive Radius")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Adaptive Radius")
plt.legend()
plt.show()
#RESULT: Number of outliers detected (ARDOD):  1180, Number of normal data points:  15820
