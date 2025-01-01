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

# Plot the results
plt.figure(figsize=(10, 6))

# Plot normal points
normal = df[df['Outlier_ARDOD'] == False]
plt.scatter(normal['longitude'], normal['latitude'], c='blue', label='Normal Data', alpha=0.5)

# Plot outliers
outliers = df[df['Outlier_ARDOD'] == True]
plt.scatter(outliers['longitude'], outliers['latitude'], c='red', label='Outliers', alpha=0.7)

# Labels and title
plt.title("Outlier Detection using ARDOD (Adaptive Radius)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()

# Show the plot
plt.show()

# Optional: Print summary of clusters and outliers
print("Number of outliers detected (ARDOD): ", len(outliers))
print("Number of normal data points: ", len(normal))
