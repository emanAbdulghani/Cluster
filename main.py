import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
# Load Iris dataset
iris = load_iris()
# Select only the first four columns of the dataset which contain the length and width of sepals and petals
X = iris.data[:, :4]
# Get target values
y = iris.target

# Define function to calculate Euclidean distance between two points
def euclidean_distance(x1, x2):
    # Convert lists to numpy arrays
    x1 = np.array(x1)
    x2 = np.array(x2)
    # Calculate Euclidean distance between two points
    return np.sqrt(np.sum((x1 - x2)**2))
# Define class for KMeans clustering
class KMeans:
    def _init_(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        # Initialize centroids by selecting first k points from our dataset
        self.centroids = []
        for i in range(self.k):
            self.centroids.append(X[i])
        for i in range(self.max_iters):
            # Create empty list of clusters
            clusters = [[] for _ in range(self.k)]
            # Assign each point to its closest centroid
            for point in X:
                distances = [euclidean_distance(point, centroid) for centroid in self.centroids]
                cluster_idx = np.argmin(distances)
                clusters[cluster_idx].append(point)

            # Save old centroids before updating them
            old_centroids = self.centroids.copy()
            # Update centroids by taking mean of all points assigned to that centroid
            for j in range(self.k):
                self.centroids[j] = np.mean(clusters[j], axis=0)

            # If centroids have not moved, break out of loop
            if euclidean_distance(old_centroids, self.centroids) == 0:
                break
    def predict(self, X):
        predictions = []
        # Assign each point to its closest centroid and save its cluster label as prediction
        for point in X:
            distances = [euclidean_distance(point, centroid) for centroid in self.centroids]
            cluster_idx = np.argmin(distances)
            predictions.append(cluster_idx)
        return predictions

# Create instance of KMeans class with k=3 clusters
kmeans = KMeans(k=3)

# Fit KMeans model to Iris dataset
kmeans.fit(X)
# Plot data points using scatter plot where x-axis represents Sepal Length and y-axis represents Sepal Width. The color of each point represents its cluster label.
plt.scatter(X[:,0], X[:,1], c=kmeans.predict(X), cmap='magma')
plt.xlabel('Sepal Length', fontsize=10)
plt.ylabel('Sepal Width', fontsize=10)
plt.show()