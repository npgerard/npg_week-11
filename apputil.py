import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans
from seaborn import load_dataset
import time


# Global variables
# Load the diamonds dataset from seaborn
df_diamonds = load_dataset('diamonds')
# reduce the dataset to only include numeric columns
df_diamonds = df_diamonds.select_dtypes(include=[np.number])



# Exercise 1: K-Means Clustering
def kmeans(X, k=3):
    '''Performs K-Means clustering on the given input dataset (np array) with k clusters.
    Results:
    - labels: The cluster labels for each data point (ROW in X)
    - centroids: The coordinates of the cluster centers (centroids)
    '''

    # Initialize KMeans model with the specified number of clusters
    kmeans = KMeans(n_clusters=k)

    # fit the model to the data
    kmeans.fit(X)

    # return the labels and cluster centers
    return kmeans.labels_, kmeans.cluster_centers_




# testting the kmeans function
# # note to self; each row is a data point, each column is a feature
# X = np.array(
#             [[1, 2, 3],
#             [4, 5, 6],
#              [7, 8, 9]])

# labels, centroids = kmeans(X, k=2)


# print ("Labels:\n", labels)
# print ("Centroids:\n", centroids)





# Exercise 2: K-Means on Diamonds Dataset
def kmeans_diamonds(n=1000, k=5):
    '''Performs K-Means clustering on the diamonds dataset.
    Samples n data points from the dataset.
    Results:
    - labels: The cluster labels for each data point (ROW in X)
    - centroids: The coordinates of the cluster centers (centroids)
    - df_sample: The sampled dataframe used for clustering (for testing/inspection)
    '''
    # Sample n data points from the diamonds dataset
    df_sample = df_diamonds.sample(n=n, random_state=42)

    # creaate the array of values to cluster on
    X = df_sample.values

    # Perform K-Means clustering
    labels, centroids = kmeans(X, k=k)
    return labels, centroids, df_sample


# Exercise 3: kmeans timing
def kmeans_timer(n, k, n_iter=5):
    '''Times the kmeans_diamond function over n_iter iterations.
    Returns the average time taken to perform kmeans_diamond with n samples and k clusters.
    '''
    
    # initialize the timer at 0
    total_time = 0.0

    # loop for n_iter iterations
    for _ in range(n_iter):
        # set the start time
        start_time = time.time()
        # run the kmeans_diamond function
        kmeans_diamonds(n=n, k=k)
        # record the end time
        end_time = time.time()
        # add the difference in the start and end time to the running total 
        total_time += (end_time - start_time)

    average_time = total_time / n_iter
    return average_time


# # testing kmeans_timer function
# kmeans_timer_result = kmeans_timer(n=1000, k=5, n_iter=5)

# # Test the kmeans_diamond function
# diamond_labels, diamond_centroids, df_diamond_sample = kmeans_diamond(n=1000, k=4)

# # Print out the results
# for cluster_id in np.unique(diamond_labels):
#     points_in_cluster = df_diamond_sample[diamond_labels == cluster_id]  # <-- use original data
#     print(f"Label: {cluster_id}; Centroid: {diamond_centroids[cluster_id]}")
#     print(points_in_cluster, "\n")