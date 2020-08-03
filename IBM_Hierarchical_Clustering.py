# %%  Agglomerative Hierarchical Clustering(bottom up)
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from sklearn import datasets, manifold
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs

# %% Generating Random Data
"""
Parameters into make_blobs:
n_samples: The total number of points equally divided among clusters.
    Choose a number from 10-1500
centers: The number of centers to generate, or the fixed center locations.
    Choose arrays of x,y coordinates for generating the centers. Have 1-10 centers (ex. centers=[[1,1], [2,5]])
cluster_std: The standard deviation of the clusters. The larger the number, the further apart the clusters
    Choose a number between 0.5-1.5
"""
X1, y1 = make_blobs(n_samples=50, centers=[[4, 4], [-2, -1], [1, 1], [10, 4]], cluster_std=0.9)
plt.scatter(X1[:, 0], X1[:, 1], marker='o')
plt.show()

# %% Agglomerative Clustering
"""
The Agglomerative Clustering class will require two inputs:

n_clusters: The number of clusters to form as well as the number of centroids to generate.
    Value will be: 4
linkage: Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion.
    Value will be: 'complete'
    Note: It is recommended you try everything with 'average' as well
"""
agglom = AgglomerativeClustering(n_clusters=4, linkage='average')
agglom.fit(X1, y1)

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(6, 4))

# These two lines of code are used to scale the data points down,
# Or else the data points will be scattered very far apart.

# Create a minimum and maximum range of X1.
x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0)

# Get the average distance for X1.
X1 = (X1 - x_min) / (x_max - x_min)

# This loop displays all of the datapoints.
for i in range(X1.shape[0]):
    # Replace the data points with their respective cluster value
    # (ex. 0) and is color coded with a colormap (plt.cm.spectral)
    plt.text(X1[i, 0], X1[i, 1], str(y1[i]),
             color=plt.cm.nipy_spectral(agglom.labels_[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})

# Remove the x ticks, y ticks, x and y axis
plt.xticks([])
plt.yticks([])
# plt.axis('off')


# Display the plot of the original data before clustering
plt.scatter(X1[:, 0], X1[:, 1], marker='.')
# Display the plot
plt.show()

# %% Dendrogram Associated for the Agglomerative Hierarchical Clustering
dist_matrix = distance_matrix(X1, X1)
print(dist_matrix)

# Use the linkage class from hierarchy, pass in the parameters
Z = hierarchy.linkage(dist_matrix, 'complete')

dendro = hierarchy.dendrogram(Z)
plt.show()

# %% Clustering on Vehicle dataset
# read data
pdf = pd.read_csv('cars_clus.csv')
print(pdf.shape)
print(pdf.head())

# %% PreProcessing
# Data Cleaning
print("Shape of dataset before cleaning: ", pdf.shape)
pdf[['sales', 'resale', 'type', 'price', 'engine_s',
     'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
     'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
                               'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
                               'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)
print("Shape of dataset after cleaning: ", pdf.shape)
print(pdf.head())

# Feature Selection
featureset = pdf[['engine_s', 'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]

# Normalization
# MinMaxScaler transforms features by scaling each feature to a given range. It is by default (0, 1)
from sklearn.preprocessing import MinMaxScaler

x = featureset.values
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)
print(feature_mtx[:5])

# %% Clustering using Scipy
import scipy
import pylab
import scipy.cluster.hierarchy

leng = feature_mtx.shape[0]
D = np.zeros([leng, leng])
for i in range(leng):
    for j in range(leng):
        D[i, j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])

# Supported parameters: single - complete - average - weighted - centroid
Z = hierarchy.linkage(D, 'complete')

# %%
# Essentially, Hierarchical clustering does not require a pre-specified number of clusters.
# However, in some applications we want a partition of disjoint clusters just as in flat clustering.
# So you can use a cutting line:

from scipy.cluster.hierarchy import fcluster

max_d = 3
clusters = fcluster(Z, max_d, criterion='distance')
print(clusters)

# you can determine the number of clusters directly:
k = 5
clusters = fcluster(Z, k, criterion='maxclust')
print(clusters)

# %% Plot Dendogram
fig = pylab.figure(figsize=(18, 50))


def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])))


dendro = hierarchy.dendrogram(Z, leaf_label_func=llf, leaf_rotation=0, leaf_font_size=12, orientation='right')
plt.show()

# %% Clustering using scikit-learn
dist_matrix = distance_matrix(feature_mtx, feature_mtx)
print(dist_matrix)

agglom = AgglomerativeClustering(n_clusters=6, linkage='complete')
agglom.fit(feature_mtx)
print(agglom.labels_)

# add a new field to our dataframe to show the cluster of each row
pdf['cluster_'] = agglom.labels_
print(pdf.head())

# %%
import matplotlib.cm as cm

n_clusters = max(agglom.labels_) + 1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(16, 14))

for color, label in zip(colors, cluster_labels):
    subset = pdf[pdf.cluster_ == label]
    for i in subset.index:
        plt.text(subset.horsepow[i], subset.mpg[i], str(subset['model'][i]), rotation=25)
    plt.scatter(subset.horsepow, subset.mpg, s=subset.price * 10, c=color, label='cluster' + str(label), alpha=0.5)
#    plt.scatter(subset.horsepow, subset.mpg)
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
plt.show()

# %%
pdf.groupby(['cluster_', 'type'])['cluster_'].count()

agg_cars = pdf.groupby(['cluster_', 'type'])['horsepow', 'engine_s', 'mpg', 'price'].mean()
print(agg_cars)

# %%
plt.figure(figsize=(16, 10))
for color, label in zip(colors, cluster_labels):
    subset = agg_cars.loc[(label,),]
    for i in subset.index:
        plt.text(subset.loc[i][0] + 5, subset.loc[i][2],
                 'type=' + str(int(i)) + ', price=' + str(int(subset.loc[i][3])) + 'k')
    plt.scatter(subset.horsepow, subset.mpg, s=subset.price * 20, c=color, label='cluster' + str(label))
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
plt.show()
