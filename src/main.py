import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Carregar Iris
iris = load_iris()
X = iris.data
y = iris.target

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means "normal"
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
print("Silhouette (K-Means):", silhouette_score(X_scaled, kmeans_labels))

# Exemplo de linkage hierárquico
linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, labels=y)
plt.title("Dendrograma (Linkage Ward)")
plt.show()

# Cortar o dendrograma
cluster_labels = fcluster(linked, t=3, criterion='maxclust')
print("Silhouette (Hierárquico):", silhouette_score(X_scaled, cluster_labels))
