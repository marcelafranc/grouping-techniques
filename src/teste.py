from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine

# Base Iris
iris = load_wine()
X = iris.data

# Criando o linkage (pode ser 'single', 'complete', 'average', 'ward')
linked = linkage(X, method='average')

# Dendrograma
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title('Dendrograma WINE - Linkage Average')
plt.show()
