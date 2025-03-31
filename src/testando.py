import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris

# Carrega os dados
X, _ = load_iris(return_X_y=True)

# Calcula a ligação hierárquica
Z = linkage(X, method='ward')

# Plota o dendrograma
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.axhline(y=250, color='r', linestyle='--')  # Linha de corte para observar os clusters
plt.show()
