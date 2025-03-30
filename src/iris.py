import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import MinMaxScaler

# Carregar o dataset Iris
iris = load_iris()

def printIris():
    # Carregar o dataset Iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Normalizar os dados
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # Atributos para o gráfico
    petal_length = X[:, 2]  # Petal Length
    petal_width = X[:, 3]   # Petal Width

    # Formas geométricas baseadas na espécie
    markers = ['o', '^', 's']  # 'o' para Setosa, '^' para Versicolor, 's' para Virginica

    # Plotando o gráfico sem agrupamento
    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.scatter(petal_width[y == i], petal_length[y == i], marker=markers[i], label=iris.target_names[i], edgecolor='k', alpha=0.7)

    plt.xlabel('Petal Width (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.title("Visualização do Dataset Iris sem Agrupamento")
    plt.legend(title='Species')
    plt.show()

# Função para determinar automaticamente o número de clusters
def determinar_n_clusters(X):
    Z = linkage(X)
    distancias = Z[:, 2]  # Coluna das distâncias das fusões
    dif_dist = np.diff(distancias)  # Diferença entre alturas consecutivas
    maior_salto = np.argmax(dif_dist)  # Índice do maior salto
    n_clusters = len(X) - maior_salto  # Número de clusters
    return n_clusters

# Função para plotar dendrograma
def plot_dendrogram(X):
    Z = linkage(X)
    plt.figure(figsize=(10, 6))
    dendrogram(Z)
    plt.title("Dendrograma")
    plt.xlabel("Amostras")
    plt.ylabel("Distância")
    plt.show()

# Agrupamento Hierárquico da Iris
def hierarquicoIris():
    X = iris.data

    # Normalizar os dados
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # Exibir dendrograma
    plot_dendrogram(X)
    
    # DETERMINA AUTOMATICAMENTE O NUMERO DE CLUSTERS
    n_clusters = determinar_n_clusters(X)
    print(f"Número de clusters sugerido: {n_clusters}")
    
    # Figuras grafico dispersao
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Lista metodos linkage e titulos
    linkages = ["ward", "complete", "average", "single"]
    titles = [
        "Método Ward",
        "Método Complete Linkage",
        "Método Average Linkage",
        "Método Single Linkage"
    ]

    # Aplicacao dos metodos linkage
    for i, linkage_method in enumerate(linkages):
        # Aplicacao AGLOMERATIVO
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        y_hr = clustering.fit_predict(X)

        # Grafico de dispersao
        axes[i//2, i%2].scatter(X[:, 0], X[:, 1], c=y_hr, cmap="viridis", s=50, edgecolor='k')
        axes[i//2, i%2].set_title(titles[i])

    # Ajusta layout
    fig.tight_layout()

    # Mostrar graficos
    plt.show()

# Função para a Técnica do Cotovelo
def tecnica_elbow(X, max_k=10):
    inertias = []
    k_values = range(2, max_k + 1)  # Começamos em 2 porque k=1 não faz sentido para clustering

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)  # Soma dos erros quadráticos dentro dos clusters

    # Plotando a curva do método do cotovelo
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, inertias, marker='o', linestyle='-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Método do Cotovelo para Determinar k (Iris Dataset)')
    plt.show()

    # Determina o melhor k pelo maior "joelho" da curva
    dif_inertias = np.diff(inertias)
    k_best = np.argmax(dif_inertias) + 2  # +2 pois o range começa em 2
    return k_best

# Função de K-means iterativo
def particionalIris(num_iterations=10):
    X = iris.data[:, [2, 3]]  # PetalLength e PetalWidth (atributos que queremos usar)
    y = iris.target  # As espécies da planta

    # Normalizar os dados
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # Encontrar o número ótimo de clusters com a técnica do cotovelo
    k_optimal = tecnica_elbow(X_normalized)

    # Inicializar variáveis para armazenar o melhor modelo
    best_inertia = np.inf
    best_labels = None
    best_centroids = None

    # Iterar várias vezes para minimizar o erro quadrático
    for i in range(num_iterations):
        kmeans = KMeans(n_clusters=k_optimal, random_state=None, n_init=1)  # n_init=1 para inicialização aleatória
        kmeans.fit(X_normalized)
        
        # Verificar o erro quadrático (inertia) e atualizar o melhor modelo
        if kmeans.inertia_ < best_inertia:
            best_inertia = kmeans.inertia_
            best_labels = kmeans.labels_
            best_centroids = kmeans.cluster_centers_

    # Definir uma paleta de cores fixa para os clusters
    cluster_colors = plt.cm.viridis(np.linspace(0, 1, k_optimal))

    # Definir os símbolos para as espécies (Triângulo, Círculo, Quadrado)
    markers = ['^', 'o', 's']  # Triângulo, Círculo, Quadrado
    species = ['Setosa', 'Versicolor', 'Virginica']  # As espécies no dataset Iris

    # Plotar o resultado final
    plt.figure(figsize=(8, 6))

    # Plotar os pontos com a cor de cada cluster e formato baseado na espécie
    for i, species_label in enumerate(np.unique(y)):
        species_indices = np.where(y == species_label)[0]
        plt.scatter(X_normalized[species_indices, 0], X_normalized[species_indices, 1], 
                    c=cluster_colors[best_labels[species_indices]],  # Usando as cores fixas para clusters
                    marker=markers[i], label=species[i], alpha=0.7)

    # Plotando os centróides (com as cores fixas de seus respectivos clusters)
    plt.scatter(best_centroids[:, 0], best_centroids[:, 1], s=200, c=cluster_colors, marker='X', label='Centroids')

    # Configurações do gráfico
    plt.xlabel('PetalWidth (Coordenada X)')
    plt.ylabel('PetalLength (Coordenada Y)')
    plt.title(f'Clusters com K={k_optimal} - Melhor Distribuição de Centróides')

    # Exibir apenas a legenda das espécies (com os formatos) e os centróides
    plt.legend(loc='best')
    plt.show()