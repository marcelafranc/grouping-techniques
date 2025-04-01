import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_wine
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import MinMaxScaler

# Carregar o dataset Wine
wine = load_wine()

def pairplotWine():
    df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
    df_wine['target'] = wine.target

    sns.pairplot(df_wine, hue='target', diag_kind='hist', palette='tab10')
    plt.suptitle("Pairplot Completo do Dataset Wine", y=1.02)
    plt.show()

def printWine():
    # Carregar o dataset Wine
    wine = load_wine()
    X = wine.data
    y = wine.target
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # Atributos para o gráfico
    alcohol = X[:, 0]        # Coordenada Y (Álcool)
    malic_acid = X[:, 1]     # Coordenada X (Ácido málico)
    ash = X_normalized[:, 2]            # Cor do ponto (Cinzas)

    # Plotar gráfico
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(malic_acid, alcohol, c='gray', s=ash*100, cmap='viridis', edgecolor='k')
    
    # Adicionar título e rótulos
    plt.xlabel('Malic Acid ( g/L )')
    plt.ylabel('Alcohol %')
    plt.title("Visualização do Dataset Wine")

    # Adicionar colorbar para mostrar a escala da cor (ash)
    #plt.colorbar()

    plt.legend(['Pontos maiores indicam maior quantidade de Ash (Material inorgânico) no vinho', 'Pontos menores indicam menor quantidade de Ash (Material inorgânico) no vinho'], loc='upper right')

    # Mostrar o gráfico
    plt.show()

# Método para determinar o número de clusters usando o Silhouette Score!!!!!!
def metodo_silhouette(X_normalized):
    scores = []
    # testa de 2 a 10
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_normalized)
        score = silhouette_score(X_normalized, kmeans.labels_)
        scores.append(score)

    # Plotar o grafico do silhouete !!
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, 11), scores, marker='o')
    plt.title('Silhouette Score para Determinação de k')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.show()

    # encontrar o k com o melhor silhouete (definicao)
    best_k = scores.index(max(scores)) + 2  # comeca em 2
    print(f"O melhor número de clusters (k) baseado no Silhouette Score é: {best_k}")
    return best_k

# Agrupamento Hierárquico do Wine
def hierarquicoWine():
    # Carregar o dataset Wine
    wine = load_wine()
    X = wine.data[:, [0, 1, 2]]
    
    # Normalizar os dados
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
   
    # 1- DENDROGRAMAS PRA DECIDIR ONDE TRACAR A LINHA (DEFINIR K)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    linkages = ["ward", "complete", "average", "single"]
    titles = [
        "Método Ward",
        "Método Complete Linkage",
        "Método Average Linkage",
        "Método Single Linkage"
    ]
    
    for i, linkage_method in enumerate(linkages):
        Z = linkage(X_normalized, method=linkage_method)
        dendrogram(Z, ax=axes[i//2, i%2])
        axes[i//2, i%2].set_title(f"Dendrograma - {titles[i]}")
        axes[i//2, i%2].set_xlabel("Amostras")
        axes[i//2, i%2].set_ylabel("Distância")
    
    plt.tight_layout()
    plt.show()

    # 2 - UTILIZAR SILHOUETTE SCORE PARA DETERMINAR O MELHOR K
    best_k = metodo_silhouette(X_normalized)

    # 3 - GRAFICOS DE DISPERSAO USANDO K DEFINIDO (MELHOR k)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    linkages = ["ward", "complete", "average", "single"]
    titles = [
        "Método Ward",
        "Método Complete Linkage",
        "Método Average Linkage",
        "Método Single Linkage"
    ]
    
    for i, linkage_method in enumerate(linkages):
        clustering = AgglomerativeClustering(n_clusters=best_k, linkage=linkage_method)
        y_hr = clustering.fit_predict(X_normalized)
        
        scatter = axes[i//2, i%2].scatter(X[:, 1], X[:, 0], c=y_hr, cmap='viridis', s=X[:, 2] * 50, edgecolor='k')
        axes[i//2, i%2].set_title(titles[i])
        axes[i//2, i%2].set_xlabel('Malic Acid ( g/L )')
        axes[i//2, i%2].set_ylabel('Alcohol %')
    
    plt.tight_layout()
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
def particionalWine(num_iterations=1000):
    wine = load_wine()
    X = wine.data[:, [0, 1, 2]]  # Alcohol, Malic Acid, Ash

    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    k_optimal = tecnica_elbow(X_normalized)

    best_silhouette = -1
    best_labels = None
    best_centroids = None

    for i in range(num_iterations):
        kmeans = KMeans(n_clusters=k_optimal, random_state=None, n_init=1, init='random')
        kmeans.fit(X_normalized)
        
        silhouette_avg = silhouette_score(X_normalized, kmeans.labels_)
        print(f"Iteração {i + 1}, Silhouette Score: {silhouette_avg}")
        
        if silhouette_avg > best_silhouette:
            best_silhouette = silhouette_avg
            best_labels = kmeans.labels_
            best_centroids = kmeans.cluster_centers_
            best_trial = i

    print(f"Melhor Silhueta: {best_silhouette}, Obtido no trial {best_trial + 1}")

    best_centroids_original = scaler.inverse_transform(best_centroids)
    ash_scaled = np.interp(X[:, 2], (np.min(X[:, 2]), np.max(X[:, 2])), (10, 200))
    colors = plt.cm.get_cmap('tab20', k_optimal)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, 1], X[:, 0], c=best_labels, cmap=colors, s=ash_scaled, alpha=0.7, edgecolor='k')

    for cluster_id in range(k_optimal):
        plt.scatter(
            best_centroids_original[cluster_id, 1],
            best_centroids_original[cluster_id, 0],
            c=[colors(cluster_id / k_optimal)],
            s=200,
            marker='X',
            edgecolor='black',
            label=f'Centroid {cluster_id}'
        )

    plt.xlabel('Malic Acid (g/L)')
    plt.ylabel('Alcohol (%)')
    plt.title(f'Agrupamento particional do dataset WINE, 1000 iterações')
    plt.legend(loc='best')
    plt.show()