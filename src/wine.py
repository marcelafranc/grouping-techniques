import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import MinMaxScaler

# Carregar o dataset Wine
wine = load_wine()



def printWine():
    # Carregar o dataset Wine
    wine = load_wine()
    X = wine.data
    y = wine.target
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    df_wine = pd.DataFrame(X, columns=wine.feature_names)

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


    plt.legend(['Pontos maiores indicam maior quantidade de Ash (Material inorgânico) no vinho', 'Pontos menores indicam menor quantidade de Ash (Material inorgânico) no vinho'], loc='upper right')

    # Mostrar o gráfico
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

# Agrupamento Hierárquico do Wine
def hierarquicoWine():
    X = wine.data

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
def tecnica_elbow(X, max_k=15):
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
def particionalWine(num_iterations=100):
    wine = load_wine()
    X = wine.data[:, [0, 1, 2]]  # Alcohol, malic acid, ash e color intensity(atributos que queremos usar)

    # Normalizar os dados para o KMeans
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
        kmeans = KMeans(n_clusters=k_optimal, random_state=None, n_init=1, init='random')  # n_init=1 para inicialização aleatória
        kmeans.fit(X_normalized)
        
        # Verificar o erro quadrático (inertia) e atualizar o melhor modelo
        if kmeans.inertia_ < best_inertia:
            best_inertia = kmeans.inertia_
            best_labels = kmeans.labels_
            best_centroids = kmeans.cluster_centers_
        print("INERCIA ", i ,": ", kmeans.inertia_)
    print("Melhor inercia: ", best_inertia)

    # Desnormalizar os centróides
    best_centroids_original = scaler.inverse_transform(best_centroids)

    # Ajustando o tamanho dos pontos com base no valor de "Ash"
    ash_scaled = np.interp(X[:, 2], (np.min(X[:, 2]), np.max(X[:, 2])), (10, 200))  # Escalona o tamanho para visualização

    # Definindo uma paleta de cores distintas
    colors = plt.cm.get_cmap('tab20', k_optimal)

    # Plotando o gráfico
    plt.figure(figsize=(10, 8))

    # Mapeando a cor dos clusters
    scatter = plt.scatter(
        X[:, 1],  # Malic Acid → eixo X
        X[:, 0],  # Alcohol → eixo Y
        c=best_labels,  # Cor do ponto baseada no cluster
        cmap=colors,
        s=ash_scaled,  # Tamanho do ponto (com base em "Ash")
        alpha=0.7,
        edgecolor='k'
    )

    # Plotando os centróides com a mesma cor dos seus clusters
    for cluster_id in range(k_optimal):
        plt.scatter(
            best_centroids_original[cluster_id, 1],  # Malic Acid (X)
            best_centroids_original[cluster_id, 0],  # Alcohol (Y)
            c=[colors(cluster_id / k_optimal)],  # Cor do centróide com base no cluster
            s=200,  # Tamanho fixo para centróides
            marker='X',  # Marcador de centróide
            edgecolor='black',
            label=f'Centroid {cluster_id}'
        )

    min_alcohol = np.min(X[:, 0])
    plt.ylim(min_alcohol - 0.5, np.max(X[:, 0]) + 0.5)
    min_malic = np.min(X[:, 1])
    plt.xlim(min_malic - 0.5, np.max(X[:, 1]) + 0.5)

    # Configurações do gráfico
    plt.xlabel('Malic Acid (g/L)')
    plt.ylabel('Alcohol (%)')
    plt.title(f'Clusters com K={k_optimal} - Melhor Distribuição de Centróides')

    plt.legend(loc='best')
    plt.show()