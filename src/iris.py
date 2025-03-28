import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
iris = load_iris()

def printIris():
    X = iris.data
    y = iris.target
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Scatter plot de duas features
    #sns.scatterplot(x=df_iris.iloc[:, 0], y=df_iris.iloc[:, 1])
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title("Visualização do Dataset Iris sem Agrupamento")
    plt.colorbar(label='Classe')
    plt.show()

#def particionalIris():

#def hierarquicoIris():