import pandas as pd
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import seaborn as sns
wine = load_wine()

def printWine():
    
    X = wine.data
    y = wine.target
    df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
    

    # Scatter plot de duas features
    #sns.scatterplot(x=df_wine.iloc[:, 0], y=df_wine.iloc[:, 1])
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
    plt.xlabel(wine.feature_names[0])
    plt.ylabel(wine.feature_names[1])
    plt.title("Visualização do Dataset Wine sem Agrupamento")
    plt.colorbar(label='Classe')
    plt.show()

#def particionalWine():

#def hierarquicoWine():