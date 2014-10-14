
import csv
from sklearn.decomposition import PCA
import numpy as np

class Contexto:

    def __init__(self, nome, dados):
        self.nome = nome
        self.dados = dados

contextos = []
dadosPCADblp = []
numMaxComponentes = 15

with open('../dados/dblp_dataset.csv') as f:
    for line in csv.reader(f, delimiter=";"):
        item = list([float(x) for x in line[0:-2]])
        dadosPCADblp.append(item)

contextos.append(Contexto("dblp", dadosPCADblp))

for contexto in contextos:

    X = np.array(contexto.dados)

    for numComponentes in range(1, numMaxComponentes):
        pca = PCA(n_components=numComponentes)
        pca.fit(X)
        print(pca.explained_variance_ratio_)

