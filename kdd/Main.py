
import csv
from sklearn.decomposition import PCA
import numpy as np

class Contexto:

    def __init__(self, nome, dados):
        self.nome = nome
        self.dados = dados

contextos = []
dadosPCADblp = []
dadosPCAAmazon = []
dadosPCAFlickr = []

numMaxComponentes = 15

with open('../dados/dblp.csv') as f:
    for line in csv.reader(f, delimiter=";"):
        item = list([float(x) for x in line[0:-2]])
        dadosPCADblp.append(item)

with open('../dados/amazon.csv') as f:
    for line in csv.reader(f, delimiter=";"):
        item = list([float(x) for x in line[0:-2]])
        dadosPCAAmazon.append(item)

with open('../dados/flickr.csv') as f:
    for line in csv.reader(f, delimiter=";"):
        item = list([float(x) for x in line[0:-2]])
        dadosPCAFlickr.append(item)

contextos.append(Contexto("dblp", dadosPCADblp))
contextos.append(Contexto("amazon", dadosPCAAmazon))
contextos.append(Contexto("flickr", dadosPCAFlickr))

for contexto in contextos:

    X = np.array(contexto.dados)

    for numComponentes in range(1, numMaxComponentes + 1):
        pca = PCA(n_components=numComponentes)
        pca.fit(X)
        print(pca.explained_variance_ratio_)

