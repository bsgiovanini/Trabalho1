
import csv
from sklearn.decomposition import PCA
import numpy as np

class BaseDados:

    def __init__(self, nome, dados):
        self.nome = nome
        self.dados = dados


class Main:

    bases = []
    numMaxComponentesPCA = 10
    numFolds = 10

    def run(self):
        self.carregaContextos()

        for base in self.bases:

            for numComponentesPCA in range(1, self.numMaxComponentesPCA + 1):

                for fold in range(1, self.numFolds+1):
                    conjuntos = self.separaConjuntos(fold, base.dados)
                    treinoPCA = self.calculaTransformacaoPCA(conjuntos['treino'], numComponentesPCA)
                    self.classificaSVM(treinoPCA, conjuntos['treino'])
                    testePCA = self.calculaTransformacaoPCA(conjuntos['teste'], numComponentesPCA)
                    self.classificaSVM(testePCA, conjuntos['teste'])



    def calculaTransformacaoPCA(self, dados, numComponentes):

        dadosPCA = []

        for tupla in dados:
            tuplaPCA = list([x for x in tupla[0:-2]])
            dadosPCA.append(tuplaPCA)
        pca = PCA(n_components=numComponentes)
        X = np.array(dadosPCA)
        return pca.fit_transform(X)

    def classificaSVM(self, dadosPCA, dados):

        i = 0

        dadosSVM = []
        for tupla in dadosPCA:
            tuplaSVM = np.append(tupla, [dados[i][-2]])
            dadosSVM.append(tuplaSVM)
            i+=1

        print "falta classificar!!! "



    def carregaContextos(self):
        dadosDblp = []
        dadosAmazon = []
        dadosFlickr = []

        with open('../dados/dblp.csv') as f:
            for line in csv.reader(f, delimiter=";"):
                item = list([float(x) for x in line[:]])
                dadosDblp.append(item)

        with open('../dados/amazon.csv') as f:
            for line in csv.reader(f, delimiter=";"):
                item = list([float(x) for x in line[:]])
                dadosAmazon.append(item)

        with open('../dados/flickr.csv') as f:
            for line in csv.reader(f, delimiter=";"):
                item = list([float(x) for x in line[:]])
                dadosFlickr.append(item)

        self.bases.append(BaseDados("dblp", dadosDblp))
        self.bases.append(BaseDados("amazon", dadosAmazon))
        self.bases.append(BaseDados("flickr", dadosFlickr))

    def separaConjuntos(self, fold, dados):
        conjuntos = {}
        fold = float(fold)
        conjuntos['treino'] = [elem for elem in dados if elem[-1] != fold]
        conjuntos['teste'] = [elem for elem in dados if elem[-1] == fold]
        return conjuntos



Main().run()





