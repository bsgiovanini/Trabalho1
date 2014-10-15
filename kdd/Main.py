from sklearn.decomposition import PCA
import pandas as pd


class BaseDados:
    def __init__(self, nome, dados):
        self.nome = nome
        self.dados = dados


class Main:
    bases = []
    numMaxComponentesPCA = 10
    numFolds = 10

    def run(self):
        self.carregaContextosDF()

        for base in self.bases:
            for numComponentesPCA in range(1, self.numMaxComponentesPCA + 1):
                for fold in range(1, self.numFolds + 1):
                    conjuntos = self.separaConjuntosDF(fold, base.dados)

                    modeloPCA, treinoPCA = self.calculaTransformacaoPCADF(conjuntos['treino'], numComponentesPCA)

                    #Com o modelo retornado, e possivel aplicar o modelo treinado no conjunto de testes.
                    X = conjuntos['teste'].as_matrix(conjuntos['teste'].columns[0:-2])
                    testePCA = modeloPCA.transform(X)

                    self.classificaSVM(treinoPCA, conjuntos['treino'])
                    self.classificaSVM(testePCA, conjuntos['teste'])


    # def calculaTransformacaoPCA(self, dados, numComponentes):
    #
    # dadosPCA = []
    #
    # for tupla in dados:
    # tuplaPCA = list([x for x in tupla[0:-2]])
    # dadosPCA.append(tuplaPCA)
    # pca = PCA(n_components=numComponentes)
    # X = np.array(dadosPCA)
    #     return pca.fit_transform(X)

    def classificaSVM(self, dadosPCA, dados):

        # i = 0
        #
        # dadosSVM = []
        # for tupla in dadosPCA:
        #     tuplaSVM = np.append(tupla, [dados[i][-2]])
        #     dadosSVM.append(tuplaSVM)
        #     i += 1

        print "falta classificar!!! "


    # def carregaContextos(self):
    #     dadosDblp = []
    #     dadosAmazon = []
    #     dadosFlickr = []
    #
    #     with open('../dados/dblp.csv') as f:
    #         for line in csv.reader(f, delimiter=";"):
    #             item = list([float(x) for x in line[:]])
    #             dadosDblp.append(item)
    #
    #     with open('../dados/amazon.csv') as f:
    #         for line in csv.reader(f, delimiter=";"):
    #             item = list([float(x) for x in line[:]])
    #             dadosAmazon.append(item)
    #
    #     with open('../dados/flickr.csv') as f:
    #         for line in csv.reader(f, delimiter=";"):
    #             item = list([float(x) for x in line[:]])
    #             dadosFlickr.append(item)
    #
    #     self.bases.append(BaseDados("dblp", dadosDblp))
    #     self.bases.append(BaseDados("amazon", dadosAmazon))
    #     self.bases.append(BaseDados("flickr", dadosFlickr))

    # def separaConjuntos(self, fold, dados):
    #     conjuntos = {}
    #     fold = float(fold)
    #     conjuntos['treino'] = [elem for elem in dados if elem[-1] != fold]
    #     conjuntos['teste'] = [elem for elem in dados if elem[-1] == fold]
    #     return conjuntos

    def carregaContextosDF(self):
        dadosDblpDF = pd.read_csv('../dados/dblp.csv', sep=';', header=None, names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', 'classe', 'fold'])
        dadosAmazonDF = pd.read_csv('../dados/amazon.csv', sep=';', header=None, names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', 'classe', 'fold'])
        dadosFlickrDF = pd.read_csv('../dados/flickr.csv', sep=';', header=None, names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', 'classe', 'fold'])

        self.bases.append(BaseDados("dblp", dadosDblpDF))
        self.bases.append(BaseDados("amazon", dadosAmazonDF))
        self.bases.append(BaseDados("flickr", dadosFlickrDF))


    def separaConjuntosDF(self, fold_num, dados):
        conjuntos = {}
        conjuntos['treino'] = dados[dados.fold != fold_num]
        conjuntos['teste'] = dados[dados.fold == fold_num]
        return conjuntos

    def calculaTransformacaoPCADF(self, dados, numComponentes):
        pca = PCA(n_components=numComponentes)
        X = dados.as_matrix(dados.columns[0:-2])
        #A linha a seguir ja transforma o X em PCA, nos precisamos do modelo
        # return pca.fit_transform(X)
        #Agora nos temos o modelo sendo retornado assim como o X transformado
        pca.fit(X)
        return pca, pca.transform(X)


if __name__ == '__main__':
    Main().run()







