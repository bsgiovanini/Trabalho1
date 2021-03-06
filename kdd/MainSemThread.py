import multiprocessing

from sklearn.decomposition import PCA
from sklearn import svm
import pandas as pd


class BaseDados:
    def __init__(self, nome, dados):
        self.nome = nome
        self.dados = dados


class Main:
    bases = []
    numMaxComponentesPCA = 15
    numFolds = 10
    out_q = multiprocessing.Queue()
    resultados = []


    # parametros opcionais para o classificador, caso queiramos variar para avaliar o resultado
    def run(self, pSVMGama=0.01, pSVMC=100.):
        self.carregaContextosDF()

        for base in self.bases:

            print "base: ", base.nome

            for numComponentesPCA in range(1, self.numMaxComponentesPCA + 1):

                print "numero de componentes PCA: ", numComponentesPCA

                processos = []

                classifier = svm.SVC(gamma=pSVMGama, C=pSVMC)
                modeloPCA = self.calculaTransformacaoPCADF(base.dados, numComponentesPCA)

                for fold in range(1, self.numFolds + 1):
                    conjuntos = self.separaConjuntosDF(fold, base.dados)

                    Y = conjuntos['treino'].as_matrix(conjuntos['treino'].columns[0:-2])
                    treinoPCA = modeloPCA.transform(Y)

                    # Com o modelo retornado, e possivel aplicar o modelo treinado no conjunto de testes.
                    X = conjuntos['teste'].as_matrix(conjuntos['teste'].columns[0:-2])
                    testePCA = modeloPCA.transform(X)

                    classesTreino = conjuntos['treino']['classe']
                    classesTeste = conjuntos['teste']['classe']

                    self.get_classifier_accuracy(classifier, treinoPCA, classesTreino, testePCA, classesTeste, fold, self.resultados)

                results = []
                for result in self.resultados:
                    results.append(result[1])
                    print numComponentesPCA, ';', result[0], ';', result[1]

                accuracy = sum(results) / len(results)


                # Montar uma tabela e grafico com o nome da base, numero de componentes, acuracia media
                print "acuracia media para o numero de componentes", accuracy


    def get_classifier_accuracy(self, classifier, treinoPCA, classesTreino, testePCA, classesTeste, foldNum, resultados):
        # realiza a classificacao e determina o percentual em cima dos dados de teste
        score = classifier.fit(treinoPCA, classesTreino).score(testePCA, classesTeste)

        # pega os resultados
        resultados.append((foldNum, score))
        return


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
        # A linha a seguir ja transforma o X em PCA, nos precisamos do modelo
        # Agora nos temos o modelo sendo retornado
        pca.fit(X)
        return pca


if __name__ == '__main__':
    Main().run()







