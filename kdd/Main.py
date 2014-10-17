from sklearn.decomposition import PCA
from sklearn import svm
import multiprocessing
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

            print "base: ", base.nome

            for numComponentesPCA in range(1, self.numMaxComponentesPCA + 1):

                print "numero de componentes PCA: ", numComponentesPCA

                processos = []
                resultados = []

                classifier = svm.SVC(gamma=0.01, C=100.)

                for fold in range(1, self.numFolds + 1):
                    conjuntos = self.separaConjuntosDF(fold, base.dados)

                    modeloPCA, treinoPCA = self.calculaTransformacaoPCADF(conjuntos['treino'], numComponentesPCA)

                    #Com o modelo retornado, e possivel aplicar o modelo treinado no conjunto de testes.
                    X = conjuntos['teste'].as_matrix(conjuntos['teste'].columns[0:-2])
                    testePCA = modeloPCA.transform(X)

                    classesTreino = conjuntos['treino']['classe']
                    classesTeste = conjuntos['teste']['classe']

                    # appenda os resultados obtidos para cada fold para futura media
                    resultados.append(self.classificaSVM(classifier, processos, treinoPCA, classesTreino, testePCA, classesTeste))

                for p in processos:
                    p.join()

                accuracy = sum(resultados)/len(resultados)

                print "acuracia para o numero de componentes", accuracy



    def get_classifier_accuracy(self, classifier, treinoPCA, classesTreino, testePCA, classesTeste, out_score):
        #realiza a classificacao e determina o percentual em cima dos dados de teste
        score = classifier.fit(treinoPCA, classesTreino).score(testePCA, classesTeste)

        #pega os resultados
        out_score.put(score)
        return

    def classificaSVM(self, classifier, processos, treinoPCA, classesTreino,  testePCA, classesTeste):
        out_q = multiprocessing.Queue()
        p = multiprocessing.Process(target=Main.get_classifier_accuracy, args=(self, classifier, treinoPCA, classesTreino,  testePCA, classesTeste,  out_q))
        processos.append(p)
        p.start()
        return out_q.get()



    def carregaContextosDF(self):
        dadosDblpDF = pd.read_csv('../dados/dblp.csv', sep=';', header=None, names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', 'classe', 'fold'])
        dadosAmazonDF = pd.read_csv('../dados/amazon.csv', sep=';', header=None, names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', 'classe', 'fold'])
        dadosFlickrDF = pd.read_csv('../dados/flickr.csv', sep=';', header=None, names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', 'classe', 'fold'])

        self.bases.append(BaseDados("dblp", dadosDblpDF))
        #self.bases.append(BaseDados("amazon", dadosAmazonDF))
        #self.bases.append(BaseDados("flickr", dadosFlickrDF))


    def separaConjuntosDF(self, fold_num, dados):
        conjuntos = {}
        conjuntos['treino'] = dados[dados.fold != fold_num]
        conjuntos['teste'] = dados[dados.fold == fold_num]
        return conjuntos

    def calculaTransformacaoPCADF(self, dados, numComponentes):
        pca = PCA(n_components=numComponentes)
        X = dados.as_matrix(dados.columns[0:-2])
        #A linha a seguir ja transforma o X em PCA, nos precisamos do modelo
        #Agora nos temos o modelo sendo retornado assim como o X transformado
        pca.fit(X)
        return pca, pca.transform(X)


if __name__ == '__main__':
    Main().run()







