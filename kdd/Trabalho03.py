# -*- coding: utf-8 -*-

import multiprocessing
import random

from sklearn.decomposition import PCA
from sklearn import svm
import pandas as pd


class BaseDados:
    def __init__(self, nome, dados, type):
        self.nome = nome
        self.dados = dados
        self.type = type


class Main:
    bases = []
    numMaxComponentesPCA = 13
    numFolds = 1
    out_q = multiprocessing.Queue()
    pca = False


    # parametros opcionais para o classificador, caso queiramos variar para avaliar o resultado
    def run(self, pSVMGama=0.01, pSVMC=100.):
        self.carregaContextosDF()

        for base in self.bases:

            print "base: ", base.nome
            print "Numero de PCAs;Fold (Vulgo divisao aleatoria das amostras);acuracia"

            classifier = svm.SVC(gamma=pSVMGama, C=pSVMC)
            regression = svm.SVR(gamma=pSVMGama, C=pSVMC)
            self.numMaxComponentesPCA = len(base.dados.columns) - 1

            for fold in range(1, self.numFolds + 1):
                # Como nao foi passado a coluna contendo os folds, Vair rodar N vezes, gerando N conjuntos aleatorios de teste e treino
                conjuntos = self.separaConjuntosDF(fold, base)
                processos = []
                resultados = []

                if self.pca:
                    for numComponentesPCA in range(1, self.numMaxComponentesPCA + 1):
                        modeloPCA = self.calculaTransformacaoPCADF(base.dados, numComponentesPCA)

                        Y = conjuntos['treino'].as_matrix(conjuntos['treino'].columns[0:-1])
                        treinoPCA = modeloPCA.transform(Y)

                        # Com o modelo retornado, e possivel aplicar o modelo treinado no conjunto de testes.
                        X = conjuntos['teste'].as_matrix(conjuntos['teste'].columns[0:-1])
                        testePCA = modeloPCA.transform(X)

                        classesTreino = conjuntos['treino']['Classe']
                        classesTeste = conjuntos['teste']['Classe']

                        if base.type == 'classification':
                            self.classificaSVM(classifier, processos, treinoPCA, classesTreino, testePCA, classesTeste, fold, self.out_q, numComponentesPCA)
                        else:
                            self.classificaSVM(regression, processos, treinoPCA, classesTreino, testePCA, classesTeste, fold, self.out_q, numComponentesPCA)
                else:
                    Y = conjuntos['treino'].as_matrix(conjuntos['treino'].columns[0:-1])
                    # Com o modelo retornado, e possivel aplicar o modelo treinado no conjunto de testes.
                    X = conjuntos['teste'].as_matrix(conjuntos['teste'].columns[0:-1])

                    classesTreino = conjuntos['treino']['Classe']
                    classesTeste = conjuntos['teste']['Classe']

                    if base.type == 'classification':
                        self.classificaSVM(classifier, processos, Y, classesTreino, X, classesTeste, fold, self.out_q, 0)
                    else:
                        self.classificaSVM(regression, processos, Y, classesTreino, X, classesTeste, fold, self.out_q, 0)

                for p in processos:
                    p.join()
                    resultados.append(self.out_q.get())

                results = []
                for result in resultados:
                    results.append(result[2])
                    print result[0], ';', result[1], ';', result[2]

                accuracy = sum(results) / len(results)


                # Montar uma tabela e grafico com o nome da base, numero de componentes, acuracia media
                print "acuracia media para o numero de componentes", accuracy


    def get_classifier_accuracy(self, classifier, treinoPCA, classesTreino, testePCA, classesTeste, foldNum, out_score, pcaNum):
        # realiza a classificacao e determina o percentual em cima dos dados de teste
        score = classifier.fit(treinoPCA, classesTreino).score(testePCA, classesTeste)
        # pega os resultados
        out_score.put((pcaNum, foldNum, score))
        return

    def classificaSVM(self, classifier, processos, treinoPCA, classesTreino, testePCA, classesTeste, foldNum, out_q, pcaNum):
        p = multiprocessing.Process(target=Main.get_classifier_accuracy, args=(self, classifier, treinoPCA, classesTreino, testePCA, classesTeste, foldNum, out_q, pcaNum))
        p.start()
        processos.append(p)


    def carregaContextosDF(self):
        creditoDF = pd.read_csv('../dados/credito.csv', sep=';')

        # Todas as colunas sao mapeadas para um valor inteiro
        zooDF = pd.read_csv('../dados/zoo.txt', sep='\t',
                            converters={0: self.mapToInt, 1: self.mapToInt, 2: self.mapToInt, 3: self.mapToInt, 4: self.mapToInt, 5: self.mapToInt, 6: self.mapToInt, 7: self.mapToInt,
                                        8: self.mapToInt, 9: self.mapToInt, 10: self.mapToInt, 11: self.mapToInt, 12: self.mapToInt, 13: self.mapToInt, 14: self.mapToInt, 15: self.mapToInt,
                                        16: self.mapToInt, 17: self.mapToInt})

        irisDF = pd.read_csv('../dados/iris.data', sep=',', header=None, names=['1', '2', '3', '4', 'Classe'])

        wineDF = pd.read_csv('../dados/wine.csv', sep=';', header=None, names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', 'Classe'])

        heartDF = pd.read_csv('../dados/processed.cleveland.data', sep=',', na_values='?', header=None, names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', 'Classe'])
        heartDF.dropna(inplace=True)

        housingDF = pd.read_csv('../dados/housing.data', sep=r"\s+", na_values='?', header=None, names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', 'Classe'])

        self.bases.append(BaseDados("credito", creditoDF, 'classification'))
        self.bases.append(BaseDados("zoo", zooDF, 'classification'))
        self.bases.append(BaseDados("iris", irisDF, 'classification'))
        self.bases.append(BaseDados("wine", wineDF, 'classification'))
        self.bases.append(BaseDados("heart", heartDF, 'classification'))
        self.bases.append(BaseDados("housing", housingDF, 'regression'))

    def mapToInt(self, valor):
        if valor == 'Sim':
            return 1
        if valor == 'Não':
            return 0
        return int(valor)


    def separaConjuntosDF(self, fold_num, base):
        conjuntos = dict()
        dados = base.dados
        # 10% das linhas serao para teste e 90 para treino
        if base.type == 'classification':
            conjuntos['treino'] = pd.DataFrame()
            conjuntos['teste'] = pd.DataFrame()
            classes = dados.groupby('Classe').Classe.unique()
            for key, value in classes.iteritems():
                classe_xpto = dados[dados.Classe == key]
                num_rows = len(classe_xpto.index) / 10
                if num_rows < 1:
                    num_rows = 1
                rows = random.sample(classe_xpto.index, num_rows)
                conjuntos['treino'] = conjuntos['treino'].append(classe_xpto.drop(rows))
                conjuntos['teste'] = conjuntos['teste'].append(classe_xpto.ix[rows])
        else:
            num_rows = len(dados.index) / 10
            rows = random.sample(dados.index, num_rows)
            conjuntos['treino'] = dados.drop(rows)
            conjuntos['teste'] = dados.ix[rows]
        return conjuntos

    def calculaTransformacaoPCADF(self, dados, numComponentes):
        pca = PCA(n_components=numComponentes)
        X = dados.as_matrix(dados.columns[0:-1])
        # Retorna o modelo para ser aplicado no treino e teste
        pca.fit(X)
        return pca


if __name__ == '__main__':
    Main().run()







