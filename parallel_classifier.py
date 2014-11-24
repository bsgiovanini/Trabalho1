import sys
import multiprocessing

import numpy as np
import sklearn as sk
from sklearn import svm
from sklearn import cross_validation
from sklearn import tree
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import ensemble


def get_classifier_accuracy(classifier, dataset, pair_class, fold, out_score):
    score = classifier.fit(dataset[fold[0]], pair_class[fold[0]]).score(dataset[fold[1]], pair_class[fold[1]])
    out_score.put(score)
    return


def parallel_classifier(classifier, dataset, pair_class, folds):
    out_q = multiprocessing.Queue()
    procs = []
    nprocs = len(folds)

    for fold in folds:
        p = multiprocessing.Process(target=get_classifier_accuracy, args=(classifier, dataset, pair_class, fold, out_q))
        procs.append(p)
        p.start()

    result = []
    for i in range(nprocs):
        result.append(out_q.get())

    for p in procs:
        p.join()

    accuracy = sum(result) / len(result)

    return accuracy


if __name__ == "__main__":

    input_f = sys.argv[1]
    data_set = open(input_f)
    lines = data_set.readlines()

    n_l = len(lines)
    n_c = len(lines[0].strip().split(';'))

    var = range(n_c - 2)

    metrics = np.zeros((n_l, len(var)))
    pair_class = np.zeros([n_l])
    fold = np.zeros([n_l])

    i = 0
    map_folds = {}
    for line in lines:
        col = line.strip().split(';')
        metrics[i] = [float(x) for x in col[0:len(var)]]
        r = metrics[:]
        pair_class[i] = int(col[-2])
        fold[i] = int(col[-1])
        dict.setdefault(map_folds, fold[i], set())
        map_folds[fold[i]].add(i)
        i += 1

    train_test_folds = []

    for k, v in map_folds.items():
        train_test_folds.append((list(set(range(n_l)) - v), list(v)))

    print "Predicao de Links - Arquivo: %s\n%s elementos, %s variaveis" % (input_f, n_l, len(var))

    classifier = svm.SVC(gamma=0.01, C=100.)
    accuracy = parallel_classifier(classifier, r, pair_class, train_test_folds)

    print "Algoritmo: SVM"
    print "Acuracia: %s" % accuracy
