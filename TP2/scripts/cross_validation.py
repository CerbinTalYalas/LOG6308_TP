import pandas as pd
import numpy as np
import random


def gen_sets(adjacent, k=10):
    adjacent_values = adjacent.values
    l, c = np.where(adjacent_values == 1)
    i = len(l)
    sets = []
    test_indexes = list(range(i))
    random.shuffle(test_indexes)
    fold_size = int(i / k)
    for fold in range(k):
        df_train = adjacent.copy()
        df_test = pd.DataFrame(columns=adjacent.columns, index=adjacent.index, data=0.0)
        fold_indexes = test_indexes[fold * fold_size: (fold + 1) * fold_size]
        for index in fold_indexes:
            test = (df_train.index[l[index]], df_train.columns[c[index]])
            df_train.at[test[0], test[1]] = 0
            df_test.at[test[0], test[1]] = 1
        sets.append((df_train, df_test))
    return sets


def fn_taux_rappel(df_adjacence_test, recommandations):
    mx_test = df_adjacence_test.values.astype(bool)
    mx_reco = recommandations.values.astype(bool)

    recall = mx_test & mx_reco  # Link is in test set AND is predicted

    recall_rate = np.sum(recall) / np.sum(mx_test)

    return recall_rate


def tx_rappel_total(df_adjacence, abstract, fn_recommandations, n=5):
    recall_mx = []
    printProgressBar(0, 50, prefix='Progress:', suffix='Complete', length=50)
    i = 0
    for _ in range(n):
        exp = []
        sets = gen_sets(df_adjacence)
        for df_adjacence_entrainement, df_adjacence_test in sets:
            recommandations = fn_recommandations(df_adjacence_entrainement, abstract)
            recall = fn_taux_rappel(df_adjacence_test, recommandations)
            exp.append(recall)

            i += 1
            printProgressBar(i, n*10, prefix='Progress:', suffix='complete', length=50)
        recall_mx.append(exp)

    return np.mean(recall_mx)


# Code from : https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
