import json
import math
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC


def cal_test(label, pred):
    acc = np.mean(label == pred)
    mae = np.mean(np.abs(label - pred))
    rmse = math.sqrt(np.mean(np.square(label - pred)))
    print("Acc:", acc, "MAE:", mae, "RMSE:", rmse)


def load_dset(use_sw=False):
    if use_sw:
        f_train = np.load("features_x_sw_train.npy")
        f_test = np.load("features_x_sw_test.npy")
    else:
        f_train = np.load("features_x_train.npy")
        f_test = np.load("features_x_test.npy")
    with open("dset_y.json", "r") as f:
        y = json.load(f)
        y_train = y["train"]
        y_test = y["test"]
    return f_train, f_test, np.array(y_train), np.array(y_test)


def train_and_test(base, algo, f_train, f_test, y_train, y_test, num):
    length = len(f_train)
    test_len = len(f_test)
    if algo == "BAGGING":
        # train
        models = []
        for i in range(num):
            print("step", i)
            if base == "TREE":
                model = DecisionTreeClassifier()
            elif base == "SVM":
                model = LinearSVC(multi_class="ovr")
            indices = np.random.choice(length, length, replace=True)
            print(len(set(indices)), length)
            model.fit(f_train[indices], y_train[indices])
            models.append(model)
        # test
        preds = []
        for i in range(test_len):
            pred = [0 for _ in range(6)]
            for j in range(num):
                pred[models[j].predict(np.array([f_test[i]]))[0]] += 1
            preds.append(np.argmax(pred))
            # preds.append(5)
        print("Bagging")
        cal_test(y_test, preds)
        print("Single")
        cal_test(y_test, models[0].predict(f_test))
    elif algo == "ADABOOST":
        models = []
        betas = []
        ets = []
        weights = np.array([1 / length] * length)
        for i in range(num):
            if base == "TREE":
                model = DecisionTreeClassifier()
            elif base == "SVM":
                model = LinearSVC(multi_class="ovr", max_iter=10000)
            indices = np.random.choice(length, length, replace=True, p=weights)
            model.fit(f_train[indices], y_train[indices])
            # model.fit(f_train, y_train, sample_weight=weights * length)
            preds = model.predict(f_train)
            et = float(np.dot(y_train != preds, weights))
            print(i, et)
            ets.append(et)
            if et > 1 / 2:
                break
            beta = et / (1.0 - et)
            weights *= [beta if y_train[j] == preds[j] else 1 for j in range(length)]
            weights /= np.sum(weights)
            models.append(model)
            betas.append(beta)
        preds = []
        print(ets)
        for i in range(test_len):
            pred = [0 for _ in range(6)]
            for j in range(len(models)):
                p = models[j].predict(np.array([f_test[i]]))[0]
                pred[p] += np.log(1 / betas[j])
            preds.append(np.argmax(pred))
        cal_test(y_test, preds)


ENSEMBLE_NUM = 1

# SVM TREE
BASE = "SVM"
# BAGGING ADABOOST
ALGO = "ADABOOST"

use_stopwords = True
train_and_test(BASE, ALGO, *load_dset(use_stopwords), ENSEMBLE_NUM)
