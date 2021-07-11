import random
import os
import re
import math
import json
import enchant
from collections import defaultdict
from bs4 import BeautifulSoup

eng = enchant.Dict("en_US")


def split_dataset():
    random.seed(1234)
    folds = [[] for _ in range(5)]
    with open("./trec06p/label/index", "r") as f:
        lines = f.readlines()
        for line in lines:
            lst = line.strip().split(" ")
            label = 1 if lst[0] == "spam" else 0
            lst2 = lst[1].split("/")
            path = "./trec06p/data/{}/{}".format(lst2[2], lst2[3])
            rand = random.random()
            if rand < 0.2:
                folds[0].append((label, path))
            elif rand < 0.4:
                folds[1].append((label, path))
            elif rand < 0.6:
                folds[2].append((label, path))
            elif rand < 0.8:
                folds[3].append((label, path))
            else:
                folds[4].append((label, path))
    with open("setting", "w") as f:
        json.dump(folds, f)


def preprocess_all():
    with open("./trec06p/label/index", "r") as f:
        lines = f.readlines()
        for line in lines:
            lst = line.strip().split(" ")
            lst2 = lst[1].split("/")
            path = "./trec06p/data/{}/{}".format(lst2[2], lst2[3])
            parse_words_and_save(path)


def parse_words(filename):
    words = []
    try:
        with open(filename, "r") as f:
            raw = f.read().lower()
            raw = BeautifulSoup(raw, "html.parser").get_text()
            text = re.sub("[^A-Za-z]", " ", raw)
            lst = text.split(" ")
            for word in lst:
                if len(word) > 2 and eng.check(word):
                    words.append(word)
            mail = re.findall(re.compile(u"from.*@.*"), raw)
            if len(mail) > 0:
                mail = (
                    mail[0]
                    .split("@")[1]
                    .split(">")[0]
                    .replace("<", " ")
                    .replace(">", " ")
                    .replace("(", " ")
                    .replace(")", " ")
                    .replace("[", " ")
                    .replace("]", " ")
                    .replace("*", " ")
                    .replace('"', " ")
                    .replace(":", " ")
                    .replace("?", " ")
                    .replace("=", " ")
                    .replace("\\", " ")
                    .split(" ")[0]
                    .strip()
                )
            else:
                mail = None
    except:
        return [], None
    return words, mail


def parse_words_and_save(filename):
    words, mail = parse_words(filename)
    if len(words) > 0:
        with open(filename + "_", "w") as f:
            json.dump({"words": words, "mail": mail}, f)


def get_parsed(filename):
    try:
        with open(filename + "_", "r") as f:
            dic = json.load(f)
        return dic["words"], dic["mail"]
    except:
        return [], None


def train(sample_ratio=1.0, fold=4):
    random.seed(1234)
    training_set = []
    with open("setting", "r") as f:
        folds = json.load(f)
        for i in range(5):
            if i != fold:
                for e in folds[i]:
                    training_set.append(e)
    freqs_y = defaultdict(int)
    freqs_x_y = defaultdict(dict)
    freqs_x_y_mail = defaultdict(dict)
    freqs_y_mail = defaultdict(int)
    for label, path in training_set:
        if random.random() > sample_ratio:
            continue
        words, mail = get_parsed(path)
        for word in words:
            freqs_y[label] += 1
            if word not in freqs_x_y[label].keys():
                freqs_x_y[label][word] = 1
            else:
                freqs_x_y[label][word] += 1
        if mail is not None:
            freqs_y_mail[label] += 1
            if mail not in freqs_x_y_mail[label].keys():
                freqs_x_y_mail[label][mail] = 1
            else:
                freqs_x_y_mail[label][mail] += 1
    with open("freqs_y", "w") as f:
        json.dump(freqs_y, f)
    with open("freqs_x_y", "w") as f:
        json.dump(freqs_x_y, f)
    with open("freqs_y_mail", "w") as f:
        json.dump(freqs_y_mail, f)
    with open("freqs_x_y_mail", "w") as f:
        json.dump(freqs_x_y_mail, f)


def test(smoothing_alpha=1e-1, mail_weight=1.0, fold=4, smooth=True):
    with open("setting", "r") as f:
        testing_set = json.load(f)[fold]
    with open("freqs_y", "r") as f:
        freqs_y = json.load(f)
    with open("freqs_x_y", "r") as f:
        freqs_x_y = json.load(f)
    with open("freqs_y_mail", "r") as f:
        freqs_y_mail = json.load(f)
    with open("freqs_x_y_mail", "r") as f:
        freqs_x_y_mail = json.load(f)
    true_p, true_n, false_p, false_n = 0, 0, 0, 0
    for true_label, path in testing_set:
        max_logp = -math.inf
        label = -1
        flag = False
        words, mail = get_parsed(path)
        if len(words) == 0:
            continue
        for y in [0, 1]:
            y = str(y)
            prob, prob_mail = math.log(float(freqs_y[y]) / float(freqs_y["0"] + freqs_y["1"])), math.log(float(freqs_y_mail[y]) / float(freqs_y_mail["0"] + freqs_y_mail["1"]))
            for word in words:
                if smooth:
                    if not word in freqs_x_y[y].keys():
                        prob += math.log(float(smoothing_alpha / (freqs_y[y] + 2 * smoothing_alpha)))
                    else:
                        prob += math.log(float((freqs_x_y[y][word] + smoothing_alpha) / (freqs_y[y] + 2 * smoothing_alpha)))
                elif word in freqs_x_y[y].keys():
                    prob += math.log(float(freqs_x_y[y][word] / freqs_y[y]))
            if mail is not None:
                if smooth:
                    if not mail in freqs_x_y_mail[y].keys():
                        prob_mail += math.log(float(smoothing_alpha / (freqs_y_mail[y] + 2 * smoothing_alpha)))
                    else:
                        prob_mail += math.log(float((freqs_x_y_mail[y][mail] + smoothing_alpha) / (freqs_y_mail[y] + 2 * smoothing_alpha)))
                elif mail in freqs_x_y_mail[y].keys():
                    prob_mail += math.log(float(freqs_x_y_mail[y][mail] / freqs_y_mail[y]))
            prob += mail_weight * prob_mail
            if prob > max_logp:
                max_logp = prob
                label = int(y)
        if true_label == 0 and label == 0:
            true_n += 1
        if true_label == 0 and label == 1:
            false_p += 1
        if true_label == 1 and label == 0:
            false_n += 1
        if true_label == 1 and label == 1:
            true_p += 1
    return true_p, true_n, false_p, false_n


def run(sample_ratio=1.0, smoothing_alpha=1e-1, mail_weight=0.0, smooth=True):
    true_p, true_n, false_p, false_n = 0, 0, 0, 0
    # 5-fold cross validation
    for fold in range(5):
        train(sample_ratio, fold)
        a, b, c, d = test(smoothing_alpha, mail_weight, fold, smooth)
        true_p += a
        true_n += b
        false_p += c
        false_n += d

    accuracy = (true_p + true_n) / (true_p + true_n + false_p + false_n)
    precision = true_p / (true_p + false_p)
    recall = true_p / (true_p + false_n)
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f1
    print("Accuracy", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


if __name__ == "__main__":
    # 初次运行时调用，需几十分钟预处理
    # preprocess_all()
    # split_dataset()
    accuracy, precision, recall, f1 = run(sample_ratio=1.0, smoothing_alpha=0, mail_weight=0.0, smooth=False)
    print("Testing no smoothing")
    print("Accuracy", accuracy)
    for i in range(-50, -9, 5):
        print("Testing alpha 1e%d" % i)
        accuracy, precision, recall, f1 = run(sample_ratio=1.0, smoothing_alpha=10 ** i, mail_weight=0.0)
        print("Accuracy", accuracy)
    for i in range(-9, 5):
        print("Testing alpha 1e%d" % i)
        accuracy, precision, recall, f1 = run(sample_ratio=1.0, smoothing_alpha=10 ** i, mail_weight=0.0)
        print("Accuracy", accuracy)
    for i in [0.05, 0.5, 1.0]:
        print("Testing sample ratio %f" % i)
        accuracy, precision, recall, f1 = run(sample_ratio=i, smoothing_alpha=1e-5, mail_weight=0.0)
        print("Accuracy", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
    for i in [0.05, 0.5, 1.0]:
        print("Testing sample ratio %f no smoothing" % i)
        accuracy, precision, recall, f1 = run(sample_ratio=i, smoothing_alpha=1e-5, mail_weight=0.0, smooth=False)
        print("Accuracy", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
    for i in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
        print("Testing mail weight %f" % i)
        accuracy, precision, recall, f1 = run(sample_ratio=1.0, smoothing_alpha=1e-5, mail_weight=i)
        print("Accuracy", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)