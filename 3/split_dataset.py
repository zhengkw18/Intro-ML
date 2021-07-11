import random
import json
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords

sw = stopwords.words("english")


def parse_list(s):
    lst = (
        s.lower()
        .replace(":", " ")
        .replace(",", " ")
        .replace(")", " ")
        .replace(".", " ")
        .replace("-", " ")
        .replace("!", " ")
        .replace("?", " ")
        .replace('"', " ")
        .replace("/", " ")
        .replace("'", " ")
        .replace("&", " ")
        .replace("#", " ")
        .replace(";", " ")
        .replace("*", " ")
        .replace("`", " ")
        .replace("$", " ")
        .replace("~", " ")
        .split()
    )
    lst = s.lower().split()
    return [word for word in lst]


def remove_sw(lst):
    return [word for word in lst if word not in sw]


with open("exp3-reviews.csv", "r", encoding="utf-8") as f:
    lines = f.readlines()[1:]

x, x_sw, y = [], [], []
cnt = 0
for line in lines:
    if cnt > len(lines) / 10:
        break
    if cnt % 1000 == 0:
        print(cnt)
    words = line.split("\t")
    lst_sw = parse_list(words[-2])
    if len(lst_sw) == 0:
        lst_sw.extend(parse_list(words[-1]))
    lst = remove_sw(lst_sw)
    if len(lst) == 0:
        lst.extend(parse_list(words[-1]))
        lst = remove_sw(lst)
    x.append(" ".join(lst))
    x_sw.append(" ".join(lst_sw))
    y.append(int(float(words[0])))
    cnt += 1

x_train, x_test, x_sw_train, x_sw_test = [], [], [], []
y_train, y_test = [], []
random.seed(1234)
for i in range(len(y)):
    if random.random() < 0.9:
        x_train.append(x[i])
        x_sw_train.append(x_sw[i])
        y_train.append(y[i])
    else:
        x_test.append(x[i])
        x_sw_test.append(x_sw[i])
        y_test.append(y[i])

with open("dset_x.json", "w") as f:
    json.dump({"all": x, "train": x_train, "test": x_test}, f)
with open("dset_x_sw.json", "w") as f:
    json.dump({"all": x_sw, "train": x_sw_train, "test": x_sw_test}, f)

with open("dset_y.json", "w") as f:
    json.dump({"all": y, "train": y_train, "test": y_test}, f)
