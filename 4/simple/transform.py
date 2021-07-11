# Preprocessing the data and save the embeded word vectors as tensor

import random
import numpy as np

# load word vector

wordVectors = {}

drive_dir = "/content/drive/My Drive/app/"
drive_dir = ""

with open(drive_dir + "glove.6B.50d.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        splitLine = line.split(" ")
        wordVectors[splitLine[0]] = list(map(lambda x: float(x), splitLine[1:]))
vectors = list(wordVectors.values())

print("Word Vectors loaded.")


def transform(filename, has_lables=True):
    labels = []
    inputs = []
    with open(drive_dir + filename, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
        for line in lines:
            splitLine = line.split("\t")
            if has_lables:
                labels.append(int(float(splitLine[0])) - 1)
            s = splitLine[-2] + " " + splitLine[-1]
            words = (
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
            currentWordVectors = []
            cnt = 0
            for word in words:
                if cnt >= 64:
                    break
                if word in wordVectors:
                    currentWordVectors.append(wordVectors[word])
                else:
                    currentWordVectors.append(random.choice(vectors))
                cnt += 1
            while cnt < 64:
                currentWordVectors.append([0] * 50)
                cnt += 1
            inputs.append(currentWordVectors)
    if has_lables:
        np.save(drive_dir + filename.split(".")[0] + "-labels.npy", np.array(labels))
    np.save(drive_dir + filename.split(".")[0] + "-50d.npy", np.array(inputs, dtype=np.float32))


transform("train.csv")
