import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

use_tfidf = True


def tfidf(src, dst):
    with open(src, "r") as f:
        dic = json.load(f)
    counter = CountVectorizer(max_features=1000)
    counter.fit_transform(dic["all"])
    counter_new = CountVectorizer(vocabulary=counter.vocabulary_)
    counter_train = counter_new.transform(dic["train"])
    counter_test = counter_new.transform(dic["test"])
    tfidf = TfidfTransformer()
    if use_tfidf:
        features_train = tfidf.fit_transform(counter_train).toarray()
    else:
        features_train = counter_train.toarray()
    print(np.sum(np.sum(features_train, axis=1) < 1e-5) / len(features_train))
    if use_tfidf:
        features_test = tfidf.transform(counter_test).toarray()
    else:
        features_test = counter_test.toarray()
    np.save(dst + "_train.npy", features_train)
    np.save(dst + "_test.npy", features_test)


tfidf("dset_x.json", "features_x")
tfidf("dset_x_sw.json", "features_x_sw")