import random
import torch
import numpy as np
import torchvision
from skimage.feature import hog
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json


def get_features(data, K, do_hog=False):
    if do_hog:
        lst = []
        for i in range(data.shape[0]):
            lst.append(hog(data[i], orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1)))
        return np.asarray(lst)
    return data.reshape(data.shape[0], -1).astype(np.float64) / 255


def get_label_array(labels):
    lst = [[] for _ in range(10)]
    for i in range(len(labels)):
        lst[labels[i]].append(i)
    return lst


def generate_centers(features, labels, K, do_random=False):
    ndim = features.shape[1]
    if do_random:
        return [np.random.rand(ndim) for _ in range(K)]
    else:
        return [features[random.choice(labels[random.randint(0, 9)])] for _ in range(K)]


def get_clusters(features, centers, K):
    clusters = [[] for _ in range(K)]
    for i in range(features.shape[0]):
        min_norm = float("inf")
        k = -1
        for j in range(K):
            norm = np.linalg.norm(features[i] - centers[j])
            if norm < min_norm:
                min_norm = norm
                k = j
        clusters[k].append(i)
    return clusters


def get_new_centers(features, clusters):
    centers = []
    for cluster in clusters:
        if len(cluster) == 0:
            return
        centers.append(np.mean(features[cluster, :], axis=0).reshape(-1))
    return centers


def kmeans(features, initial_centers):
    old_clusters = get_clusters(features, initial_centers, K)
    cnt = 0
    while 1:
        if cnt % 10 == 0:
            print("Iteration", cnt)
        try:
            centers = get_new_centers(features, old_clusters)
        except:
            return
        new_clusters = get_clusters(features, centers, K)
        if old_clusters == new_clusters:
            break
        old_clusters = new_clusters
        cnt += 1
    print("Convergence at step", cnt)
    return get_new_centers(features, old_clusters), old_clusters, cnt


def to_lst(lst_of_ndarray):
    lst = []
    for ndarray in lst_of_ndarray:
        lst.append(ndarray.tolist())
    return lst


def train_and_save(data, labels, K, use_hog, random_centers):
    features = get_features(data, K, use_hog)
    initial_centers = generate_centers(features, labels, K, random_centers)
    try:
        centers, clusters, cnt = kmeans(features, initial_centers)
    except:
        return
    filename = "data_{}_{}_{}".format(str(K), str(use_hog), str(random_centers))
    with open(filename + ".json", "w") as f:
        json.dump({"cnt": cnt, "centers": to_lst(centers), "clusters": clusters}, f)


def evaluate_and_visualize(data, targets, K, use_hog, random_centers):
    features = get_features(data, K, use_hog)
    filename = "data_{}_{}_{}".format(str(K), str(use_hog), str(random_centers))
    try:
        with open(filename + ".json", "r") as f:
            dic = json.load(f)
    except:
        print("File", filename, "failed")
        return
    print("Testing K={} use_hog={} random_centers={}".format(str(K), str(use_hog), str(random_centers)))
    print("Convergence at step", dic["cnt"])
    centers, clusters = dic["centers"], dic["clusters"]
    voting = []
    pred = 0
    for cluster in clusters:
        cnt = [0 for _ in range(10)]
        for e in cluster:
            cnt[targets[e]] += 1
        v = np.argmax(cnt)
        voting.append(v)
        for e in cluster:
            if targets[e] == v:
                pred += 1
    print("Accuracy", pred / len(targets))
    n_per_cluster = 1000 // K
    good_points, bad_points = [], []
    chosen_lst = []
    lst_lst = []
    for (i, cluster) in enumerate(clusters):
        n = min(n_per_cluster, len(cluster))
        lst = np.random.choice(cluster, n)
        lst_lst.append(lst)
        chosen_lst.extend(list(lst))
    transformed = TSNE(n_components=2, init="pca", random_state=0).fit_transform(np.vstack((features[chosen_lst, :], np.asarray(centers))))
    cnt = 0
    for (i, cluster) in enumerate(clusters):
        good, bad = [], []
        for e in lst_lst[i]:
            if targets[e] == voting[i]:
                good.append(transformed[cnt])
            else:
                bad.append(transformed[cnt])
            cnt += 1
        good_points.append(np.asarray(good))
        bad_points.append(np.asarray(bad))
    plt.figure(figsize=(6, 5), dpi=200)
    for i in range(K):
        v = voting[i]
        plt.scatter(good_points[i][:, 0], good_points[i][:, 1], s=5, color="w", edgecolor=plt.cm.Set1(v / 10.0), marker="o", linewidths=0.5)
        if len(bad_points[i]) > 0:
            plt.scatter(bad_points[i][:, 0], bad_points[i][:, 1], s=5, color=plt.cm.Set1(v / 10.0), marker="x", linewidths=0.5)
    plt.scatter(transformed[len(chosen_lst) :, 0], transformed[len(chosen_lst) :, 1], s=30, marker="*", edgecolor="k")
    patch_lst = []
    for i in range(10):
        patch_lst.append(mpatches.Patch(color=plt.cm.Set1(i / 10.0), label=str(i)))
    patch_lst.append(mpatches.Patch(color="k", label="centroid"))
    plt.legend(handles=patch_lst, fontsize=6)
    plt.savefig(filename + ".png")


if __name__ == "__main__":
    dset = torchvision.datasets.MNIST("./data", train=True, download=True)
    labels = get_label_array(dset.targets)
    # train and save
    # for K in [10]:
    #     for use_hog in [True, False]:
    #         for random_centers in [True, False]:
    #             train_and_save(dset.data.numpy(), labels, K, use_hog, random_centers)
    # for K in [10, 20, 30, 50, 80]:
    #     for use_hog in [True,False]:
    #         for random_centers in [False]:
    #             train_and_save(dset.data.numpy(), labels, K, use_hog, random_centers)

    # load, evaluate and visualize
    for K in [10]:
        for use_hog in [True, False]:
            for random_centers in [True, False]:
                evaluate_and_visualize(dset.data.numpy(), dset.targets, K, use_hog, random_centers)
    for K in [10, 20, 30, 50, 80]:
        for use_hog in [True, False]:
            for random_centers in [False]:
                evaluate_and_visualize(dset.data.numpy(), dset.targets, K, use_hog, random_centers)