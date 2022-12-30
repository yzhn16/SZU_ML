import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import cv2
import matplotlib.pyplot as plt

cancer = load_breast_cancer()

data = pd.DataFrame(cancer.data, columns=cancer.feature_names)
data['target'] = cancer.target

plt.figure(dpi=300, figsize=(15, 5))
for i, reducer in enumerate([PCA, TSNE, UMAP]):
    res = reducer(n_components=2, random_state=2022).fit_transform(cancer.data)

    # plt.figure(dpi=300)
    plt.subplot(1, 3, i + 1)
    plt.scatter(res[cancer.target == 0][:, 0], res[cancer.target == 0][:, 1], marker='o', )
    plt.scatter(res[cancer.target == 1][:, 0], res[cancer.target == 1][:, 1], marker='^')
    
    # plt.savefig(f'{reducer.__name__}_2.png', bbox_inches='tight', pad_inches=0.05)

plt.savefig('2.png', bbox_inches='tight', pad_inches=0.05)

plt.figure(dpi=300, figsize=(15, 5))
for i, reducer in enumerate([PCA, TSNE, UMAP]):
    res = reducer(n_components=3, random_state=2022).fit_transform(cancer.data)
    
    plt.subplot(1, 3, i + 1)
    # fig = plt.figure(dpi=300)
    ax = plt.gca(projection="3d")
    
    xs0 = res[cancer.target == 0][:, 0]
    ys0 = res[cancer.target == 0][:, 1]
    zs0 = res[cancer.target == 0][:, 2]
    xs1 = res[cancer.target == 1][:, 0]
    ys1 = res[cancer.target == 1][:, 1]
    zs1 = res[cancer.target == 1][:, 2]
    
    ax.scatter(xs0, ys0, zs0, c="#00DDAA", marker="o")
    ax.scatter(xs1, ys1, zs1, c="#FF5511", marker="^")

    # plt.savefig(f'{reducer.__name__}_3.png', bbox_inches='tight', pad_inches=0.05)
plt.savefig('3.png', bbox_inches='tight', pad_inches=0.05)
