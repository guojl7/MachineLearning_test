#-*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans

# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本4个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.2, 0.2, 0.2], random_state =9)
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.show()

y_pred = KMeans(n_clusters=4, random_state=9).fit_predict(X)
print(metrics.calinski_harabaz_score(X, y_pred))
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()

plt.subplots_adjust(left=.02, right=.98, bottom=.096, top=.96, wspace=.1, hspace=.1)
for index, k in enumerate((2,3,4,5)):
    plt.subplot(2,2,index+1)
    y_pred = MiniBatchKMeans(n_clusters=k, batch_size = 200, random_state=9).fit_predict(X)
    score= metrics.calinski_harabaz_score(X, y_pred)  
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.text(.99, .01, ('k=%d, score: %.2f' % (k,score)), transform=plt.gca().transAxes, size=10, horizontalalignment='right')
plt.show()
