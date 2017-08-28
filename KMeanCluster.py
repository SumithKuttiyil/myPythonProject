#Simple clustering algorithm using K-Mean clustering
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


n_samples = 1500
random_state = 200
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X)

print(y_pred)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("KMean Cluster")
plt.show()
