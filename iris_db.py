import joblib
from sklearn.metrics import adjusted_rand_score
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# 載入 Iris 資料集
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
true_labels = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X_scaled)
labels = dbscan.labels_

predicted_clusters = dbscan.labels_
joblib.dump(dbscan, "dbscan_iris_model.pkl")
# 把群集標籤加入 DataFrame
X['cluster'] = dbscan.labels_

print(X)

score = adjusted_rand_score(true_labels, predicted_clusters)
print(f"Adjusted Rand Score: {score:.4f}")
