import joblib
from sklearn.metrics import adjusted_rand_score
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans

# 載入 Iris 資料集
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
true_labels = iris.target

# 建立 KMeans 分群模型（分為 3 群）
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
predicted_clusters = kmeans.labels_
joblib.dump(kmeans, "kmeans_iris_model.pkl")
# 把群集標籤加入 DataFrame
X['cluster'] = kmeans.labels_

print(X)

score = adjusted_rand_score(true_labels, predicted_clusters)
print(f"Adjusted Rand Score: {score:.4f}")
