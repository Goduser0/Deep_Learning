from collections import Counter
from sklearn.datasets import make_classification, load_breast_cancer, load_wine
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import EditedNearestNeighbours, InstanceHardnessThreshold
import matplotlib.pyplot as plt
from numpy import where
import seaborn as sns

data = load_wine()

X = data.data
y = data.target

print(f"类别为 0 的样本数: {X[y == 0].shape[0]}, 类别为 1 的样本数: {X[y == 1].shape[0]}")

sns.set_style("darkgrid")
sns.scatterplot(data=data, x=X[:, 0], y=X[:, 1], hue=y)
plt.xlabel(f"{data.feature_names[0]}")
plt.ylabel(f"{data.feature_names[1]}")
plt.title("Original")
plt.show()

#Edited Nearest Neighbor
sampler = EditedNearestNeighbours(n_neighbors=5)

X1, y1 = sampler.fit_resample(X, y)

while True:
    a, b = X1[y1 == 0].shape[0], X1[y1 == 1].shape[0]
    X1, y1 = sampler.fit_resample(X1, y1)
    if a == X1[y1 == 0].shape[0] and b == X1[y1 == 1].shape[0]:
        break
print(
    f"EditedNearestNeighbours: 类别为 0 的样本数: {X1[y1 == 0].shape[0]}, 类别为 1 的样本数: {X1[y1 == 1].shape[0]}"
)

sns.scatterplot(data=data, x=X1[:, 0], y=X1[:, 1], hue=y1)
plt.title("EditedNearestNeighbours")
plt.show()

iht = InstanceHardnessThreshold(estimator=LogisticRegression(), cv=10, random_state=42)
X_res, y_res = iht.fit_resample(X1, y1)
print(
    f"EditedNearestNeighbours: 类别为 0 的样本数: {X_res[y_res == 0].shape[0]}, 类别为 1 的样本数: {X_res[y_res == 1].shape[0]}"
)

sns.scatterplot(data=data, x=X_res[:, 0], y=X_res[:, 1], hue=y_res)
plt.title("EditedNearestNeighbours")
plt.show()