import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

X = np.random.rand(1000).reshape(-1, 1)
y = np.random.rand(1000)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k_values = list(range(1, 20))
train_scores = []
test_scores = []
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))

best_k = k_values[np.argmax(test_scores)]

# 6. Візуалізація отриманих рішень
plt.plot(k_values, train_scores, label='Train score')
plt.plot(k_values, test_scores, label='Test score')
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best k: {best_k}')
plt.xlabel('k')
plt.ylabel('R^2 score')
plt.legend()
plt.show()
