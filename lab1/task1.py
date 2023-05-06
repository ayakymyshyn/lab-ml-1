from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

random.seed(42)
random.shuffle(iris.data)

scaler = StandardScaler()
iris.data = scaler.fit_transform(iris.data)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

K_values = [1, 3, 5, 7, 9]
best_K = None
best_accuracy = 0
for K in K_values:
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy for K={K}: {accuracy}')
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_K = K

print(f'\nThe best K value is {best_K} with accuracy {best_accuracy}')

