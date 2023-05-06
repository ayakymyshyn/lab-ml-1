from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# завантаження набору даних MNIST
mnist = fetch_openml('mnist_784')
X, y = mnist['data'], mnist['target']

# розділення набору даних на тренувальний та тестовий
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# нормалізація даних
X_train = X_train / 255.0
X_test = X_test / 255.0

# навчання моделі MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, alpha=0.0001, solver='adam', verbose=10, tol=0.0001, random_state=1, learning_rate_init=0.001)
mlp.fit(X_train, y_train)

# навчання моделі SVM
svm = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', verbose=10, random_state=1)
svm.fit(X_train, y_train)

# оцінка точності моделей
mlp_accuracy = accuracy_score(y_test, mlp.predict(X_test))
svm_accuracy = accuracy_score(y_test, svm.predict(X_test))

# виведення результатів
print(f"Точність MLPClassifier: {mlp_accuracy:.4f}")
print(f"Точність SVM: {svm_accuracy:.4f}")
