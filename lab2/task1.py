import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


# виконання завдання зі стандартним файлом, вказаним у завданні.

# data = sio.loadmat('digits.mat')
# X = data['X']
# y = data['y']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Завантажуємо MNIST dataset і повертаємо словник
mnist_dict = fetch_openml('mnist_784', version=1, cache=True)

X = mnist_dict['data']
y = mnist_dict['target']

y = y.astype(int)

# навчальна та тестова вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# одношарова мережа
mlp_1 = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, max_iter=200)

# двошарова мережа
mlp_2 = MLPClassifier(hidden_layer_sizes=(3,3), activation='relu', solver='adam', alpha=0.0001, max_iter=200)

# тришарова мережа
mlp_3 = MLPClassifier(hidden_layer_sizes=(20,7,10), activation='relu', solver='adam', alpha=0.0001, max_iter=200)

# Одношарова мережа
model1 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, solver='adam', alpha=0.0001, activation='relu',
                      verbose=10, random_state=1, learning_rate_init=0.001)
model1.fit(X_train, y_train)

# Двошарова мережа
model2 = MLPClassifier(hidden_layer_sizes=(3, 3), max_iter=500, solver='adam', alpha=0.0001, activation='relu',
                      verbose=10, random_state=1, learning_rate_init=0.001)
model2.fit(X_train, y_train)

# Тришарова мережа
model3 = MLPClassifier(hidden_layer_sizes=(20, 7, 10), max_iter=500, solver='adam', alpha=0.0001, activation='relu',
                      verbose=10, random_state=1, learning_rate_init=0.001)
model3.fit(X_train, y_train)


print(f"Точність одношарової мережі на тестових даних: {model1.score(X_test, y_test):.4f}")
print(f"Точність двошарової мережі на тестових даних: {model2.score(X_test, y_test):.4f}")
print(f"Точність тришарової мережі на тестових даних: {model3.score(X_test, y_test):.4f}")


