import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('./texture_features.csv')

X = data.drop('category', axis=1)
y = data['category']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.6, random_state=42, stratify=y_encoded)

# Support Vector Machine
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)

# Predykcja
y_pred_svm = svm_classifier.predict(X_test)

# Obliczanie dokładności
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Dokładność klasyfikacji SVM: {accuracy_svm:.2f}")

# K-Nearest Neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Predykcja
y_pred_knn = knn_classifier.predict(X_test)

# Obliczanie dokładności
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Dokładność klasyfikacji KNN: {accuracy_knn:.2f}")