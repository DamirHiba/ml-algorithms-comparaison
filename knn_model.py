# knn_model.py

# Étape 1 : Importer les bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Étape 2 : Charger les données
data = pd.read_csv('classification_dataset.csv')

# Étape 3 : Séparer les features (X) et la target (y)
X = data.drop('target', axis=1)
y = data['target']

# Étape 4 : Diviser en jeu d'entraînement et de test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Étape 5 : Créer le modèle KNN
knn_model = KNeighborsClassifier(n_neighbors=5)

# Étape 6 : Entraîner le modèle avec les données d'entraînement
knn_model.fit(X_train, y_train)

# Étape 7 : Faire des prédictions avec les données de test
y_pred_knn = knn_model.predict(X_test)

# Étape 8 : Calculer la précision du modèle et afficher un rapport de classification
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'Précision du modèle KNN : {accuracy_knn * 100:.2f}%')

# Étape 9 : Afficher le rapport de classification du modèle KNN
print('\nRapport de classification du modèle KNN :\n', classification_report(y_test, y_pred_knn))
