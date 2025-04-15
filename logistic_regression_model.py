# logistic_regression_model.py

# Étape 1 : Importation des bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Étape 2 : Charger le dataset
data = pd.read_csv('classification_dataset.csv')

# Étape 3 : Séparer les features (X) et la target (y)
X = data.drop('target', axis=1)
y = data['target']

# Étape 4 : Diviser en jeu d'entraînement et de test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Étape 5 : Créer l'objet du modèle de régression logistique
model = LogisticRegression()

# Étape 6 : Entraîner le modèle avec les données d'entraînement
model.fit(X_train, y_train)

# Étape 7 : Faire des prédictions avec les données de test
y_pred = model.predict(X_test)

# Étape 8 : Calculer la précision du modèle et afficher un rapport de classification
accuracy = accuracy_score(y_test, y_pred)
print(f'Précision du modèle : {accuracy * 100:.2f}%')

# Étape 9 : Afficher le rapport de classification
print('\nRapport de classification :\n', classification_report(y_test, y_pred))
