# decision_tree_model.py

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Charger le dataset
data = pd.read_csv('classification_dataset.csv')
X = data.drop('target', axis=1)
y = data['target']

# Séparer en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le modèle (max_depth limite la profondeur pour éviter l'overfitting)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation
print("📊 Précision :", accuracy_score(y_test, y_pred))
print("\n📋 Rapport de classification :\n", classification_report(y_test, y_pred))
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Taille de la figure
plt.figure(figsize=(20,10))

# Visualisation de l'arbre
plot_tree(model, filled=True, feature_names=X.columns, class_names=["Classe 0", "Classe 1"])

# Afficher l'arbre
plt.title("Visualisation de l'Arbre de Décision")
plt.show()
