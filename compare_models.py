# compare_models.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Chargement des données
data = pd.read_csv('classification_dataset.csv')
X = data.drop('target', axis=1)
y = data['target']

# 2. Séparation en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Définition des modèles
models = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier()
}

# 4. Entraînement et évaluation
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    })

# 5. Résultats sous forme de tableau
results_df = pd.DataFrame(results)
print("\n📊 Résumé des performances :\n")
print(results_df)
import matplotlib.pyplot as plt

# Définir les métriques à afficher
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# Création du graphique
plt.figure(figsize=(12, 6))

for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i + 1)  # 2 lignes, 2 colonnes
    plt.bar(results_df['Model'], results_df[metric], color=['blue', 'orange', 'green', 'red'])
    plt.title(f'Comparaison des modèles - {metric}')
    plt.ylabel(metric)
    plt.ylim(0, 1)  # Les scores vont de 0 à 1
    plt.xticks(rotation=15)

plt.tight_layout()
plt.show()
