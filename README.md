#  ml-algorithms-comparaison

Ce projet présente une comparaison de plusieurs algorithmes de **classification supervisée** (supervised classification algorithms) en Machine Learning, à l’aide d’un dataset synthétique contenant des variables numériques et une cible binaire (0 ou 1).

## 📌 Objectif

L'objectif est de :
- Comprendre le fonctionnement des principaux algorithmes de classification
- Entraîner et évaluer chaque modèle
- Visualiser et comparer les performances à travers des métriques standards

## 📂 Contenu du projet
ml-algorithms-comparaison/ │ ├── classification_dataset.csv # Jeu de données utilisé ├── logistic_regression_model.py # Régression Logistique (Logistic Regression) ├── knn_model.py # K-Nearest Neighbors (KNN) ├── svm_model.py # Support Vector Machine (SVM) ├── decision_tree_model.py # Arbre de Décision (Decision Tree) ├── compare_models.py # Comparaison globale des performances └── README.md # Description du projet


## 📊 Algorithmes comparés

- ✅ Régression Logistique (Logistic Regression)
- ✅ K-plus proches voisins (K-Nearest Neighbors - KNN)
- ✅ Machine à vecteurs de support (Support Vector Machine - SVM)
- ✅ Arbre de Décision (Decision Tree)

Chaque modèle a été entraîné, testé, puis évalué sur les métriques suivantes :
- **Accuracy** : Taux de bonnes prédictions
- **Precision** : Capacité à éviter les faux positifs
- **Recall** : Capacité à capturer les vrais positifs
- **F1-score** : Moyenne harmonique de la précision et du rappel

## 📈 Résultats

Chaque script affiche un **rapport de classification** (`classification report`) généré avec `scikit-learn`, ainsi qu’un tableau comparatif global dans `compare_models.py`.

### Exemple de sortie :
```text
              precision    recall  f1-score   support
           0       0.96      0.96      0.96        26
           1       0.93      0.93      0.93        14
    accuracy                           0.95        40
