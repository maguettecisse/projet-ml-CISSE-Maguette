# Projet 2 — Détection des attaques réseau avec le Machine Learning

**Nom et prénom :** Maguette CISSE  
**Niveau :** Master 2 RESI  
**Langage :** Python 3.12

## 1. Présentation du projet

Ce projet consiste à développer un système de détection d'intrusion basé sur le Machine Learning. L'objectif est d'analyser les flux réseau afin de distinguer le trafic normal des activités malveillantes.

Le système effectue une classification binaire :

- `0` : trafic réseau normal.
- `1` : trafic réseau correspondant à une attaque.

Les attaques concernées sont notamment les scans de ports, les tentatives de force brute SSH/Telnet et les attaques par déni de service (DoS).

## 2. Technologies utilisées

- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- Joblib

## 3. Structure du projet

```text
projet-ml-CISSE-Maguette/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── flux_train.csv
│
├── src/
│   ├── train.py
│   └── predict.py
│
└── models/
    └── model.joblib
```

## 4. Préparation et exploration des données

Le jeu de données contient 3 000 flux réseau et 10 colonnes.

Les données comprennent des caractéristiques telles que la durée du flux, le protocole réseau, le port de destination, le nombre d'octets échangés et le nombre de paquets.

La variable cible est `attaque`.

L'exploration a révélé :

- 2 167 flux normaux, soit 72,23 %.
- 833 flux d'attaque, soit 27,77 %.
- 90 valeurs manquantes dans la colonne `duration`.

Les valeurs manquantes sont remplacées par la médiane lors du prétraitement.

## 5. Méthodologie

Le travail a été réalisé en plusieurs étapes :

1. Exploration et analyse des données.
2. Séparation des variables explicatives et de la variable cible.
3. Imputation des valeurs manquantes.
4. Standardisation des variables numériques.
5. Encodage de la variable catégorielle `protocol`.
6. Séparation des données en apprentissage (80 %) et validation (20 %).
7. Entraînement et comparaison de deux modèles.
8. Sélection et sauvegarde du meilleur modèle.

Les traitements sont intégrés dans un Pipeline Scikit-learn.

## 6. Résultats

Les deux modèles ont été évalués sur le même jeu de validation de 600 flux.

| Modèle | F1-score (attaque) | Rappel (attaque) | Précision (attaque) |
|---|---:|---:|---:|
| Régression logistique | 0,9202 | 0,8982 | 0,9434 |
| Forêt aléatoire | 0,9760 | 0,9760 | 0,9760 |

### Modèle retenu

Le modèle retenu est la **Forêt aléatoire (Random Forest)**, car il présente le meilleur F1-score sur la classe attaque, avec une valeur de 0,9760.

Ce résultat correspond à une performance de 97,60 % selon le F1-score de la classe attaque sur le jeu de validation.

### Matrice de confusion du modèle retenu

```text
[[429   4]
 [  4 163]]
```

Interprétation :

- Vrais négatifs : 429.
- Faux positifs : 4.
- Faux négatifs : 4.
- Vrais positifs : 163.

### Analyse des erreurs

Le modèle a identifié correctement 163 attaques sur les 167 présentes dans le jeu de validation.

Il a manqué 4 attaques (faux négatifs). En sécurité informatique, ces erreurs sont particulièrement préoccupantes, car elles correspondent à des activités malveillantes qui peuvent passer inaperçues.

Le modèle a également généré 4 faux positifs, c'est-à-dire des flux normaux signalés à tort comme des attaques. Ces alertes peuvent nécessiter une vérification par un analyste de sécurité.

## 7. Utilisation du projet

### Installation des dépendances

```bash
python -m pip install -r requirements.txt
```

### Entraînement du modèle

```bash
python src/train.py
```

Le meilleur modèle est enregistré dans :

```text
models/model.joblib
```

### Effectuer des prédictions

La fonction `predict()` du fichier `src/predict.py` permet de prédire la classe de nouveaux flux réseau.

Exemple :

```python
import pandas as pd
from src.predict import predict

df = pd.read_csv("data/flux_train.csv")
X = df.drop(columns=["attaque"])

predictions = predict(X.head(10))

print(predictions)
```

## 8. Conclusion

Ce projet a permis de mettre en œuvre une approche de détection d'attaques réseau basée sur le Machine Learning.

La comparaison entre la régression logistique et la forêt aléatoire a montré de meilleures performances pour le second modèle sur le jeu de validation.

Le système développé permet de classifier automatiquement les flux réseau et constitue une base pour la mise en place d'un dispositif intelligent de détection d'intrusion.

Les performances obtenues devront également être confirmées sur des données jamais vues, notamment le jeu de test caché utilisé par l'enseignant.
