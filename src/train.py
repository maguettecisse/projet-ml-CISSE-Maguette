
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)


# 1. Définir les chemins du projet
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "flux_train.csv"
MODEL_PATH = BASE_DIR / "models" / "model.joblib"


# 2. Charger les données
print("Chargement des données...")

df = pd.read_csv(DATA_PATH)

print("Dimensions du jeu de données :", df.shape)


# 3. Séparer les variables et la cible
X = df.drop(columns=["attaque"])
y = df["attaque"]

numeric_features = [
    "duration",
    "dst_port",
    "src_bytes",
    "dst_bytes",
    "nb_packets",
    "nb_failed_logins",
    "nb_conn_same_host",
    "syn_rate"
]

categorical_features = ["protocol"]


# 4. Prétraiter les données numériques
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# 5. Prétraiter les données catégorielles
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# 6. Construire le prétraitement complet
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# 7. Séparer les données en apprentissage et validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDonnées d'apprentissage :", X_train.shape)
print("Données de validation :", X_valid.shape)


# 8. Définir les modèles à comparer
models = {
    "Régression logistique": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Forêt aléatoire": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


# 9. Entraîner et évaluer les modèles
results = {}
best_f1 = -1
best_model_name = None
best_pipeline = None

for name, classifier in models.items():

    print("\n" + "=" * 55)
    print("Modèle :", name)

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_valid)

    f1 = f1_score(y_valid, predictions, pos_label=1)
    precision = precision_score(
        y_valid, predictions, pos_label=1, zero_division=0
    )
    recall = recall_score(
        y_valid, predictions, pos_label=1, zero_division=0
    )

    results[name] = {
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

    print("\nRapport de classification :")
    print(classification_report(
        y_valid,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Attaque"],
        zero_division=0
    ))

    matrix = confusion_matrix(
        y_valid, predictions, labels=[0, 1]
    )

    print("Matrice de confusion :")
    print(matrix)

    print("F1-score attaque :", round(f1, 4))
    print("Précision attaque :", round(precision, 4))
    print("Rappel attaque :", round(recall, 4))

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_pipeline = pipeline


# 10. Afficher la comparaison
print("\n" + "=" * 55)
print("COMPARAISON DES MODELES")

for name, metrics in results.items():
    print(
        f"{name} : "
        f"F1={metrics['f1']:.4f} | "
        f"Précision={metrics['precision']:.4f} | "
        f"Rappel={metrics['recall']:.4f}"
    )


# 11. Réentraîner le meilleur modèle sur toutes les données
print("\nMeilleur modèle :", best_model_name)
print("Réentraînement sur toutes les données...")

best_pipeline.fit(X, y)


# 12. Enregistrer le modèle
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(best_pipeline, MODEL_PATH)

print("\nModèle enregistré avec succès :", MODEL_PATH)
print("Entraînement terminé !")
