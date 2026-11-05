
from pathlib import Path
import joblib
import pandas as pd


# Chemin vers le modèle enregistré
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "model.joblib"


def predict(df: pd.DataFrame) -> list:
    """
    Prédit si chaque flux réseau est normal ou une attaque.

    Paramètre :
        df : DataFrame contenant les caractéristiques des flux,
             sans la colonne attaque.

    Retour :
        Liste de 0 (normal) ou 1 (attaque).
    """

    # Charger le modèle entraîné
    model = joblib.load(MODEL_PATH)

    # Effectuer les prédictions
    predictions = model.predict(df)

    # Retourner une liste d'entiers
    return predictions.astype(int).tolist()
