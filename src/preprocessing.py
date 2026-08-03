"""
Prétraitement du texte pour le projet Rakuten.

Reproduit à l'identique le nettoyage utilisé dans les notebooks
(nettoyage HTML + fusion designation/description).
"""
import re

import pandas as pd


def nettoyer_texte(texte) -> str:
    """Nettoie un texte : supprime les balises HTML, remplace les entités,
    normalise les espaces."""
    if pd.isna(texte) or texte == "nan":
        return ""
    texte = str(texte)
    # Supprimer les balises HTML
    texte = re.sub(r"<[^>]+>", "", texte)
    # Remplacer les entités HTML courantes
    remplacements = {
        "&nbsp;": " ",
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
        "&eacute;": "é",
        "&egrave;": "è",
        "&ecirc;": "ê",
        "&agrave;": "à",
    }
    for cle, valeur in remplacements.items():
        texte = texte.replace(cle, valeur)
    # Normaliser les espaces
    texte = re.sub(r"\s+", " ", texte)
    return texte.strip()


def construire_texte_complet(designation, description) -> str:
    """Concatène designation + description après nettoyage (une seule paire)."""
    return (nettoyer_texte(designation) + " " + nettoyer_texte(description)).strip()


def preparer_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne `texte_complet` à un DataFrame contenant
    `designation` et `description`."""
    df = df.copy()
    df["texte_complet"] = (
        df["designation"].apply(nettoyer_texte)
        + " "
        + df["description"].apply(nettoyer_texte)
    ).str.strip()
    return df
