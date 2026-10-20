"""Prépare le dataset Bitext pour SupportLLM : dédoublonnage sémantique
et split train/eval stratifié, reproductibles.

Usage : poetry run python -m src.data_prep
"""

import logging
import random
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SEUIL_QUASI_DUPLICAT = 0.9
TAILLE_EVAL = 500
PLANCHER_MIN_PAR_INTENT = 15
SEED = 42

DOSSIER_SORTIE = Path(__file__).resolve().parent.parent / "data" / "processed"


def charger_dataset() -> pd.DataFrame:
    """Télécharge (ou lit depuis le cache HF) le dataset Bitext et le renvoie en DataFrame."""
    ds = load_dataset(DATASET_ID)
    df = ds["train"].to_pandas()
    logger.info("Dataset chargé : %d exemples, colonnes=%s", len(df), list(df.columns))
    return df


def detecter_quasi_duplicats(
    df: pd.DataFrame, model: SentenceTransformer, seuil: float = SEUIL_QUASI_DUPLICAT
) -> list[tuple[str, int, int, float]]:
    """Repère les paires de phrases quasi-identiques au sein de chaque intention.

    Comparer uniquement au sein d'une même intention (pas sur tout le dataset)
    est à la fois suffisant — deux phrases d'intentions différentes ne peuvent
    pas se faire concurrence au moment du split — et nécessaire pour la
    performance (une matrice de similarité N×N sur 27k lignes serait ~2.9 Go).
    """
    paires: list[tuple[str, int, int, float]] = []
    for intent, groupe in df.groupby("intent"):
        textes = groupe["instruction"].tolist()
        embeddings = model.encode(textes, normalize_embeddings=True, convert_to_numpy=True)
        similarity_matrix = util.cos_sim(embeddings, embeddings)
        matrice_triu = torch.triu(similarity_matrix, diagonal=1)
        for i, j in (matrice_triu > seuil).nonzero().tolist():
            paires.append((intent, i, j, similarity_matrix[i][j].item()))

    logger.info("%d quasi-duplicats détectés (seuil=%.2f)", len(paires), seuil)
    return paires


def dedupliquer(df: pd.DataFrame, paires: list[tuple[str, int, int, float]]) -> pd.DataFrame:
    """Exclut un représentant par paire quasi-duplicate (toujours `j`, jamais `i`).

    On exclut directement les paires détectées plutôt que de fusionner les
    composantes connexes du graphe de similarité : la transitivité (A~B et
    B~C n'implique pas A~C) ferait sinon fusionner des dizaines de phrases
    par groupe, avec un dataset au vocabulaire restreint par intention comme
    Bitext — constaté empiriquement (85% du dataset éliminé avec l'approche
    par composantes connexes, contre un taux de duplication réel ~3% des paires).
    """
    a_exclure = {(intent, j) for intent, _, j, _ in paires}

    df = df.copy()
    df["position_locale"] = df.groupby("intent").cumcount()
    masque = df.apply(
        lambda row: (row["intent"], row["position_locale"]) in a_exclure, axis=1
    )
    df_filtre = df[~masque].drop(columns=["position_locale"]).reset_index(drop=True)

    logger.info("%d phrases gardées sur %d (dédoublonnage)", len(df_filtre), len(df))
    return df_filtre


def split_stratifie(
    df: pd.DataFrame,
    taille_eval: int = TAILLE_EVAL,
    plancher_min: int = PLANCHER_MIN_PAR_INTENT,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split train/eval proportionnel par intention, avec un plancher minimum.

    Un split purement proportionnel laisserait les intentions les plus rares
    avec trop peu d'exemples en eval pour que leur score individuel soit
    fiable (une seule erreur ferait varier leur F1 de ±20%) ; le plancher
    garantit un minimum mesurable pour chaque intention.
    """
    eval_parts, train_parts = [], []

    for _, groupe in df.groupby("intent"):
        n_intent = len(groupe)
        part_proportionnelle = round(taille_eval * n_intent / len(df))
        n_eval = min(max(plancher_min, part_proportionnelle), n_intent)

        groupe_shuffle = groupe.sample(frac=1, random_state=seed)
        eval_parts.append(groupe_shuffle.iloc[:n_eval])
        train_parts.append(groupe_shuffle.iloc[n_eval:])

    df_eval = pd.concat(eval_parts).reset_index(drop=True)
    df_train = pd.concat(train_parts).reset_index(drop=True)

    logger.info("Split final : %d train / %d eval", len(df_train), len(df_eval))
    return df_train, df_eval


def sauvegarder(df_train: pd.DataFrame, df_eval: pd.DataFrame, dossier: Path = DOSSIER_SORTIE) -> None:
    """Écrit train/eval en JSONL. Non versionné dans Git (cf. .gitignore) : ce sont
    des artefacts reproductibles via ce script, pas du code."""
    dossier.mkdir(parents=True, exist_ok=True)
    df_train.to_json(dossier / "train.jsonl", orient="records", lines=True, force_ascii=False)
    df_eval.to_json(dossier / "eval.jsonl", orient="records", lines=True, force_ascii=False)
    logger.info("Fichiers écrits dans %s", dossier)


def main() -> None:
    random.seed(SEED)

    df = charger_dataset()
    model = SentenceTransformer(EMBEDDING_MODEL)

    paires = detecter_quasi_duplicats(df, model)
    df_filtre = dedupliquer(df, paires)
    df_train, df_eval = split_stratifie(df_filtre)
    sauvegarder(df_train, df_eval)


if __name__ == "__main__":
    main()
