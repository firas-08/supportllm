"""
Un baseline pour appeler le LLM avant sans entraînement (zéro shot).
On veut voir ce que le LLM arrive à faire sur les questions support client.
Ce baseline sert comme une comparaison utltérieurment.


Contenu : charger eval → charger modèle → générer une réponse par exemple 
→ parser le JSON → calculer les métriques → résumer → sauvegarder

Usage : poetry run python -m src.baseline
"""

import pandas as pd
import transformers

from rouge_score import rouge_scorer
from pathlib import Path
import json
import re

MODEL = "Qwen/Qwen2.5-3B-Instruct"
PROMPT_SYSTEME = (
    "You are a customer support agent. Given the customer's message, reply with "
    "a JSON object with exactly two fields: \"intent\" (the single best matching "
    "intent label, in lowercase with underscores, e.g. cancel_order) and "
    "\"response\" (a short, helpful reply, maximum 2 short sentences, under 40 "
    "words total). Output ONLY the JSON object, nothing else — no explanation, "
    "no markdown, no step-by-step instructions."
)
CHEMIN = Path(__file__).resolve().parent.parent / "data" / "processed" / "eval.jsonl"

def charger_eval(chemin=CHEMIN) -> pd.DataFrame:
    """ Charger la partie évaluation de la dataset """
    df = pd.read_json(chemin, lines=True)
    return df

def charger_model(model=MODEL):
    """ Charger le modèle """
    llm = transformers.AutoModelForCausalLM.from_pretrained(model)
    tokenizer_llm = transformers.AutoTokenizer.from_pretrained(model)
    return llm, tokenizer_llm

def generate_answer(question : str, model : any,  tokenizer : any, max_new_tokens=160) -> str:
    """ Génère une réponse avec le système prompt
    La réponse coule de la partie évaluation dans 
    le dataset. """
    messages = [
        {"role" : "system", "content" : PROMPT_SYSTEME},
        {"role" : "user", "content" : question},
    ]
    texte = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(texte, return_tensors='pt')
    texte_genere = model.generate(**inputs, max_new_tokens=max_new_tokens)
    answer = tokenizer.decode(texte_genere[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return answer

def parser_sortie_json(texte_brut: str) -> dict:
    """ Extrait {"intent":..., "response":...} d'une sortie LLM imparfaite.
    Le modèle n'entoure pas toujours son JSON proprement (texte parasite
    autour) : on isole le premier bloc {...} plutôt que de supposer une
    sortie parfaite. """
    match = re.search(r"\{.*\}", texte_brut, flags=re.DOTALL)
    if not match:
        return {"intent": "", "response": texte_brut.strip()}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"intent": "", "response": texte_brut.strip()}
    return {
        "intent": str(data.get("intent", "")).strip().lower(),
        "response": str(data.get("response", "")).strip(),
    }

def dossier_modele(model: str = MODEL) -> Path:
    """ Dossier dédié experiments/baseline/<model> pour ranger tous les
    fichiers d'un modèle ensemble, au lieu de les préfixer en vrac. """
    dossier = Path(__file__).resolve().parent.parent / "experiments" / "baseline" / model.replace("/", "_")
    dossier.mkdir(parents=True, exist_ok=True)
    return dossier

def evaluate(df_eval, model, tokenizer):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # Fichier de reprise : un dossier par modèle, pour ne jamais mélanger deux
    # runs et pouvoir relancer le script après une interruption sans tout refaire.
    chemin_partiel = dossier_modele() / "partiel.jsonl"

    deja_traitees = set()
    if chemin_partiel.exists():
        df_deja = pd.read_json(chemin_partiel, lines=True)
        if not df_deja.empty:
            deja_traitees = set(df_deja["instruction"])
        print(f"Reprise : {len(deja_traitees)} exemples déjà traités, on les saute.")

    # Compteurs cumulés (incluent les exemples repris d'un run précédent) pour
    # afficher un F1 intermédiaire tous les 25 exemples, sans attendre la fin.
    nb_traites = len(deja_traitees)
    nb_corrects = 0
    if deja_traitees:
        df_deja = pd.read_json(chemin_partiel, lines=True)
        nb_corrects = int(df_deja["intent_correct"].sum())

    for idx, ligne in df_eval.iterrows():
        if ligne["instruction"] in deja_traitees:
            continue

        sortie_brute = generate_answer(ligne["instruction"], model, tokenizer)
        parsed = parser_sortie_json(sortie_brute)

        rouge_l = scorer.score(ligne["response"], parsed["response"])["rougeL"].fmeasure
        intent_correct = parsed["intent"] == ligne["intent"]

        resultat = {
            "instruction": ligne["instruction"],
            "intent_reel": ligne["intent"],
            "intent_predit": parsed["intent"],
            "intent_correct": intent_correct,
            "response_reelle": ligne["response"],
            "response_predite": parsed["response"],
            "rouge_l": rouge_l,
        }

        # Écriture incrémentale : chaque exemple est sauvegardé dès qu'il est
        # traité (mode append), pas seulement à la toute fin de la boucle.
        with open(chemin_partiel, "a", encoding="utf-8") as f:
            f.write(json.dumps(resultat, ensure_ascii=False) + "\n")

        nb_traites += 1
        nb_corrects += int(intent_correct)

        if nb_traites % 5 == 0:
            f1_intermediaire = nb_corrects / nb_traites
            print(f"Progression : {nb_traites} / {len(df_eval)} — F1 intent cumulé = {f1_intermediaire:.3f}")

    return pd.read_json(chemin_partiel, lines=True)

def resumer(df_resultats: pd.DataFrame) -> dict:
    """ Agrège les métriques ligne à ligne en un résumé comparable entre runs. """
    return {
        "model_id": MODEL,
        "n_exemples": len(df_resultats),
        "f1_intent": df_resultats["intent_correct"].mean(),
        "rouge_l_moyen": df_resultats["rouge_l"].mean(),
    }

def save(resume: dict, df_resultats: pd.DataFrame) -> None:
    """ Sauvegarde le résumé et le détail ligne à ligne dans
    experiments/baseline/<model>/ (results.json, detail.csv). """
    dossier = dossier_modele()

    chemin_resultats = dossier / "results.json"
    chemin_resultats.write_text(json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")

    chemin_detail = dossier / "detail.csv"
    df_resultats.to_csv(chemin_detail, index=False)
    print(f"Résultats écrits dans {chemin_resultats}")
    print(f"Détail écrit dans {chemin_detail}")

def main():
    df_eval = charger_eval(CHEMIN)
    llm, tokenizer_llm = charger_model(MODEL)
    df_resultats = evaluate(df_eval, llm, tokenizer_llm)
    resume = resumer(df_resultats)
    print("Résumé :", resume)
    save(resume, df_resultats)
    return 0

if __name__ == "__main__":
    main()