"""
Dans ce fichier, nous allons évaluer les deux versions :
    * le modèle fine-tuné avec LoRA
    * le modèle baseline zero-shot
L'évaluation se fait grâce à la partie évaluation de la dataset.

Usage : poetry run python -m src.evaluate
"""

import json
import time
import argparse
import transformers
import torch
import peft

from src.baseline import charger_eval, parser_sortie_json, PROMPT_SYSTEME
from rouge_score import rouge_scorer
from pathlib import Path

import pandas as pd


CHEMIN = Path(__file__).resolve().parent.parent / "data" / "processed" / "eval.jsonl"
MODEL = "Qwen/Qwen2.5-3B-Instruct"

def charger_model(adapter:Path=None, model=MODEL) -> tuple[peft.PeftModel, transformers.PreTrainedTokenizer]:
    """ Charger le modèle en fonction de l'entrée : zero shot / fine tuned """
    bnb_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model_charge = transformers.AutoModelForCausalLM.from_pretrained(model, quantization_config=bnb_config, dtype=torch.float16)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    if adapter :
        model_charge = peft.PeftModel.from_pretrained(model_charge, adapter)
    return model_charge, tokenizer

def generer_reponse(question : str, model : any,  tokenizer : any, max_new_tokens=400) -> str:
    """ Génère une réponse avec le système prompt
    La réponse coule de la partie évaluation dans 
    le dataset. """
    messages = [
        {"role" : "system", "content" : PROMPT_SYSTEME},
        {"role" : "user", "content" : question},
    ]
    texte = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(texte, return_tensors='pt').to(model.device)
    texte_genere = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    answer = tokenizer.decode(texte_genere[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return answer

def evaluate(df_eval, model, tokenizer, dossier:Path):
    """ Evaluer le modèle avec la dataset d'évaluation. """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    chemin = dossier / "partiel.jsonl"
    deja_traitees = set()
    if chemin.exists():
        df_deja = pd.read_json(chemin, lines=True)
        if not df_deja.empty:
            deja_traitees = set(df_deja["instruction"])
        print(f"Reprise : {len(deja_traitees)} exemples déjà traités, on les saute.")

    # Compteurs cumulés (incluent les exemples repris d'un run précédent) pour
    # afficher un F1 intermédiaire tous les 25 exemples, sans attendre la fin.
    nb_traites = len(deja_traitees)
    nb_corrects = 0
    if deja_traitees:
        df_deja = pd.read_json(chemin, lines=True)
        nb_corrects = int(df_deja["intent_correct"].sum())

    for idx, ligne in df_eval.iterrows():
        if ligne["instruction"] in deja_traitees:
            continue
        debut = time.perf_counter()    
        sortie_brute = generer_reponse(ligne["instruction"], model, tokenizer)
        latence = time.perf_counter() - debut
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
            "latence_s" : latence,
            "rouge_l": rouge_l,
        }

        # Écriture incrémentale : chaque exemple est sauvegardé dès qu'il est
        # traité (mode append), pas seulement à la toute fin de la boucle.
        with open(chemin, "a", encoding="utf-8") as f:
            f.write(json.dumps(resultat, ensure_ascii=False) + "\n")

        nb_traites += 1
        nb_corrects += int(intent_correct)

        if nb_traites % 5 == 0:
            f1_intermediaire = nb_corrects / nb_traites
            print(f"Progression : {nb_traites} / {len(df_eval)} — F1 intent cumulé = {f1_intermediaire:.3f}")

    return pd.read_json(chemin, lines=True)

def resumer(df_resultats: pd.DataFrame) -> dict:
    """ Agrège les métriques ligne à ligne en un résumé comparable entre runs. """
    return {
        "model_id": MODEL,
        "n_exemples": len(df_resultats),
        "f1_intent": df_resultats["intent_correct"].mean(),
        "rouge_l_moyen": df_resultats["rouge_l"].mean(),
        "latence_p50_s": df_resultats["latence_s"].quantile(0.5),
        "latence_p95_s": df_resultats["latence_s"].quantile(0.95),
    }

def save(resume: dict, df_resultats: pd.DataFrame, dossier:Path) -> None:
    """ Sauvegarde le résumé et le détail ligne à ligne dans
    on en sait pas encore """

    chemin_resultats = dossier / "results.json"
    chemin_resultats.write_text(json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")

    chemin_detail = dossier / "detail.csv"
    df_resultats.to_csv(chemin_detail, index=False)
    print(f"Résultats écrits dans {chemin_resultats}")
    print(f"Détail écrit dans {chemin_detail}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    eval_dataset = charger_eval(CHEMIN)
    if args.limit:
        eval_dataset = eval_dataset.head(args.limit)
    model, tokenizer = charger_model(args.adapter)
    dossier = Path(__file__).resolve().parent.parent / "experiments" / "evaluation" / MODEL.replace("/", "_") / args.run_name
    dossier.mkdir(parents=True, exist_ok=True)
    resultats = evaluate(eval_dataset, model, tokenizer, dossier)
    resume = resumer(resultats)
    save(resume, resultats, dossier)
    return 0

if __name__ == "__main__" :
    main()