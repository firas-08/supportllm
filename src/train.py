""" 
Dans ce fichier, nous allons procéder à un fine-tuning (QLoRA) 
du modèle utilisé dans le fichier de référence. La référence est 
un modèle zero-shot, tandis que le résultat de ce fichier est un 
modèle affiné sur le jeu de données d'entraînement présent dans 
data/train.jsonl. L'objectif de ce fine-tuning est d'ajuster la 
sortie du modèle pour qu'elle respecte le format JSON, avec un 
ton plus proche de celui d'un service client.

Utilisation : poetry run python -m src.train
"""

import logging
import transformers
import torch
import datasets
import peft
import bitsandbytes
import trl
import json
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHEMIN = Path(__file__).resolve().parent.parent / "data" / "processed" / "train.jsonl"
DOSSIER_MODELS = Path(__file__).resolve().parent.parent / "models"

MODEL = "Qwen/Qwen2.5-3B-Instruct"
PROMPT_SYSTEME = (
    "You are a customer support agent. Given the customer's message, reply with "
    "a JSON object with exactly two fields: \"intent\" (the single best matching "
    "intent label, in lowercase with underscores, e.g. cancel_order) and "
    "\"response\" (a short, helpful reply, maximum 2 short sentences, under 40 "
    "words total). Output ONLY the JSON object, nothing else — no explanation, "
    "no markdown, no step-by-step instructions."
)

def charge_datatraining(chemin = CHEMIN) -> pd.DataFrame:
    """ Charger la partie training de la dataset. """
    df = pd.read_json(chemin, lines=True)
    return df

def get_conversations(df : pd.DataFrame) -> list[list[dict]]:
    """ Transformer la dataset vers une format de conversation """
    liste_conversations = []
    for idx, ligne in df.iterrows() :
        instruct = ligne["instruction"]
        intent = ligne["intent"]
        response = ligne["response"]
        json_attendu = json.dumps({"intent" : intent, "response": response})
        conversation_system = {"role" : "system", "content": PROMPT_SYSTEME}
        conversation_user = {"role": "user", "content": instruct}
        conversation_assistant = {"role" : "assistant", "content": json_attendu}
        liste_conversations.append([conversation_system, conversation_user, conversation_assistant])
    return liste_conversations

def get_quantized_model(model=MODEL) -> tuple[peft.PeftModel, transformers.PreTrainedTokenizer]:
    """ Charger le model en quantization 4-bits (QLoRA) """
    bnb_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model_quant = transformers.AutoModelForCausalLM.from_pretrained(model, quantization_config=bnb_config, dtype=torch.float16)
    model_quant = peft.prepare_model_for_kbit_training(model_quant)
    lora_config = peft.LoraConfig(r=16, lora_alpha=32,
                                  target_modules=["q_proj", "v_proj"],
                                  lora_dropout=0.05,
                                  task_type="CAUSAL_LM")
    ready_model = peft.get_peft_model(model_quant, lora_config)
    for p in ready_model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    logger.info("dtypes entraînables : %s", {p.dtype for p in ready_model.parameters() if p.requires_grad})

    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    return ready_model, tokenizer

def train(conversations : list[list[dict]], model_quantized:peft.PeftModel, tokenizer:transformers.PreTrainedTokenizer, chemin=DOSSIER_MODELS) -> trl.SFTTrainer:
    """ Fine tuner le modèle avec des loggings réguliers.

    Reprise : save_strategy="steps" + save_steps régulier écrit des
    checkpoints dans `chemin` (output_dir). Si ce dossier contient déjà des
    checkpoints (run précédent interrompu), on reprend dessus au lieu de
    repartir de zéro — même logique que la reprise dans baseline.py, mais
    gérée nativement par transformers.Trainer via resume_from_checkpoint.
    """
    dataset = datasets.Dataset.from_dict({"messages": conversations})
    sft_config = trl.SFTConfig(
        output_dir=str(chemin),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        logging_steps=5,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        bf16=False,
        fp16=True,
    )
    trainer = trl.SFTTrainer(
        model_quantized,
        sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # SFTTrainer force les paramètres LoRA en bf16 pour tout modèle quantisé ;
    # le GradScaler fp16 les refuse, donc on les repasse en fp32 avant train().
    for p in model_quantized.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    logger.info("dtypes entraînables : %s", {p.dtype for p in model_quantized.parameters() if p.requires_grad})

    checkpoint_existant = transformers.trainer_utils.get_last_checkpoint(str(chemin))
    if checkpoint_existant:
        logger.info("Reprise depuis le checkpoint %s", checkpoint_existant)
    else:
        logger.info("Aucun checkpoint trouvé, entraînement depuis le début.")

    trainer.train(resume_from_checkpoint=checkpoint_existant)
    return trainer

def dossier_modele(model: str = MODEL) -> Path:
    """ Dossier dédié experiments/finetuning/<model>/, même logique que
    dossier_modele() dans baseline.py (un dossier par modèle, pas des fichiers
    préfixés en vrac dans un seul dossier partagé). """
    dossier = Path(__file__).resolve().parent.parent / "experiments" / "finetuning" / model.replace("/", "_")
    dossier.mkdir(parents=True, exist_ok=True)
    return dossier

def save(training_results: trl.SFTTrainer) -> None:
    """ Sauvegarder l'adaptateur LoRA, un résumé, et l'historique de loss. """
    dossier = dossier_modele()

    dossier_adapter = dossier / "adapter"
    training_results.save_model(dossier_adapter)

    resume = {
        "model_id": MODEL,
        "adapter_dir": str(dossier_adapter),
        "n_exemples_train": len(training_results.train_dataset),
    }
    chemin_resultats = dossier / "results.json"
    chemin_resultats.write_text(json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")

    chemin_loss = dossier / "loss_history.json"
    loss = training_results.state.log_history
    chemin_loss.write_text(json.dumps(loss, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Adaptateur écrit dans %s", dossier_adapter)
    logger.info("Résultats écrits dans %s", chemin_resultats)
    logger.info("Loss écrite dans %s", chemin_loss)


def main():
    df = charge_datatraining(CHEMIN)
    conversations = get_conversations(df)
    model_quantized, tokenizer = get_quantized_model(MODEL)
    training_results = train(conversations, model_quantized, tokenizer, DOSSIER_MODELS)
    save(training_results)
    return 0

if __name__ == "__main__" :
    main()
