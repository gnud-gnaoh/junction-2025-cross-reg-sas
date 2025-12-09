import os
os.environ["TORCHINDUCTOR_DISABLED"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["DISABLE_EXPERIMENTAL_FEATURES"] = "1"

import pandas as pd
import numpy as np
import random
from transformers import pipeline
from tqdm import tqdm

# ---------- SAFE PRINT FOR WINDOWS UNICODE ---------
def safe_print(text):
    try:
        print(text)
    except Exception:
        try:
            print(str(text).encode("utf-8", "ignore").decode("utf-8"))
        except Exception:
            print("[[UNPRINTABLE TEXT SKIPPED]]")

# ---------- LOAD GPU NLI PIPELINE ----------
pipe = pipeline(
    "text-classification",
    model="tasksource/ModernBERT-large-nli",
    device=0,                     # <<< GPU
    torch_dtype="auto"
)

# ---------- CONFIG ----------
csv_path = "final_combined_classified_data.csv"
text_column = "Paragraph"
class_column = "classified_class"

similarity_threshold = 0.93
batch_size = 32  # batch NLI calls to avoid crashing

# ---------- LOAD DATA ----------
df = pd.read_csv(csv_path, delimiter="|")
df[text_column] = df[text_column].astype(str)
df = df[df[text_column].str.strip() != ""].reset_index(drop=True)
#classes = df[class_column].unique()
classes = ["Risk_MarketLiquidity", "Risk_DepositRunOff", "Risk_MaturityMismatch", "Risk_ReputationImpact"]
safe_print(f"Classes found: {classes}")

# ---------- HELPER: classify a batch ----------
def classify_batch(pipe, batch_pairs):
    """
    batch_pairs: list of tuples (text1, text2)
    Returns list of results from pipe
    """
    try:
        return pipe([{"text": a, "text_pair": b} for (a, b) in batch_pairs])
    except Exception as e:
        safe_print(f"Batch error: {e}")
        return [{} for _ in batch_pairs]

# ---------- CREATE OUTPUT DIRECTORY ----------
os.makedirs("nli_results", exist_ok=True)

# ---------- PROCESS EACH CLASS ----------
for cls in classes:
    safe_print(f"\n=== Processing class: {cls} ===")
    subset = df[df[class_column] == cls].reset_index(drop=True)
    texts = subset[text_column].tolist()
    n = len(texts)

    sim_file = f"sim_matrices/similarity_{cls}.npy"
    if not os.path.exists(sim_file):
        safe_print(f"Missing similarity file: {sim_file}, skipping class {cls}")
        continue

    similarity_matrix = np.load(sim_file)
    if similarity_matrix.shape[0] != n:
        safe_print(f"Mismatch between similarity matrix ({similarity_matrix.shape}) and #texts ({n}). Skipping.")
        continue

    # --- Collect all high-similarity pairs ---
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            score = float(similarity_matrix[i, j])
            if score >= similarity_threshold:
                pairs.append((i, j, score))

    safe_print(f"Found {len(pairs)} pairs with similarity >= {similarity_threshold}")
    if not pairs:
        continue

    # --- Classify in batches ---
    entailments = []
    contradictions = []
    batched = []

    pbar = tqdm(total=len(pairs), desc=f"Class {cls}", ncols=100)

    for i, j, sim_score in pairs:
        t1 = texts[i]
        t2 = texts[j]
        batched.append((i, j, sim_score, t1, t2))

        if len(batched) == batch_size:
            batch_texts = [(x[3], x[4]) for x in batched]
            results = classify_batch(pipe, batch_texts)

            for info, res in zip(batched, results):
                if not res:
                    continue
                label = res.get("label", "").upper()
                score = res.get("score", 0)

                if label == "ENTAILMENT":
                    entailments.append((info[0], info[1], info[2], score, info[3], info[4]))
                elif label == "CONTRADICTION":
                    contradictions.append((info[0], info[1], info[2], score, info[3], info[4]))
            batched = []
            # free GPU memory
            import torch
            torch.cuda.empty_cache()
            pbar.update(batch_size)

    # process leftover batch
    if batched:
        batch_texts = [(x[3], x[4]) for x in batched]
        results = classify_batch(pipe, batch_texts)

        for info, res in zip(batched, results):
            if not res:
                continue
            label = res.get("label", "").upper()
            score = res.get("score", 0)

            if label == "ENTAILMENT":
                entailments.append((info[0], info[1], info[2], score, info[3], info[4]))
            elif label == "CONTRADICTION":
                contradictions.append((info[0], info[1], info[2], score, info[3], info[4]))
        pbar.update(len(batched))

    pbar.close()

    safe_print(f"Class {cls}: Found {len(entailments)} entailments and {len(contradictions)} contradictions")

    # --- SAVE TO CSV ---
    if entailments:
        ent_df = pd.DataFrame(entailments, columns=["index_i", "index_j", "similarity", "nli_score", "text_i", "text_j"])
        ent_file = f"nli_results/entailments_{cls}.csv"
        ent_df.to_csv(ent_file, index=False, sep="|")
        safe_print(f"Saved {len(entailments)} entailments to {ent_file}")

    if contradictions:
        con_df = pd.DataFrame(contradictions, columns=["index_i", "index_j", "similarity", "nli_score", "text_i", "text_j"])
        con_file = f"nli_results/contradictions_{cls}.csv"
        con_df.to_csv(con_file, index=False, sep="|")
        safe_print(f"Saved {len(contradictions)} contradictions to {con_file}")

    # --- OPTIONAL: show one example ---
    if entailments:
        safe_print("\nExample ENTAILMENT:")
        i, j, sim, score, t1, t2 = entailments[0]
        safe_print(f"Similarity={sim:.4f}, Score={score:.4f}")
        safe_print("A:" + t1)
        safe_print("B:"+ t2)

    if contradictions:
        safe_print("\nExample CONTRADICTION:")
        i, j, sim, score, t1, t2 = contradictions[0]
        safe_print(f"Similarity={sim:.4f}, Score={score:.4f}")
        safe_print("A:"+ t1)
        safe_print("B:"+ t2)

safe_print("\nALL DONE")