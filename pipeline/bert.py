import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np

# ---------- CONFIG ----------
csv_path = "final_combined_classified_data.csv"
text_column = "cleaned_paragraph"   # use only cleaned text
class_column = "classified_class"   # filter within this class

model_name = "nlpaueb/legal-bert-base-uncased"
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available())
print(f"Using device: {device}")

# ---------- LOAD DATA ----------
df = pd.read_csv(csv_path, delimiter="|")
df[text_column] = df[text_column].astype(str)

# Drop empty cleaned paragraphs
df = df[df[text_column].str.strip() != ""].reset_index(drop=True)

print(f"Loaded {len(df)} rows with cleaned paragraphs.")

# ---------- LOAD MODEL ----------
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

# ---------- EMBEDDING FUNCTION ----------
def embed_texts(text_list, batch_size=16):
    all_embs = []

    for i in tqdm(range(0, len(text_list), batch_size), desc="Embedding"):
        batch_texts = text_list[i:i+batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc)
            batch_embs = outputs.last_hidden_state.mean(dim=1)

        all_embs.append(batch_embs.cpu())

    return torch.cat(all_embs, dim=0)

# ---------- PROCESS EACH CLASS SEPARATELY ----------
classes = df[class_column].unique()
print("Classes found:", classes)

for cls in classes:
    print(f"\n=== Processing class: {cls} ===")
    
    subset = df[df[class_column] == cls].reset_index(drop=True)
    texts = subset[text_column].astype(str).tolist()
    
    print(f"{len(texts)} items in this class.")

    # ---- Create embeddings for this class only ----
    embeddings = embed_texts(texts, batch_size=batch_size)
    print(f"Embeddings shape for class {cls}: {embeddings.shape}")

    # ---- Cosine similarity matrix ----
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    similarity_matrix = embeddings_norm @ embeddings_norm.T

    print(f"Similarity matrix shape for class {cls}: {similarity_matrix.shape}")

    # ---- Save ----
    filename = f"similarity_{cls}.npy"
    np.save(filename, similarity_matrix.numpy())
    print(f"Saved:", filename)
