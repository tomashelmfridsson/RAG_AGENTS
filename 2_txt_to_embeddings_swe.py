import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import tiktoken  # Optional: only if you chunk by token count

# === INSTÄLLNINGAR ===
TEXT_DIRECTORY = Path("datatxt")
CHUNK_DIRECTORY = Path("datachunks")
EMBEDDINGS_FOLDER = Path("dataembedding")
MAX_TOKENS = 512  # Justera efter modell

# === SE TILL ATT MAPPAR FINNS ===
CHUNK_DIRECTORY.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_FOLDER.mkdir(parents=True, exist_ok=True)

# === LÄS IN MODELL ===
embedding_model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")

# === TOKENIZER ===
try:
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
except Exception:
    encoding = tiktoken.get_encoding("cl100k_base")  # fallback

# === HJÄLPFUNKTION FÖR CHUNKNING ===
def split_into_chunks(text, max_tokens=MAX_TOKENS):
    tokens = encoding.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [encoding.decode(chunk) for chunk in chunks]

# === HUVUDLOGIK ===
def generate_and_store_embeddings():
    embeddings = []
    file_names = []

    text_files = list(TEXT_DIRECTORY.glob("*.txt"))
    if not text_files:
        print("❌ Inga .txt-filer hittades i 'datatxt/'")
        return

    print(f"🔍 Bearbetar {len(text_files)} filer...")

    for file_path in tqdm(text_files, desc="⏳ Genererar embeddings"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = split_into_chunks(content)
        for idx, chunk in enumerate(chunks):
            chunk_name = f"{file_path.stem}_part_{idx+1}"
            txt_chunk_path = CHUNK_DIRECTORY / f"{chunk_name}.txt"
            with open(txt_chunk_path, "w", encoding="utf-8") as chunk_file:
                chunk_file.write(chunk)

            embedding = embedding_model.encode([chunk])[0]
            embeddings.append(embedding)
            file_names.append(chunk_name)

    # === Konvertera och spara ===
    embeddings_np = np.array(embeddings).astype('float32')
    file_names_np = np.array(file_names)

    np.save(EMBEDDINGS_FOLDER / "embeddings.npy", embeddings_np)
    np.save(EMBEDDINGS_FOLDER / "file_names.npy", file_names_np)

    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    faiss.write_index(index, str(EMBEDDINGS_FOLDER / "faiss_index.index"))

    print("✅ Alla embeddings och index har sparats.")

# === MAIN ===
if __name__ == "__main__":
    generate_and_store_embeddings()
