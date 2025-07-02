import os
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Modell: svensk, lokal
EMBEDDING_MODEL_NAME = "KBLab/sentence-bert-swedish-cased"

# In-/ut-mappar
TEXT_DIRECTORY = Path("datatxt")
EMBEDDINGS_FOLDER = Path("dataembedding")
EMBEDDINGS_FOLDER.mkdir(parents=True, exist_ok=True)

# Chunkning: dela upp stora texter i fasta teckenlängder (valfritt)
CHUNK_SIZE = 1000

# Ladda svensk modell (endast 1 gång)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def split_text(text, max_length=CHUNK_SIZE):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def create_embedding(text):
    try:
        return embedding_model.encode(text)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def generate_and_store_embeddings():
    embeddings = []
    file_names = []

    text_files = list(TEXT_DIRECTORY.glob("*.txt"))
    if not text_files:
        print("No text files found in the datatxt folder.")
        return

    print(f"Found {len(text_files)} text files to process.")

    for file_path in tqdm(text_files, desc="Processing text files"):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            chunks = split_text(content)

            for idx, chunk in enumerate(chunks):
                embedding = create_embedding(chunk)
                if embedding is not None:
                    embeddings.append(embedding)
                    file_names.append(f"{file_path.stem}_part_{idx+1}")

    # Skapa numpy-matriser
    embeddings_np = np.array(embeddings).astype('float32')
    file_names_np = np.array(file_names)

    # Spara
    np.save(EMBEDDINGS_FOLDER / "embeddings.npy", embeddings_np)
    np.save(EMBEDDINGS_FOLDER / "file_names.npy", file_names_np)

    # Skapa FAISS-index
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    faiss.write_index(index, str(EMBEDDINGS_FOLDER / "faiss_index.index"))

    print("Embeddings and FAISS index successfully saved.")

# Kör
if __name__ == "__main__":
    print("Starting the embedding generation process (local, Swedish)...")
    generate_and_store_embeddings()
    print("All embeddings have been generated and stored successfully.")
