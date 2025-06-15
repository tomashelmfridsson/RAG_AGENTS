import os
import openai
import faiss
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import tiktoken

# Load OpenAI API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define input and output directories
TEXT_DIRECTORY = Path("datatxt")
EMBEDDINGS_FOLDER = Path("dataembedding")

# Ensure the embedding folder exists
EMBEDDINGS_FOLDER.mkdir(parents=True, exist_ok=True)

# Initialize tokenizer for token counting
encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
MAX_TOKENS = 8192  # Maximum tokens for text-embedding-ada-002

# Function to split text into chunks based on the token limit
def split_into_chunks(text, max_tokens=MAX_TOKENS):
    tokens = encoding.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [encoding.decode(chunk) for chunk in chunks]

# Function to create embeddings using OpenAI's API
def create_embedding(text):
    try:
        response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# Function to process text files and generate embeddings
def generate_and_store_embeddings():
    embeddings = []
    file_names = []

    # Iterate over all .txt files in the directory
    text_files = list(TEXT_DIRECTORY.glob("*.txt"))
    if not text_files:
        print("No text files found in the datatxt folder.")
        return

    print(f"Found {len(text_files)} text files to process.")

    for file_path in tqdm(text_files, desc="Processing text files"):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

            # Split content into chunks based on token limit
            chunks = split_into_chunks(content)

            for idx, chunk in enumerate(chunks):
                embedding = create_embedding(chunk)
                if embedding:
                    embeddings.append(embedding)
                    # Use unique names for each chunk
                    file_names.append(f"{file_path.stem}_part_{idx+1}")

    # Convert embeddings and file names to NumPy arrays
    embeddings_np = np.array(embeddings).astype('float32')  # FAISS requires float32 type
    file_names_np = np.array(file_names)

    # Save embeddings and file names
    np.save(EMBEDDINGS_FOLDER / "embeddings.npy", embeddings_np)
    np.save(EMBEDDINGS_FOLDER / "file_names.npy", file_names_np)

    # Create FAISS index and add embeddings
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
    index.add(embeddings_np)

    # Save FAISS index
    faiss.write_index(index, str(EMBEDDINGS_FOLDER / "faiss_index.index"))

    print("Embeddings and FAISS index successfully saved.")

# Main function
if __name__ == "__main__":
    print("Starting the embedding generation process...")
    generate_and_store_embeddings()
    print("All embeddings have been generated and stored successfully.")
