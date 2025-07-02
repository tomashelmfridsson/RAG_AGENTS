# Vanilla RAG pipeline by [Toufique Hasan - 2025]
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
import ollama  # ← Alternativ 2: lokal LLaMA

load_dotenv()

# Define the path for embeddings and FAISS index
EMBEDDINGS_FILE = "dataembedding/embeddings.npy"
FILE_NAMES_FILE = "dataembedding/file_names.npy"
FAISS_INDEX_FILE = "dataembedding/faiss_index.index"

# Load embeddings, file names, and FAISS index
def load_faiss_index_and_embeddings():
    embeddings = np.load(EMBEDDINGS_FILE)
    file_names = np.load(FILE_NAMES_FILE)
    index = faiss.read_index(FAISS_INDEX_FILE)
    return index, embeddings, file_names

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("KBLab/sentence-bert-swedish-cased")

# Create query embedding using sentence-transformers
def create_query_embedding(query):
    try:
        embedding_model = get_embedding_model()
        embedding = embedding_model.encode([query])[0]
        return embedding
    except Exception as e:
        st.error(f"Error creating query embedding: {e}")
        return None

# Search for similar chunks based on query embedding
def search_similar_chunks(query_embedding, index, file_names, top_k=10, distance_threshold=None):
    D, I = index.search(np.array([query_embedding]).astype('float32'), top_k)
    results = []
    print("Avstånd (D):", D)
    print("Index (I):", I)
    print("Antal giltiga träffar:", sum(i != -1 for i in I[0]))
    for idx, i in enumerate(I[0]):
        if i == -1:
            continue
        if distance_threshold is None or D[0][idx] < distance_threshold:
            results.append((file_names[i], D[0][idx]))
    results = sorted(results, key=lambda x: x[1])[:5]
    return results

# Generate answer using LLaMA 3 via Ollama
def generate_answer_llama(context, query):
    try:
        prompt = f"""
Du är en hjälpsam AI-assistent. Här är relevant kontext från dokument:

{context}

Fråga: {query}

Svara så tydligt och korrekt du kan baserat på kontexten ovan.
"""
        response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response['message']['content']
    except Exception as e:
        st.error(f"Error generating answer with LLaMA: {e}")
        return "Kunde inte generera ett svar just nu."

# Main function for RAG retrieval and answer generation
def retrieve_and_generate_answer(query, top_k=10, distance_threshold=None):
    query_embedding = create_query_embedding(query)
    if query_embedding is None:
        return "Failed to create query embedding."

    index, embeddings, file_names = load_faiss_index_and_embeddings()

    print("Index dimension:", index.d)
    print("Query shape:", np.array([query_embedding]).astype('float32').shape)

    similar_chunks = search_similar_chunks(query_embedding, index, file_names, top_k, distance_threshold)

    if not similar_chunks:
        return "No relevant context found."

    context = ""
    sources = []
    for fname, _ in similar_chunks:
        pdf_name = fname.split("_part")[0]
        try:
            with open(f"datachunks/{fname}.txt", "r", encoding="utf-8") as f:
                context += f.read() + "\n"
            sources.append(pdf_name)
        except FileNotFoundError:
            st.warning(f"Saknad chunk-fil: {fname}")
            continue

    sources = list(dict.fromkeys(sources))
    print("===== Kontext till GPT START =====")
    print(context[:1000])
    print("===== Kontext till GPT SLUT =====")

    answer = generate_answer_llama(context, query)
    sources_text = "\n".join([f"[{idx+1}] {source}" for idx, source in enumerate(sources[:5])])
    answer_with_sources = f"{answer}\n\n**Sources:**\n{sources_text}"
    return answer_with_sources

# Streamlit UI
def main():
    st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")

    st.markdown(
        """
        <h1 style='text-align: center;'>🤖 RAG Assistant</h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style='text-align: center; font-size: 20px;'>
            <strong>En AI-assistent med Retrieval-Augmented Generation (RAG) som levererar kontextmedvetna svar från dina egna dokument.</strong>
        </p>
        """,
        unsafe_allow_html=True
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.markdown("### 🕘 Tidigare frågor")
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])

    if user_input := st.chat_input("Ställ en fråga:"):
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Genererar svar..."):
                answer = retrieve_and_generate_answer(user_input, distance_threshold=None)
                st.write(answer)
        st.session_state.chat_history.append({"question": user_input, "answer": answer})

    add_footer()

# Footer
def add_footer():
    st.markdown(
        """
        <!-- Developed by Toufique Hasan, modded by Tomas -->
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #d3d3d3;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
            color: black;
            border-top: 1px solid #ccc;
            z-index: 999;
        }
        body {
            margin: 0;
            padding-bottom: 50px;
        }
        </style>
        <div class="footer">
            2025 © GPT-Lab + Tomas Helmfridsson
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
