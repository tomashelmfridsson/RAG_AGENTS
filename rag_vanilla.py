# Vanilla RAG pipeline by [Toufique Hasan - 2025]
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv

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


# Create query embedding using OpenAI
def create_query_embedding(query):
    try:
        # response = openai.Embedding.create(input=[query], model="text-embedding-ada-002")
        # return response['data'][0]['embedding']
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

# Generate answer using OpenAI's GPT-4 model
def generate_answer_gpt4(context, query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error generating answer with GPT-4: {e}")
        return "Unable to generate an answer at the moment."

# Main function for RAG retrieval and answer generation
def retrieve_and_generate_answer(query, top_k=10, distance_threshold=None):
    query_embedding = create_query_embedding(query)
    #if not query_embedding:
    if query_embedding is None:
        return "Failed to create query embedding."

    index, embeddings, file_names = load_faiss_index_and_embeddings()

    import numpy as np
    print("Index dimension:", index.d)
    print("Query shape:", np.array([query_embedding]).astype('float32').shape)

    similar_chunks = search_similar_chunks(query_embedding, index, file_names, top_k, distance_threshold)

    if not similar_chunks:
        return "No relevant context found."

    context = ""
    sources = []
    for fname, _ in similar_chunks:
        pdf_name = fname.split("_part")[0]
        #context += f"{fname}\n"
    with open(f"datachunks/{fname}.txt", "r", encoding="utf-8") as f:
        context += f.read() + "\n"
        sources.append(pdf_name)

    sources = list(dict.fromkeys(sources))
    print("===== Kontext till GPT START =====")
    print(context[:1000])  # max 1000 tecken för terminalen
    print("===== Kontext till GPT SLUT =====")
    answer = generate_answer_gpt4(context, query)
    sources_text = "\n".join([f"[{idx+1}] {source}" for idx, source in enumerate(sources[:5])])
    answer_with_sources = f"{answer}\n\n**Sources:**\n{sources_text}"
    return answer_with_sources

# Streamlit UI
def main():
    st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")

    # Title
    st.markdown(
        """
        <h1 style='text-align: center;'>🤖 RAG Assistant</h1>
        """,
        unsafe_allow_html=True
    )

    # Subtitle
    st.markdown(
        """
        <p style='text-align: center; font-size: 20px;'>
            <strong>An AI-powered assistant using <strong>Retrieval-Augmented Generation (RAG) to deliver accurate, context-aware answers from your knowledge sources.</strong>
        </p>
        """,
        unsafe_allow_html=True
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.markdown("### 🕘 Previous Interactions")
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])

    if user_input := st.chat_input("Ask your question:"):
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                answer = retrieve_and_generate_answer(user_input, distance_threshold=None)
                st.write(answer)
        st.session_state.chat_history.append({"question": user_input, "answer": answer})

    # Add footer
    add_footer()

# Footer
def add_footer():
    st.markdown(
        """
        <!-- Developed by Toufique Hasan, 2025 -->
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #d3d3d3; /* Light gray background */
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
            color: black; /* Black text */
            border-top: 1px solid #ccc;
            z-index: 999;
        }
        body {
            margin: 0;
            padding-bottom: 50px;
        }
        </style>
        <div class="footer">
            2025 © GPT-Lab
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
