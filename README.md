# 🧠+🤖 From Static to Smart: RAG Agents in Action

This repository contains a modular implementation of a Retrieval-Augmented Generation (RAG) system. It supports both **Vanilla** and **Agentic** RAG pipelines and can be easily extended or adapted. The system converts PDF documents into plain text, generates embeddings, and enables interactive question answering via Streamlit.

---

## 📁 Project Structure

```
.
├── dataembedding/             # Stores generated embeddings
├── datapdf/                   # Input PDFs
├── datatxt/                   # Extracted plain-text from PDFs
├── 1_pdf_to_txt.py            # PDF → TXT conversion
├── 2_txt_to_embeddings.py     # TXT → Vector embeddings
├── rag_vanilla.py             # Standard RAG pipeline
├── rag_agentic.py             # Agentic RAG pipeline with step-by-step reasoning
├── SampleQA.txt               # Sample medical QA cases
├── requirements.txt           # Project dependencies
├── .env                       # Environment variables (API key)
└── README.md                  # You're reading it!
```

---

## 🔧 Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/GPT-Laboratory/RAG_Agents.git
   cd RAG_Agents
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your OpenAI key to `.env`**
   ```env
   OPENAI_API_KEY=your-api-key-here
   ```

---

## 🧾 Pipeline Overview

### 1️⃣ Convert PDFs to TXT

Use `1_pdf_to_txt.py` to convert PDF documents in `datapdf/` into plain-text files in `datatxt/`.

```bash
python 1_pdf_to_txt.py
```

> ⚠️ Uses `pymupdf`, `opencv-python`, and `easyocr` to handle text + scanned content.

---

### 2️⃣ Generate Embeddings

Run `2_txt_to_embeddings.py` to process the text files into vector embeddings and save them in `dataembedding/`.

```bash
python 2_txt_to_embeddings.py
```

> 📌 Uses `tiktoken`, `faiss-cpu`, and `numpy`.

---

### 3️⃣ Start the QA Interface

#### Vanilla RAG (Simple retrieval + generation)
```bash
streamlit run rag_vanilla.py
```

#### Agentic RAG (Reasoning chain and multi-step answers)
```bash
streamlit run rag_agentic.py
```

Both scripts launch a user interface where you can ask domain-specific questions based on your embedded documents.

---

## 🧪 Sample Use Case

The `SampleQA.txt` includes clinical decision-making prompts for aneurysm treatment. You can paste these questions into the Streamlit interface after running either RAG mode.

Example:
```
40-year old female patient as a 10mm unruptured multilobular Acom aneurysm...
```

---

## 🧩 Dependencies

Here are the core libraries used:

- `openai`
- `faiss-cpu`
- `tiktoken`
- `streamlit`
- `pymupdf`, `easyocr`, `opencv-python`
- `python-dotenv`, `tqdm`, `numpy`

---

## 🚀 Features

- End-to-end RAG pipeline from PDF ingestion to QA.
- Vanilla vs. Agentic modes.
- Customizable for domain-specific corpora.
- Integration-ready with other LLM backends.

---

## 🧠 Future Improvements

- Add support for document-level permissions.
- Integrate with local LLMs.
- Organize chunking and indexing strategies for better recall.

---

## 🧠 Themes and Data for Hands On Session

- Link: [https://tuni-my.sharepoint.com/:f:/g/personal/mdtoufique_hasan_tuni_fi/Em5fFIOJH7FApTl8PO-y2JEBzJfD6jAUrQqmCEHaMDx7Yg?e=4UmQpf](https://tinyurl.com/5fwer8wz)

---

## 📬 Contact

For any inquiries or collaborations:

**Toufique Hasan, Doctoral Researcher, GPT-Lab (Tampere University)**  
Email: mdtoufique.hasan@tuni.fi  


För att köra lokalt installer Ollama 
brew install ollama

Installera Modell
ollama pull llama3

pip install sentence-transformers

