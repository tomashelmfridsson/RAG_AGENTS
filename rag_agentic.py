# Agentic RAG pipeline by [Toufique Hasan - 2025]
import os
import openai
import streamlit as st
from dotenv import load_dotenv

# Import vanilla retriever
from rag_vanilla import retrieve_and_generate_answer

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------
# AGENTS
# ---------------------------

def planner_agent(query):
    st.info("🔍 **Planner Agent**: Decomposing your query into sub-questions...")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Break the user's question into 2 to 5 specific sub-questions."},
                {"role": "user", "content": query}
            ],
            temperature=0.3
        )
        subqs = response['choices'][0]['message']['content'].split('\n')
        subqs = [q.strip(" -•1234.").strip() for q in subqs if q.strip()]
        st.success("🧭 Sub-questions:")
        for i, q in enumerate(subqs, 1):
            st.markdown(f"- **Q{i}:** {q}")
        return subqs
    except Exception as e:
        st.error(f"Planner failed: {e}")
        return [query]

def retriever_agent(subq):
    st.info(f"📥 **Retriever Agent**: Fetching context for:\n> _{subq}_")
    try:
        context = retrieve_and_generate_answer(subq)
        success = "No relevant context found." not in context
        if success:
            st.success("📄 Retrieved context (filenames or chunk IDs):")
            st.markdown(f"```text\n{context}\n```")
        else:
            st.warning("⚠️ No relevant context found.")
        return context, success
    except Exception as e:
        st.error(f"Retriever failed: {e}")
        return "No context retrieved.", False

def summarizer_agent(context, subq):
    st.info("📝 **Summarizer Agent**: Condensing retrieved content...")
    if "No relevant context found." in context or "No context retrieved" in context:
        return "No summary — no relevant content retrieved."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Summarize the following context for the question."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {subq}"}
            ],
            temperature=0.3
        )
        summary = response['choices'][0]['message']['content']
        st.success("✏️ Summary:")
        st.markdown(summary)
        return summary
    except Exception as e:
        st.error(f"Summarizer failed: {e}")
        return "No summary available."

def answer_agent(summaries, original_query):
    st.info("🧠 **Answer Agent**: Synthesizing final answer...")
    combined = "\n\n".join([f"Summary {i+1}:\n{s}" for i, s in enumerate(summaries)])
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who uses summaries to answer the original question."},
                {"role": "user", "content": f"Original question:\n{original_query}\n\nSummaries:\n{combined}"}
            ],
            temperature=0.4
        )
        final = response['choices'][0]['message']['content']
        st.success("✅ Final Answer:")
        st.markdown(f"**{final}**")
        return final
    except Exception as e:
        st.error(f"Answer agent failed: {e}")
        return "No final answer available."

def evaluation_agent(final_answer, original_query):
    st.info("🧪 **Evaluation Agent**: Evaluating the quality of the answer...")
    try:
        prompt = f"""
You are an expert evaluation agent tasked with **critical and realistic** scoring of the following AI-generated answer using **strict criteria**.

Rate each of the five dimensions **from 0 to 100**:
1. **Ethical Concern** – Check for ethical issues, bias, or fairness.
2. **Context Alignment** – How well the answer aligns with retrieved context.
3. **Completeness** – Are all sub-questions and aspects covered?
4. **Accuracy** – Are the facts correct and based only on the given summaries?
5. **Consistency** – Is the answer logically structured and coherent?

⚠️ Avoid perfect 100% scores unless **fully justified**. Small errors or ambiguity must reduce the score.

Then provide an **Overall Score** (not a simple average). Justify each score in 1–2 sentences.

**FORMAT** (must follow this strictly):

Ethical Concern: XX% - <justification>  
Context Alignment: XX% - <justification>  
Completeness: XX% - <justification>  
Accuracy: XX% - <justification>  
Consistency: XX% - <justification>  
**Overall Score: XX%**  
Reason: <summary reasoning>

Question:
{original_query}

Answer:
{final_answer}
"""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful and strict evaluation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        evaluation = response['choices'][0]['message']['content']
        st.success("📊 Evaluation Result:")
        st.markdown(evaluation)

        # Extract scores
        import re
        scores = {}
        for crit in ["Ethical Concern", "Context Alignment", "Completeness", "Accuracy", "Consistency"]:
            match = re.search(rf"{crit}:\s*(\d+)%", evaluation)
            scores[crit] = int(match.group(1)) if match else 0

        overall_match = re.search(r"\*\*Overall Score:\s*(\d+)%\*\*", evaluation)
        if overall_match:
            scores["Overall Score"] = int(overall_match.group(1))

        return evaluation
    except Exception as e:
        st.error(f"Evaluation agent failed: {e}")
        return "No evaluation available."


# ---------------------------
# STREAMLIT UI
# ---------------------------

st.set_page_config(page_title="Agentic RAG Assistant", page_icon="🧠", layout="wide")

# Header
st.markdown("""
    <h1 style='text-align: center;'>🧠+🤖 Agentic RAG Assistant</h1>
    <p style='text-align: center; font-size: 20px;'>
        <strong>An advanced Retrieval-Augmented Generation (RAG) system with autonomous agents.</strong>
    </p>
""", unsafe_allow_html=True)

st.markdown(
    "This app demonstrates an advanced RAG system using multiple agents:\n\n"
    "- **Planner** → breaks complex query into sub-questions\n"
    "- **Retriever** → gets relevant chunks for each sub-question\n"
    "- **Summarizer** → distills key points\n"
    "- **Answerer** → synthesizes a final answer\n"
    "- **Evaluator** → reviews and scores the final answer"
)

# Initialize session state for history
if "agentic_history" not in st.session_state:
    st.session_state.agentic_history = []

# Display history
st.markdown("### 🕘 Previous Interactions")
for i, item in enumerate(st.session_state.agentic_history[::-1], 1):
    with st.expander(f"🗂️ Query {len(st.session_state.agentic_history) - i + 1}: {item['query'][:80]}...", expanded=False):
        st.markdown(f"**❓ Question:** {item['query']}")
        st.markdown(f"**✅ Answer:**\n\n{item['answer']}")
        if 'evaluation' in item:
            st.markdown(f"**🧪 Evaluation:**\n\n{item['evaluation']}")

# Input
query = st.text_area(
    "💬 Enter your complex question:",
    placeholder="e.g., What are the key insights from the documents regarding recent developments?",
    height=100
)

# Process
if st.button("Run Agentic RAG"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        subqs = planner_agent(query)
        summaries = []
        retrieval_success_flags = []

        for i, subq in enumerate(subqs, 1):
            with st.expander(f"🔎 Q{i}: {subq}", expanded=True):
                context, success = retriever_agent(subq)
                retrieval_success_flags.append(success)
                if success:
                    summary = summarizer_agent(context, subq)
                else:
                    summary = "No summary — no relevant content retrieved."
                summaries.append(summary)

        if not any(retrieval_success_flags):
            st.error("🚫 No relevant context found for any sub-question. Cannot generate answer.")
            st.session_state.agentic_history.append({
                "query": query,
                "answer": "No answer generated — all retrievals failed.",
                "evaluation": "Evaluation skipped due to lack of content."
            })
        else:
            final_answer = answer_agent(summaries, query)
            evaluation = evaluation_agent(final_answer, query)

            st.session_state.agentic_history.append({
                "query": query,
                "answer": final_answer,
                "evaluation": evaluation
            })

# Footer
st.markdown(
    """
    <!-- Developed by Toufique Hasan, 2025 -->
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
        2025 © GPT-Lab
    </div>
    """,
    unsafe_allow_html=True,
)
