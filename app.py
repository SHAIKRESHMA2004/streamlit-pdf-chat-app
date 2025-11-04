import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import os, json
import numpy as np
from typing import List, Dict

# -------------------------
# Load environment
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in .env. Add it and restart.")
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# Helper functions for login
# -------------------------
USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def register_user(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = password
    save_users(users)
    return True

def check_login(username, password):
    users = load_users()
    return users.get(username) == password

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="📚 RAG PDF Chatbot", page_icon="🤖", layout="wide")

# -------------------------
# Authentication page
# -------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

def login_page():
    st.title("🔐 Login to RAG PDF Chatbot")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if check_login(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome {username}!")
            st.rerun()
        else:
            st.error("Invalid username or password.")
    st.markdown("Don't have an account?")
    if st.button("Go to Register"):
        st.session_state.show_register = True
        st.rerun()

def register_page():
    st.title("📝 Register for RAG PDF Chatbot")
    username = st.text_input("Create Username")
    password = st.text_input("Create Password", type="password")
    if st.button("Register"):
        if username.strip() == "" or password.strip() == "":
            st.warning("Please enter both username and password.")
        elif register_user(username, password):
            st.success("Registration successful! Please log in.")
            st.session_state.show_register = False
            st.rerun()
        else:
            st.error("Username already exists.")
    if st.button("Back to Login"):
        st.session_state.show_register = False
        st.rerun()

# -------------------------
# If not logged in, show login/register
# -------------------------
if not st.session_state.logged_in:
    if "show_register" in st.session_state and st.session_state.show_register:
        register_page()
    else:
        login_page()
    st.stop()

# -------------------------
# After login — Main Chatbot Page
# -------------------------
st.markdown(f"<h1 style='text-align:center'>📚 RAG PDF Chatbot — Welcome {st.session_state.username}</h1>", unsafe_allow_html=True)
st.markdown("Upload PDFs, let the app process them, then ask any question related to the uploaded documents. Answers are grounded in retrieved document chunks. ✨")

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

# -------------------------
# Utility functions (RAG)
# -------------------------
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not text: return []
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return chunks

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def embed_texts(texts: List[str], model="text-embedding-3-small"):
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]

def retrieve_top_k(query: str, k=4):
    if "embeddings" not in st.session_state or len(st.session_state["embeddings"]) == 0:
        return []
    q_emb = embed_texts([query])[0]
    sims = []
    for idx, emb in enumerate(st.session_state["embeddings"]):
        sim = cosine_similarity(np.array(q_emb), np.array(emb))
        sims.append((idx, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in sims[:k]:
        meta = st.session_state["metadatas"][idx]
        text = st.session_state["chunks"][idx]
        results.append({"score": float(score), "text": text, "meta": meta})
    return results

def build_prompt(question: str, retrieved: List[Dict], chat_history: List[Dict]) -> List[Dict]:
    system_msg = (
        "You are an expert assistant. Answer the user's question using ONLY the provided document excerpts. "
        "If the answer is not contained in the sources, say you don't know."
    )
    context_texts = []
    for i, r in enumerate(retrieved):
        src = r["meta"].get("source", "unknown")
        page = r["meta"].get("page", "")
        header = f"Source {i+1} (file: {src} page:{page}):"
        context_texts.append(header + "\n" + r["text"])
    context_combined = "\n\n---\n\n".join(context_texts)
    user_intro = (
        f"Here are the relevant document excerpts:\n\n{context_combined}\n\n"
        f"Question: {question}\n\nPlease answer concisely and cite which source(s) you used."
    )
    messages = [{"role": "system", "content": system_msg}]
    for m in chat_history[-6:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_intro})
    return messages

def ask_openai_chat(messages: List[Dict], model="gpt-3.5-turbo", temperature=0.1) -> str:
    response = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    return response.choices[0].message.content

# -------------------------
# Sidebar - Upload & Controls
# -------------------------
with st.sidebar:
    st.header("📁 Document Controls")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    process_btn = st.button("🧠 Process PDFs")
    top_k = st.slider("Retrieve top K chunks", 1, 8, 4)
    chunk_size = st.number_input("Chunk size", 300, 4000, 1000, 100)
    overlap = st.number_input("Chunk overlap", 0, 1000, 200, 50)
    if st.button("🧹 Clear Data"):
        for k in ["documents", "chunks", "metadatas", "embeddings", "chat_history"]:
            st.session_state.pop(k, None)
        st.success("Data cleared.")

# -------------------------
# Initialize states
# -------------------------
for key in ["documents", "chunks", "metadatas", "embeddings", "chat_history"]:
    st.session_state.setdefault(key, [])

# -------------------------
# Process uploads
# -------------------------
if process_btn:
    if not uploaded_files:
        st.warning("Please upload at least one PDF file first.")
    else:
        st.info("Processing PDFs...")
        for up in uploaded_files:
            reader = PdfReader(up)
            full_text = ""
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                full_text += f"\n\n[[[PAGE {i+1}]]]\n" + page_text
            st.session_state["documents"].append({"filename": up.name, "text": full_text})
            pages = full_text.split("\n\n[[[PAGE ")
            for p in pages:
                if p.strip():
                    if "]]]" in p:
                        try:
                            page_no_str, page_body = p.split("]]]", 1)
                            page_no = page_no_str.strip()
                            page_body = page_body.strip()
                        except Exception:
                            page_no, page_body = "", p
                    else:
                        page_no, page_body = "", p
                    for c in chunk_text(page_body, int(chunk_size), int(overlap)):
                        if c.strip():
                            st.session_state["chunks"].append(c)
                            st.session_state["metadatas"].append({"source": up.name, "page": page_no})
        st.success("PDFs processed. Creating embeddings...")
        embeddings_all = []
        for i in range(0, len(st.session_state["chunks"]), 50):
            batch = st.session_state["chunks"][i:i+50]
            embs = embed_texts(batch)
            embeddings_all.extend(embs)
        st.session_state["embeddings"] = embeddings_all
        st.success("Embeddings created!")

# -------------------------
# Chat Section
# -------------------------
st.subheader("💬 Chat with your documents")
user_question = st.text_input("🧠 Ask a question:")
if st.button("🚀 Ask"):
    if not user_question.strip():
        st.warning("Please write a question.")
    elif not st.session_state["embeddings"]:
        st.warning("No processed documents yet.")
    else:
        retrieved = retrieve_top_k(user_question, k=int(top_k))
        messages = build_prompt(user_question, retrieved, st.session_state["chat_history"])
        try:
            answer = ask_openai_chat(messages)
        except Exception as e:
            answer = f"OpenAI API error: {e}"
        st.session_state["chat_history"].append({"role": "user", "content": user_question})
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
        st.markdown("**🤖 Answer:**")
        st.markdown(answer)

st.markdown("---")
st.markdown("**Conversation**")
for msg in st.session_state["chat_history"]:
    if msg["role"] == "user":
        st.markdown(f"🧑‍💻 **You:** {msg['content']}")
    else:
        st.markdown(f"🤖 **AI:** {msg['content']}")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by SK.RESHMA — uses OpenAI for embeddings & chat.")
