import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document
import pypdf
import io

load_dotenv()

# ── PAGE CONFIG ────────────────────────────────────────
st.set_page_config(
    page_title="Company Knowledge Base",
    page_icon="🏢",
    layout="centered"
)

st.title("🏢 Company Knowledge Base")
st.caption("Upload your company documents and ask questions across all of them")

# ── SESSION STATE ──────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "ready" not in st.session_state:
    st.session_state.ready = False
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []

# ── FILE READER ────────────────────────────────────────
def read_file(uploaded_file) -> str:
    if uploaded_file.type == "application/pdf":
        pdf_reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    else:
        return uploaded_file.read().decode("utf-8")

# ── BUILD VECTORSTORE ──────────────────────────────────
def build_vectorstore(uploaded_files):
    all_chunks = []
    doc_names = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    for file in uploaded_files:
        text = read_file(file)
        doc = Document(
            page_content=text,
            metadata={"source": file.name}
        )
        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)
        doc_names.append(file.name)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name="knowledge_base"
    )

    return vectorstore, doc_names

# ── ASK QUESTION ───────────────────────────────────────
def ask_question(question: str):
    vectorstore = st.session_state.vectorstore
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)

    context = "\n\n".join([
        f"[From: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
        for doc in docs
    ])

    sources = list(set([doc.metadata.get('source', 'Unknown') for doc in docs]))

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = f"""You are a helpful company knowledge assistant.
Answer the question using ONLY the provided document context below.
Always mention which document your answer comes from.
If the answer is not in the documents, say "I could not find this information in the uploaded documents."

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return response.content, sources

# ── UPLOAD UI ──────────────────────────────────────────
if not st.session_state.ready:
    st.markdown("### Upload your documents")
    st.caption("Supports PDF and TXT — upload as many as you need")

    uploaded_files = st.file_uploader(
        "Upload company documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="doc_upload"
    )

    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) selected: {', '.join([f.name for f in uploaded_files])}")

        if st.button("Build Knowledge Base"):
            with st.spinner(f"Processing {len(uploaded_files)} document(s)..."):
                vectorstore, doc_names = build_vectorstore(uploaded_files)
                st.session_state.vectorstore = vectorstore
                st.session_state.doc_names = doc_names
                st.session_state.ready = True
            st.rerun()
    else:
        st.info("Upload one or more documents to get started")

# ── CHAT UI ────────────────────────────────────────────
if st.session_state.ready:
    st.success(f"Knowledge base ready — {len(st.session_state.doc_names)} document(s) loaded")

    with st.expander("Loaded documents"):
        for name in st.session_state.doc_names:
            st.write(f"- {name}")

    if st.button("Upload new documents"):
        st.session_state.ready = False
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.session_state.doc_names = []
        st.rerun()

    # Suggested questions
    st.markdown("### Suggested questions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("What topics are covered?"):
            st.session_state.chat_history.append({
                "role": "user",
                "content": "What topics and subjects are covered across all the uploaded documents?"
            })
    with col2:
        if st.button("Summarize all documents"):
            st.session_state.chat_history.append({
                "role": "user",
                "content": "Give me a brief summary of each uploaded document."
            })

    col3, col4 = st.columns(2)
    with col3:
        if st.button("What are the key policies?"):
            st.session_state.chat_history.append({
                "role": "user",
                "content": "What are the key policies or rules mentioned across the documents?"
            })
    with col4:
        if st.button("What are action items?"):
            st.session_state.chat_history.append({
                "role": "user",
                "content": "What are the main action items or next steps mentioned in the documents?"
            })

    # Chat input
    st.markdown("### Chat")
    user_input = st.chat_input("Ask anything about your documents...")
    if user_input:
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

    # Process last question
    if st.session_state.chat_history:
        last = st.session_state.chat_history[-1]
        if last["role"] == "user":
            with st.spinner("Searching documents..."):
                answer, sources = ask_question(last["content"])
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                st.caption(f"Sources: {', '.join(message['sources'])}")

# ── FOOTER ─────────────────────────────────────────────
st.divider()
st.caption("Built with LangChain · ChromaDB · Groq · Streamlit")