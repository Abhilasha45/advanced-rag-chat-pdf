from src.retrieval.context_compressor import compress_context
import time
from xml.dom.minidom import Document
from src.llm.gemini import generate_answer
from src.retrieval.vector_store import create_vector_store, retrieve_chunks
from src.embeddings.embedding_model import create_embeddings
from src.chunking.text_splitter import split_text
from src.parser.pdf_parser import extract_text_from_pdf
from src.utils.file_handler import save_uploaded_files

import streamlit as st

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Advanced RAG Chat",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>

div.stButton > button {
    width:100%;
    background:transparent;
    border:none;
    color:white;
    text-align:left;
    font-size:18px;
    padding:12px 16px;
    border-radius:12px;
    transition:0.2s;
}

div.stButton > button:hover{
    background:#22252e;
    border:1px solid #444;
    cursor:pointer;
}

div.stButton > button:focus{
    outline:none;
    box-shadow:none;
}

</style>
""", unsafe_allow_html=True)


# ----------------------------
# Session State Initialization
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:

    st.title("📂 Documents")
    st.caption("Upload your PDF files to start chatting.")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.divider()

    #st.subheader("Uploaded Files")

    if uploaded_files:

        saved_files = save_uploaded_files(uploaded_files)

        pages = extract_text_from_pdf(saved_files[0])

        chunks = split_text(pages)
        embeddings = create_embeddings(chunks)

        
        #st.success(f"Total Embeddings:( {len(embeddings)} , {len(embeddings[0])})")
        vector_store = create_vector_store(chunks)
        st.session_state.vector_store = vector_store

        #st.success("✅ FAISS Index Created") 

        #st.success(f"Total Chunks: {len(chunks)}")

        #st.write(chunks[0])
        
        #st.success(f"{len(saved_files)} PDF(s) saved successfully!")

        #for file in saved_files:
            #st.write(file.name)
    #else: 
        #st.info("No PDF uploaded.")

# ----------------------------
# Main Page
# ----------------------------
st.title("📄 AI Document Assistant")

st.caption("Chat intelligently with your uploaded PDF documents ")

if not st.session_state.messages:
    st.markdown("### ✨ Try asking")

suggestions = [
    "📄 Summarize this PDF",
    "🧠 Explain this document",
    "💡 What are the main points?",
    "📝 Create short notes",
    "🔍 Simplify difficult concepts",
    "❓ What is this PDF about?"
]

for text in suggestions:
    if st.button(
        text,
        key=f"suggestion_{text}",
        use_container_width=True
    ):
        st.session_state.clicked_question = text
        st.rerun()

st.divider()


# ----------------------------
# Chat History
# ----------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------------
# Chat Input
# ----------------------------
user_question = st.session_state.pop("clicked_question", None)

if uploaded_files:

    typed_question = st.chat_input("Ask anything about your PDFs...")

    if typed_question:
        user_question = typed_question

else:
    st.chat_input(
        "Upload a PDF first...",
        disabled=True
    )


if user_question:

    print("Reached retrieve_chunks()")

    docs = retrieve_chunks(
        st.session_state.vector_store,
        user_question
    )

    print("Finished retrieve_chunks()")

    print("Retrieved", len(docs), "documents")

    for i, doc in enumerate(docs):
        print(f"Chunk {i+1}")
        print(doc.metadata)
    #st.write("Retrieved Chunks:")
    #for i, doc in enumerate(docs):
        #st.write(f"Chunk {i+1}")
        #st.write(doc.page_content)
        #st.divider()

    raw_context = "\n\n".join([doc.page_content for doc in docs])
    
    context =raw_context

    #context = compress_context(raw_context)
    #st.subheader("Compressed Context:")
    #st.write(context)

    print("=" * 60)
    print(context)
    print("=" * 60)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    thinking = st.empty()

    thinking.info("📄 Reading document...")
    time.sleep(0.5)

    thinking.info("🔎 Searching relevant chunks...")
    time.sleep(0.5)

    thinking.info("🧠 Understanding your question...")
    time.sleep(0.5)

    thinking.info("✍️ Generating answer...")

    try:
        answer = generate_answer(
            context,
            user_question,
            st.session_state.messages,
            docs
        )

        answer_generated = True

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
        answer_generated = False
        error_message = str(e).lower()

        if "resource_exhausted" in error_message or "429" in error_message:
            answer = (
                "⚠️ The AI service has reached its current request limit. "
                "Please try again later."
            )

        elif "not_found" in error_message or "404" in error_message:
            answer = (
                "⚠️ The selected AI model is currently unavailable. "
                "Please try again later."
            )

        elif "api_key" in error_message or "authentication" in error_message:
            answer = (
                "⚠️ AI service authentication failed. "
                "Please contact the application administrator."
            )

        else:
            answer = (
                "⚠️ Something went wrong while generating the answer. "
                "Please try again."
            )

    if answer_generated:
        thinking.success("✅ Answer generated!")
    else:
        thinking.error("⚠️ Unable to generate answer.")

    time.sleep(0.3)
    thinking.empty()

    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed = ""

        for char in answer:
            streamed += char
            placeholder.markdown(streamed + "▌")
            time.sleep(0.003)

        placeholder.markdown(streamed)

    st.session_state.messages.append(
    {
        "role": "assistant",
        "content": answer,
        "is_error": not answer_generated
    }
)
# st.subheader("Retrieved Chunks")

# for doc in docs:
#     st.write(doc.page_content)
#     st.divider()