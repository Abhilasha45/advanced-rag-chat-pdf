# 📄 Advanced RAG Chat with PDF Assistant

An AI-powered Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and interact with them through a conversational interface. The system retrieves relevant information from uploaded PDFs using semantic search and generates context-aware responses using Google Gemini.

## ✨ Features

- Upload and chat with PDF documents
- AI-powered question answering
- PDF summarization and short notes generation
- Semantic search using FAISS vector database
- Context-aware responses using Google Gemini
- Page-wise source citations
- Conversation history
- Interactive Streamlit interface
- Robust API error handling

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini API
- FAISS
- Hugging Face Embeddings (all-MiniLM-L6-v2)
- PyMuPDF
- Git & GitHub

## 📂 Project Structure

```
advanced-rag-chat/
│── app.py
│── requirements.txt
│── README.md
│── config/
│── prompts/
│── src/
│── data/
└── logs/
```

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/advanced-rag-chat-pdf.git
```

Move into the project directory:

```bash
cd advanced-rag-chat-pdf
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file and add your Google Gemini API key.

Run the application:

```bash
streamlit run app.py
```

## 📌 Future Enhancements

- Premium UI redesign
- FAISS index caching
- Confidence score for responses
- Enhanced source citations
- Multi-document comparison
- Performance optimization

## 👩‍💻 Author

**Abhilasha Mishra**

B.Tech in Computer Science & Engineering (Artificial Intelligence)