# 🎓 Study Assistant

Turn PDF notes into summaries, quizzes, flashcards, and a chat assistant that remembers the conversation.

Built with **LangChain**, **ChromaDB**, **Groq**, and **Streamlit**.

---

## ✨ Features

- **📝 Summarize** — structured summaries via map-reduce, for long documents.
- **🧠 Quiz** — auto-generated multiple-choice quizzes, with score history tracked in SQLite.
- **🗂️ Flashcards** — auto-generated Q/A cards scheduled with SM-2 spaced repetition.
- **💬 Chat** — history-aware RAG; ask follow-ups, get answers grounded in the doc with source page numbers.
- **⚡ Persistent vectors** — documents are embedded once; re-uploading the same PDF reuses the existing ChromaDB index instead of re-embedding.

## 🛠 Tech Stack

Python · LangChain · ChromaDB · HuggingFace Embeddings · Groq · Streamlit · SQLite

## 📂 Project Structure

```text
study_assistant/
├── app.py                     # Streamlit app
├── config.py                  # Central configuration
├── backend/
│   ├── chat.py                # History-aware RAG chat
│   ├── db.py                  # SQLite operations
│   ├── document_loader.py     # PDF loading & chunking
│   ├── flashcards.py          # Flashcard generation + SM-2
│   ├── models.py              # LLM & embedding models
│   ├── quiz.py                # Quiz generation
│   ├── summarizer.py          # Map-reduce summarization
│   └── vectorstore.py         # ChromaDB wrapper
├── data/
│   ├── chroma_db/             # Persistent vector store
│   └── study_assistant.db     # Quiz history & flashcards
└── sample_docs/
```

## 🚀 Getting Started

```bash
git clone <repository-url>
cd study_assistant

uv venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

uv sync
```

Create a `.env` file in the project root with a free key from [console.groq.com/keys](https://console.groq.com/keys):

```text
GROQ_API_KEY=your_api_key
```

Run it:

```bash
streamlit run app.py
```

Open `http://localhost:8501`, upload a PDF from the sidebar, and start studying.

## 📦 How It Works

Upload PDF → chunk → embed (HuggingFace) → store in ChromaDB → retrieve relevant chunks per request → generate summaries/quizzes/flashcards/chat answers with Groq.

## 💾 Data

Everything persists in `data/` (vector store + SQLite). Delete this folder to fully reset the app.

## ⚠️ Troubleshooting

| Issue | Solution |
|---|---|
| `GROQ_API_KEY` not found | Check `.env` exists in the project root with a valid key. |
| Slow first run | The embedding model downloads and caches on first use. |
| Empty quiz/flashcards | Model returned malformed output — retry generation. |
| Slow install | PyTorch + sentence-transformers can take a few minutes. |

## 📄 License

For educational purposes.