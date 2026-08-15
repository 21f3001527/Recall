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
├── evaluation/
│   ├── evaluate_chat.py         # Runs the chat pipeline against a test set
│   ├── ragas_eval_chat.py       # Scores results with RAGAS (LLM-as-judge)
│   ├── analyze_chat_results.py  # Quick, free retrieval sanity check
│   ├── dataset/
│   │   └── chat_eval.json       # 20-question chat benchmark
│   └── results/                 # Generated results (see Evaluation section)
└── sample_docs/
    └── Numpy Notes.pdf        # Sample document used for RAG evaluation
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
uv run streamlit run app.py
```

Open `http://localhost:8501`, upload a PDF from the sidebar, and start studying.

## 📦 How It Works

Upload PDF → chunk → embed (HuggingFace) → store in ChromaDB → retrieve relevant chunks per request → generate summaries/quizzes/flashcards/chat answers with Groq.

## 📊 Evaluation

RAG pipelines can *look* correct while silently hallucinating or missing relevant context. To catch this, the **Chat** pipeline is evaluated end-to-end using [RAGAS](https://github.com/explodinggradients/ragas), scoring retrieval quality and generation quality separately rather than relying on a single "looks right" judgment.

> **Scope:** The current automated evaluation covers the **Chat/RAG pipeline only**. Summarize, Quiz, and Flashcards do not yet have automated evaluation harnesses.

### Method

- A 20-question benchmark (`evaluation/dataset/chat_eval.json`) was hand-written against a sample document (NumPy reference notes).
- `evaluate_chat.py` runs each question through the actual production chat chain (retriever + Groq `llama-3.3-70b-versatile`), capturing the real retrieved contexts and generated answers.
- `ragas_eval_chat.py` scores each response using an LLM-as-judge (Groq `llama-3.1-8b-instant`) across four metrics — two on retrieval, two on generation:

| Metric | Measures |
|---|---|
| Context Precision | Are retrieved chunks relevant, not noisy? |
| Context Recall | Does retrieval surface everything needed to answer? |
| Faithfulness | Is the answer grounded in retrieved context (low hallucination)? |
| Answer Relevancy | Does the answer actually address the question? |

### Results

Two identical runs (`RETRIEVAL_K=4`) were scored to check stability, and one run with `RETRIEVAL_K=6` was tested as a tuning experiment:

| Metric | K=4 (run 1) | K=4 (run 2) | K=6 |
|---|---:|---:|---:|
| Faithfulness | 0.94 | 0.87 | 0.91 |
| Answer Relevancy | 0.87 | 0.81 | 0.84 |
| Context Precision | 0.93 | 0.92 | 0.88 |
| Context Recall | 0.91 | 0.91 | 0.88 |

Retrieval baseline (`analyze_chat_results.py`, no LLM calls): **100% of questions retrieved at least one context chunk**, averaging 3.15 chunks/question.

### Findings

- **K=4 outperformed K=6 across all four reported metrics in this experiment.** Increasing the number of retrieved chunks did not improve context recall and reduced context precision, suggesting that the additional chunks introduced more irrelevant context. `RETRIEVAL_K` was therefore kept at **4**.
- **Faithfulness and Answer Relevancy show real run-to-run variance** (0.87–0.94 and 0.81–0.87 across two identical K=4 runs), while Context Precision and Recall stayed stable (~0.91–0.93). This reflects inherent non-determinism in LLM-as-judge scoring, not a change in the underlying system — worth reporting as a range rather than a single number.
- The lowest-scoring individual questions pointed to two concrete, fixable issues: one case where the LLM claimed information wasn't in the document despite it being present in the retrieved context (a generation issue), and a couple of cases where relevant chunks weren't retrieved at all for comparison-style questions (a retrieval-coverage gap). Neither was large enough to justify a config change on its own, but both are documented for future tuning.

### Reproducing this evaluation

```bash
uv run python -m evaluation.evaluate_chat        # regenerate chat_results.json
uv run python -m evaluation.analyze_chat_results # free retrieval sanity check
uv run python -m evaluation.ragas_eval_chat      # full RAGAS scoring (LLM-judged, can take 30-90 min on Groq's free tier)
```

### Planned

- Evaluation harnesses for **Quiz** (factual accuracy + distractor quality), **Flashcards** (faithfulness of front/back pairs), and **Summarize** (faithfulness + topic coverage), following the same LLM-as-judge pattern used for Chat.

### Sample Document

The `sample_docs/` directory contains the PDF used for testing and evaluating the RAG pipeline. The current evaluation uses `Numpy Notes.pdf` as the knowledge source for the Chat benchmark.

## 💾 Data

Everything persists in `data/` (vector store + SQLite). Delete this folder to fully reset the app.

## ⚠️ Troubleshooting

| Issue | Solution |
|---|---|
| `GROQ_API_KEY` not found | Check `.env` exists in the project root with a valid key. |
| Slow first run | The embedding model downloads and caches on first use. |
| Empty quiz/flashcards | Model returned malformed output — retry generation. |
| Slow install | PyTorch + sentence-transformers can take a few minutes. |
| Groq `RateLimitError` (429) during evaluation | Free-tier daily/per-minute token limits — wait for reset, or use a separate API key for evaluation runs. |

## 📄 License

For educational purposes.