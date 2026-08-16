# 🎓 Study Assistant

Turn PDF notes into summaries, quizzes, flashcards, and a history-aware chat assistant.

Built with **LangChain**, **ChromaDB**, **Groq**, **HuggingFace Embeddings**, and **Streamlit**.

---

## ✨ Features

- **📝 Summarize** — map-reduce summaries for long documents.
- **🧠 Quiz** — auto-generated MCQ quizzes with score history in SQLite.
- **🗂️ Flashcards** — auto-generated Q/A flashcards scheduled with SM-2 spaced repetition.
- **💬 Chat** — history-aware RAG chat, grounded in the document, with source page references.
- **⚡ Persistent vectors** — documents embedded once and stored in ChromaDB; re-uploads reuse the index.
- **💾 Persistent study data** — quiz history and flashcard scheduling stored in SQLite.
- **📊 Evaluation framework** — automated evaluation for both the Chat/RAG pipeline and the Flashcard pipeline.

---

## 🛠 Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| LLM Framework | LangChain |
| LLM Provider | Groq |
| Embeddings | HuggingFace Embeddings |
| Vector Database | ChromaDB |
| Database | SQLite |
| UI | Streamlit |
| Evaluation | RAGAS, LLM-as-Judge |
| Package Management | uv |

---

## 📂 Project Structure

```text
study_assistant/
├── app.py
├── config.py
├── backend/
│   ├── chat.py
│   ├── db.py
│   ├── document_loader.py
│   ├── flashcards.py
│   ├── models.py
│   ├── quiz.py
│   ├── summarizer.py
│   └── vectorstore.py
├── data/
│   ├── chroma_db/
│   └── study_assistant.db
├── evaluation/
│   ├── analyze_chat_results.py
│   ├── analyze_flashcards_results.py
│   ├── evaluate_chat.py
│   ├── evaluate_flashcards.py
│   ├── judge_flashcards.py
│   ├── ragas_eval_chat.py
│   ├── dataset/
│   │   ├── chat_eval.json
│   │   └── flashcards_eval.json
│   └── results/
│       ├── chat_results.json
│       ├── chat_ragas_results.json
│       ├── chat_ragas_summary.json
│       ├── flashcards_results.json
│       ├── flashcards_judge_results.json
│       └── flashcards_judge_summary.json
└── sample_docs/
    └── Numpy Notes.pdf
```

---

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone <repository-url>
cd Recall
```

**2. Create and activate a virtual environment** (using `uv`)
```bash
# Windows
uv venv
.venv\Scripts\activate

# Linux / macOS
uv venv
source .venv/bin/activate
```

**3. Install dependencies**
```bash
uv sync
```

**4. Configure Groq API key** — create a `.env` file in the project root:
```
GROQ_API_KEY=your_api_key
```

**5. Run the app**
```bash
uv run streamlit run app.py
```
Open `http://localhost:8501`, upload a PDF, and start studying.

---

## 📦 How It Works

```
PDF Upload → PDF Loading → Chunking → HuggingFace Embeddings → ChromaDB
       ├── Chat        → Groq LLM (+ retrieval) → Grounded answer
       ├── Summarize   → Groq LLM
       ├── Quiz        → Groq LLM
       └── Flashcards  → Groq LLM → SM-2 scheduling
```

**Chat / RAG:** user question → history-aware query → ChromaDB retriever → top-K chunks → Groq LLM → grounded answer with source pages.

---

## 📊 Evaluation

Two separate evaluation pipelines exist, each matched to its component's failure modes. Evaluations run against the **actual production functions**, not duplicated logic.

### 💬 Chat / RAG Evaluation
- **Dataset:** `evaluation/dataset/chat_eval.json` — 20 questions on the sample NumPy document.
- **Pipeline:** `evaluate_chat.py` → `chat_results.json` → `analyze_chat_results.py` (free sanity checks) + `ragas_eval_chat.py` (RAGAS metrics).
- **Metrics (RAGAS):** Context Precision, Context Recall, Faithfulness, Answer Relevancy.

**Results:**

| Metric | K=4 Run 1 | K=4 Run 2 | K=6 |
|---|---|---|---|
| Faithfulness | 0.94 | 0.87 | 0.91 |
| Answer Relevancy | 0.87 | 0.81 | 0.84 |
| Context Precision | 0.93 | 0.92 | 0.88 |
| Context Recall | 0.91 | 0.91 | 0.88 |

- Free sanity check: 100% of questions retrieved ≥1 chunk; avg. 3.15 chunks/question.
- **K=4 outperformed K=6** — more chunks added noise without improving recall, so `RETRIEVAL_K=4` was kept.
- LLM-as-judge scores show run-to-run variance (Faithfulness 0.87–0.94, Relevancy 0.81–0.87); Context Precision/Recall were more stable.
- Identified issues: one case where the model claimed missing info despite it being retrieved, and a few comparison-style questions with poor retrieval — logged as future work.

**Commands:**
```bash
uv run python -m evaluation.evaluate_chat        # regenerate chat_results.json
uv run python -m evaluation.analyze_chat_results # free retrieval sanity check
uv run python -m evaluation.ragas_eval_chat      # full RAGAS scoring (LLM-judged, can take 30-90 min on Groq's free tier)
```

### 🗂️ Flashcard Evaluation
Flashcards lack a fixed question/retrieval structure, so RAG metrics don't apply. Instead:
- **Free structural sanity checks**
- **LLM-as-judge quality scoring**
- **Topic coverage analysis**

**Pipeline:** `evaluate_flashcards.py` → `flashcards_results.json` → `analyze_flashcards_results.py` (sanity checks) + `judge_flashcards.py` (LLM judge + topic coverage) → `flashcards_judge_results.json` / `flashcards_judge_summary.json`.

**Dataset:** `evaluation/dataset/flashcards_eval.json` — topic checklist based on `Numpy Notes.pdf` (ndarray basics, indexing/slicing, broadcasting, reshaping, aggregation, vectorized ops, boolean/fancy indexing, linear algebra, random generation, stacking/splitting).

**Step 1 — Generation** (runs production `generate_flashcards()`):
```bash
uv run python -m evaluation.evaluate_flashcards
uv run python -m evaluation.evaluate_flashcards --num-cards 20
```

**Step 2 — Free sanity checks** (card count, timing, empty Q/A, length, near-duplicates via `SequenceMatcher`, threshold 0.85):
```bash
uv run python -m evaluation.analyze_flashcards_results
```

**Step 3 — LLM-as-judge** (1–5 scale on Faithfulness, Clarity, Correctness, Appropriate Scope, plus a topic coverage check):
```bash
uv run python -m evaluation.judge_flashcards
```

**Resumable:** progress is checkpointed. If Groq rate-limits or errors mid-run, completed cards are saved and skipped on the next run — just re-run the same command.

**Current result (15/15 cards judged):**

| Metric | Score |
|---|---|
| Faithfulness | 5.0 / 5 |
| Clarity | 5.0 / 5 |
| Correctness | 5.0 / 5 |
| Appropriate Scope | 5.0 / 5 |

Topic coverage: **10/10 (100%)**

> Interpret these as the outcome of the current LLM-as-judge run, not absolute ground truth.

### 📌 Evaluation Philosophy
- **Chat/RAG:** retrieval evaluation + generation evaluation via RAGAS.
- **Flashcards:** structural sanity checks + LLM-as-judge + topic coverage.

Each component is evaluated with metrics suited to its actual failure modes rather than one-size-fits-all metrics.

---

## 💾 Data Persistence

```
data/
├── chroma_db/        # document embeddings (ChromaDB)
└── study_assistant.db  # quiz history, flashcard scheduling (SQLite)
```
Delete `data/` to fully reset the app; both stores are recreated automatically.

## 🧠 Flashcard Spaced Repetition
Flashcards use the **SM-2** algorithm to track review performance and schedule future reviews, turning generated cards into an ongoing study workflow.

## 🔐 Configuration
Centralized in `config.py`. Store secrets (e.g., `GROQ_API_KEY`) in `.env`, not in Git.

---

## 🐛 Troubleshooting

| Issue | Solution |
|---|---|
| `GROQ_API_KEY not found` | Check `.env` exists in the project root with a valid key. |
| Slow first run | HuggingFace embedding models download/cache on first use. |
| Empty quiz/flashcards | LLM returned malformed output — retry and check logs. |
| Slow installation | PyTorch / sentence-transformers can take a few minutes. |
| Groq 429 during evaluation | Wait for the rate limit to reset and rerun; flashcard eval resumes automatically. |
| Flashcard judge stops partway | Re-run `judge_flashcards`; completed cards are skipped. |
| Poor retrieval results | Run `analyze_chat_results.py` and inspect retrieved contexts before tuning retrieval params. |

---

## 🚧 Future Improvements
- Quiz factual-accuracy and distractor-quality evaluation
- Summarization faithfulness & topic coverage evaluation
- Semantic (vs. keyword) topic coverage
- Human vs. LLM-as-judge comparison
- Multi-document evaluation
- Statistical confidence intervals across repeated judge runs
- Cost/latency tracking
- Automated regression testing in CI/CD

## 📄 Sample Document
`sample_docs/Numpy Notes.pdf` — used as the source for both the Chat benchmark and Flashcard evaluation.

## 📄 License
For educational purposes.