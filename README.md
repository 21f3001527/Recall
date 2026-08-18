<div align="center">

# 🎓 Study Assistant

**An AI-powered study workspace that turns PDF notes into summaries, quizzes, flashcards, and a history-aware RAG chat assistant.**

Built with **LangChain, ChromaDB, Groq, HuggingFace Embeddings, Streamlit, and RAGAS**.

<br>

[![CI](https://img.shields.io/github/actions/workflow/status/21f3001527/Recall/evaluation-ci.yml?style=for-the-badge&logo=githubactions&logoColor=white&label=CI&color=2EA44F&labelColor=181717)](https://github.com/21f3001527/Recall/actions/workflows/evaluation-ci.yml)
[![Stars](https://img.shields.io/github/stars/21f3001527/Recall?style=for-the-badge&logo=github&logoColor=white&color=FFD21E&labelColor=181717)](https://github.com/21f3001527/Recall/stargazers)
[![Last Commit](https://img.shields.io/github/last-commit/21f3001527/Recall?style=for-the-badge&logo=git&logoColor=white&color=F05032&labelColor=181717)](https://github.com/21f3001527/Recall/commits)
![License](https://img.shields.io/badge/License-Educational-6B7280?style=for-the-badge&labelColor=181717)

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Orchestration-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Inference-F55036?style=for-the-badge&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-7C3AED?style=for-the-badge&logoColor=white)
![uv](https://img.shields.io/badge/uv-Package%20Manager-DE5FE9?style=for-the-badge&logoColor=white)

</div>

---

## ✨ Features

- 📝 **Summarization** — generates structured summaries from long PDF notes
- 💬 **RAG Chat** — ask questions about your documents, with source-page references
- 🧠 **Quiz Generation** — auto-generated MCQs with persistent score history
- 🗂️ **Flashcards** — Q&A cards with **SM-2 spaced repetition**
- ⚡ **Persistent Vector Store** — ChromaDB avoids re-embedding previously processed documents
- 💾 **Persistent Study Data** — SQLite stores quiz history and flashcard schedules
- 📊 **LLM Evaluation** — RAGAS + LLM-as-Judge across Chat, Quiz, Flashcards, and Summary
- 🔄 **CI Regression Tests** — GitHub Actions automatically validates committed evaluation results

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python |
| LLM Orchestration | LangChain |
| LLM Inference | Groq |
| Embeddings | HuggingFace |
| Vector Store | ChromaDB |
| Database | SQLite |
| UI | Streamlit |
| Evaluation | RAGAS, LLM-as-Judge |
| Package Management | uv |
| CI/CD | GitHub Actions |

---

## 🖥️ Application

Upload a PDF and use the same document across Chat, Summary, Quiz, and Flashcards.

### 💬 RAG Chat
History-aware conversations grounded in the uploaded document, with source-page references.

![RAG Chat](assets/study_assistant_chat.png)

### 🧠 Quiz
Automatically generated MCQs with persistent score history across attempts.

![Quiz](assets/study_assistant_quiz.png)

### 🗂️ Flashcards
Generated Q&A flashcards with SM-2 spaced-repetition scheduling.

![Flashcards](assets/study_assistant_flashcards.png)

### 📝 Summary
Structured, source-grounded summaries generated from the uploaded PDF.

![Summary](assets/study_assistant_summary.png)

---

## 🏗️ How It Works

```text
PDF → Loader → Chunking → HuggingFace Embeddings → ChromaDB
                                                        │
                              ┌─────────────┬───────────┼───────────┐
                              ▼             ▼           ▼           ▼
                          RAG Chat      Summary       Quiz     Flashcards
                              │                          └──────┬──────┘
                              ▼                                 ▼
                          Groq LLM                      SQLite + SM-2
```

**RAG Chat flow:** conversation history is used to reformulate follow-up questions → the ChromaDB retriever selects the top-K relevant chunks → the Groq LLM generates a grounded response with source-page references.

---

## 📊 Evaluation

Each component is evaluated according to its failure mode using deterministic checks and LLM-based evaluation.

| Component | Evaluation | Current Results |
|---|---|---|
| 💬 Chat / RAG | Retrieval sanity checks + RAGAS | 100% questions retrieved context (sanity check) · Faithfulness 0.87–0.94 · Context Recall 0.91 (RAGAS) |
| 🗂️ Flashcards | Structural checks + LLM-as-Judge | 5.0/5 across all metrics · 100% topic coverage |
| 🧠 Quiz | Structural checks + LLM-as-Judge | 4.0–5.0/5 · 70% topic coverage · 22 unique questions |
| 📝 Summary | Structural checks + LLM-as-Judge | Faithfulness 5.0/5 · Coverage 5.0/5 · ~9.4% compression ratio |

### Key Findings

- Deterministic checks caught near-duplicate quiz options that the LLM judge rated highly.
- Keyword-based topic matching produced a false negative (`reshape` vs `reshaping`).
- Increasing retrieval from K=4 to K=6 did not improve recall, so K=4 was retained.
- Quiz generation reached 22 unique questions because duplicate filtering rejected repeated content from the small source document.

Detailed results are stored as JSON under `evaluation/results/` — the numbers above are the summary, the JSON is the underlying evidence. Quiz/flashcard judging is resumable via checkpointing.

### 📄 Sample Document

The evaluation benchmark uses `sample_docs/Numpy Notes.pdf`, shared across the Chat, Flashcard, Quiz, and Summary evaluation pipelines so results are comparable across components.

---

## 🔄 CI / Regression Testing

Evaluation results are protected by GitHub Actions using two tiers.

### 1. Sanity Checks — Every Push / PR

- Runs automatically on pushes to `main` and pull requests targeting `main`
- No LLM/API calls
- Validates committed evaluation JSON files
- Checks structural integrity of generated outputs
- Fast and free

### 2. Full Evaluation — Manual / Weekly

- Triggered manually (`workflow_dispatch`) or every Monday via cron
- Uses `GROQ_API_KEY` stored in GitHub Secrets
- Regenerates Chat, Flashcard, Quiz, and Summary evaluations
- Runs RAGAS and LLM-as-Judge pipelines
- Uploads refreshed results as GitHub Actions artifacts
- Does not automatically commit regenerated results

![GitHub Actions](assets/github-actions-evaluatio.png)

---

## 📂 Project Structure

```text
Recall/
├── app.py
├── config.py
├── backend/                    # chat, quiz, flashcards, summarizer, vectorstore, db
├── evaluation/
│   ├── dataset/                 # eval question sets
│   ├── results/                 # generated JSON evaluation output
│   ├── evaluate_*.py            # generation pipelines
│   ├── analyze_*_results.py     # deterministic checks
│   ├── judge_*.py               # LLM-as-Judge
│   └── ragas_eval_chat.py
├── assets/                      # screenshots
├── data/
│   ├── chroma_db/                # embeddings
│   └── study_assistant.db        # quiz + flashcard history
├── sample_docs/
│   └── Numpy Notes.pdf
├── .github/
│   └── workflows/
│       └── evaluation-ci.yml
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 🚀 Getting Started & Commands

All commands, in the order you'd typically run them.

### Setup

```bash
git clone <repository-url>
cd Recall
uv sync

# create .env in project root
echo "GROQ_API_KEY=your_api_key" > .env
```

### Run the App

```bash
uv run streamlit run app.py
# open http://localhost:8501
```

### Run Evaluations

**Chat / RAG**
```bash
uv run python -m evaluation.evaluate_chat
uv run python -m evaluation.analyze_chat_results
uv run python -m evaluation.ragas_eval_chat
```

**Flashcards**
```bash
uv run python -m evaluation.evaluate_flashcards
uv run python -m evaluation.analyze_flashcards_results
uv run python -m evaluation.judge_flashcards
```

**Quiz**
```bash
uv run python -m evaluation.evaluate_quiz --num-questions 50 --batch-size 10
uv run python -m evaluation.analyze_quiz_results
uv run python -m evaluation.judge_quiz
```

**Summary**
```bash
uv run python -m evaluation.evaluate_summary
uv run python -m evaluation.analyze_summary_results
uv run python -m evaluation.judge_summary
```

> `evaluate_*` regenerates results, `analyze_*_results` runs free deterministic checks, `judge_*` runs LLM-as-Judge. Chat/RAG uses `ragas_eval_chat` instead of `judge_chat`. Quiz and flashcard runs are resumable/checkpointed, so an interrupted run can continue without restarting.

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `GROQ_API_KEY not found` | Check `.env` exists and has a valid key |
| Slow first run | HuggingFace embedding models download & cache on first use |
| Groq 429 during evaluation | Wait for rate limit reset and rerun |
| Judge/quiz run stops partway | Re-run — completed items are checkpointed |
| Fewer quiz questions than requested | Source doc lacks enough distinct content after dedup filtering |
| Poor retrieval | Run `analyze_chat_results.py` to inspect retrieved contexts |

---

## 🚧 Future Improvements

- Multi-document RAG
- Better semantic topic-coverage evaluation
- Human vs LLM-as-Judge comparison
- Statistical confidence intervals across judge runs
- Cost and latency tracking
- Improved quiz distractor generation
- More comprehensive CI evaluation gates

---

## 📜 License

For educational purposes.