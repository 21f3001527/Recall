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
- **📊 Evaluation framework** — automated evaluation for the Chat/RAG, Flashcard, Quiz, and Summary pipelines.

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
│   ├── analyze_quiz_results.py
│   ├── analyze_summary_results.py
│   ├── evaluate_chat.py
│   ├── evaluate_flashcards.py
│   ├── evaluate_quiz.py
│   ├── evaluate_summary.py
│   ├── judge_flashcards.py
│   ├── judge_quiz.py
│   ├── judge_summary.py
│   ├── ragas_eval_chat.py
│   ├── dataset/
│   │   ├── chat_eval.json
│   │   ├── flashcards_eval.json
│   │   ├── quiz_eval.json
│   │   └── summary_eval.json
│   └── results/
│       ├── chat_results.json
│       ├── chat_ragas_results.json
│       ├── chat_ragas_summary.json
│       ├── flashcards_results.json
│       ├── flashcards_judge_results.json
│       ├── flashcards_judge_summary.json
│       ├── quiz_results.json
│       ├── quiz_judge_results.json
│       ├── quiz_judge_summary.json
│       ├── summary_results.json
│       ├── summary_judge_results.json
│       └── summary_judge_summary.json
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

Four separate evaluation pipelines exist, each matched to its component's failure modes. Evaluations run against the **actual production functions**, not duplicated logic.

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

### 🧠 Quiz Evaluation
Like flashcards, quiz questions don't have a fixed retrieval-QA structure, so the same approach is used:
- **Free structural sanity checks**
- **LLM-as-judge quality scoring**
- **Topic coverage analysis**

**Pipeline:** `evaluate_quiz.py` → `quiz_results.json` → `analyze_quiz_results.py` (sanity checks) + `judge_quiz.py` (LLM judge + topic coverage) → `quiz_judge_results.json` / `quiz_judge_summary.json`.

**Dataset:** `evaluation/dataset/quiz_eval.json` — same 10-topic checklist used for flashcards, applied to quiz coverage.

**Step 1 — Generation** (runs production `generate_quiz()`, in resumable batches to survive Groq rate limits):
```bash
uv run python -m evaluation.evaluate_quiz --num-questions 50 --batch-size 10
```
Progress is saved after every batch and merged across runs, so a 429 mid-run doesn't lose work — just re-run the same command once quota refills.

**Step 2 — Free sanity checks** (option formatting, answer-key validity, near-duplicate questions **and** near-duplicate answer options within a question, via `SequenceMatcher`, threshold 0.85):
```bash
uv run python -m evaluation.analyze_quiz_results
```

**Step 3 — LLM-as-judge** (1–5 scale on Correctness, Clarity, Distractor Quality, Difficulty Appropriateness, plus a topic coverage check):
```bash
uv run python -m evaluation.judge_quiz
```

**Current result (22/22 questions judged, 0 failed):**

| Metric | Score |
|---|---|
| Correctness | 5.0 / 5 |
| Clarity | 5.0 / 5 |
| Distractor Quality | 5.0 / 5 |
| Difficulty Appropriateness | 4.0 / 5 |

Topic coverage: **7/10 (70%)** — missing: broadcasting rules, aggregation functions, stacking/splitting arrays.

**Findings**

1. **50 requested, only 22 unique questions generated.** Near-duplicate filtering (similarity ≥ 0.85) is applied at generation time, not just at analysis time — every batch that returned only rephrased repeats of earlier questions was rejected outright. The 23-page source document does not contain enough distinct content to support 50 non-overlapping medium-difficulty questions; 22 is the honest ceiling for this document at this difficulty, not a bug or a quota issue.

2. **The LLM judge missed a defect the free sanity check caught.** `analyze_quiz_results.py` flagged 3 questions where two of the four answer options were near-duplicates of each other (effectively reducing a 4-option question to 3 real choices). Despite this, the LLM judge scored Distractor Quality a perfect 5.0/5 across the board. This is a concrete demonstration of why both check types are run: deterministic sanity checks catch structural defects an LLM judge can silently overlook, and an LLM judge catches semantic/quality issues a deterministic check can't detect.

3. **Topic coverage gaps line up with the duplicate ceiling.** With only 22 unique questions available, 3 of the 10 checklist topics (broadcasting, aggregation, stacking/splitting) weren't touched at all — a direct consequence of finding #1, not a separate generation weakness.

> As with flashcards, interpret these as the outcome of the current run, not absolute ground truth — re-running with a longer or more topic-dense source document should raise both the unique-question ceiling and topic coverage.

### 📝 Summary Evaluation
Summarization produces a single structured document, not a list of many items, so the pipeline differs slightly: there's no batching or per-item resumability — just one generation call and one judge call, each defensively wrapped.

**Pipeline:** `evaluate_summary.py` → `summary_results.json` → `analyze_summary_results.py` (structural sanity checks) + `judge_summary.py` (LLM judge + topic coverage) → `summary_judge_results.json` / `summary_judge_summary.json`.

**Dataset:** `evaluation/dataset/summary_eval.json` — same 10-topic checklist used for flashcards and quiz.

**Step 1 — Generation** (runs production `summarise_notes()`, a map-reduce pipeline: one LLM call per document chunk, then one call to combine them into the final structured summary):
```bash
uv run python -m evaluation.evaluate_summary
```

**Step 2 — Free sanity checks** (verifies the 4 required sections — Summary, Key Concepts, Important Terms, Key Takeaways — are present and in order, flags empty sections, checks for near-duplicate bullets via `SequenceMatcher` threshold 0.85, and reports the summary-to-source length ratio):
```bash
uv run python -m evaluation.analyze_summary_results
```

**Step 3 — LLM-as-judge** (1–5 scale on Faithfulness, Coverage, Conciseness, Coherence, plus a topic coverage check):
```bash
uv run python -m evaluation.judge_summary
```

**Current result:**

| Metric | Score |
|---|---|
| Faithfulness | 5.0 / 5 |
| Coverage | 5.0 / 5 |
| Conciseness | 4.0 / 5 |
| Coherence | 5.0 / 5 |

- Source: 22,539 characters → Summary: 2,126 characters (ratio 0.094).
- Free sanity check: all 4 sections present and correctly ordered, 18 total bullets across Key Concepts/Important Terms/Key Takeaways, 0 near-duplicate bullets.
- Judge notes: faithful and well-organized, but could be more concise — some bullets repeat similar phrasing rather than staying maximally tight, which matches the 4/5 (not 5/5) Conciseness score.

Topic coverage: **7/10 (70%)** — missing per keyword-matching: reshaping/flattening, boolean/fancy indexing, stacking/splitting.

**Findings**

1. **The topic coverage gap is partly a false negative from keyword matching, not an actual content gap.** The generated summary explicitly includes *"Reshaping and transposing: operations for changing the dimensions of an array"* under Key Concepts — but the checklist keyword is `"reshape"`, which is not a substring of `"reshaping"`, so the deterministic check missed it. This is a concrete, reproducible example of why keyword-based coverage is listed as a known limitation (see Future Improvements) rather than a precise measurement — the true content coverage is likely higher than 70%.

2. **Faithfulness and Coherence scored perfectly**, consistent with the map-reduce design's internal retry logic (one transient 429 during generation was retried automatically and didn't affect the final output — no chunks were dropped this run).

> As with flashcards and quiz, interpret these as the outcome of the current run, not absolute ground truth. Re-running `judge_summary` after switching the topic checklist to lemma-aware or semantic matching (see Future Improvements) would likely raise the reported coverage percentage without any change to the summary itself.

### 📌 Evaluation Philosophy
- **Chat/RAG:** retrieval evaluation + generation evaluation via RAGAS.
- **Flashcards, Quiz & Summary:** structural sanity checks + LLM-as-judge + topic coverage.

Each component is evaluated with metrics suited to its actual failure modes rather than one-size-fits-all metrics. Free sanity checks and LLM-as-judge scoring are complementary, not redundant — the quiz evaluation surfaced a case where a structural defect (near-duplicate answer options) was caught by the deterministic check but scored perfectly by the LLM judge, while the summary evaluation surfaced the reverse pattern: a deterministic check (keyword-based topic coverage) reporting a false negative that the LLM judge's holistic Coverage score (5/5) did not.

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
| Groq 429 during evaluation | Wait for the rate limit to reset and rerun; flashcard and quiz evaluation resume automatically. |
| Flashcard judge stops partway | Re-run `judge_flashcards`; completed cards are skipped. |
| Quiz generation stops short of requested count | Expected once the source document runs out of distinct content — near-duplicate filtering rejects rephrased repeats. Re-run `evaluate_quiz` to resume, or accept the lower count as the honest ceiling for that document. |
| Summary topic coverage looks low despite a good summary | Check whether the missing topic's keyword just doesn't match the summary's actual word form (e.g. "reshape" vs "reshaping") — this is a known keyword-matching limitation, not necessarily a content gap. Read the summary directly to confirm. |
| Poor retrieval results | Run `analyze_chat_results.py` and inspect retrieved contexts before tuning retrieval params. |

---

## 🚧 Future Improvements
- Regenerate quiz questions against a longer/more topic-dense source document to close the coverage gap (broadcasting, aggregation, stacking/splitting)
- Tighten `generate_quiz()`'s prompt to enforce 4 semantically distinct distractors per question, closing the gap the free sanity check found
- Semantic (vs. keyword) topic coverage — the summary evaluation found a concrete false negative ("reshaping" vs. the keyword "reshape") that a lemma-aware or embedding-based match would avoid
- Human vs. LLM-as-judge comparison
- Multi-document evaluation
- Statistical confidence intervals across repeated judge runs
- Cost/latency tracking
- Automated regression testing in CI/CD

## 📄 Sample Document
`sample_docs/Numpy Notes.pdf` — used as the source for the Chat benchmark, Flashcard evaluation, Quiz evaluation, and Summary evaluation.

## 📄 License
For educational purposes.