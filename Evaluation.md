# 📊 Evaluation

A major focus of this project is evaluating the AI components rather than relying only on subjective inspection. Each component is evaluated according to its actual failure mode — not the same metric applied everywhere.

← [Back to README](README.md)

---

## Evaluation Strategy

| Component | Evaluation Approach |
|---|---|
| 💬 Chat / RAG | Retrieval sanity checks + RAGAS |
| 🗂️ Flashcards | Structural checks + LLM-as-Judge + topic coverage |
| 🧠 Quiz | Structural checks + LLM-as-Judge + topic coverage |
| 📝 Summary | Structural checks + LLM-as-Judge + topic coverage |

Deterministic checks and LLM-as-Judge are complementary:

- **Deterministic checks** catch structural problems such as malformed output and duplicate options.
- **LLM-as-Judge** evaluates semantic properties such as correctness, clarity, faithfulness, and quality.

All evaluation pipelines execute the actual production functions, rather than separate test implementations.

---

## 📈 Current Evaluation Results

| Component | Results |
|---|---|
| 💬 Chat / RAG | 100% of questions retrieved ≥1 context · Faithfulness 0.87–0.94 · Context Recall 0.91 |
| 🗂️ Flashcards | Faithfulness 5.0/5 · Clarity 5.0/5 · Correctness 5.0/5 · Scope 5.0/5 · Topic coverage 100% |
| 🧠 Quiz | Correctness 5.0/5 · Clarity 5.0/5 · Distractor Quality 5.0/5 · Difficulty 4.0/5 |
| 📝 Summary | Faithfulness 5.0/5 · Coverage 5.0/5 · Conciseness 4.0/5 · Coherence 5.0/5 |

> Evaluation results represent the current benchmark runs and should not be interpreted as absolute ground truth.

### Chat / RAG — retrieval depth comparison

| Metric | K=4 Run 1 | K=4 Run 2 | K=6 |
|---|---|---|---|
| Faithfulness | 0.94 | 0.87 | 0.91 |
| Answer Relevancy | 0.87 | 0.81 | 0.84 |
| Context Precision | 0.93 | 0.92 | 0.88 |
| Context Recall | 0.91 | 0.91 | 0.88 |

The current configuration uses `RETRIEVAL_K = 4`, because K=4 provided better overall retrieval quality than K=6 without introducing unnecessary context.

**Retrieval sanity check** (deterministic, no LLM calls):

```
Questions evaluated       : 20
Questions with context    : 20
Questions without context : 0
Retrieval rate             : 100%
Average contexts/question : 3.15
```

The sanity check is intentionally separate from LLM-based evaluation so retrieval failures can be detected without consuming API tokens.

---

## 🔎 Evaluation Findings

**1. Retrieval depth**

Increasing retrieval from K=4 to K=6 did not improve Context Recall. As additional chunks introduced more irrelevant context, K=4 was retained.

**2. Deterministic checks caught an LLM-judge miss**

The quiz structural analysis detected near-duplicate answer options that the LLM judge still rated highly. This demonstrates why deterministic validation is useful alongside LLM-based evaluation.

**3. Topic coverage false negative**

The summary evaluator reported a topic-coverage miss because the checklist searched for `reshape`, while the generated summary contained `reshaping`. This highlights a limitation of keyword-based coverage evaluation and motivates future semantic matching.

**4. Quiz generation ceiling**

The quiz evaluation requested 50 questions but produced 22 unique questions. Near-duplicate filtering rejected repeated or rephrased questions because the 23-page source document did not contain enough distinct material for 50 non-overlapping questions at the selected difficulty. This is treated as an evaluation finding rather than artificially inflating the dataset with duplicates.

---

## 📄 Evaluation Dataset

The benchmark uses:

```
sample_docs/Numpy Notes.pdf
```

The same source document is used across the Chat, Flashcard, Quiz, and Summary evaluation pipelines so the results can be compared consistently.

Evaluation datasets are stored under:

```
evaluation/dataset/
```

Generated results are stored under:

```
evaluation/results/
```

---

## 🧪 Running Evaluations

**Chat / RAG**

```bash
uv run python -m evaluation.evaluate_chat
uv run python -m evaluation.analyze_chat_results
uv run python -m evaluation.ragas_eval_chat
```

- `evaluate_chat` regenerates the retrieval and answer results
- `analyze_chat_results` performs free, deterministic retrieval checks
- `ragas_eval_chat` performs LLM-based RAG evaluation

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

Quiz generation is resumable across batches, allowing evaluation to continue after API rate limits.

**Summary**

```bash
uv run python -m evaluation.evaluate_summary
uv run python -m evaluation.analyze_summary_results
uv run python -m evaluation.judge_summary
```

Quiz and flashcard evaluation both support checkpointing, so interrupted runs can resume without starting from the beginning.

### Evaluation Workflow

```text
evaluate_*
     │
     ▼
Generate evaluation results
     │
     ▼
analyze_*_results
     │
     ▼
Deterministic validation
     │
     ├───────────────┐
     ▼               ▼
LLM-as-Judge       RAGAS
     │               │
     └───────┬───────┘
             ▼
      Evaluation Results
```

---

## 🔬 Evaluation Philosophy

Different application components fail in different ways, so the project deliberately uses different evaluation strategies rather than one metric for everything.

```text
                    Study Assistant
                          │
          ┌───────────────┴───────────────┐
          │                               │
       Chat/RAG                    Generated Content
          │                               │
          ▼                               ▼
    Retrieval + RAGAS           Structural + LLM Judge
          │                               │
    ┌─────┴─────┐                  ┌──────┼──────┐
    ▼           ▼                  ▼      ▼      ▼
Precision     Recall            Quiz  Cards  Summary
    │
    └─────────────── Faithfulness / Relevancy
```

← [Back to README](README.md)