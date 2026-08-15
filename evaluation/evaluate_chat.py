import json
from pathlib import Path

from backend.chat import build_chat_chain
from backend.document_loader import load_pdf, chunk_pages
from backend.vectorstore import get_or_build_vectorstore


PDF_PATH = Path("sample_docs/Numpy Notes.pdf")
DATASET_PATH = Path("evaluation/dataset/chat_eval.json")
RESULTS_DIR = Path("evaluation/results")
RESULTS_PATH = RESULTS_DIR / "chat_results.json"


def main():
    print("Loading evaluation dataset...", flush=True)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("Loading PDF...", flush=True)

    pages, _, doc_id = load_pdf(str(PDF_PATH))

    print(f"Loaded {len(pages)} pages.", flush=True)

    print("Chunking PDF...", flush=True)

    chunks = chunk_pages(pages)

    print(f"Created {len(chunks)} chunks.", flush=True)

    print("Loading/building vector store...", flush=True)

    vector_store = get_or_build_vectorstore(
        doc_id,
        chunks=chunks,
    )

    print("Vector store ready.", flush=True)

    print("Building RAG chain...", flush=True)

    chain = build_chat_chain(vector_store)

    print("RAG chain ready.", flush=True)

    results = []

    for i, item in enumerate(dataset, start=1):
        question = item["question"]

        print(
            f"\n[{i}/{len(dataset)}] Evaluating: {question}",
            flush=True,
        )

        result = chain.invoke(
            {"input": question},
            config={
                "configurable": {
                    "session_id": f"evaluation-{i}",
                }
            },
        )

        contexts = []

        for doc in result.get("context", []):
            contexts.append(
                {
                    "page": doc.metadata.get("page"),
                    "content": doc.page_content,
                }
            )

        results.append(
            {
                "question": question,
                "ground_truth": item["ground_truth"],
                "answer": result["answer"],
                "contexts": contexts,
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nEvaluation run complete.", flush=True)
    print(f"Results saved to: {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()