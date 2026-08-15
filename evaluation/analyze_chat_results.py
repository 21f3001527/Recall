import json
from pathlib import Path


RESULTS_PATH = Path("evaluation/results/chat_results.json")


def main():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    total = len(results)

    with_context = 0
    without_context = 0
    total_contexts = 0

    print("\n" + "=" * 70)
    print("RAG RETRIEVAL BASELINE")
    print("=" * 70)

    for i, result in enumerate(results, start=1):
        contexts = result.get("contexts", [])

        pages = sorted(
            {
                context["page"]
                for context in contexts
                if context.get("page") is not None
            }
        )

        total_contexts += len(contexts)

        if contexts:
            with_context += 1
        else:
            without_context += 1

        print(f"\nQ{i}: {result['question']}")
        print(f"Retrieved contexts: {len(contexts)}")
        print(f"Retrieved pages: {pages}")

    avg_contexts = total_contexts / total if total else 0

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Total questions:              {total}")
    print(f"Questions with context:       {with_context}")
    print(f"Questions without context:    {without_context}")
    print(f"Average contexts/question:    {avg_contexts:.2f}")

    if total:
        print(
            f"Context retrieval rate:       "
            f"{with_context / total:.2%}"
        )


if __name__ == "__main__":
    main()