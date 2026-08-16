"""
Flashcards evaluation -- free sanity check (no LLM calls).

Mirrors analyze_chat_results.py: a quick, zero-cost pass over the raw
generated flashcards to catch obvious problems (near-duplicate cards,
suspiciously short/long answers) before spending API calls on
judge_flashcards.py.

Usage:
    uv run python -m evaluation.analyze_flashcards_results
"""

import json
from difflib import SequenceMatcher
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "results" / "flashcards_results.json"

DUPLICATE_SIMILARITY_THRESHOLD = 0.85


def find_near_duplicates(cards: list[dict]) -> list[tuple[int, int, float]]:
    pairs = []
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            ratio = SequenceMatcher(
                None, cards[i]["question"].lower(), cards[j]["question"].lower(),
            ).ratio()
            if ratio >= DUPLICATE_SIMILARITY_THRESHOLD:
                pairs.append((i, j, round(ratio, 2)))
    return pairs


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"{RESULTS_PATH} not found -- run evaluate_flashcards.py first.")

    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)
    cards = results["cards"]

    if not cards:
        print("No cards to analyze.")
        return

    q_lengths = [len(c["question"].split()) for c in cards]
    a_lengths = [len(c["answer"].split()) for c in cards]
    empty_cards = [i for i, c in enumerate(cards) if not c["question"].strip() or not c["answer"].strip()]
    duplicates = find_near_duplicates(cards)

    print(f"Doc: {results['doc']}")
    print(f"Cards generated: {results['num_cards_generated']}/{results['num_cards_requested']}")
    print(f"Generation time: {results['generation_time_seconds']}s")
    print()
    print(f"Question length (words) -- avg: {sum(q_lengths)/len(q_lengths):.1f}, "
          f"min: {min(q_lengths)}, max: {max(q_lengths)}")
    print(f"Answer length (words)   -- avg: {sum(a_lengths)/len(a_lengths):.1f}, "
          f"min: {min(a_lengths)}, max: {max(a_lengths)}")
    print()

    if empty_cards:
        print(f"⚠️  Empty question/answer at indices: {empty_cards}")
    else:
        print("✅ No empty questions/answers.")

    if duplicates:
        print(f"⚠️  {len(duplicates)} near-duplicate question pair(s) (similarity >= {DUPLICATE_SIMILARITY_THRESHOLD}):")
        for i, j, ratio in duplicates:
            print(f"   [{i}] \"{cards[i]['question']}\"")
            print(f"   [{j}] \"{cards[j]['question']}\"  (similarity: {ratio})")
    else:
        print("✅ No near-duplicate questions detected.")


if __name__ == "__main__":
    main()