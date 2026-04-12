"""Evaluate the RAG pipeline against the golden Q&A set.

Usage:
    python evaluate_golden_set.py              # run evaluation
    python evaluate_golden_set.py --verbose    # show individual answers
"""

import json
import sys
import time
from pathlib import Path

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

from contract_intelligence.analysis import AnalysisArtifacts, analyze_contract
from contract_intelligence.rag import (
    ContractRetriever,
    answer_question,
    ollama_available,
    ollama_chat,
)

PROJ_DIR = Path(__file__).resolve().parent
VECTOR_DB_DIR = PROJ_DIR / "chroma_db"
GOLDEN_FILE = PROJ_DIR / "golden_qa.json"

CONTRACT_PATHS = {
    "Peloton-Affirm": PROJ_DIR.parent / "sec.gov_Archives_edgar_data_1820953_000110465920126927_tm2026663d5_ex10-6.htm.pdf",
    "Affirm-Amazon": PROJ_DIR.parent / "Document-2.pdf",
}
ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
BLEU_SMOOTHER = SmoothingFunction().method1


def load_golden_set() -> list[dict]:
    with open(GOLDEN_FILE, encoding="utf-8") as f:
        return json.load(f)


def build_artifacts(pdf_path: Path) -> AnalysisArtifacts:
    return analyze_contract(pdf_path)


def judge_with_llm(question: str, expected: str, actual: str) -> dict:
    """Use Ollama/Mistral to score the RAG answer against the expected answer."""
    prompt = (
        "You are an impartial evaluator comparing two answers to the same question about a legal contract.\n\n"
        "Score the ACTUAL answer on a scale of 1-5:\n"
        "  5 = Fully correct and complete, covers all key points in the EXPECTED answer.\n"
        "  4 = Mostly correct, minor details missing or slightly imprecise.\n"
        "  3 = Partially correct, captures the main idea but misses important details.\n"
        "  2 = Mostly wrong or very incomplete, only touches on the topic.\n"
        "  1 = Completely wrong, irrelevant, or no useful information.\n\n"
        "Respond ONLY with a JSON object: {\"score\": <int>, \"reason\": \"<one-line explanation>\"}\n\n"
        f"QUESTION: {question}\n\n"
        f"EXPECTED ANSWER: {expected}\n\n"
        f"ACTUAL ANSWER: {actual}\n"
    )
    try:
        raw = ollama_chat("mistral", prompt, timeout=180)
    except Exception:
        return {"score": 0, "reason": "LLM judge timed out"}

    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        result = json.loads(raw[start:end])
        return {"score": int(result.get("score", 1)), "reason": result.get("reason", "")}
    except (ValueError, json.JSONDecodeError):
        return {"score": 0, "reason": f"Could not parse LLM judge response: {raw[:200]}"}


def compute_overlap_metrics(expected: str, actual: str) -> dict:
    expected = expected or ""
    actual = actual or ""
    rouge = ROUGE_SCORER.score(expected, actual)
    bleu = (
        sentence_bleu(
            [expected.split()],
            actual.split(),
            smoothing_function=BLEU_SMOOTHER,
        )
        if expected.strip() and actual.strip()
        else 0.0
    )
    return {
        "bleu": round(float(bleu), 4),
        "rouge1_f1": round(rouge["rouge1"].fmeasure, 4),
        "rouge2_f1": round(rouge["rouge2"].fmeasure, 4),
        "rougeL_f1": round(rouge["rougeL"].fmeasure, 4),
    }


def run_evaluation(verbose: bool = False) -> dict:
    golden_set = load_golden_set()
    print(f"Loaded {len(golden_set)} golden Q&A pairs.\n")

    if not ollama_available():
        print("ERROR: Ollama is not running. Start it with 'ollama serve' and 'ollama pull mistral'.")
        sys.exit(1)

    print("Analyzing contracts...")
    artifacts_map: dict[str, AnalysisArtifacts] = {}
    retriever_map: dict[str, ContractRetriever] = {}
    risk_summary_map: dict[str, list] = {}

    for label, pdf_path in CONTRACT_PATHS.items():
        if not pdf_path.exists():
            print(f"  WARNING: {pdf_path} not found, skipping {label}")
            continue
        print(f"  Processing {label}...")
        artifacts = build_artifacts(pdf_path)
        artifacts_map[label] = artifacts
        retriever_map[label] = ContractRetriever(
            artifacts.structured_output["clauses"],
            db_dir=VECTOR_DB_DIR,
            collection_name=f"eval_{label.lower().replace('-', '_')}",
        )
        risk_summary_map[label] = artifacts.structured_output["risk_obligation_intelligence"]["risk_category_summary"]

    print()

    results = []
    total_score = 0
    for item in golden_set:
        qid = item["id"]
        contract = item["contract"]
        question = item["question"]
        expected = item["expected_answer"]

        if contract not in retriever_map:
            print(f"  Q{qid}: SKIP (contract {contract} not loaded)")
            results.append({**item, "actual_answer": "SKIPPED", "score": 0, "reason": "Contract not loaded"})
            continue

        print(f"  Q{qid}: {question[:80]}...")
        start_t = time.time()
        result = answer_question(question, retriever_map[contract], risk_summary_map[contract])
        elapsed = time.time() - start_t

        judgement = judge_with_llm(question, expected, result.answer)
        overlap_metrics = compute_overlap_metrics(expected, result.answer)
        score = judgement["score"]
        reason = judgement["reason"]
        total_score += score

        results.append(
            {
                **item,
                "actual_answer": result.answer,
                "score": score,
                "reason": reason,
                "elapsed_sec": round(elapsed, 1),
                "num_contexts": len(result.contexts),
                "top_context_score": round(result.contexts[0]["score"], 3) if result.contexts else 0,
                **overlap_metrics,
            }
        )

        if verbose:
            print(f"       Score: {score}/5 | {reason}")
            print(
                "       Overlap metrics: "
                f"BLEU={overlap_metrics['bleu']}, "
                f"ROUGE-L={overlap_metrics['rougeL_f1']}"
            )
            print(f"       Expected: {expected[:120]}...")
            print(f"       Actual:   {result.answer[:120]}...")
            print()

    avg_score = total_score / len(golden_set) if golden_set else 0
    avg_bleu = sum(r["bleu"] for r in results) / len(results) if results else 0
    avg_rouge_l = sum(r["rougeL_f1"] for r in results) / len(results) if results else 0
    score_dist = {s: sum(1 for r in results if r["score"] == s) for s in range(6)}

    summary = {
        "total_questions": len(golden_set),
        "average_score": round(avg_score, 2),
        "average_bleu": round(avg_bleu, 4),
        "average_rougeL_f1": round(avg_rouge_l, 4),
        "score_distribution": score_dist,
        "scores_per_question": [
            {
                "id": r["id"],
                "contract": r["contract"],
                "score": r["score"],
                "reason": r["reason"],
                "top_context_score": r.get("top_context_score", 0),
                "bleu": r.get("bleu", 0),
                "rougeL_f1": r.get("rougeL_f1", 0),
            }
            for r in results
        ],
    }

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Questions: {len(golden_set)}")
    print(f"  Average score: {avg_score:.2f} / 5.00")
    print(f"  Average BLEU: {avg_bleu:.4f}")
    print(f"  Average ROUGE-L F1: {avg_rouge_l:.4f}")
    print(f"  Score distribution: {score_dist}")
    print()
    for r in results:
        marker = "PASS" if r["score"] >= 4 else "WEAK" if r["score"] >= 3 else "FAIL"
        print(f"  Q{r['id']:2d} [{marker}] {r['score']}/5  {r['question'][:65]}")
    print()

    output_file = PROJ_DIR / "golden_eval_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2)
    print(f"Detailed results saved to {output_file}")

    return summary


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    run_evaluation(verbose=verbose)
