import json
import pandas as pd
from pathlib import Path
from contract_intelligence.rag import ContractRetriever, answer_question
from contract_intelligence.analysis import analyze_contract

def run_golden_tests(contract_pdf_path: str | Path, golden_dataset_path: str = "golden_dataset.json", output_path: str = "rag_test_results.json"):
    """
    Run golden dataset questions against the RAG system for a given contract.

    Args:
        contract_pdf_path: Path to the contract PDF to analyze.
        golden_dataset_path: Path to the golden dataset JSON.
        output_path: Path to save test results.

    Returns:
        List of test results with comparisons.
    """
    # Resolve paths relative to this script's directory
    script_dir = Path(__file__).resolve().parent
    if not Path(golden_dataset_path).is_absolute():
        golden_dataset_path = script_dir / golden_dataset_path
    if not Path(output_path).is_absolute():
        output_path = script_dir / output_path
    
    # Load golden dataset
    with open(golden_dataset_path, 'r') as f:
        golden_data = json.load(f)

    # Analyze the contract
    print("Analyzing contract...")
    artifacts = analyze_contract(contract_pdf_path)

    # Initialize retriever
    VECTOR_DB_DIR = Path(__file__).resolve().parent / "chroma_db"
    retriever = ContractRetriever(
        artifacts.structured_output["clauses"],
        db_dir=VECTOR_DB_DIR,
        collection_name="test_contract"
    )

    # Get risk summary
    risk_summary = artifacts.structured_output["risk_obligation_intelligence"]["risk_category_summary"]

    results = []
    print(f"Running {len(golden_data)} test questions...")
    for i, item in enumerate(golden_data):
        print(f"Testing question {i+1}/{len(golden_data)}: {item['question'][:50]}...")
        result = answer_question(item["question"], retriever, risk_summary)
        results.append({
            "question": item["question"],
            "category": item["category"],
            "expected_answer": item["expected_answer"],
            "generated_answer": result.answer,
            "retrieved_clauses_count": len(result.contexts),
            "top_clause_category": result.contexts[0]["category"] if result.contexts else "None",
            "top_clause_score": result.contexts[0]["score"] if result.contexts else 0.0
        })

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Test completed. Results saved to {output_path}")
    return results

def compute_basic_metrics(results: list) -> dict:
    """
    Compute basic metrics like retrieval success rate.
    For more advanced metrics, use libraries like rouge-score or bert-score.
    """
    total = len(results)
    retrieved_any = sum(1 for r in results if r["retrieved_clauses_count"] > 0)
    retrieval_rate = retrieved_any / total if total > 0 else 0

    # Simple keyword overlap for answer similarity (basic heuristic)
    similarities = []
    for r in results:
        gen_lower = r["generated_answer"].lower()
        exp_lower = r["expected_answer"].lower()
        common_words = set(gen_lower.split()) & set(exp_lower.split())
        similarity = len(common_words) / max(len(set(exp_lower.split())), 1)
        similarities.append(similarity)

    avg_similarity = sum(similarities) / len(similarities) if similarities else 0

    return {
        "total_questions": total,
        "retrieval_success_rate": retrieval_rate,
        "average_answer_similarity": avg_similarity
    }

if __name__ == "__main__":
    # Example usage: Run on a sample contract
    # Replace with actual PDF path
    sample_pdf = Path("sample_contract.pdf")  # Add a sample PDF or use one from week3
    if sample_pdf.exists():
        results = run_golden_tests(sample_pdf)
        metrics = compute_basic_metrics(results)
        print("Metrics:", metrics)
    else:
        print("No sample contract found. Run with a valid PDF path.")