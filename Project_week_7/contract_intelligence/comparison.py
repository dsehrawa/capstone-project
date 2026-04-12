"""Contract comparison: side-by-side diff, missing-clause detection, and
deviation analysis between two analysed contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from contract_intelligence.analysis import (
    CLAUSE_CATEGORIES,
    AnalysisArtifacts,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimum number of clauses in a category to consider it "present"
_PRESENCE_THRESHOLD = 1

# TF-IDF similarity threshold – below this a clause pair is a "deviation"
_SIMILARITY_THRESHOLD = 0.30


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MissingClause:
    """A clause category present in one contract but absent in the other."""
    category: str
    present_in: str          # label of the contract that HAS this category
    missing_from: str        # label of the contract that LACKS it
    sample_clauses: list[str]
    risk_level: str          # Low / Medium / High
    explanation: str


@dataclass
class ClauseDeviation:
    """A matched clause category where the two contracts differ materially."""
    category: str
    contract_a_text: str
    contract_b_text: str
    similarity_score: float
    risk_delta: int          # difference in overall risk score
    explanation: str


@dataclass
class ComparisonResult:
    """Full comparison output between two contracts."""
    label_a: str
    label_b: str
    missing_clauses: list[MissingClause]
    deviations: list[ClauseDeviation]
    category_coverage_df: pd.DataFrame
    risk_comparison_df: pd.DataFrame
    summary: str
    hallucination_flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Category coverage
# ---------------------------------------------------------------------------

def _category_counts(artifacts: AnalysisArtifacts) -> dict[str, int]:
    """Count clauses per category."""
    counts: dict[str, int] = {}
    for clause in artifacts.structured_output.get("clauses", []):
        cat = clause.get("category", "General Provisions")
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def _clauses_by_category(artifacts: AnalysisArtifacts) -> dict[str, list[str]]:
    """Group clause texts by their assigned category."""
    groups: dict[str, list[str]] = {}
    for clause in artifacts.structured_output.get("clauses", []):
        cat = clause.get("category", "General Provisions")
        groups.setdefault(cat, []).append(clause.get("clause_text", ""))
    return groups


# ---------------------------------------------------------------------------
# Missing clause detection
# ---------------------------------------------------------------------------

_CATEGORY_RISK_IMPORTANCE: dict[str, str] = {
    "Indemnification": "High",
    "Limitation of Liability": "High",
    "Confidentiality": "High",
    "Intellectual Property": "High",
    "Term and Termination": "High",
    "Payment": "High",
    "Governing Law": "Medium",
    "Force Majeure": "Medium",
    "Representations and Warranties": "Medium",
    "Assignment": "Medium",
    "Notices": "Low",
    "Severability": "Low",
    "Amendments": "Low",
    "Entire Agreement": "Low",
    "Parties": "Low",
    "General Provisions": "Low",
}

_CATEGORY_MISSING_EXPLANATIONS: dict[str, str] = {
    "Indemnification": (
        "Indemnification clauses allocate liability for third-party claims. "
        "Their absence leaves parties exposed to uncapped legal costs and reduces "
        "clarity on who bears loss from breaches or IP infringement."
    ),
    "Limitation of Liability": (
        "Without a liability cap the defaulting party faces unlimited financial "
        "exposure for damages, which significantly raises the overall risk profile."
    ),
    "Confidentiality": (
        "A missing confidentiality clause means there is no contractual protection "
        "against disclosure of proprietary information, trade secrets, or business data."
    ),
    "Intellectual Property": (
        "Absence of IP provisions creates ambiguity over ownership of work product, "
        "license scope, and infringement liability."
    ),
    "Term and Termination": (
        "Without clear term and termination language, neither party has a defined "
        "right to exit the arrangement, creating lock-in and revenue-disruption risk."
    ),
    "Payment": (
        "No payment clause means the economic terms are undefined, risking disputes "
        "over fees, timing, and currency."
    ),
    "Governing Law": (
        "Absent governing-law or dispute-resolution provisions leave enforcement "
        "jurisdiction unclear, increasing litigation cost and unpredictability."
    ),
    "Force Majeure": (
        "Without a force majeure clause, obligations persist even during events "
        "outside a party's control (natural disasters, pandemics), raising operational risk."
    ),
    "Representations and Warranties": (
        "Missing representations and warranties reduce the legal remedies available "
        "if a party's stated facts turn out to be false."
    ),
    "Assignment": (
        "No assignment clause means either party could freely transfer its rights "
        "or obligations to a third party without consent."
    ),
    "Notices": (
        "Without a notices clause, there is no agreed communication channel, risking "
        "missed legal notices and cure-period disputes."
    ),
    "Severability": (
        "Absence of a severability clause may cause an entire contract to be voided "
        "if a single provision is found unenforceable."
    ),
    "Amendments": (
        "No amendments clause allows verbal modifications, which can create evidentiary "
        "disputes about what was actually agreed."
    ),
    "Entire Agreement": (
        "Without an integration clause, prior representations may be enforceable, "
        "complicating what constitutes the full agreement."
    ),
}


def detect_missing_clauses(
    artifacts_a: AnalysisArtifacts,
    artifacts_b: AnalysisArtifacts,
    label_a: str,
    label_b: str,
) -> list[MissingClause]:
    """Find clause categories present in one contract but absent in the other."""
    counts_a = _category_counts(artifacts_a)
    counts_b = _category_counts(artifacts_b)
    groups_a = _clauses_by_category(artifacts_a)
    groups_b = _clauses_by_category(artifacts_b)

    all_categories = set(CLAUSE_CATEGORIES.keys()) | set(
        counts_a) | set(counts_b)
    missing: list[MissingClause] = []

    for cat in sorted(all_categories):
        if cat == "General Provisions":
            continue
        a_present = counts_a.get(cat, 0) >= _PRESENCE_THRESHOLD
        b_present = counts_b.get(cat, 0) >= _PRESENCE_THRESHOLD

        if a_present and not b_present:
            missing.append(MissingClause(
                category=cat,
                present_in=label_a,
                missing_from=label_b,
                sample_clauses=groups_a.get(cat, [])[:3],
                risk_level=_CATEGORY_RISK_IMPORTANCE.get(cat, "Medium"),
                explanation=_CATEGORY_MISSING_EXPLANATIONS.get(
                    cat,
                    f"The '{cat}' category is present in {label_a} but missing from "
                    f"{label_b}, which may create gaps in contractual coverage.",
                ),
            ))
        elif b_present and not a_present:
            missing.append(MissingClause(
                category=cat,
                present_in=label_b,
                missing_from=label_a,
                sample_clauses=groups_b.get(cat, [])[:3],
                risk_level=_CATEGORY_RISK_IMPORTANCE.get(cat, "Medium"),
                explanation=_CATEGORY_MISSING_EXPLANATIONS.get(
                    cat,
                    f"The '{cat}' category is present in {label_b} but missing from "
                    f"{label_a}, which may create gaps in contractual coverage.",
                ),
            ))

    # Sort by risk severity: High first
    order = {"High": 0, "Medium": 1, "Low": 2}
    missing.sort(key=lambda m: order.get(m.risk_level, 3))
    return missing


# ---------------------------------------------------------------------------
# Clause-level deviation detection (TF-IDF similarity)
# ---------------------------------------------------------------------------

def _best_match_similarity(
    texts_a: list[str],
    texts_b: list[str],
) -> tuple[float, str, str]:
    """Return (max_similarity, best_text_a, best_text_b) between two sets of clause texts."""
    if not texts_a or not texts_b:
        return 0.0, "", ""

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    all_texts = texts_a + texts_b
    matrix = vectorizer.fit_transform(all_texts)
    mat_a = matrix[: len(texts_a)]
    mat_b = matrix[len(texts_a):]
    sim_matrix = cosine_similarity(mat_a, mat_b)

    best_idx = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)
    best_score = float(sim_matrix[best_idx])
    return best_score, texts_a[best_idx[0]], texts_b[best_idx[1]]


def _risk_score_for_category(artifacts: AnalysisArtifacts, category: str) -> float:
    """Average overall_score for clauses in a category."""
    risks = artifacts.structured_output["risk_obligation_intelligence"]["hybrid_clause_risks"]
    scores = [
        r["overall_score"]
        for r in risks
        if r.get("original_clause_category") == category
    ]
    return float(np.mean(scores)) if scores else 0.0


def _deviation_explanation(category: str, sim: float, risk_delta: int) -> str:
    """Generate a human-readable explanation for a clause deviation."""
    parts: list[str] = []
    if sim < 0.15:
        parts.append(
            f"The '{category}' clauses share almost no common language "
            f"(similarity {sim:.0%}), suggesting fundamentally different provisions."
        )
    else:
        parts.append(
            f"The '{category}' clauses have moderate textual overlap "
            f"(similarity {sim:.0%}) but differ materially in scope or conditions."
        )

    if abs(risk_delta) >= 3:
        parts.append(
            f"The risk score differs by {abs(risk_delta)} points, indicating one "
            "contract carries significantly higher exposure in this area."
        )
    elif abs(risk_delta) >= 1:
        parts.append(
            f"The risk score differs by {abs(risk_delta)} point(s), warranting "
            "closer review of the stronger vs. weaker protections."
        )

    importance = _CATEGORY_RISK_IMPORTANCE.get(category, "Medium")
    if importance == "High":
        parts.append(
            "This category is critically important — deviations here directly "
            "affect financial exposure and legal enforceability."
        )
    return " ".join(parts)


def detect_deviations(
    artifacts_a: AnalysisArtifacts,
    artifacts_b: AnalysisArtifacts,
    label_a: str,
    label_b: str,
) -> list[ClauseDeviation]:
    """Compare matching categories and flag material deviations."""
    groups_a = _clauses_by_category(artifacts_a)
    groups_b = _clauses_by_category(artifacts_b)

    shared_categories = set(groups_a) & set(groups_b) - {"General Provisions"}
    deviations: list[ClauseDeviation] = []

    for cat in sorted(shared_categories):
        sim, best_a, best_b = _best_match_similarity(
            groups_a[cat], groups_b[cat])
        if sim >= _SIMILARITY_THRESHOLD:
            continue  # sufficiently similar — no deviation

        risk_a = _risk_score_for_category(artifacts_a, cat)
        risk_b = _risk_score_for_category(artifacts_b, cat)
        risk_delta = int(risk_a - risk_b)

        deviations.append(ClauseDeviation(
            category=cat,
            contract_a_text=best_a[:500],
            contract_b_text=best_b[:500],
            similarity_score=sim,
            risk_delta=risk_delta,
            explanation=_deviation_explanation(cat, sim, risk_delta),
        ))

    deviations.sort(key=lambda d: d.similarity_score)
    return deviations


# ---------------------------------------------------------------------------
# Hallucination control — ground every finding back to source text
# ---------------------------------------------------------------------------

def _validate_grounding(
    missing: list[MissingClause],
    deviations: list[ClauseDeviation],
) -> list[str]:
    """Flag findings that may not be grounded in the actual contract text."""
    flags: list[str] = []

    for m in missing:
        # Ensure the "present" side really has that category
        if m.present_in and not m.sample_clauses:
            flags.append(
                f"Missing-clause finding for '{m.category}' (present_in={m.present_in}) "
                "has no sample clauses — may be a false positive."
            )

    for d in deviations:
        if not d.contract_a_text.strip() or not d.contract_b_text.strip():
            flags.append(
                f"Deviation for '{d.category}' has empty clause text on one side "
                "— verify extraction quality."
            )

    return flags


# ---------------------------------------------------------------------------
# Risk comparison DataFrame
# ---------------------------------------------------------------------------

def _build_risk_comparison(
    artifacts_a: AnalysisArtifacts,
    artifacts_b: AnalysisArtifacts,
    label_a: str,
    label_b: str,
) -> pd.DataFrame:
    """Side-by-side risk scores per category."""
    cats = set()
    for r in artifacts_a.structured_output["risk_obligation_intelligence"]["hybrid_clause_risks"]:
        cats.add(r.get("predicted_risk_category", ""))
    for r in artifacts_b.structured_output["risk_obligation_intelligence"]["hybrid_clause_risks"]:
        cats.add(r.get("predicted_risk_category", ""))

    rows = []
    for cat in sorted(cats):
        score_a = _risk_score_for_category(artifacts_a, cat)
        score_b = _risk_score_for_category(artifacts_b, cat)
        rows.append({
            "risk_category": cat,
            f"{label_a}_avg_score": round(score_a, 2),
            f"{label_b}_avg_score": round(score_b, 2),
            "delta": round(score_a - score_b, 2),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("delta", key=abs, ascending=False)
    return df


# ---------------------------------------------------------------------------
# Category coverage DataFrame
# ---------------------------------------------------------------------------

def _build_category_coverage(
    artifacts_a: AnalysisArtifacts,
    artifacts_b: AnalysisArtifacts,
    label_a: str,
    label_b: str,
) -> pd.DataFrame:
    counts_a = _category_counts(artifacts_a)
    counts_b = _category_counts(artifacts_b)
    all_cats = sorted(set(CLAUSE_CATEGORIES) | set(counts_a) | set(counts_b))
    rows = []
    for cat in all_cats:
        ca = counts_a.get(cat, 0)
        cb = counts_b.get(cat, 0)
        rows.append({
            "category": cat,
            f"{label_a}_count": ca,
            f"{label_b}_count": cb,
            "status": (
                "Both" if ca >= _PRESENCE_THRESHOLD and cb >= _PRESENCE_THRESHOLD
                else f"Only {label_a}" if ca >= _PRESENCE_THRESHOLD
                else f"Only {label_b}" if cb >= _PRESENCE_THRESHOLD
                else "Neither"
            ),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary generator (rule-based — no LLM hallucination risk)
# ---------------------------------------------------------------------------

def _generate_summary(
    missing: list[MissingClause],
    deviations: list[ClauseDeviation],
    label_a: str,
    label_b: str,
) -> str:
    parts: list[str] = [
        f"## Comparison Summary: {label_a} vs {label_b}\n"
    ]

    # Missing clauses
    high_missing = [m for m in missing if m.risk_level == "High"]
    med_missing = [m for m in missing if m.risk_level == "Medium"]
    parts.append(
        f"**Missing clauses:** {len(missing)} total — "
        f"{len(high_missing)} high-risk, {len(med_missing)} medium-risk."
    )
    for m in high_missing:
        parts.append(
            f"- **{m.category}** missing from *{m.missing_from}* — {m.explanation}"
        )

    # Deviations
    parts.append(
        f"\n**Material deviations:** {len(deviations)} categories differ significantly.")
    for d in deviations[:5]:
        parts.append(
            f"- **{d.category}** (similarity {d.similarity_score:.0%}, "
            f"risk delta {d.risk_delta:+d}) — {d.explanation}"
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_contracts(
    artifacts_a: AnalysisArtifacts,
    artifacts_b: AnalysisArtifacts,
    label_a: str = "Contract A",
    label_b: str = "Contract B",
) -> ComparisonResult:
    """Run full comparison between two analysed contracts."""
    missing = detect_missing_clauses(
        artifacts_a, artifacts_b, label_a, label_b)
    deviations = detect_deviations(artifacts_a, artifacts_b, label_a, label_b)
    hallucination_flags = _validate_grounding(
        missing, deviations)
    coverage_df = _build_category_coverage(
        artifacts_a, artifacts_b, label_a, label_b)
    risk_df = _build_risk_comparison(
        artifacts_a, artifacts_b, label_a, label_b)
    summary = _generate_summary(missing, deviations, label_a, label_b)

    return ComparisonResult(
        label_a=label_a,
        label_b=label_b,
        missing_clauses=missing,
        deviations=deviations,
        category_coverage_df=coverage_df,
        risk_comparison_df=risk_df,
        summary=summary,
        hallucination_flags=hallucination_flags,
    )
