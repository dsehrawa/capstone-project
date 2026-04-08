import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from pypdf import PdfReader

try:
    import spacy
except ImportError:  # pragma: no cover
    spacy = None


RISK_TAXONOMY = {
    "Financial": {"subtypes": ["Payment Delay", "Fee Dispute", "Tax Exposure"], "keywords": ["payment", "fee", "charge", "invoice", "tax", "late payment"]},
    "Legal": {"subtypes": ["Indemnity Exposure", "Liability Cap", "Dispute Resolution"], "keywords": ["indemnify", "indemnification", "liability", "damages", "arbitration", "court"]},
    "Operational": {"subtypes": ["Service Disruption", "Force Majeure", "Delivery Failure"], "keywords": ["delay", "failure to perform", "force majeure", "service levels", "delivery"]},
    "Compliance": {"subtypes": ["Regulatory Breach", "Jurisdiction Risk", "Policy Non-Compliance"], "keywords": ["law", "jurisdiction", "compliance", "regulation", "applicable law"]},
    "Information Security": {"subtypes": ["Confidentiality Breach", "Data Misuse", "Security Incident"], "keywords": ["confidential", "proprietary", "security", "non-disclosure", "data"]},
    "Intellectual Property": {"subtypes": ["Ownership Ambiguity", "License Scope", "Infringement"], "keywords": ["intellectual property", "copyright", "trademark", "patent", "license", "ownership"]},
    "Strategic External": {"subtypes": ["Geopolitical", "Weather / Disaster", "Macroeconomic", "Supply Chain"], "keywords": ["geopolitical", "weather", "natural disaster", "inflation", "supply chain"]},
}

CLAUSE_CATEGORIES = {
    "Parties": ["merchant", "company", "party", "agreement", "affiliate", "customer", "supplier", "provider", "user", "seller", "buyer"],
    "Term and Termination": ["term", "terminate", "termination", "expiration", "renewal", "effective date", "commencement date", "end date", "duration", "period"],
    "Confidentiality": ["confidential", "disclosure", "secret", "proprietary", "non-disclosure", "nondisclosure", "confidential information"],
    "Governing Law": ["governing law", "jurisdiction", "applicable law", "venue", "dispute resolution", "arbitration", "court", "state", "country"],
    "Payment": ["payment", "fee", "charge", "invoice", "remuneration", "compensation", "price", "currency", "due date", "tax", "late payment"],
    "Indemnification": ["indemnify", "indemnification", "hold harmless", "damages", "losses", "claims", "liabilities"],
    "Limitation of Liability": ["limit liability", "limitation of liability", "maximum liability", "exclude liability", "indirect damages", "consequential damages"],
    "Representations and Warranties": ["represent", "warrant", "representation", "warranty", "guarantee", "covenant", "accuracy", "truthfulness"],
    "Intellectual Property": ["intellectual property", "patent", "trademark", "copyright", "license", "licence", "infringement", "ownership", "rights"],
    "Force Majeure": ["force majeure", "act of god", "unforeseeable", "circumstances", "event beyond control", "delay", "failure to perform"],
    "Notices": ["notice", "address", "writing", "mail", "email", "fax", "delivery"],
    "Assignment": ["assign", "assignment", "transfer", "delegate"],
    "Severability": ["severable", "invalid", "unenforceable", "void", "provision", "clause"],
    "Amendments": ["amend", "modify", "change", "alteration", "in writing"],
    "Entire Agreement": ["entire agreement", "supersedes", "prior agreements", "oral agreements", "understanding"],
    "General Provisions": [],
}

INTERNAL_RISK_RULES = [
    {"risk_name": "Termination / early exit risk", "keywords": ["terminate", "termination", "expiration", "end date", "renewal"], "category": "Contractual",
        "default_impact": "High", "explanation": "Termination clauses can create revenue disruption, service interruption, or abrupt obligation changes."},
    {"risk_name": "Payment / fee risk", "keywords": ["payment", "fee", "charge", "invoice", "tax", "late payment"], "category": "Financial",
        "default_impact": "High", "explanation": "Payment timing, fee disputes, or tax allocation can directly affect cash flow and commercial viability."},
    {"risk_name": "Confidentiality / data handling risk", "keywords": ["confidential", "proprietary", "non-disclosure", "nondisclosure", "secret"], "category": "Information Security",
        "default_impact": "High", "explanation": "Disclosure of confidential information can trigger legal exposure, reputational damage, and commercial harm."},
    {"risk_name": "Liability / indemnity exposure", "keywords": ["indemnify", "indemnification", "hold harmless", "liabilities", "damages", "limitation of liability"],
        "category": "Legal", "default_impact": "High", "explanation": "Broad indemnity or liability language can create significant financial and legal exposure."},
    {"risk_name": "Regulatory / compliance risk", "keywords": ["law", "jurisdiction", "arbitration", "court", "compliance", "applicable law"], "category": "Compliance",
        "default_impact": "Medium", "explanation": "Jurisdiction, dispute resolution, and compliance obligations can increase cost and enforcement complexity."},
    {"risk_name": "Intellectual property ownership risk", "keywords": ["intellectual property", "patent", "trademark", "copyright", "license", "ownership", "rights"],
        "category": "IP", "default_impact": "High", "explanation": "Ambiguous IP ownership or license scope can affect product rights, reuse, and future commercialization."},
    {"risk_name": "Force majeure / service disruption risk", "keywords": ["force majeure", "act of god", "delay", "failure to perform", "event beyond control"],
        "category": "Operational", "default_impact": "High", "explanation": "Disruption clauses indicate exposure to uncontrollable events that may delay or prevent performance."},
]

OBLIGATION_PATTERNS = [r"\bshall\b", r"\bmust\b", r"\bagrees? to\b",
                       r"\bis required to\b", r"\bwill\b", r"\bundertakes? to\b"]
EXTERNAL_RISK_CONTEXT = {"weather_area": "NY", "cyber_vendor_keywords": [
    "microsoft", "apache", "oracle", "linux", "openssl"], "cyber_product_keywords": ["exchange", "windows", "http server", "mysql", "vmware"]}
COUNTRY_NAME_TO_CODE = {"united states": "USA", "u.s.": "USA", "u.s.a.": "USA", "usa": "USA", "india": "IND", "united kingdom": "GBR", "uk": "GBR", "england": "GBR",
                        "canada": "CAN", "australia": "AUS", "germany": "DEU", "france": "FRA", "singapore": "SGP", "japan": "JPN", "china": "CHN", "uae": "ARE", "united arab emirates": "ARE"}
STATE_TO_COUNTRY_CODE = {"new york": "USA", "california": "USA", "texas": "USA", "florida": "USA",
                         "delaware": "USA", "massachusetts": "USA", "illinois": "USA", "washington": "USA", "new jersey": "USA"}
SCORE_MAP = {"Low": 1, "Medium": 2, "High": 3}


@dataclass
class AnalysisArtifacts:
    structured_output: dict[str, Any]
    clauses_df: pd.DataFrame
    obligation_summary_df: pd.DataFrame
    risk_category_summary_df: pd.DataFrame
    heatmap_df: pd.DataFrame
    combined_priority_risks_df: pd.DataFrame


def load_spacy_model():
    if spacy is None:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n".join(parts)


def clean_text(text: str) -> str:
    cleaned = re.sub(r"\uf0a0", " ", text)
    cleaned = re.sub(r"\u00a0", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def segment_into_clauses(text: str) -> list[str]:
    return [clause.strip() for clause in re.split(r"(?<=[.!?]) +|\n+|; +", text) if clause.strip()]


def classify_clause(clause_text: str) -> str:
    clause_text_lower = clause_text.lower()
    best_match_category = "General Provisions"
    max_keyword_matches = 0
    for category, keywords in CLAUSE_CATEGORIES.items():
        if category == "General Provisions":
            continue
        current_keyword_matches = sum(
            1 for keyword in keywords if keyword in clause_text_lower)
        if current_keyword_matches > max_keyword_matches:
            max_keyword_matches = current_keyword_matches
            best_match_category = category
    return best_match_category


def extract_entities(nlp, clause_text: str) -> list[dict[str, Any]]:
    if nlp is None:
        return []
    doc = nlp(clause_text)
    return [{"text": ent.text, "label": ent.label_, "start_char": ent.start_char, "end_char": ent.end_char} for ent in doc.ents]


def infer_level(match_count: int, low_cutoff: int = 1, high_cutoff: int = 4) -> str:
    if match_count >= high_cutoff:
        return "High"
    if match_count >= low_cutoff:
        return "Medium"
    return "Low"


def detect_party_name(clause_text: str, entities: list[dict[str, Any]]) -> str:
    for ent in entities:
        if ent.get("label") == "ORG":
            return ent.get("text", "")
    clause_text_lower = clause_text.lower()
    for party in ["merchant", "company", "affirm", "peloton", "buyer", "seller", "provider", "customer"]:
        if party in clause_text_lower:
            return party.title()
    return "Unspecified Party"


def detect_obligation_type(clause_text_lower: str) -> str:
    if any(word in clause_text_lower for word in ["pay", "payment", "fee", "invoice"]):
        return "Payment Obligation"
    if any(word in clause_text_lower for word in ["confidential", "disclose", "proprietary"]):
        return "Confidentiality Obligation"
    if any(word in clause_text_lower for word in ["comply", "law", "regulation"]):
        return "Compliance Obligation"
    if any(word in clause_text_lower for word in ["deliver", "provide", "perform", "service"]):
        return "Performance Obligation"
    if any(word in clause_text_lower for word in ["notice", "notify"]):
        return "Notice Obligation"
    return "General Obligation"


def extract_obligations(combined_legal_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    obligations = []
    for item in combined_legal_data:
        clause_text = item.get("clause_text", "")
        clause_text_lower = clause_text.lower()
        if any(re.search(pattern, clause_text_lower) for pattern in OBLIGATION_PATTERNS):
            entities = item.get("entities", [])
            obligations.append({"party": detect_party_name(clause_text, entities), "obligation_type": detect_obligation_type(
                clause_text_lower), "category": item.get("category", "General Provisions"), "clause_text": clause_text})
    return obligations


def risk_impact_from_category(category_name: str) -> str:
    if category_name in {"Legal", "Information Security", "Financial", "Intellectual Property"}:
        return "High"
    if category_name in {"Operational", "Compliance", "Strategic External"}:
        return "Medium"
    return "Low"


def _build_risk_explanation(
    clause_text: str,
    top_category: str,
    matched_keywords: list[str],
    likelihood: str,
    impact: str,
    overall_score: int,
    category_hint: str | None,
) -> str:
    """Produce a human-readable explanation of *why* this risk was assigned."""
    parts: list[str] = []

    # 1. Category assignment reason
    if matched_keywords:
        kw_display = ", ".join(f"'{k}'" for k in matched_keywords[:5])
        parts.append(
            f"Classified as **{top_category}** risk because the clause contains "
            f"the keywords {kw_display}."
        )
    elif category_hint:
        parts.append(
            f"Classified as **{top_category}** risk based on the clause-category "
            f"hint '{category_hint}' (no direct keyword matches)."
        )
    else:
        parts.append(
            f"Classified as **{top_category}** risk by default — no strong "
            "keyword signal was detected, warranting manual review."
        )

    # 2. Likelihood reasoning
    parts.append(
        f"Likelihood is **{likelihood}** ({len(matched_keywords)} keyword hit(s); "
        f"≥3 → High, 1-2 → Medium, 0 → Low)."
    )

    # 3. Impact reasoning
    high_cats = {"Legal", "Information Security",
                 "Financial", "Intellectual Property"}
    med_cats = {"Operational", "Compliance", "Strategic External"}
    if top_category in high_cats:
        parts.append(
            f"Impact is **{impact}** because {top_category} risks can directly "
            "affect financial exposure, legal liability, or data security."
        )
    elif top_category in med_cats:
        parts.append(
            f"Impact is **{impact}** because {top_category} risks primarily "
            "affect service continuity or regulatory standing."
        )
    else:
        parts.append(
            f"Impact is **{impact}** — the category does not fall into the "
            "high-impact or medium-impact tiers."
        )

    # 4. Overall score
    parts.append(
        f"Overall risk score = {overall_score} (likelihood × impact).")

    return " ".join(parts)


def hybrid_risk_classifier(clause_text: str, category_hint: str | None = None) -> dict[str, Any]:
    clause_text_lower = clause_text.lower()
    rule_scores = {}
    matched_keywords_map: dict[str, list[str]] = {}
    for category_name, config in RISK_TAXONOMY.items():
        hits = [kw for kw in config["keywords"] if kw in clause_text_lower]
        keyword_hits = len(hits)
        if category_hint and category_hint.lower() in category_name.lower():
            keyword_hits += 1
        rule_scores[category_name] = keyword_hits
        matched_keywords_map[category_name] = hits
    top_category, top_score = max(
        rule_scores.items(), key=lambda item: item[1])
    if top_score == 0:
        top_category = "Operational"
    likelihood = infer_level(top_score, low_cutoff=1, high_cutoff=3)
    impact = risk_impact_from_category(top_category)
    overall_score = SCORE_MAP[likelihood] * SCORE_MAP[impact]
    matched_keywords = matched_keywords_map.get(top_category, [])
    explanation = _build_risk_explanation(
        clause_text, top_category, matched_keywords,
        likelihood, impact, overall_score, category_hint,
    )
    return {"predicted_risk_category": top_category, "likelihood": likelihood, "impact": impact, "overall_score": overall_score, "rule_score": top_score, "llm_review_required": top_score <= 1, "matched_keywords": matched_keywords, "risk_explanation": explanation}


def safe_get_json(url: str, timeout: int = 30) -> Any:
    response = requests.get(url, timeout=timeout, headers={
                            "User-Agent": "contract-risk-intelligence-app"})
    response.raise_for_status()
    return response.json()


def level_from_count(count: int, medium_threshold: int = 1, high_threshold: int = 5) -> str:
    if count >= high_threshold:
        return "High"
    if count >= medium_threshold:
        return "Medium"
    return "Low"


def fetch_noaa_weather_risk(area: str) -> dict[str, Any]:
    url = f"https://api.weather.gov/alerts/active/area/{area}"
    payload = safe_get_json(url)
    severe_terms = ["Tornado", "Hurricane", "Flood",
                    "Storm", "Blizzard", "Fire", "Heat", "Ice"]
    severe_alerts = []
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        event = props.get("event", "")
        if any(term.lower() in event.lower() for term in severe_terms):
            severe_alerts.append({"event": event, "severity": props.get(
                "severity"), "headline": props.get("headline")})
    likelihood = level_from_count(
        len(severe_alerts), medium_threshold=1, high_threshold=3)
    impact = "High" if severe_alerts else "Medium"
    return {"risk_name": "Weather / natural disaster risk", "category": "Strategic External", "source": "NOAA Alerts API", "likelihood": likelihood, "impact": impact, "overall_score": SCORE_MAP[likelihood] * SCORE_MAP[impact], "observed_signal": len(severe_alerts), "signal_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "evidence": severe_alerts[:5]}


def fetch_cisa_kev_risk(vendor_keywords: list[str], product_keywords: list[str]) -> dict[str, Any]:
    url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
    payload = safe_get_json(url)
    matched = []
    for item in payload.get("vulnerabilities", []):
        searchable = f"{item.get('vendorProject', '')} {item.get('product', '')}".lower(
        )
        if any(keyword.lower() in searchable for keyword in vendor_keywords + product_keywords):
            matched.append({"cveID": item.get("cveID"), "vendorProject": item.get(
                "vendorProject"), "product": item.get("product"), "dateAdded": item.get("dateAdded")})
    likelihood = level_from_count(
        len(matched), medium_threshold=1, high_threshold=10)
    impact = "High" if matched else "Medium"
    return {"risk_name": "Cybersecurity risk", "category": "Information Security", "source": "CISA KEV Catalog", "likelihood": likelihood, "impact": impact, "overall_score": SCORE_MAP[likelihood] * SCORE_MAP[impact], "observed_signal": len(matched), "signal_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "evidence": matched[:10]}


def infer_country_codes_from_contract(combined_legal_data: list[dict[str, Any]], document_metadata: dict[str, Any], fallback_codes: list[str] | None = None, max_countries: int = 3) -> list[str]:
    fallback_codes = fallback_codes or ["USA"]
    candidate_counter: Counter[str] = Counter()
    for item in combined_legal_data:
        clause_text_lower = item.get("clause_text", "").lower()
        for country_name, country_code in COUNTRY_NAME_TO_CODE.items():
            if country_name in clause_text_lower:
                candidate_counter[country_code] += 2
        for state_name, country_code in STATE_TO_COUNTRY_CODE.items():
            if state_name in clause_text_lower:
                candidate_counter[country_code] += 1
        for ent in item.get("entities", []):
            entity_text = str(ent.get("text", "")).lower()
            entity_label = ent.get("label", "")
            if entity_label in ["GPE", "LOC"]:
                if entity_text in COUNTRY_NAME_TO_CODE:
                    candidate_counter[COUNTRY_NAME_TO_CODE[entity_text]] += 3
                elif entity_text in STATE_TO_COUNTRY_CODE:
                    candidate_counter[STATE_TO_COUNTRY_CODE[entity_text]] += 1
    parties = " ".join(document_metadata.get("parties", [])).lower()
    if any(suffix in parties for suffix in ["inc", "corporation", "llc"]):
        candidate_counter["USA"] += 1
    inferred_codes = [country_code for country_code,
                      _ in candidate_counter.most_common(max_countries)]
    return inferred_codes if inferred_codes else fallback_codes


def fetch_world_bank_indicator_latest(country_code: str, indicator_code: str) -> dict[str, Any]:
    normalized_country_code = str(country_code).lower()
    candidate_urls = [
        f"https://api.worldbank.org/v2/country/{normalized_country_code}/indicator/{indicator_code}?format=json&per_page=60&mrnev=1",
        f"https://api.worldbank.org/v2/country/{normalized_country_code}/indicator/{indicator_code}?format=json&per_page=60&mrv=5",
        f"https://api.worldbank.org/v2/country/{normalized_country_code}/indicator/{indicator_code}?format=json&per_page=60",
    ]
    last_error = None
    for url in candidate_urls:
        try:
            payload = safe_get_json(url)
            if not isinstance(payload, list) or len(payload) < 2:
                continue
            valid_points = [row for row in payload[1]
                            if row.get("value") is not None]
            if valid_points:
                latest = valid_points[0]
                return {"country_code": country_code, "indicator_code": indicator_code, "value": latest.get("value"), "year": latest.get("date"), "indicator_name": latest.get("indicator", {}).get("value"), "country_name": latest.get("country", {}).get("value"), "url": url}
        except Exception as exc:
            last_error = exc
    raise ValueError(
        f"World Bank lookup failed for {country_code} / {indicator_code}: {last_error}")


def score_macro_risk_from_indicators(inflation_value: Any, gdp_growth_value: Any) -> tuple[str, str, int]:
    signal_points = 0
    if inflation_value is not None:
        inflation_abs = abs(float(inflation_value))
        signal_points += 2 if inflation_abs >= 10 else 1 if inflation_abs >= 5 else 0
    if gdp_growth_value is not None:
        gdp_growth = float(gdp_growth_value)
        signal_points += 2 if gdp_growth < 0 else 1 if gdp_growth < 2 else 0
    likelihood = "High" if signal_points >= 3 else "Medium" if signal_points >= 1 else "Low"
    impact = "High" if signal_points >= 2 else "Medium"
    return likelihood, impact, signal_points


def fetch_world_bank_macro_risk(country_code: str) -> dict[str, Any]:
    inflation = fetch_world_bank_indicator_latest(
        country_code, "FP.CPI.TOTL.ZG")
    gdp_growth = fetch_world_bank_indicator_latest(
        country_code, "NY.GDP.MKTP.KD.ZG")
    likelihood, impact, signal_points = score_macro_risk_from_indicators(
        inflation.get("value"), gdp_growth.get("value"))
    return {"risk_name": f"Macroeconomic / country risk ({inflation['country_name']})", "category": "Strategic External", "source": "World Bank Indicators API", "likelihood": likelihood, "impact": impact, "overall_score": SCORE_MAP[likelihood] * SCORE_MAP[impact], "observed_signal": signal_points, "signal_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "evidence": [{"indicator": inflation["indicator_name"], "value": inflation["value"], "year": inflation["year"]}, {"indicator": gdp_growth["indicator_name"], "value": gdp_growth["value"], "year": gdp_growth["year"]}]}


def collect_internal_risk_evidence(risk_rule: dict[str, Any], combined_legal_data: list[dict[str, Any]]) -> dict[str, Any]:
    matched_clauses = [item.get("clause_text", "") for item in combined_legal_data if any(
        keyword in item.get("clause_text", "").lower() for keyword in risk_rule["keywords"])]
    likelihood = infer_level(len(matched_clauses), low_cutoff=1, high_cutoff=5)
    impact = risk_rule["default_impact"]
    return {"risk_name": risk_rule["risk_name"], "category": risk_rule["category"], "source": "Contract text", "likelihood": likelihood, "impact": impact, "overall_score": SCORE_MAP[likelihood] * SCORE_MAP[impact], "matched_clause_count": len(matched_clauses), "sample_evidence": matched_clauses[:3], "assessment": risk_rule["explanation"]}


def build_document_metadata(combined_legal_data: list[dict[str, Any]], cleaned_text: str) -> dict[str, Any]:
    org_candidates = []
    for item in combined_legal_data[:50]:
        for ent in item.get("entities", []):
            if ent.get("label") == "ORG":
                org_candidates.append(ent.get("text", "").strip())
    return {"document_type": "Contract", "parties": list(dict.fromkeys([candidate for candidate in org_candidates if candidate]))[:5], "text_length": len(cleaned_text), "clause_count": len(combined_legal_data)}


def run_external_risk_analysis(combined_legal_data: list[dict[str, Any]], document_metadata: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    live_external_risks = []
    live_external_errors = []
    try:
        live_external_risks.append(fetch_noaa_weather_risk(
            area=EXTERNAL_RISK_CONTEXT["weather_area"]))
    except Exception as exc:
        live_external_errors.append(
            {"risk_name": "Weather / natural disaster risk", "error": str(exc)})
    try:
        live_external_risks.append(fetch_cisa_kev_risk(
            EXTERNAL_RISK_CONTEXT["cyber_vendor_keywords"], EXTERNAL_RISK_CONTEXT["cyber_product_keywords"]))
    except Exception as exc:
        live_external_errors.append(
            {"risk_name": "Cybersecurity risk", "error": str(exc)})
    for country_code in infer_country_codes_from_contract(combined_legal_data, document_metadata):
        try:
            live_external_risks.append(
                fetch_world_bank_macro_risk(country_code))
        except Exception as exc:
            live_external_errors.append(
                {"country_code": country_code, "error": str(exc)})
    return live_external_risks, live_external_errors


def analyze_contract(pdf_path: str | Path) -> AnalysisArtifacts:
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)
    clauses = segment_into_clauses(cleaned_text)
    nlp = load_spacy_model()
    combined_legal_data = []
    for clause in clauses:
        combined_legal_data.append({"clause_text": clause, "category": classify_clause(
            clause), "entities": extract_entities(nlp, clause)})
    document_metadata = build_document_metadata(
        combined_legal_data, cleaned_text)
    obligations = extract_obligations(combined_legal_data)
    hybrid_clause_risks = []
    for item in combined_legal_data:
        hybrid_clause_risks.append({"clause_text": item["clause_text"], "original_clause_category": item["category"],
                                   **hybrid_risk_classifier(item["clause_text"], category_hint=item["category"])})
    internal_risks = [collect_internal_risk_evidence(
        rule, combined_legal_data) for rule in INTERNAL_RISK_RULES]
    live_external_risks, live_external_errors = run_external_risk_analysis(
        combined_legal_data, document_metadata)
    clauses_df = pd.DataFrame(hybrid_clause_risks)
    obligations_df = pd.DataFrame(obligations)
    risk_category_summary_df = clauses_df.groupby("predicted_risk_category", as_index=False).agg(clause_count=("clause_text", "count"), avg_score=(
        "overall_score", "mean"), llm_review_count=("llm_review_required", "sum")).sort_values(by=["avg_score", "clause_count"], ascending=[False, False])
    obligation_summary_df = obligations_df.groupby(["party", "obligation_type"], as_index=False).size().rename(columns={"size": "obligation_count"}).sort_values(
        by="obligation_count", ascending=False) if not obligations_df.empty else pd.DataFrame(columns=["party", "obligation_type", "obligation_count"])
    heatmap_df = clauses_df.groupby(["likelihood", "impact"]).size().unstack(fill_value=0).reindex(
        index=["Low", "Medium", "High"], columns=["Low", "Medium", "High"], fill_value=0)
    all_live_external_df = pd.DataFrame(live_external_risks)
    combined_priority_risks_df = pd.concat([clauses_df[["predicted_risk_category", "likelihood", "impact", "overall_score"]].rename(columns={"predicted_risk_category": "risk_name"}), all_live_external_df[[
                                           "risk_name", "likelihood", "impact", "overall_score"]] if not all_live_external_df.empty else pd.DataFrame(columns=["risk_name", "likelihood", "impact", "overall_score"])], ignore_index=True).sort_values(by="overall_score", ascending=False)
    structured_output = {"metadata": document_metadata, "full_text": cleaned_text, "clauses": combined_legal_data, "risk_obligation_intelligence": {"taxonomy": RISK_TAXONOMY, "internal_risks": internal_risks, "hybrid_clause_risks": hybrid_clause_risks,
                                                                                                                                                    "obligations": obligations, "risk_category_summary": risk_category_summary_df.to_dict(orient="records"), "obligation_summary": obligation_summary_df.to_dict(orient="records"), "live_external_risks": live_external_risks, "live_external_errors": live_external_errors}}
    return AnalysisArtifacts(structured_output=structured_output, clauses_df=clauses_df, obligation_summary_df=obligation_summary_df, risk_category_summary_df=risk_category_summary_df, heatmap_df=heatmap_df, combined_priority_risks_df=combined_priority_risks_df)


def structured_output_json(artifacts: AnalysisArtifacts) -> str:
    return json.dumps(artifacts.structured_output, indent=2)
