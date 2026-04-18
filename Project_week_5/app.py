import tempfile
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from contract_intelligence.analysis import AnalysisArtifacts, analyze_contract, structured_output_json
from contract_intelligence.rag import ContractRetriever, answer_question


st.set_page_config(page_title="Contract Risk Intelligence", layout="wide")
VECTOR_DB_DIR = Path(__file__).resolve().parent / "chroma_db"


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def annotate_clauses_with_document(clauses: list[dict], document_name: str) -> list[dict]:
    return [{**clause, "document_name": document_name} for clause in clauses]


def combine_analysis_artifacts(artifacts_list: list[AnalysisArtifacts], document_names: list[str]) -> AnalysisArtifacts:
    combined_clause_records = []
    for artifacts, document_name in zip(artifacts_list, document_names):
        combined_clause_records.extend(
            annotate_clauses_with_document(artifacts.structured_output["clauses"], document_name)
        )

    combined_clauses_df = pd.concat(
        [artifact.clauses_df.assign(document_name=document_name) for artifact, document_name in zip(artifacts_list, document_names)],
        ignore_index=True,
    ) if artifacts_list else pd.DataFrame()

    combined_obligation_summary_df = pd.concat(
        [artifact.obligation_summary_df.assign(document_name=document_name) for artifact, document_name in zip(artifacts_list, document_names)],
        ignore_index=True,
    ) if any(not artifact.obligation_summary_df.empty for artifact in artifacts_list) else pd.DataFrame(columns=["party", "obligation_type", "obligation_count", "document_name"])

    if not combined_obligation_summary_df.empty:
        combined_obligation_summary_df = (
            combined_obligation_summary_df.groupby(["party", "obligation_type"], dropna=False, as_index=False)["obligation_count"]
            .sum()
            .sort_values(by="obligation_count", ascending=False)
        )

    if not combined_clauses_df.empty:
        risk_category_summary_df = combined_clauses_df.groupby("predicted_risk_category", as_index=False).agg(
            clause_count=("clause_text", "count"),
            avg_score=("overall_score", "mean"),
            llm_review_count=("llm_review_required", "sum"),
        ).sort_values(by=["avg_score", "clause_count"], ascending=[False, False])
        heatmap_df = (
            combined_clauses_df.groupby(["likelihood", "impact"]).size().unstack(fill_value=0)
            .reindex(index=["Low", "Medium", "High"], columns=["Low", "Medium", "High"], fill_value=0)
        )
        combined_priority_risks_df = combined_clauses_df[["predicted_risk_category", "likelihood", "impact", "overall_score"]].rename(columns={"predicted_risk_category": "risk_name"})
    else:
        risk_category_summary_df = pd.DataFrame(columns=["predicted_risk_category", "clause_count", "avg_score", "llm_review_count"])
        heatmap_df = pd.DataFrame(0, index=["Low", "Medium", "High"], columns=["Low", "Medium", "High"])
        combined_priority_risks_df = pd.DataFrame(columns=["risk_name", "likelihood", "impact", "overall_score"])

    all_live_external_risks = []
    all_live_external_errors = []
    for artifacts in artifacts_list:
        risk_block = artifacts.structured_output["risk_obligation_intelligence"]
        all_live_external_risks.extend(risk_block.get("live_external_risks", []) or [])
        all_live_external_errors.extend(risk_block.get("live_external_errors", []) or [])

    if not all_live_external_risks:
        all_live_external_risks = []

    if all_live_external_risks:
        external_risks_df = pd.DataFrame(all_live_external_risks)[["risk_name", "likelihood", "impact", "overall_score"]]
        combined_priority_risks_df = pd.concat([combined_priority_risks_df, external_risks_df], ignore_index=True)

    combined_priority_risks_df = combined_priority_risks_df.sort_values(by="overall_score", ascending=False)

    structured_output = {
        "metadata": {
            "document_count": len(document_names),
            "documents": document_names,
            "clause_count": len(combined_clause_records),
            "parties": list(dict.fromkeys(
                party
                for artifact in artifacts_list
                for party in artifact.structured_output.get("metadata", {}).get("parties", [])
            )),
        },
        "full_text": "\n\n".join(
            artifact.structured_output.get("full_text", "") for artifact in artifacts_list
        ),
        "clauses": combined_clause_records,
        "risk_obligation_intelligence": {
            "taxonomy": artifacts_list[0].structured_output["risk_obligation_intelligence"]["taxonomy"] if artifacts_list else {},
            "internal_risks": artifacts_list[0].structured_output["risk_obligation_intelligence"]["internal_risks"] if artifacts_list else [],
            "hybrid_clause_risks": [
                {**row, "document_name": doc_name}
                for artifact, doc_name in zip(artifacts_list, document_names)
                for row in artifact.structured_output["risk_obligation_intelligence"]["hybrid_clause_risks"]
            ],
            "obligations": [
                {**item, "document_name": doc_name}
                for artifact, doc_name in zip(artifacts_list, document_names)
                for item in artifact.structured_output["risk_obligation_intelligence"]["obligations"]
            ],
            "risk_category_summary": risk_category_summary_df.to_dict(orient="records"),
            "obligation_summary": combined_obligation_summary_df.to_dict(orient="records"),
            "live_external_risks": all_live_external_risks,
            "live_external_errors": all_live_external_errors,
        },
    }

    return AnalysisArtifacts(
        structured_output=structured_output,
        clauses_df=combined_clauses_df,
        obligation_summary_df=combined_obligation_summary_df,
        risk_category_summary_df=risk_category_summary_df,
        heatmap_df=heatmap_df,
        combined_priority_risks_df=combined_priority_risks_df,
    )


def render_heatmap(artifacts: AnalysisArtifacts) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    image = ax.imshow(artifacts.heatmap_df.values, cmap="YlOrRd")
    ax.set_xticks(range(len(artifacts.heatmap_df.columns)))
    ax.set_yticks(range(len(artifacts.heatmap_df.index)))
    ax.set_xticklabels(artifacts.heatmap_df.columns)
    ax.set_yticklabels(artifacts.heatmap_df.index)
    ax.set_xlabel("Impact")
    ax.set_ylabel("Likelihood")
    ax.set_title("Contract Risk Heatmap")
    for row_index in range(artifacts.heatmap_df.shape[0]):
        for col_index in range(artifacts.heatmap_df.shape[1]):
            ax.text(col_index, row_index, int(artifacts.heatmap_df.iloc[row_index, col_index]), ha="center", va="center", color="black")
    plt.colorbar(image, ax=ax, shrink=0.8)
    st.pyplot(fig)


def render_structured_report(artifacts: AnalysisArtifacts) -> None:
    metadata = artifacts.structured_output["metadata"]
    risk_block = artifacts.structured_output["risk_obligation_intelligence"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Clauses", metadata.get("clause_count", 0))
    col2.metric("Parties Found", len(metadata.get("parties", [])))
    col3.metric("Live External Risks", len(risk_block.get("live_external_risks", [])))

    if metadata.get("documents"):
        st.markdown(f"**Documents processed:** {', '.join(metadata['documents'])}")

    st.subheader("Top Risk Categories")
    st.dataframe(artifacts.risk_category_summary_df, use_container_width=True)

    selected_risk_category = st.selectbox(
        "Drill down into a risk category",
        options=["Select a category"] + artifacts.risk_category_summary_df["predicted_risk_category"].tolist(),
        index=0,
    )
    if selected_risk_category != "Select a category":
        filtered_risks = artifacts.clauses_df[
            artifacts.clauses_df["predicted_risk_category"] == selected_risk_category
        ][["predicted_risk_category", "likelihood", "impact", "overall_score", "clause_text"]]
        st.markdown(f"**Clauses for {selected_risk_category}**")
        st.dataframe(filtered_risks, use_container_width=True)

    st.subheader("Obligation Summary")
    st.dataframe(artifacts.obligation_summary_df, use_container_width=True)

    if not artifacts.obligation_summary_df.empty:
        obligation_labels = (
            artifacts.obligation_summary_df["party"].fillna("Unspecified Party")
            + " | "
            + artifacts.obligation_summary_df["obligation_type"].fillna("General Obligation")
        ).tolist()
        selected_obligation_label = st.selectbox(
            "Drill down into an obligation summary row",
            options=["Select an obligation"] + obligation_labels,
            index=0,
        )
        if selected_obligation_label != "Select an obligation":
            selected_party, selected_type = selected_obligation_label.split(" | ", 1)
            obligations = pd.DataFrame(risk_block.get("obligations", []))
            filtered_obligations = obligations[
                (obligations["party"].fillna("Unspecified Party") == selected_party)
                & (obligations["obligation_type"].fillna("General Obligation") == selected_type)
            ][["party", "obligation_type", "category", "clause_text"]]
            st.markdown(f"**Supporting clauses for {selected_party} / {selected_type}**")
            st.dataframe(filtered_obligations, use_container_width=True)

    st.subheader("Priority Risks")
    st.dataframe(artifacts.combined_priority_risks_df.head(20), use_container_width=True)

    st.subheader("Risk Heatmap")
    render_heatmap(artifacts)

    st.subheader("Live External Risks")
    live_external_df = pd.DataFrame(risk_block.get("live_external_risks", []))
    if not live_external_df.empty:
        st.dataframe(live_external_df[["risk_name", "source", "likelihood", "impact", "overall_score", "observed_signal", "signal_date"]], use_container_width=True)
    else:
        st.info("No live external risks were returned.")

    if risk_block.get("live_external_errors"):
        st.subheader("Live External Risk Errors")
        st.dataframe(pd.DataFrame(risk_block["live_external_errors"]), use_container_width=True)

    st.subheader("Structured JSON Output")
    structured_json = structured_output_json(artifacts)
    with st.expander("View Structured JSON", expanded=False):
        st.code(structured_json, language="json")
        st.download_button(
            label="Download Structured Output (JSON)",
            data=structured_json.encode("utf-8"),
            file_name="contract_structured_output.json",
            mime="application/json",
        )

    st.subheader("Export Tables")
    export_col1, export_col2, export_col3 = st.columns(3)
    export_col1.download_button(
        label="Download Risk Summary (CSV)",
        data=dataframe_to_csv_bytes(artifacts.risk_category_summary_df),
        file_name="risk_category_summary.csv",
        mime="text/csv",
    )
    export_col2.download_button(
        label="Download Obligation Summary (CSV)",
        data=dataframe_to_csv_bytes(artifacts.obligation_summary_df),
        file_name="obligation_summary.csv",
        mime="text/csv",
    )
    export_col3.download_button(
        label="Download Priority Risks (CSV)",
        data=dataframe_to_csv_bytes(artifacts.combined_priority_risks_df),
        file_name="priority_risks.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("Contract Risk and Obligation Intelligence")
    st.write("Upload one or more contract PDFs to generate a structured risk report and ask grounded questions over the documents.")

    uploaded_files = st.file_uploader("Upload contract PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_files and st.button("Run Analysis", type="primary"):
        artifacts_list = []
        document_names = []
        contract_clauses = []

        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_pdf_path = Path(temp_file.name)
            document_name = uploaded_file.name
            document_names.append(document_name)
            with st.spinner(f"Processing {document_name}..."):
                contract_artifacts = analyze_contract(temp_pdf_path)
                artifacts_list.append(contract_artifacts)
                contract_clauses.extend(
                    annotate_clauses_with_document(contract_artifacts.structured_output["clauses"], document_name)
                )
            os.unlink(temp_pdf_path)

        if artifacts_list:
            combined_artifacts = combine_analysis_artifacts(artifacts_list, document_names)
            st.session_state["artifacts"] = combined_artifacts
            st.session_state["retriever"] = ContractRetriever(
                combined_artifacts.structured_output["clauses"],
                db_dir=VECTOR_DB_DIR,
                collection_name="uploaded_contracts",
            )
            st.success(f"Processed {len(artifacts_list)} contract(s): {', '.join(document_names)}")

    artifacts = st.session_state.get("artifacts")
    if artifacts is None:
        st.info("Upload at least one PDF and run the analysis to see the structured result.")
        return

    render_structured_report(artifacts)

    st.subheader("Ask Questions About the Contracts")
    question = st.text_input("Ask a grounded question", placeholder="What are the termination risks?")
    if question and st.button("Answer Question"):
        with st.spinner("Retrieving evidence and generating answer..."):
            result = answer_question(
                question,
                st.session_state["retriever"],
                artifacts.structured_output["risk_obligation_intelligence"]["risk_category_summary"],
            )
        st.markdown("**Answer**")
        st.write(result.answer)
        st.markdown("**Retrieved Clauses**")
        for idx, item in enumerate(result.contexts, start=1):
            st.write(f"{idx}. [{item['document_name']}] [{item['category']}] score={item['score']:.3f}")
            st.caption(item["clause_text"])

    # Golden Dataset Testing Section
    st.subheader("Golden Dataset Testing")
    st.write("Run baseline questions from the golden dataset to evaluate RAG performance.")
    if st.button("Run Golden Tests", type="secondary"):
        if "artifacts" not in st.session_state:
            st.error("Please upload and analyze at least one contract first.")
        else:
            with st.spinner("Running golden dataset tests..."):
                # Import and run tests
                from test_rag import run_golden_tests, compute_basic_metrics

                # Save the first uploaded file temporarily for testing
                first_uploaded_file = uploaded_files[0]
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(first_uploaded_file.getbuffer())
                    temp_pdf_path = Path(temp_file.name)

                try:
                    results = run_golden_tests(temp_pdf_path)
                    metrics = compute_basic_metrics(results)

                    st.success("Tests completed!")
                    st.metric("Total Questions", metrics["total_questions"])
                    st.metric("Retrieval Success Rate", f"{metrics['retrieval_success_rate']:.2%}")
                    st.metric("Avg Answer Similarity", f"{metrics['average_answer_similarity']:.2f}")

                    # Display results table
                    results_df = pd.DataFrame(results)
                    with st.expander("View Detailed Test Results", expanded=False):
                        st.dataframe(results_df, use_container_width=True)
                        st.download_button(
                            label="Download Test Results (CSV)",
                            data=dataframe_to_csv_bytes(results_df),
                            file_name="golden_test_results.csv",
                            mime="text/csv",
                        )
                finally:
                    os.unlink(temp_pdf_path)


if __name__ == "__main__":
    main()
