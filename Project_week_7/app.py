import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from contract_intelligence.analysis import AnalysisArtifacts, analyze_contract, structured_output_json
from contract_intelligence.comparison import ComparisonResult, compare_contracts
from contract_intelligence.rag import ContractRetriever, answer_question


st.set_page_config(page_title="Contract Risk Intelligence", layout="wide")
VECTOR_DB_DIR = Path(__file__).resolve().parent / "chroma_db"


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


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
            ax.text(col_index, row_index, int(
                artifacts.heatmap_df.iloc[row_index, col_index]), ha="center", va="center", color="black")
    plt.colorbar(image, ax=ax, shrink=0.8)
    st.pyplot(fig)


def render_structured_report(artifacts: AnalysisArtifacts) -> None:
    metadata = artifacts.structured_output["metadata"]
    risk_block = artifacts.structured_output["risk_obligation_intelligence"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Clauses", metadata.get("clause_count", 0))
    col2.metric("Parties Found", len(metadata.get("parties", [])))
    col3.metric("Live External Risks", len(
        risk_block.get("live_external_risks", [])))

    st.subheader("Top Risk Categories")
    st.dataframe(artifacts.risk_category_summary_df, use_container_width=True)

    selected_risk_category = st.selectbox(
        "Drill down into a risk category",
        options=["Select a category"] +
        artifacts.risk_category_summary_df["predicted_risk_category"].tolist(),
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
            artifacts.obligation_summary_df["party"].fillna(
                "Unspecified Party")
            + " | "
            + artifacts.obligation_summary_df["obligation_type"].fillna("General Obligation")
        ).tolist()
        selected_obligation_label = st.selectbox(
            "Drill down into an obligation summary row",
            options=["Select an obligation"] + obligation_labels,
            index=0,
        )
        if selected_obligation_label != "Select an obligation":
            selected_party, selected_type = selected_obligation_label.split(
                " | ", 1)
            obligations = pd.DataFrame(risk_block.get("obligations", []))
            filtered_obligations = obligations[
                (obligations["party"].fillna(
                    "Unspecified Party") == selected_party)
                & (obligations["obligation_type"].fillna("General Obligation") == selected_type)
            ][["party", "obligation_type", "category", "clause_text"]]
            st.markdown(
                f"**Supporting clauses for {selected_party} / {selected_type}**")
            st.dataframe(filtered_obligations, use_container_width=True)

    st.subheader("Priority Risks")
    st.dataframe(artifacts.combined_priority_risks_df.head(
        20), use_container_width=True)

    st.subheader("Risk Heatmap")
    render_heatmap(artifacts)

    st.subheader("Live External Risks")
    live_external_df = pd.DataFrame(risk_block.get("live_external_risks", []))
    if not live_external_df.empty:
        st.dataframe(live_external_df[["risk_name", "source", "likelihood", "impact",
                     "overall_score", "observed_signal", "signal_date"]], use_container_width=True)
    else:
        st.info("No live external risks were returned.")

    if risk_block.get("live_external_errors"):
        st.subheader("Live External Risk Errors")
        st.dataframe(pd.DataFrame(
            risk_block["live_external_errors"]), use_container_width=True)

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


def render_explainability_panel(artifacts: AnalysisArtifacts) -> None:
    """Show per-clause risk explanations."""
    st.subheader("Risk Explainability — Why Each Risk Was Assigned")
    risks = artifacts.structured_output["risk_obligation_intelligence"]["hybrid_clause_risks"]
    for idx, risk in enumerate(risks, 1):
        explanation = risk.get("risk_explanation", "")
        if not explanation:
            continue
        with st.expander(
            f"Clause {idx}: {risk['predicted_risk_category']} "
            f"(score {risk['overall_score']})",
            expanded=False,
        ):
            st.markdown(f"**Clause text:** {risk['clause_text'][:300]}...")
            keywords = risk.get("matched_keywords", [])
            if keywords:
                st.markdown(f"**Matched keywords:** {', '.join(keywords)}")
            st.markdown(f"**Explanation:** {explanation}")


def render_comparison(comparison: ComparisonResult) -> None:
    """Render the full comparison result between two contracts."""
    st.header(f"Comparison: {comparison.label_a} vs {comparison.label_b}")
    st.markdown(comparison.summary)

    # Hallucination control flags
    if comparison.hallucination_flags:
        st.warning("**Grounding warnings** — the following findings may not be "
                   "fully supported by the source text:")
        for flag in comparison.hallucination_flags:
            st.write(f"- {flag}")

    # Category coverage
    st.subheader("Clause Category Coverage")
    st.dataframe(comparison.category_coverage_df, use_container_width=True)

    # Missing clauses
    st.subheader("Missing Clauses")
    if comparison.missing_clauses:
        for mc in comparison.missing_clauses:
            icon = {"High": "🔴", "Medium": "🟡",
                    "Low": "🟢"}.get(mc.risk_level, "⚪")
            with st.expander(
                f"{icon} {mc.category} — missing from {mc.missing_from} "
                f"({mc.risk_level} risk)",
                expanded=mc.risk_level == "High",
            ):
                st.markdown(f"**Present in:** {mc.present_in}")
                st.markdown(f"**Risk level:** {mc.risk_level}")
                st.markdown(f"**Why this matters:** {mc.explanation}")
                if mc.sample_clauses:
                    st.markdown("**Sample clauses from the other contract:**")
                    for i, clause in enumerate(mc.sample_clauses, 1):
                        st.caption(f"{i}. {clause[:300]}")
    else:
        st.success(
            "No missing clause categories detected between the two contracts.")

    # Deviations
    st.subheader("Material Deviations")
    if comparison.deviations:
        for dev in comparison.deviations:
            with st.expander(
                f"{dev.category} — similarity {dev.similarity_score:.0%}, "
                f"risk delta {dev.risk_delta:+d}",
                expanded=True,
            ):
                st.markdown(f"**Why flagged:** {dev.explanation}")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{comparison.label_a}:**")
                    st.caption(dev.contract_a_text)
                with col2:
                    st.markdown(f"**{comparison.label_b}:**")
                    st.caption(dev.contract_b_text)
    else:
        st.success("No material deviations detected in shared clause categories.")

    # Risk comparison
    st.subheader("Risk Score Comparison")
    if not comparison.risk_comparison_df.empty:
        st.dataframe(comparison.risk_comparison_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        df_plot = comparison.risk_comparison_df.set_index("risk_category")
        col_a = f"{comparison.label_a}_avg_score"
        col_b = f"{comparison.label_b}_avg_score"
        x = range(len(df_plot))
        width = 0.35
        ax.bar([i - width / 2 for i in x], df_plot[col_a],
               width, label=comparison.label_a)
        ax.bar([i + width / 2 for i in x], df_plot[col_b],
               width, label=comparison.label_b)
        ax.set_xticks(list(x))
        ax.set_xticklabels(df_plot.index, rotation=45, ha="right")
        ax.set_ylabel("Avg Risk Score")
        ax.set_title("Risk Score by Category")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    # Export
    st.subheader("Export Comparison")
    export_col1, export_col2 = st.columns(2)
    export_col1.download_button(
        label="Download Category Coverage (CSV)",
        data=dataframe_to_csv_bytes(comparison.category_coverage_df),
        file_name="category_coverage.csv",
        mime="text/csv",
    )
    if not comparison.risk_comparison_df.empty:
        export_col2.download_button(
            label="Download Risk Comparison (CSV)",
            data=dataframe_to_csv_bytes(comparison.risk_comparison_df),
            file_name="risk_comparison.csv",
            mime="text/csv",
        )


def main() -> None:
    st.title("Contract Risk and Obligation Intelligence")

    tab_single, tab_compare = st.tabs(
        ["Single Contract Analysis", "Compare Two Contracts"])

    # ---- Tab 1: Single contract analysis (existing behaviour) ----
    with tab_single:
        st.write(
            "Upload a contract PDF to generate a structured risk report and ask grounded questions over the document.")

        uploaded_file = st.file_uploader("Upload contract PDF", type=[
                                         "pdf"], key="single_upload")
        if uploaded_file and st.button("Run Analysis", type="primary", key="run_single"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_pdf_path = Path(temp_file.name)
            with st.spinner("Running contract analysis..."):
                artifacts = analyze_contract(temp_pdf_path)
                st.session_state["artifacts"] = artifacts
                st.session_state["retriever"] = ContractRetriever(
                    artifacts.structured_output["clauses"],
                    db_dir=VECTOR_DB_DIR,
                    collection_name="uploaded_contract",
                )

        artifacts = st.session_state.get("artifacts")
        if artifacts is None:
            st.info("Upload a PDF and run the analysis to see the structured result.")
        else:
            render_structured_report(artifacts)

            render_explainability_panel(artifacts)

            st.subheader("Ask Questions About the Contract")
            question = st.text_input("Ask a grounded question",
                                     placeholder="What are the termination risks?")
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
                    st.write(
                        f"{idx}. [{item['category']}] score={item['score']:.3f}")
                    st.caption(item["clause_text"])

    # ---- Tab 2: Contract comparison ----
    with tab_compare:
        st.write(
            "Upload **two** contract PDFs to compare them side-by-side. "
            "The system will detect missing clauses, material deviations, "
            "and explain the risk differences."
        )
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            label_a = st.text_input(
                "Contract A label", value="Contract A", key="label_a")
            upload_a = st.file_uploader("Upload Contract A", type=[
                                        "pdf"], key="compare_upload_a")
        with col_up2:
            label_b = st.text_input(
                "Contract B label", value="Contract B", key="label_b")
            upload_b = st.file_uploader("Upload Contract B", type=[
                                        "pdf"], key="compare_upload_b")

        if upload_a and upload_b and st.button("Compare Contracts", type="primary", key="run_compare"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_a:
                tmp_a.write(upload_a.getbuffer())
                path_a = Path(tmp_a.name)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_b:
                tmp_b.write(upload_b.getbuffer())
                path_b = Path(tmp_b.name)

            with st.spinner("Analysing Contract A..."):
                artifacts_a = analyze_contract(path_a)
            with st.spinner("Analysing Contract B..."):
                artifacts_b = analyze_contract(path_b)
            with st.spinner("Computing comparison..."):
                comparison = compare_contracts(
                    artifacts_a, artifacts_b, label_a, label_b)
                st.session_state["comparison"] = comparison
                st.session_state["artifacts_a"] = artifacts_a
                st.session_state["artifacts_b"] = artifacts_b

        comparison = st.session_state.get("comparison")
        if comparison is None:
            st.info("Upload two PDFs and click 'Compare Contracts' to start.")
        else:
            render_comparison(comparison)

            # Show explainability panels for both contracts
            st.subheader(f"Explainability — {comparison.label_a}")
            render_explainability_panel(st.session_state["artifacts_a"])
            st.subheader(f"Explainability — {comparison.label_b}")
            render_explainability_panel(st.session_state["artifacts_b"])


if __name__ == "__main__":
    main()
