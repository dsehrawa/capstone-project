import tempfile
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
    st.write("Upload a contract PDF to generate a structured risk report and ask grounded questions over the document.")

    uploaded_file = st.file_uploader("Upload contract PDF", type=["pdf"])
    if uploaded_file and st.button("Run Analysis", type="primary"):
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
        return

    render_structured_report(artifacts)

    st.subheader("Ask Questions About the Contract")
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
            st.write(f"{idx}. [{item['category']}] score={item['score']:.3f}")
            st.caption(item["clause_text"])


if __name__ == "__main__":
    main()
