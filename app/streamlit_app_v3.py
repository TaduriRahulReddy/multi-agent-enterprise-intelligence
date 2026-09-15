import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from orchestration.orchestrator_v15 import run_orchestrator


# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Multi-Agent Enterprise Intelligence",
    page_icon="🤖",
    layout="wide",
)


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

st.title("Multi-Agent Enterprise Intelligence System")

st.markdown(
    """
    Ask an enterprise business question and the system will automatically
    route it to the appropriate specialized agent.

    **Available capabilities**
    - Structured data analytics
    - Customer insights
    - Enterprise document research
    - Retrieval-Augmented Generation
    - Grounded executive reporting
    - Automated claim validation
    """
)


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

with st.sidebar:
    st.header("System Architecture")

    st.markdown(
        """
        **Planner Agent**  
        Routes each question.

        **Data Analyst Agent**  
        Performs structured SQL analytics.

        **Customer Insights Agent**  
        Handles customer-focused analysis.

        **Research Agent**  
        Retrieves and validates document evidence.

        **Business Insights Agent**  
        Generates business interpretation.

        **Reporting Agent**  
        Produces grounded executive reports.

        **Validation Layer**  
        Checks factual and numerical claims.
        """
    )

    st.divider()

    st.caption("Backend: Orchestrator V15")
    st.caption("Report Validator: V17")
    st.caption("Local LLM: Qwen 2.5 7B")


# -------------------------------------------------------------------
# Example questions
# -------------------------------------------------------------------

st.subheader("Example Questions")

example_col1, example_col2 = st.columns(2)

with example_col1:
    st.markdown(
        """
        **Structured Analytics**
        - What are the top 5 product categories by total product revenue?
        - How many repeat customers do we have?
        """
    )

with example_col2:
    st.markdown(
        """
        **Enterprise Research**
        - What pricing changes were introduced in Q2?
        - What customer problems were reported in the quarterly report?
        """
    )


# -------------------------------------------------------------------
# User input
# -------------------------------------------------------------------

st.divider()

question = st.text_area(
    "Ask an enterprise question",
    placeholder="Example: What are the top 5 product categories by total product revenue?",
    height=100,
)

run_button = st.button(
    "Run Multi-Agent Analysis",
    type="primary",
    use_container_width=True,
)


# -------------------------------------------------------------------
# Execute orchestrator
# -------------------------------------------------------------------

if run_button:

    if not question.strip():
        st.warning("Please enter a question before running the analysis.")

    else:
        with st.spinner("Agents are analyzing your question..."):

            try:
                result = run_orchestrator(question.strip())

            except Exception as exc:
                st.error("The multi-agent system encountered an error.")
                st.exception(exc)

            else:

                st.success("Analysis completed successfully.")

                # ---------------------------------------------------
                # Routing information
                # ---------------------------------------------------

                st.subheader("Agent Routing")

                selected_agent = result.get("selected_agent", "Unknown")

                st.info(f"Selected Agent: {selected_agent}")

                # ---------------------------------------------------
                # Main results
                # ---------------------------------------------------

                st.subheader("Analysis Results")

                analytical_result = result.get("analytical_result")

                research_result = result.get("research_result")

                business_insights = result.get("business_insights")

                report = result.get("executive_report")

                if analytical_result is not None:
                    st.markdown("#### Structured Analytical Result")

                    try:
                        st.dataframe(
                            analytical_result,
                            use_container_width=True,
                        )
                    except Exception:
                        st.write(analytical_result)

                if research_result:
                    st.markdown("#### Research Result")
                    st.write(research_result)

                if business_insights:
                    st.markdown("#### Business Insights")
                    st.write(business_insights)

                # ---------------------------------------------------
                # Grounded executive report
                # ---------------------------------------------------

                st.subheader("Grounded Executive Report")

                if report:
                    st.markdown(report)
                else:
                    st.info(
                        "No executive report was generated for this request."
                    )

                # ---------------------------------------------------
                # Validation information
                # ---------------------------------------------------

                st.subheader("Validation & Reliability")

                validation_col1, validation_col2 = st.columns(2)

                report_attempts = result.get(
                    "report_validation_attempts"
                )

                research_attempts = result.get(
                    "research_validation_attempts"
                )

                with validation_col1:
                    st.metric(
                        "Report Validation Attempts",
                        report_attempts
                        if report_attempts is not None
                        else "N/A",
                    )

                with validation_col2:
                    st.metric(
                        "Research Validation Attempts",
                        research_attempts
                        if research_attempts is not None
                        else "N/A",
                    )

                # ---------------------------------------------------
                # Debug / system details
                # ---------------------------------------------------

                with st.expander("System Output Details"):
                    st.write(result)
