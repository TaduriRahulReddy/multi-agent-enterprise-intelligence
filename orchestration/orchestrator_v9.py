from agents.planner_agent_v4 import plan
from agents.data_analyst_agent_v4 import analyze
from agents.customer_insights_agent_v2 import analyze_customer_question
from agents.research_agent_v2 import research
from agents.business_insights_agent import generate_insights
from agents.reporting_agent_v3 import generate_grounded_report


# ============================================================
# MULTI-AGENT ORCHESTRATOR - VERSION 9
# DATA + CUSTOMER + RAG RESEARCH ROUTING
# ============================================================


def run_orchestrator(question: str):
    """
    Full enterprise intelligence workflow.

    User Question
        ↓
    Planner Agent V4
        ↓
    Specialist Routing
        ├── Data Analyst V4
        ├── Customer Insights V2
        └── Research Agent V2
                ↓
        Specialist Result
                ↓
        Final Business / Research Output
    """

    print("\n" + "=" * 70)
    print("MULTI-AGENT ENTERPRISE INTELLIGENCE SYSTEM")
    print("ORCHESTRATOR VERSION 9")
    print("=" * 70)

    print("\nUser Question:")
    print(question)

    # --------------------------------------------------------
    # STEP 1: PLANNER
    # --------------------------------------------------------

    print(
        "\nPlanner Agent V4 is creating "
        "an execution plan..."
    )

    execution_plan = plan(
        question
    )

    print("\nExecution Plan:")
    print(
        execution_plan
    )

    selected_agent = execution_plan.get(
        "agent",
        "unsupported",
    )

    task = execution_plan.get(
        "task",
        question,
    )

    requires_database = execution_plan.get(
        "requires_database",
        False,
    )

    print("\nSelected Agent:")
    print(
        selected_agent
    )

    print("\nPlanned Task:")
    print(
        task
    )

    print("\nRequires Database:")
    print(
        requires_database
    )

    # ========================================================
    # DATA ANALYST ROUTE
    # ========================================================

    if selected_agent == "data_analyst":

        print(
            "\nRouting request to "
            "Data Analyst Agent Version 4..."
        )

        analytical_result = analyze(
            question
        )

        if analytical_result is None:

            print(
                "\nData Analyst Agent did not return "
                "a validated result."
            )

            return {
                "plan": execution_plan,
                "selected_agent": selected_agent,
                "analytical_result": None,
                "business_insights": None,
                "executive_report": None,
                "research_result": None,
            }

        # ----------------------------------------------------
        # Business insights
        # ----------------------------------------------------

        print(
            "\nRouting validated analytical result "
            "to Business Insights Agent..."
        )

        business_insights = generate_insights(
            question=question,
            analytical_result=analytical_result,
        )

        # ----------------------------------------------------
        # Grounded reporting
        # ----------------------------------------------------

        print(
            "\nRouting analytical result and business "
            "insights to Reporting Agent Version 3..."
        )

        executive_report = generate_grounded_report(
            question=question,
            analytical_result=analytical_result,
            business_insights=business_insights,
        )

        return {
            "plan": execution_plan,
            "selected_agent": selected_agent,
            "analytical_result": analytical_result,
            "business_insights": business_insights,
            "executive_report": executive_report,
            "research_result": None,
        }

    # ========================================================
    # CUSTOMER INSIGHTS ROUTE
    # ========================================================

    elif selected_agent == "customer_insights":

        print(
            "\nRouting request to "
            "Customer Insights Agent Version 2..."
        )

        analytical_result = analyze_customer_question(
            question
        )

        if analytical_result is None:

            print(
                "\nCustomer Insights Agent did not return "
                "a validated result."
            )

            return {
                "plan": execution_plan,
                "selected_agent": selected_agent,
                "analytical_result": None,
                "business_insights": None,
                "executive_report": None,
                "research_result": None,
            }

        # ----------------------------------------------------
        # Business insights
        # ----------------------------------------------------

        print(
            "\nRouting validated customer result "
            "to Business Insights Agent..."
        )

        business_insights = generate_insights(
            question=question,
            analytical_result=analytical_result,
        )

        # ----------------------------------------------------
        # Grounded reporting
        # ----------------------------------------------------

        print(
            "\nRouting customer result and business "
            "insights to Reporting Agent Version 3..."
        )

        executive_report = generate_grounded_report(
            question=question,
            analytical_result=analytical_result,
            business_insights=business_insights,
        )

        return {
            "plan": execution_plan,
            "selected_agent": selected_agent,
            "analytical_result": analytical_result,
            "business_insights": business_insights,
            "executive_report": executive_report,
            "research_result": None,
        }

    # ========================================================
    # RESEARCH / RAG ROUTE
    # ========================================================

    elif selected_agent == "research":

        print(
            "\nRouting request to "
            "Research Agent Version 2..."
        )

        research_output = research(
            question=question,
            top_k=3,
        )

        research_answer = research_output.get(
            "answer"
        )

        if not research_answer:

            print(
                "\nResearch Agent did not return "
                "an evidence-based answer."
            )

            return {
                "plan": execution_plan,
                "selected_agent": selected_agent,
                "analytical_result": None,
                "business_insights": None,
                "executive_report": None,
                "research_result": None,
            }

        print(
            "\nResearch Agent completed "
            "document-grounded analysis."
        )

        return {
            "plan": execution_plan,
            "selected_agent": selected_agent,
            "analytical_result": None,
            "business_insights": None,
            "executive_report": None,
            "research_result": research_answer,
        }

    # ========================================================
    # UNSUPPORTED ROUTE
    # ========================================================

    else:

        print(
            "\nThe selected route is currently unsupported."
        )

        return {
            "plan": execution_plan,
            "selected_agent": selected_agent,
            "analytical_result": None,
            "business_insights": None,
            "executive_report": None,
            "research_result": None,
        }


# ============================================================
# TEST ORCHESTRATOR VERSION 9
# ============================================================

if __name__ == "__main__":

    print(
        "Testing Multi-Agent "
        "Orchestrator Version 9"
    )

    test_questions = [
        (
            "What are the top 5 product categories "
            "by total product revenue?"
        ),
        (
            "How many repeat customers do we have?"
        ),
        (
            "What pricing changes were introduced in Q2?"
        ),
        (
            "What customer problems were reported "
            "in the quarterly report?"
        ),
    ]

    for question in test_questions:

        print("\n\n" + "#" * 70)
        print("NEW TEST QUESTION")
        print("#" * 70)

        output = run_orchestrator(
            question
        )

        print("\n" + "=" * 70)
        print("FINAL ORCHESTRATOR OUTPUT")
        print("=" * 70)

        print("\nExecution Plan:")
        print(
            output["plan"]
        )

        print("\nSelected Specialist:")
        print(
            output["selected_agent"]
        )

        # ----------------------------------------------------
        # Structured analytical route output
        # ----------------------------------------------------

        if output["analytical_result"] is not None:

            print(
                "\nValidated Analytical Result:"
            )

            print(
                output["analytical_result"]
            )

            print(
                "\nBusiness Insights:"
            )

            print("-" * 70)

            print(
                output["business_insights"]
            )

            print("-" * 70)

            print(
                "\nFinal Grounded Executive Report:"
            )

            print("-" * 70)

            print(
                output["executive_report"]
            )

            print("-" * 70)

        # ----------------------------------------------------
        # Research route output
        # ----------------------------------------------------

        elif output["research_result"] is not None:

            print(
                "\nEvidence-Grounded Research Result:"
            )

            print("-" * 70)

            print(
                output["research_result"]
            )

            print("-" * 70)

        else:

            print(
                "\nNo final result was produced."
            )

        # ----------------------------------------------------
        # Test success
        # ----------------------------------------------------

        if (
            output["analytical_result"] is not None
            or output["research_result"] is not None
        ):

            print(
                "\nTest completed successfully."
            )

        else:

            print(
                "\nTest did not complete successfully."
            )

    print(
        "\nMulti-Agent Orchestrator Version 9 "
        "testing completed."
    )