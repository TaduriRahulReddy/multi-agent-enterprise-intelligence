from agents.planner_agent_v3 import plan
from agents.data_analyst_agent_v3 import analyze
from agents.customer_insights_agent_v2 import analyze_customer_question
from agents.business_insights_agent import generate_insights
from agents.reporting_agent import generate_report


# ============================================================
# MULTI-AGENT ORCHESTRATOR - VERSION 6
# MULTI-SPECIALIST + BUSINESS INSIGHTS + EXECUTIVE REPORTING
# ============================================================


def run_orchestrator(question: str):
    """
    Full multi-agent enterprise intelligence workflow.

    User Question
        ↓
    Planner Agent V3
        ↓
    Structured Execution Plan
        ↓
    Specialist Agent
        ↓
    Validated Analytical Result
        ↓
    Business Insights Agent
        ↓
    Reporting Agent
        ↓
    Executive Report
    """

    print("\n" + "=" * 70)
    print("MULTI-AGENT ENTERPRISE INTELLIGENCE SYSTEM")
    print("ORCHESTRATOR VERSION 6")
    print("=" * 70)

    print("\nUser Question:")
    print(question)

    # --------------------------------------------------------
    # STEP 1: PLANNER
    # --------------------------------------------------------

    print("\nPlanner Agent V3 is creating an execution plan...")

    execution_plan = plan(question)

    print("\nExecution Plan:")
    print(execution_plan)

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
    print(selected_agent)

    print("\nPlanned Task:")
    print(task)

    print("\nRequires Database:")
    print(requires_database)

    # --------------------------------------------------------
    # STEP 2: DATA ANALYST ROUTE
    # --------------------------------------------------------

    if selected_agent == "data_analyst":

        print(
            "\nRouting request to "
            "Data Analyst Agent Version 3..."
        )

        analytical_result = analyze(question)

    # --------------------------------------------------------
    # STEP 3: CUSTOMER INSIGHTS ROUTE
    # --------------------------------------------------------

    elif selected_agent == "customer_insights":

        print(
            "\nRouting request to "
            "Customer Insights Agent Version 2..."
        )

        analytical_result = analyze_customer_question(
            question
        )

    # --------------------------------------------------------
    # STEP 4: UNSUPPORTED ROUTE
    # --------------------------------------------------------

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
        }

    # --------------------------------------------------------
    # STEP 5: CHECK SPECIALIST RESULT
    # --------------------------------------------------------

    if analytical_result is None:

        print(
            "\nThe specialist agent did not return "
            "a validated analytical result."
        )

        return {
            "plan": execution_plan,
            "selected_agent": selected_agent,
            "analytical_result": None,
            "business_insights": None,
            "executive_report": None,
        }

    # --------------------------------------------------------
    # STEP 6: BUSINESS INSIGHTS
    # --------------------------------------------------------

    print(
        "\nRouting validated analytical result to "
        "Business Insights Agent..."
    )

    business_insights = generate_insights(
        question=question,
        analytical_result=analytical_result,
    )

    # --------------------------------------------------------
    # STEP 7: REPORTING AGENT
    # --------------------------------------------------------

    print(
        "\nRouting analytical result and business insights "
        "to Reporting Agent Version 1..."
    )

    executive_report = generate_report(
        question=question,
        analytical_result=analytical_result,
        business_insights=business_insights,
    )

    # --------------------------------------------------------
    # STEP 8: RETURN FINAL OBJECT
    # --------------------------------------------------------

    return {
        "plan": execution_plan,
        "selected_agent": selected_agent,
        "analytical_result": analytical_result,
        "business_insights": business_insights,
        "executive_report": executive_report,
    }


# ============================================================
# TEST ORCHESTRATOR VERSION 6
# ============================================================

if __name__ == "__main__":

    print(
        "Testing Multi-Agent "
        "Orchestrator Version 6"
    )

    test_questions = [
        (
            "What are the top 5 product categories "
            "by total product revenue?"
        ),
        (
            "How many repeat customers do we have?"
        ),
    ]

    for question in test_questions:

        print("\n\n" + "#" * 70)
        print("NEW TEST QUESTION")
        print("#" * 70)

        output = run_orchestrator(question)

        print("\n" + "=" * 70)
        print("FINAL ORCHESTRATOR OUTPUT")
        print("=" * 70)

        print("\nExecution Plan:")
        print(output["plan"])

        print("\nSelected Specialist:")
        print(output["selected_agent"])

        print("\nValidated Analytical Result:")
        print(output["analytical_result"])

        print("\nBusiness Insights:")
        print("-" * 70)
        print(output["business_insights"])
        print("-" * 70)

        print("\nExecutive Report:")
        print("-" * 70)
        print(output["executive_report"])
        print("-" * 70)

        if (
            output["analytical_result"] is not None
            and output["business_insights"] is not None
            and output["executive_report"] is not None
        ):

            print(
                "\nTest completed successfully."
            )

        else:

            print(
                "\nTest did not complete successfully."
            )

    print(
        "\nMulti-Agent Orchestrator Version 6 "
        "testing completed."
    )