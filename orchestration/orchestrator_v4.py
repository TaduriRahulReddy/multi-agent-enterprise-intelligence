from agents.planner_agent_v3 import plan
from agents.data_analyst_agent_v3 import analyze
from agents.customer_insights_agent import analyze_customer_question
from agents.business_insights_agent import generate_insights


# ============================================================
# MULTI-AGENT ORCHESTRATOR - VERSION 4
# MULTI-SPECIALIST ROUTING + BUSINESS INSIGHTS
# ============================================================


def run_orchestrator(question: str):
    """
    Full multi-agent orchestration workflow.

    User Question
        ↓
    Planner Agent V3
        ↓
    Structured Execution Plan
        ↓
    Route to Specialist Agent
        ↓
    Validated Analytical Result
        ↓
    Business Insights Agent
        ↓
    Final Business Response
    """

    print("\n" + "=" * 70)
    print("MULTI-AGENT ENTERPRISE INTELLIGENCE SYSTEM")
    print("ORCHESTRATOR VERSION 4")
    print("=" * 70)

    print("\nUser Question:")
    print(question)

    # --------------------------------------------------------
    # STEP 1: CREATE EXECUTION PLAN
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
    # STEP 2: ROUTE TO DATA ANALYST AGENT
    # --------------------------------------------------------

    if selected_agent == "data_analyst":

        print(
            "\nRouting request to "
            "Data Analyst Agent Version 3..."
        )

        analytical_result = analyze(question)

    # --------------------------------------------------------
    # STEP 3: ROUTE TO CUSTOMER INSIGHTS AGENT
    # --------------------------------------------------------

    elif selected_agent == "customer_insights":

        print(
            "\nRouting request to "
            "Customer Insights Agent Version 1..."
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
        }

    # --------------------------------------------------------
    # STEP 6: BUSINESS INSIGHTS
    # --------------------------------------------------------

    print(
        "\nRouting validated result to "
        "Business Insights Agent..."
    )

    business_insights = generate_insights(
        question=question,
        analytical_result=analytical_result,
    )

    # --------------------------------------------------------
    # STEP 7: FINAL OUTPUT OBJECT
    # --------------------------------------------------------

    return {
        "plan": execution_plan,
        "selected_agent": selected_agent,
        "analytical_result": analytical_result,
        "business_insights": business_insights,
    }


# ============================================================
# TEST ORCHESTRATOR VERSION 4
# ============================================================

if __name__ == "__main__":

    print(
        "Testing Multi-Agent "
        "Orchestrator Version 4"
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

        if (
            output["analytical_result"] is not None
            and output["business_insights"] is not None
        ):

            print(
                "\nTest completed successfully."
            )

        else:

            print(
                "\nTest did not complete successfully."
            )

    print(
        "\nMulti-Agent Orchestrator Version 4 "
        "testing completed."
    )