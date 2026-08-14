from agents.planner_agent_v2 import plan
from agents.data_analyst_agent_v3 import analyze
from agents.business_insights_agent import generate_insights


# ============================================================
# MULTI-AGENT ORCHESTRATOR - VERSION 3
# DATA ANALYSIS + BUSINESS INSIGHTS
# ============================================================


def run_orchestrator(question: str):
    """
    Main orchestration workflow.

    User Question
        ↓
    Planner Agent V2
        ↓
    Structured Execution Plan
        ↓
    Data Analyst Agent V3
        ↓
    Validated Analytical Result
        ↓
    Business Insights Agent
        ↓
    Final Business Answer
    """

    print("\n" + "=" * 70)
    print("MULTI-AGENT ENTERPRISE INTELLIGENCE SYSTEM")
    print("ORCHESTRATOR VERSION 3")
    print("=" * 70)

    print("\nUser Question:")
    print(question)

    # --------------------------------------------------------
    # Step 1: Planner creates structured plan
    # --------------------------------------------------------

    print("\nPlanner Agent V2 is creating an execution plan...")

    execution_plan = plan(question)

    print("\nExecution Plan:")
    print(execution_plan)

    selected_agent = execution_plan.get(
        "agent",
        "unsupported",
    )

    requires_database = execution_plan.get(
        "requires_database",
        False,
    )

    print("\nSelected Agent:")
    print(selected_agent)

    print("\nRequires Database:")
    print(requires_database)

    # --------------------------------------------------------
    # Step 2: Route to Data Analyst Agent
    # --------------------------------------------------------

    if selected_agent == "data_analyst":

        print(
            "\nRouting request to "
            "Data Analyst Agent Version 3..."
        )

        analytical_result = analyze(question)

        # ----------------------------------------------------
        # Step 3: Check analytical result
        # ----------------------------------------------------

        if analytical_result is None:

            print(
                "\nData Analyst Agent did not return "
                "a validated result."
            )

            return {
                "plan": execution_plan,
                "analytical_result": None,
                "business_insights": None,
            }

        # ----------------------------------------------------
        # Step 4: Send validated result to Business Insights
        # ----------------------------------------------------

        print(
            "\nRouting validated analytical result to "
            "Business Insights Agent..."
        )

        business_insights = generate_insights(
            question=question,
            analytical_result=analytical_result,
        )

        # ----------------------------------------------------
        # Step 5: Return complete output
        # ----------------------------------------------------

        return {
            "plan": execution_plan,
            "analytical_result": analytical_result,
            "business_insights": business_insights,
        }

    # --------------------------------------------------------
    # Unsupported route
    # --------------------------------------------------------

    print(
        "\nThe selected route is currently unsupported."
    )

    return {
        "plan": execution_plan,
        "analytical_result": None,
        "business_insights": None,
    }


# ============================================================
# TEST ORCHESTRATOR VERSION 3
# ============================================================

if __name__ == "__main__":

    print(
        "Testing Multi-Agent "
        "Orchestrator Version 3"
    )

    test_question = (
        "What are the top 5 product categories "
        "by total product revenue?"
    )

    output = run_orchestrator(test_question)

    print("\n" + "=" * 70)
    print("FINAL ORCHESTRATOR OUTPUT")
    print("=" * 70)

    print("\nExecution Plan:")
    print(output["plan"])

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
            "\nMulti-Agent Orchestrator Version 3 "
            "test completed successfully."
        )

    else:

        print(
            "\nMulti-Agent Orchestrator Version 3 "
            "did not complete successfully."
        )