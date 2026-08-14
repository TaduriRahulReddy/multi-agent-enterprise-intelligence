from agents.planner_agent import plan
from agents.data_analyst_agent_v3 import analyze


# ============================================================
# MULTI-AGENT ORCHESTRATOR - VERSION 1
# ============================================================


def run_orchestrator(question: str):
    """
    Main orchestration workflow.

    User Question
        ↓
    Planner Agent
        ↓
    Route Decision
        ↓
    Data Analyst Agent V3
        ↓
    Final Result
    """

    print("\n" + "=" * 70)
    print("MULTI-AGENT ENTERPRISE INTELLIGENCE SYSTEM")
    print("=" * 70)

    print("\nUser Question:")
    print(question)

    # --------------------------------------------------------
    # Step 1: Planner decides where to route the question
    # --------------------------------------------------------

    print("\nPlanner Agent is analyzing the request...")

    route = plan(question)

    print("\nSelected Route:")
    print(route)

    # --------------------------------------------------------
    # Step 2: Route to the appropriate specialist
    # --------------------------------------------------------

    if route == "data_analyst":

        print(
            "\nRouting request to "
            "Data Analyst Agent Version 3..."
        )

        result = analyze(question)

        return result

    # --------------------------------------------------------
    # Unsupported route
    # --------------------------------------------------------

    print(
        "\nThis request is currently unsupported "
        "by the available specialist agents."
    )

    return None


# ============================================================
# TEST ORCHESTRATOR
# ============================================================

if __name__ == "__main__":

    print("Testing Multi-Agent Orchestrator Version 1")

    test_question = (
        "What are the top 5 product categories "
        "by total product revenue?"
    )

    result = run_orchestrator(test_question)

    if result is not None:

        print(
            "\nMulti-Agent Orchestrator "
            "test completed successfully."
        )

    else:

        print(
            "\nMulti-Agent Orchestrator "
            "test did not return a result."
        )