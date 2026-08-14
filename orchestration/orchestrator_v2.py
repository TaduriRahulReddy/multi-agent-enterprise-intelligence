from agents.planner_agent_v2 import plan
from agents.data_analyst_agent_v3 import analyze


# ============================================================
# MULTI-AGENT ORCHESTRATOR - VERSION 2
# STRUCTURED PLAN EXECUTION
# ============================================================


def run_orchestrator(question: str):
    """
    Main orchestration workflow.

    User Question
        ↓
    Planner Agent V2
        ↓
    Structured Plan
        ↓
    Specialist Agent
        ↓
    Final Result
    """

    print("\n" + "=" * 70)
    print("MULTI-AGENT ENTERPRISE INTELLIGENCE SYSTEM")
    print("ORCHESTRATOR VERSION 2")
    print("=" * 70)

    print("\nUser Question:")
    print(question)

    # --------------------------------------------------------
    # Step 1: Planner creates structured execution plan
    # --------------------------------------------------------

    print("\nPlanner Agent V2 is creating an execution plan...")

    execution_plan = plan(question)

    print("\nExecution Plan:")
    print(execution_plan)

    # --------------------------------------------------------
    # Step 2: Extract plan fields
    # --------------------------------------------------------

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
    # Step 3: Route to Data Analyst Agent
    # --------------------------------------------------------

    if selected_agent == "data_analyst":

        if not requires_database:

            print(
                "\nPlanner selected the Data Analyst Agent "
                "but marked database access as unnecessary."
            )

        print(
            "\nRouting request to "
            "Data Analyst Agent Version 3..."
        )

        # IMPORTANT:
        # We pass the ORIGINAL user question to the
        # Data Analyst Agent so business context is preserved.
        result = analyze(question)

        return {
            "plan": execution_plan,
            "result": result,
        }

    # --------------------------------------------------------
    # Unsupported route
    # --------------------------------------------------------

    print(
        "\nThe selected route is currently unsupported."
    )

    return {
        "plan": execution_plan,
        "result": None,
    }


# ============================================================
# TEST ORCHESTRATOR VERSION 2
# ============================================================

if __name__ == "__main__":

    print(
        "Testing Multi-Agent "
        "Orchestrator Version 2"
    )

    test_question = (
        "What are the top 5 product categories "
        "by total product revenue?"
    )

    output = run_orchestrator(test_question)

    print("\n" + "=" * 70)
    print("ORCHESTRATOR OUTPUT")
    print("=" * 70)

    print("\nExecution Plan:")
    print(output["plan"])

    print("\nFinal Result:")
    print(output["result"])

    if output["result"] is not None:

        print(
            "\nMulti-Agent Orchestrator Version 2 "
            "test completed successfully."
        )

    else:

        print(
            "\nMulti-Agent Orchestrator Version 2 "
            "did not return a result."
        )