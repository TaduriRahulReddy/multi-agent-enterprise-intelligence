from tools.llm_tool import ask_llm


# ============================================================
# PLANNER AGENT - VERSION 2
# STRUCTURED TASK ROUTING
# ============================================================


SUPPORTED_AGENTS = [
    "data_analyst",
]


def build_planner_prompt(question: str) -> str:
    """
    Build a prompt that asks the local LLM to create
    a structured execution plan.
    """

    return f"""
You are the Planner Agent for a multi-agent enterprise
intelligence system.

Your job is to analyze the user's request and determine
which specialist agent should handle it.

AVAILABLE AGENTS:

1. data_analyst
   Use this agent for:
   - SQL queries
   - database questions
   - revenue analysis
   - customer analysis
   - product analysis
   - order analysis
   - payment analysis
   - review analysis
   - seller analysis
   - business metrics
   - aggregations
   - counts
   - averages
   - rankings

USER QUESTION:
{question}

Return your answer using EXACTLY this format:

agent: <agent_name>
task: <short description of the task>
requires_database: <true_or_false>

RULES:

1. Select only an agent listed above.
2. Do not invent new agents.
3. Keep the task description short and specific.
4. Set requires_database to true when answering the
   question requires querying enterprise data.
5. Return only the structured plan.
6. Do not include explanations.
7. Do not include markdown.
"""


def parse_plan(response: str) -> dict:
    """
    Convert the LLM response into a Python dictionary.
    """

    plan = {
        "agent": "unsupported",
        "task": "",
        "requires_database": False,
    }

    for line in response.strip().splitlines():

        line = line.strip()

        if line.startswith("agent:"):
            plan["agent"] = (
                line.split(":", 1)[1]
                .strip()
                .lower()
            )

        elif line.startswith("task:"):
            plan["task"] = (
                line.split(":", 1)[1]
                .strip()
            )

        elif line.startswith("requires_database:"):

            value = (
                line.split(":", 1)[1]
                .strip()
                .lower()
            )

            plan["requires_database"] = (
                value == "true"
            )

    return plan


def validate_plan(plan: dict) -> dict:
    """
    Validate the structured plan before the
    orchestrator uses it.
    """

    agent = plan.get("agent")

    if agent not in SUPPORTED_AGENTS:

        plan["agent"] = "unsupported"
        plan["requires_database"] = False

    return plan


def plan(question: str) -> dict:
    """
    Analyze a user question and return a structured
    execution plan.
    """

    prompt = build_planner_prompt(question)

    response = ask_llm(prompt)

    structured_plan = parse_plan(response)

    structured_plan = validate_plan(structured_plan)

    return structured_plan


# ============================================================
# TEST PLANNER AGENT VERSION 2
# ============================================================

if __name__ == "__main__":

    print("Testing Planner Agent Version 2")
    print("=" * 70)

    test_questions = [
        "What are the top 5 product categories by revenue?",
        "How many repeat customers do we have?",
        "What is the average review score?",
    ]

    for question in test_questions:

        print("\nQuestion:")
        print(question)

        result = plan(question)

        print("\nExecution Plan:")
        print(result)

        print("-" * 70)

    print(
        "\nPlanner Agent Version 2 "
        "test completed."
    )