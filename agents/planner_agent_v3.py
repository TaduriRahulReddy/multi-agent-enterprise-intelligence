from tools.llm_tool import ask_llm


# ============================================================
# PLANNER AGENT - VERSION 3
# MULTI-SPECIALIST ROUTING
# ============================================================


SUPPORTED_AGENTS = [
    "data_analyst",
    "customer_insights",
]


def build_planner_prompt(question: str) -> str:
    """
    Build a routing prompt for multiple specialist agents.
    """

    return f"""
You are the Planner Agent for a multi-agent enterprise
intelligence system.

Your job is to analyze the user's request and choose the
most appropriate specialist agent.

AVAILABLE AGENTS:

1. data_analyst

Use data_analyst for:
- revenue analysis
- product analysis
- order analysis
- seller analysis
- payment analysis
- review analysis
- business metrics
- rankings
- averages
- aggregations
- general SQL analysis
- structured database questions that are not primarily
  about customer identity or customer behavior

2. customer_insights

Use customer_insights for:
- repeat customer analysis
- customer behavior
- customer retention-style questions
- customer geography
- customer-level order frequency
- customer reviews
- customer purchase patterns
- customer identity across multiple orders
- questions involving customer_unique_id

USER QUESTION:
{question}

Return your answer using EXACTLY this format:

agent: <agent_name>
task: <short description of the task>
requires_database: <true_or_false>

ROUTING RULES:

1. If the question is primarily customer-focused,
   choose customer_insights.

2. If the question is primarily about products, revenue,
   orders, payments, sellers, reviews, rankings, or general
   business metrics, choose data_analyst.

3. If customer_unique_id, repeat customers, customer
   purchase frequency, or customer geography is required,
   prefer customer_insights.

4. Select only one of these agents:
   data_analyst
   customer_insights

5. Do not invent additional agents.

6. Keep the task short and specific.

7. Set requires_database to true when the question
   requires enterprise data.

8. Return only the structured plan.

9. Do not include markdown.

10. Do not include explanations.
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
    Validate that the planner selected a supported agent.
    """

    selected_agent = plan.get(
        "agent",
        "unsupported",
    )

    if selected_agent not in SUPPORTED_AGENTS:

        plan["agent"] = "unsupported"
        plan["requires_database"] = False

    return plan


def plan(question: str) -> dict:
    """
    Generate and validate the structured execution plan.
    """

    prompt = build_planner_prompt(question)

    response = ask_llm(prompt)

    structured_plan = parse_plan(response)

    validated_plan = validate_plan(
        structured_plan
    )

    return validated_plan


# ============================================================
# TEST PLANNER AGENT VERSION 3
# ============================================================

if __name__ == "__main__":

    print("Testing Planner Agent Version 3")
    print("=" * 70)

    test_questions = [
        (
            "What are the top 5 product categories "
            "by total revenue?"
        ),
        (
            "How many repeat customers do we have?"
        ),
        (
            "Which customer states have the most customers?"
        ),
        (
            "What is the average review score?"
        ),
        (
            "Which sellers generate the most revenue?"
        ),
    ]

    for question in test_questions:

        print("\nQuestion:")
        print(question)

        execution_plan = plan(question)

        print("\nExecution Plan:")
        print(execution_plan)

        print("-" * 70)

    print(
        "\nPlanner Agent Version 3 "
        "test completed."
    )