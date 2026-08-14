from tools.llm_tool import ask_llm


# ============================================================
# PLANNER AGENT - VERSION 4
# MULTI-SPECIALIST ROUTING INCLUDING RAG / RESEARCH
# ============================================================


SUPPORTED_AGENTS = [
    "data_analyst",
    "customer_insights",
    "research",
]


def build_planner_prompt(question: str) -> str:
    """
    Build a routing prompt for all currently available
    specialist agents.
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
  customer-identity questions

2. customer_insights

Use customer_insights for:
- repeat customer analysis
- customer purchase frequency
- customer behavior
- customer geography
- customer-level order activity
- questions involving customer_unique_id
- customer identity across multiple orders
- customer-focused database analysis

3. research

Use research for questions that require document evidence,
reports, policies, written business documents, or RAG.

Examples:
- What pricing changes were introduced in Q2?
- What problems were mentioned in the quarterly report?
- What does the policy document say?
- What customer issues were reported?
- What evidence is documented about delivery delays?
- Summarize findings from company reports.
- What does the document say about pricing?
- According to the report, what changed?

USER QUESTION:
{question}

Return your answer using EXACTLY this format:

agent: <agent_name>
task: <short description of the task>
requires_database: <true_or_false>

ROUTING RULES:

1. If the question asks about structured numerical data,
   SQL, revenue, products, orders, sellers, payments,
   rankings, averages, or aggregations:
   choose data_analyst.

2. If the question is primarily about customer identity,
   repeat customers, customer purchase frequency,
   customer geography, or customer_unique_id:
   choose customer_insights.

3. If the question asks what a report, policy, document,
   written evidence, quarterly report, or other enterprise
   document says:
   choose research.

4. If the question asks about changes, findings, issues,
   policies, explanations, or evidence that are described
   in documents rather than calculated from database rows:
   choose research.

5. For research:
   requires_database must be false.

6. For data_analyst and customer_insights:
   requires_database should normally be true.

7. Select only one of these agents:
   data_analyst
   customer_insights
   research

8. Do not invent additional agents.

9. Keep the task short and specific.

10. Return only the structured plan.

11. Do not include markdown.

12. Do not include explanations.
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
    Validate the planner output.
    """

    selected_agent = plan.get(
        "agent",
        "unsupported",
    )

    if selected_agent not in SUPPORTED_AGENTS:

        plan["agent"] = "unsupported"
        plan["requires_database"] = False

        return plan

    # --------------------------------------------------------
    # Enforce resource expectations
    # --------------------------------------------------------

    if selected_agent == "research":

        plan["requires_database"] = False

    elif selected_agent in [
        "data_analyst",
        "customer_insights",
    ]:

        plan["requires_database"] = True

    return plan


def plan(question: str) -> dict:
    """
    Generate a validated structured execution plan.
    """

    prompt = build_planner_prompt(
        question
    )

    response = ask_llm(
        prompt
    )

    structured_plan = parse_plan(
        response
    )

    validated_plan = validate_plan(
        structured_plan
    )

    return validated_plan


# ============================================================
# TEST PLANNER AGENT VERSION 4
# ============================================================

if __name__ == "__main__":

    print(
        "Testing Planner Agent Version 4"
    )

    print("=" * 70)

    test_questions = [
        (
            "What are the top 5 product categories "
            "by total product revenue?"
        ),
        (
            "How many repeat customers do we have?"
        ),
        (
            "Which customer states have the most customers?"
        ),
        (
            "What pricing changes were introduced in Q2?"
        ),
        (
            "What customer problems were reported "
            "in the quarterly report?"
        ),
        (
            "Which sellers generate the most revenue?"
        ),
    ]

    for question in test_questions:

        print("\nQuestion:")
        print(
            question
        )

        execution_plan = plan(
            question
        )

        print(
            "\nExecution Plan:"
        )

        print(
            execution_plan
        )

        print("-" * 70)

    print(
        "\nPlanner Agent Version 4 "
        "test completed."
    )