from tools.llm_tool import ask_llm


# ============================================================
# PLANNER AGENT - VERSION 1
# ============================================================


SUPPORTED_AGENTS = [
    "data_analyst",
]


def build_planner_prompt(question: str) -> str:
    """
    Build a routing prompt that asks the LLM
    which specialist agent should handle the request.
    """

    return f"""
You are the Planner Agent for a multi-agent enterprise
intelligence system.

Your job is to understand the user's question and decide
which specialized agent should handle it.

CURRENTLY AVAILABLE AGENTS:

1. data_analyst

The data_analyst agent is responsible for:
- SQL analysis
- revenue analysis
- customer analysis
- product analysis
- order analysis
- review analysis
- seller analysis
- business metrics
- aggregations
- trends based on structured database data
- answering analytical questions using DuckDB

USER QUESTION:
{question}

ROUTING RULES:

1. If the question requires structured data analysis,
   SQL, metrics, aggregations, rankings, counts,
   revenue calculations, customer analysis, product
   analysis, seller analysis, review analysis, or
   order analysis, route to:

   data_analyst

2. If the request cannot currently be handled by the
   available agent, route to:

   unsupported

3. Return exactly one word.

Allowed responses:

data_analyst

or

unsupported

Do not provide explanations.

Return only the selected route.
"""


def clean_route(response: str) -> str:
    """
    Clean and validate the route returned by the LLM.
    """

    route = response.strip().lower()

    if route in SUPPORTED_AGENTS:
        return route

    return "unsupported"


def plan(question: str) -> str:
    """
    Decide which specialist agent should handle
    the user's question.
    """

    prompt = build_planner_prompt(question)

    response = ask_llm(prompt)

    route = clean_route(response)

    return route


# ============================================================
# TEST PLANNER AGENT
# ============================================================

if __name__ == "__main__":

    print("Testing Planner Agent")
    print("=" * 70)

    test_questions = [
        "What are the top 5 product categories by revenue?",
        "How many repeat customers do we have?",
        "What is the average review score?",
        "Write a marketing email to customers.",
    ]

    for question in test_questions:

        print("\nQuestion:")
        print(question)

        route = plan(question)

        print("Selected Agent:")
        print(route)

        print("-" * 70)

    print("\nPlanner Agent Version 1 test completed.")