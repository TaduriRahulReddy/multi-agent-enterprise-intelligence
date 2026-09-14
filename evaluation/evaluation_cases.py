EVALUATION_CASES = [
    {
        "id": "structured_revenue_top5",
        "question": "What are the top 5 product categories by total product revenue?",
        "expected_agent": "data_analyst",
        "expected_success": True,
        "expected_values": {
            "health_beauty": 1258681.34,
            "watches_gifts": 1205005.68,
            "bed_bath_table": 1036988.68,
            "sports_leisure": 988048.97,
            "computers_accessories": 911954.32,
        },
    },
    {
        "id": "customer_repeat_count",
        "question": "How many repeat customers do we have?",
        "expected_agent": "customer_insights",
        "expected_success": True,
        "expected_values": {
            "repeat_customer_count": 2997,
        },
    },
    {
        "id": "research_q2_pricing",
        "question": "What pricing changes were introduced in Q2?",
        "expected_agent": "research",
        "expected_success": True,
        "expected_phrases": [
            "approximately 7 percent",
            "premium product categories",
        ],
    },
    {
        "id": "research_customer_problems",
        "question": "What customer problems were reported in the quarterly report?",
        "expected_agent": "research",
        "expected_success": True,
        "expected_phrases": [
            "delivery delays",
            "negative feedback",
        ],
    },
]
