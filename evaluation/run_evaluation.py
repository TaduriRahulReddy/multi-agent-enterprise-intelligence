from evaluation.evaluation_cases import EVALUATION_CASES
from orchestration.orchestrator_v15 import (
    is_successful_output,
    run_orchestrator,
)


def evaluate_case(case: dict) -> dict:
    """
    Run one evaluation case and return a structured result.
    """

    print("\n" + "=" * 80)
    print(f"EVALUATION CASE: {case['id']}")
    print("=" * 80)

    output = run_orchestrator(
        case["question"]
    )

    routing_pass = (
        output.get("selected_agent")
        == case["expected_agent"]
    )

    success_pass = (
        is_successful_output(output)
        == case["expected_success"]
    )

    value_pass = True
    phrase_pass = True

    analytical_result = output.get(
        "analytical_result"
    )

    research_result = output.get(
        "research_result"
    )

    # ========================================================
    # STRUCTURED VALUE VALIDATION
    # ========================================================

    if "expected_values" in case:

        expected_values = case[
            "expected_values"
        ]

        if analytical_result is None:
            value_pass = False

        else:
            result_text = (
                analytical_result
                .to_string()
                .lower()
            )

            for key, expected_value in (
                expected_values.items()
            ):

                normalized_key = (
                    str(key)
                    .lower()
                    .strip()
                )

                expected_value_text = (
                    str(expected_value)
                    .lower()
                )

                if (
                    normalized_key
                    not in result_text
                ):
                    value_pass = False

                if (
                    expected_value_text
                    not in result_text
                ):
                    value_pass = False

    # ========================================================
    # RESEARCH PHRASE VALIDATION
    # ========================================================

    if "expected_phrases" in case:

        if not research_result:
            phrase_pass = False

        else:
            normalized_answer = (
                research_result
                .lower()
            )

            for phrase in case[
                "expected_phrases"
            ]:

                if (
                    phrase.lower()
                    not in normalized_answer
                ):
                    phrase_pass = False

    overall_pass = all(
        [
            routing_pass,
            success_pass,
            value_pass,
            phrase_pass,
        ]
    )

    return {
        "id": case["id"],
        "question": case["question"],
        "routing_pass": routing_pass,
        "success_pass": success_pass,
        "value_pass": value_pass,
        "phrase_pass": phrase_pass,
        "overall_pass": overall_pass,
        "selected_agent": output.get(
            "selected_agent"
        ),
        "report_validation_attempts": (
            output.get(
                "report_validation_attempts"
            )
        ),
        "research_validation_attempts": (
            output.get(
                "research_validation_attempts"
            )
        ),
    }


def run_evaluation():
    """
    Execute the complete evaluation suite.
    """

    print("\n" + "#" * 80)
    print("MULTI-AGENT ENTERPRISE INTELLIGENCE")
    print("AUTOMATED EVALUATION SUITE - VERSION 1")
    print("#" * 80)

    results = []

    for case in EVALUATION_CASES:

        result = evaluate_case(
            case
        )

        results.append(
            result
        )

        print("\nEvaluation Result:")
        print(result)

    passed = sum(
        1
        for result in results
        if result["overall_pass"]
    )

    failed = len(results) - passed

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print(
        f"Passed: {passed}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Total:  {len(results)}"
    )

    if results:
        pass_rate = (
            passed
            / len(results)
            * 100
        )
    else:
        pass_rate = 0.0

    print(
        f"Pass Rate: {pass_rate:.1f}%"
    )

    print("\nDetailed Results:")

    for result in results:

        status = (
            "PASS"
            if result["overall_pass"]
            else "FAIL"
        )

        print(
            f"- {result['id']}: "
            f"{status}"
        )

    return results


if __name__ == "__main__":
    run_evaluation()
