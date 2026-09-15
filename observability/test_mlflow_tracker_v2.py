from observability.mlflow_tracker_v2 import (

    run_with_mlflow_tracking,

)
from orchestration.orchestrator_v15 import run_orchestrator


def main():
    question = "How many repeat customers do we have?"

    print("\nRunning MLflow observability test...")
    print(f"Question: {question}")

    result = run_with_mlflow_tracking(
        question=question,
        orchestrator_function=run_orchestrator,
    )

    print("\n" + "=" * 60)
    print("MLFLOW TRACKING TEST RESULT")
    print("=" * 60)

    print(
        "Selected Agent:",
        result.get("selected_agent"),
    )

    print(
        "Report Validation Attempts:",
        result.get("report_validation_attempts"),
    )

    print(
        "Research Validation Attempts:",
        result.get("research_validation_attempts"),
    )

    print(
        "Latency Seconds:",
        round(
            result.get("latency_seconds", 0),
            2,
        ),
    )

    print(
        "MLflow Run ID:",
        result.get("mlflow_run_id"),
    )

    print("\nMLflow tracking test completed.")


if __name__ == "__main__":
    main()