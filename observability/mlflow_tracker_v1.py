import time
from typing import Any

import mlflow


# ------------------------------------------------------------
# MLflow configuration
# ------------------------------------------------------------

EXPERIMENT_NAME = "multi-agent-enterprise-intelligence"

mlflow.set_experiment(EXPERIMENT_NAME)


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def safe_log_param(name: str, value: Any) -> None:
    """
    Log a parameter only when the value is available.
    """
    if value is not None:
        mlflow.log_param(name, str(value))


def safe_log_metric(name: str, value: Any) -> None:
    """
    Log a numeric metric only when the value can be converted
    to a float.
    """
    if value is None:
        return

    try:
        mlflow.log_metric(name, float(value))
    except (TypeError, ValueError):
        pass


# ------------------------------------------------------------
# Main tracking function
# ------------------------------------------------------------

def run_with_mlflow_tracking(
    question: str,
    orchestrator_function,
) -> dict:
    """
    Execute the multi-agent orchestrator while recording
    operational metadata in MLflow.

    This wrapper does not modify the orchestrator itself.
    """

    start_time = time.perf_counter()

    with mlflow.start_run() as run:

        safe_log_param(
            "question",
            question,
        )

        safe_log_param(
            "orchestrator_version",
            "v15",
        )

        safe_log_param(
            "local_llm",
            "qwen2.5:7b",
        )

        try:
            result = orchestrator_function(question)

            latency_seconds = (
                time.perf_counter() - start_time
            )

            selected_agent = result.get(
                "selected_agent"
            )

            safe_log_param(
                "selected_agent",
                selected_agent,
            )

            safe_log_metric(
                "latency_seconds",
                latency_seconds,
            )

            safe_log_metric(
                "report_validation_attempts",
                result.get(
                    "report_validation_attempts"
                ),
            )

            safe_log_metric(
                "research_validation_attempts",
                result.get(
                    "research_validation_attempts"
                ),
            )

            report_validation = result.get(
                "report_validation"
            )

            research_validation = result.get(
                "research_validation"
            )

            report_valid = None

            if isinstance(
                report_validation,
                dict,
            ):
                report_valid = report_validation.get(
                    "is_valid"
                )

            research_valid = None

            if isinstance(
                research_validation,
                dict,
            ):
                research_valid = research_validation.get(
                    "is_valid"
                )

            if report_valid is not None:
                safe_log_metric(
                    "report_valid",
                    int(bool(report_valid)),
                )

            if research_valid is not None:
                safe_log_metric(
                    "research_valid",
                    int(bool(research_valid)),
                )

            safe_log_param(
                "execution_status",
                "success",
            )

            result["mlflow_run_id"] = run.info.run_id
            result["latency_seconds"] = latency_seconds

            return result

        except Exception as exc:

            latency_seconds = (
                time.perf_counter() - start_time
            )

            safe_log_metric(
                "latency_seconds",
                latency_seconds,
            )

            safe_log_param(
                "execution_status",
                "failed",
            )

            safe_log_param(
                "error_type",
                type(exc).__name__,
            )

            raise