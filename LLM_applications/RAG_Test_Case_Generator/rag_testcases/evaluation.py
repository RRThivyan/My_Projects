
from typing import Dict, Any, List


def validate_output_schema(obj: Dict[str, Any]) -> List[str]:
    """Basic evaluation hook: check JSON structure & non-empty use_cases."""
    errors: List[str] = []

    if not isinstance(obj, dict):
        return ["Output is not a JSON object"]

    for key in ["query", "use_cases", "assumptions", "missing_info"]:
        if key not in obj:
            errors.append(f"Missing top-level field: {key}")

    use_cases = obj.get("use_cases", [])
    if not isinstance(use_cases, list):
        errors.append("use_cases is not a list")
    elif len(use_cases) == 0:
        errors.append("use_cases is empty")

    required_fields = [
        "use_case_title",
        "goal",
        "preconditions",
        "test_data",
        "steps",
        "expected_results",
        "negative_cases",
        "boundary_cases",
    ]

    if isinstance(use_cases, list) and use_cases:
        sample = use_cases[0]
        for f in required_fields:
            if f not in sample:
                errors.append(f"First use_case missing field: {f}")

    return errors
