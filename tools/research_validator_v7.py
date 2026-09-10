import json
import re
import requests


# ============================================================
# RESEARCH GROUNDING VALIDATOR - VERSION 7
# CLAIM-LEVEL VALIDATION WITH PYTHON-OWNED CLAIM IDENTITY
# ============================================================


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"


# ============================================================
# DETERMINISTIC GUARDRAILS
# ============================================================


RISKY_PHRASES = [
    "supply chain",
    "logistical issues",
    "logistics issues",
    "customer loyalty",
    "strong demand",
    "market demand",
    "customer preference",
    "customer satisfaction",
    "satisfaction levels",
    "customer dissatisfaction",
    "dissatisfaction levels",
    "negative impact on satisfaction",
    "impacted satisfaction",
    "affected satisfaction",
    "retention strength",
    "competitive position",
    "successful strategy",
    "operational inefficiency",
    "operational issues",
    "business success",
    "business growth",
]


SUCCESS_PHRASES = [
    "if successful",
    "if it is successful",
    "if this is successful",
    "if the change is successful",
    "if the strategy is successful",
    "when successful",
    "once successful",
    "if the strategy succeeds",
    "if the change succeeds",
    "depending on success",
]


UNSUPPORTED_QUALIFIERS = [
    "frequently",
    "very frequently",
    "significantly",
    "substantially",
    "strongly",
    "widely",
    "extensively",
    "major",
    "severe",
    "highly",
    "large-scale",
]


CAUSAL_PATTERNS = [
    r"\bcaused by\b",
    r"\bcausing\b",
    r"\bdue to\b",
    r"\bresulted from\b",
    r"\bled to\b",
    r"\bbecause of\b",
    r"\bresulting in\b",
]


SATISFACTION_PATTERNS = [
    r"negatively impacted .*satisfaction",
    r"impacted .*satisfaction",
    r"affected .*satisfaction",
    r"reduced .*satisfaction",
    r"lowered .*satisfaction",
    r"caused .*dissatisfaction",
    r"resulted in .*dissatisfaction",
]


# ============================================================
# META / CONTROL LANGUAGE
# ============================================================


META_CONTROL_SENTENCES = [
    (
        "the retrieved evidence directly answers the question, "
        "and no additional interpretation is necessary."
    ),
    (
        "no additional interpretation is provided because "
        "the available evidence is insufficient."
    ),
    (
        "the retrieved evidence does not provide enough information "
        "to answer the question confidently."
    ),
    (
        "unable to produce a fully grounded research answer."
    ),
    (
        "retrieved evidence was available, but the generated answer "
        "could not pass grounding validation."
    ),
    (
        "no relevant enterprise evidence was retrieved."
    ),
    (
        "no relevant factual evidence was found."
    ),
    (
        "no additional interpretation is provided because relevant "
        "evidence was not retrieved."
    ),
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================


def normalize_text(text: str) -> str:
    """
    Normalize text for deterministic comparison.
    """

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# META-CONTROL DETECTION
# ============================================================


def is_meta_control_statement(
    statement: str,
) -> bool:
    """
    Return True when a sentence is system/meta language rather than
    a factual business claim.
    """

    normalized_statement = normalize_text(
        statement
    )

    if not normalized_statement:
        return True

    for meta_sentence in META_CONTROL_SENTENCES:

        normalized_meta = normalize_text(
            meta_sentence
        )

        if normalized_statement == normalized_meta:
            return True

    return False


# ============================================================
# DETERMINISTIC VALIDATION
# ============================================================


def deterministic_grounding_check(
    research_answer: str,
    retrieved_context: str,
) -> dict | None:
    """
    Run deterministic checks for clearly unsupported language.
    """

    answer_lower = normalize_text(
        research_answer
    )

    evidence_lower = normalize_text(
        retrieved_context
    )

    # --------------------------------------------------------
    # Risky unsupported phrases
    # --------------------------------------------------------

    for phrase in RISKY_PHRASES:

        phrase_lower = normalize_text(
            phrase
        )

        if (
            phrase_lower in answer_lower
            and phrase_lower not in evidence_lower
        ):

            return {
                "is_valid": False,
                "reason": (
                    f"Unsupported interpretation detected: "
                    f"'{phrase}' appears in the research answer "
                    f"but is not supported by the retrieved evidence."
                ),
                "source": "deterministic_rule",
            }

    # --------------------------------------------------------
    # Unsupported success-condition language
    # --------------------------------------------------------

    for phrase in SUCCESS_PHRASES:

        phrase_lower = normalize_text(
            phrase
        )

        if (
            phrase_lower in answer_lower
            and phrase_lower not in evidence_lower
        ):

            return {
                "is_valid": False,
                "reason": (
                    f"Unsupported conditional interpretation detected: "
                    f"'{phrase}' is not stated in the retrieved evidence."
                ),
                "source": "deterministic_rule",
            }

    # --------------------------------------------------------
    # Unsupported qualifiers
    # --------------------------------------------------------

    for qualifier in UNSUPPORTED_QUALIFIERS:

        qualifier_lower = normalize_text(
            qualifier
        )

        if (
            qualifier_lower in answer_lower
            and qualifier_lower not in evidence_lower
        ):

            return {
                "is_valid": False,
                "reason": (
                    f"Unsupported qualifier detected: "
                    f"'{qualifier}' strengthens the evidence beyond "
                    f"what is explicitly stated."
                ),
                "source": "deterministic_rule",
            }

    # --------------------------------------------------------
    # Unsupported causal claims
    # --------------------------------------------------------

    for pattern in CAUSAL_PATTERNS:

        answer_match = re.search(
            pattern,
            answer_lower,
        )

        evidence_match = re.search(
            pattern,
            evidence_lower,
        )

        if (
            answer_match
            and not evidence_match
        ):

            phrase = answer_match.group(0)

            return {
                "is_valid": False,
                "reason": (
                    f"Unsupported causal wording detected: "
                    f"'{phrase}' appears in the answer but is not "
                    f"supported by the evidence."
                ),
                "source": "deterministic_rule",
            }

    # --------------------------------------------------------
    # Satisfaction semantic patterns
    # --------------------------------------------------------

    for pattern in SATISFACTION_PATTERNS:

        answer_match = re.search(
            pattern,
            answer_lower,
        )

        evidence_match = re.search(
            pattern,
            evidence_lower,
        )

        if (
            answer_match
            and not evidence_match
        ):

            phrase = answer_match.group(0)

            return {
                "is_valid": False,
                "reason": (
                    f"Unsupported customer-impact interpretation "
                    f"detected: '{phrase}' is not supported by the "
                    f"retrieved evidence."
                ),
                "source": "deterministic_rule",
            }

    return None


# ============================================================
# CLAIM EXTRACTION
# ============================================================


def extract_answer_claims(
    research_answer: str,
) -> list[str]:
    """
    Extract meaningful factual claims from the answer.

    Version 7 excludes:
    - section headings
    - source filenames
    - evidence sufficiency labels
    - known meta/control statements

    Real factual statements inside INTERPRETATION are still validated.
    """

    if not research_answer.strip():
        return []

    section_headers = {
        "DOCUMENT FINDING",
        "SUPPORTING EVIDENCE",
        "SOURCES",
        "INTERPRETATION",
        "EVIDENCE SUFFICIENCY",
    }

    ignored_values = {
        "sufficient",
        "insufficient",
    }

    claims = []

    current_section = None

    for raw_line in research_answer.splitlines():

        line = raw_line.strip()

        if not line:
            continue

        if line in section_headers:

            current_section = line
            continue

        # ----------------------------------------------------
        # Ignore source metadata
        # ----------------------------------------------------

        if current_section == "SOURCES":
            continue

        # ----------------------------------------------------
        # Ignore sufficiency labels
        # ----------------------------------------------------

        if current_section == "EVIDENCE SUFFICIENCY":
            continue

        # ----------------------------------------------------
        # Remove bullet prefix
        # ----------------------------------------------------

        if line.startswith("-"):

            line = line[1:].strip()

        if not line:
            continue

        if normalize_text(line) in ignored_values:
            continue

        # ----------------------------------------------------
        # Sentence-level claim extraction
        # ----------------------------------------------------

        sentences = re.split(
            r"(?<=[.!?])\s+",
            line,
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            if is_meta_control_statement(
                sentence
            ):
                continue

            if sentence not in claims:

                claims.append(
                    sentence
                )

    return claims


# ============================================================
# OLLAMA CALL
# ============================================================


def call_ollama(
    prompt: str,
) -> str:
    """
    Call local Qwen model through Ollama.
    """

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    result = response.json()

    return result.get(
        "response",
        "",
    ).strip()


# ============================================================
# JSON EXTRACTION
# ============================================================


def extract_json_object(
    text: str,
) -> dict | None:
    """
    Extract a JSON object from an LLM response.
    """

    if not text:
        return None

    text = text.strip()

    # --------------------------------------------------------
    # Direct JSON
    # --------------------------------------------------------

    try:

        return json.loads(
            text
        )

    except json.JSONDecodeError:

        pass

    # --------------------------------------------------------
    # Remove markdown fences
    # --------------------------------------------------------

    cleaned = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    cleaned = cleaned.replace(
        "```",
        "",
    ).strip()

    try:

        return json.loads(
            cleaned
        )

    except json.JSONDecodeError:

        pass

    # --------------------------------------------------------
    # Extract embedded JSON
    # --------------------------------------------------------

    match = re.search(
        r"\{.*\}",
        cleaned,
        flags=re.DOTALL,
    )

    if not match:
        return None

    try:

        return json.loads(
            match.group(0)
        )

    except json.JSONDecodeError:

        return None


# ============================================================
# SINGLE CLAIM VALIDATION
# ============================================================


def validate_claim_with_llm(
    question: str,
    claim: str,
    retrieved_context: str,
) -> dict:
    """
    Validate exactly one claim against retrieved evidence.

    Version 7 improvement:
    The LLM no longer returns or controls the claim text.

    Python already knows exactly which claim is being validated.
    """

    prompt = f"""
You are a strict enterprise grounding validator.

Validate exactly ONE factual claim against retrieved evidence.

USER QUESTION:
{question}

CLAIM TO VALIDATE:
{claim}

RETRIEVED EVIDENCE:
{retrieved_context}

IMPORTANT:

The claim has already been extracted by the Python application.

Do NOT rewrite the claim.
Do NOT return the claim.
Do NOT invent another claim.

Your only job is to determine whether the supplied claim is supported.

SUPPORTED means the claim is:
- directly stated in the evidence, or
- a faithful paraphrase preserving the same meaning.

UNSUPPORTED means the claim introduces information not supported by
the retrieved evidence, including:

- a new fact
- a new causal relationship
- customer satisfaction
- customer dissatisfaction
- customer loyalty
- business impact
- business recommendations
- stronger frequency
- stronger severity
- stronger certainty
- success conditions
- operational explanations
- supply-chain explanations
- conclusions that cannot be traced to evidence

IMPORTANT:

Do not reject a claim merely because its wording differs slightly from
the evidence.

Semantic equivalence is allowed.

Example:

Evidence:
"During Q2, the company increased average selling prices for selected
premium product categories by approximately 7 percent."

Claim:
"The company increased average selling prices for selected premium
product categories by approximately 7 percent during Q2."

SUPPORTED.

Example:

Evidence:
"Management planned to monitor conversion rates and customer response
before expanding the pricing change to additional categories."

Claim:
"Management planned to monitor conversion rates and customer response
before expanding the pricing change to additional categories."

SUPPORTED.

Example:

Evidence:
"Management planned to monitor conversion rates and customer response
before expanding the pricing change."

Claim:
"Management planned to expand the pricing change if it was successful."

UNSUPPORTED.

Return ONLY valid JSON.

Use exactly:

{{
    "supported": true,
    "supporting_evidence": "<short evidence sentence>",
    "reason": "<short explanation>"
}}

or:

{{
    "supported": false,
    "supporting_evidence": "",
    "reason": "<short explanation>"
}}
"""

    raw_response = call_ollama(
        prompt
    )

    parsed = extract_json_object(
        raw_response
    )

    # ========================================================
    # JSON FAILURE
    # ========================================================

    if parsed is None:

        return {
            "claim": claim,
            "supported": False,
            "supporting_evidence": "",
            "reason": (
                "Validator did not return valid structured JSON."
            ),
        }

    # ========================================================
    # PARSE SUPPORTED VALUE
    # ========================================================

    supported = parsed.get(
        "supported",
        False,
    )

    if isinstance(
        supported,
        str,
    ):

        supported = (
            supported.strip().lower()
            == "true"
        )

    # ========================================================
    # PYTHON OWNS CLAIM IDENTITY
    # ========================================================

    return {
        "claim": claim,
        "supported": bool(
            supported
        ),
        "supporting_evidence": str(
            parsed.get(
                "supporting_evidence",
                "",
            )
        ).strip(),
        "reason": str(
            parsed.get(
                "reason",
                "",
            )
        ).strip(),
    }


# ============================================================
# MAIN RESEARCH VALIDATOR
# ============================================================


def validate_research_answer(
    question: str,
    research_answer: str,
    retrieved_context: str,
) -> dict:
    """
    Validate a complete research answer.

    Version 7 pipeline:

    Research Answer
         ↓
    Deterministic Guardrails
         ↓
    Claim Extraction
         ↓
    Meta-Control Filtering
         ↓
    Python-Owned Claim Identity
         ↓
    Claim-by-Claim LLM Grounding
         ↓
    Final Validation
    """

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    if not research_answer.strip():

        return {
            "is_valid": False,
            "reason": "Research answer is empty.",
            "source": "input_validation",
        }

    if not retrieved_context.strip():

        return {
            "is_valid": False,
            "reason": "Retrieved evidence is empty.",
            "source": "input_validation",
        }

    # ========================================================
    # STEP 1: DETERMINISTIC GUARDRAILS
    # ========================================================

    deterministic_result = deterministic_grounding_check(
        research_answer=research_answer,
        retrieved_context=retrieved_context,
    )

    if deterministic_result is not None:

        return deterministic_result

    # ========================================================
    # STEP 2: EXTRACT FACTUAL CLAIMS
    # ========================================================

    claims = extract_answer_claims(
        research_answer
    )

    if not claims:

        return {
            "is_valid": False,
            "reason": (
                "No meaningful factual claims were found "
                "in the research answer."
            ),
            "source": "claim_extraction",
        }

    claim_results = []

    # ========================================================
    # STEP 3: VALIDATE EACH CLAIM
    # ========================================================

    for claim in claims:

        claim_result = validate_claim_with_llm(
            question=question,
            claim=claim,
            retrieved_context=retrieved_context,
        )

        claim_results.append(
            claim_result
        )

        if not claim_result["supported"]:

            return {
                "is_valid": False,
                "reason": (
                    f"Unsupported claim detected: "
                    f"'{claim}'. "
                    f"{claim_result['reason']}"
                ),
                "source": "claim_level_llm_validator",
                "failed_claim": claim,
                "claim_validation": claim_result,
                "claim_results": claim_results,
            }

    # ========================================================
    # STEP 4: SUCCESS
    # ========================================================

    return {
        "is_valid": True,
        "reason": (
            f"All {len(claims)} factual claims in the research "
            f"answer are supported by the retrieved evidence."
        ),
        "source": "claim_level_llm_validator",
        "claim_results": claim_results,
    }


# ============================================================
# LOCAL TESTS
# ============================================================


if __name__ == "__main__":

    print("=" * 70)
    print("RESEARCH GROUNDING VALIDATOR - VERSION 7")
    print("PYTHON-OWNED CLAIM IDENTITY")
    print("=" * 70)

    retrieved_context = """
SOURCE 1
Document: q2_business_report.txt
Chunk: 1

Quarterly Business Report - Q2

During Q2, the company increased average selling prices for selected
premium product categories by approximately 7 percent.

Management planned to monitor conversion rates and customer response
before expanding the pricing change to additional categories.

Customer service teams also reported that delivery delays were a
recurring source of negative feedback in some regions.
""".strip()

    tests = [
        {
            "name": "Grounded Customer Answer",
            "question": "What customer problems were reported?",
            "answer": """
DOCUMENT FINDING
Delivery delays were reported as a recurring source of negative feedback.

SUPPORTING EVIDENCE
- Customer service teams also reported that delivery delays were a recurring source of negative feedback in some regions.

SOURCES
- q2_business_report.txt

INTERPRETATION
The retrieved evidence directly answers the question, and no additional interpretation is necessary.

EVIDENCE SUFFICIENCY
sufficient
""".strip(),
            "expected": True,
        },
        {
            "name": "Unsupported Supply Chain Answer",
            "question": "What customer problems were reported?",
            "answer": """
DOCUMENT FINDING
Delivery delays were reported.

INTERPRETATION
The delays were caused by supply chain problems.

EVIDENCE SUFFICIENCY
sufficient
""".strip(),
            "expected": False,
        },
        {
            "name": "Unsupported Satisfaction Answer",
            "question": "What customer problems were reported?",
            "answer": """
DOCUMENT FINDING
Delivery delays were reported.

INTERPRETATION
Delivery delays negatively affected customer satisfaction.

EVIDENCE SUFFICIENCY
sufficient
""".strip(),
            "expected": False,
        },
        {
            "name": "Grounded Pricing Answer",
            "question": "What pricing changes were introduced in Q2?",
            "answer": """
DOCUMENT FINDING
The company increased average selling prices for selected premium product categories by approximately 7 percent during Q2.

SUPPORTING EVIDENCE
- During Q2, the company increased average selling prices for selected premium product categories by approximately 7 percent.
- Management planned to monitor conversion rates and customer response before expanding the pricing change to additional categories.

SOURCES
- q2_business_report.txt

INTERPRETATION
The retrieved evidence directly answers the question, and no additional interpretation is necessary.

EVIDENCE SUFFICIENCY
sufficient
""".strip(),
            "expected": True,
        },
        {
            "name": "Unsupported Success Condition",
            "question": "What pricing changes were introduced in Q2?",
            "answer": """
DOCUMENT FINDING
The company increased prices by approximately 7 percent.

INTERPRETATION
Management planned to expand the pricing change if it was successful.

EVIDENCE SUFFICIENCY
sufficient
""".strip(),
            "expected": False,
        },
        {
            "name": "False-Rejection Regression Test",
            "question": "What customer problems were reported?",
            "answer": """
DOCUMENT FINDING
The customer problem reported was delivery delays.

SUPPORTING EVIDENCE
- Customer service teams also reported that delivery delays were a recurring source of negative feedback in some regions.

SOURCES
- q2_business_report.txt

INTERPRETATION
The retrieved evidence directly answers the question, and no additional interpretation is necessary.

EVIDENCE SUFFICIENCY
sufficient
""".strip(),
            "expected": True,
        },
        {
            "name": "Actual Unsupported Interpretation Still Rejected",
            "question": "What customer problems were reported?",
            "answer": """
DOCUMENT FINDING
Delivery delays were reported as a recurring source of negative feedback.

SUPPORTING EVIDENCE
- Customer service teams also reported that delivery delays were a recurring source of negative feedback in some regions.

SOURCES
- q2_business_report.txt

INTERPRETATION
This suggests the company has broader operational problems affecting customers.

EVIDENCE SUFFICIENCY
sufficient
""".strip(),
            "expected": False,
        },
    ]

    passed = 0
    failed = 0

    for test in tests:

        print(
            "\n" + "-" * 70
        )

        print(
            test["name"]
        )

        print(
            "-" * 70
        )

        result = validate_research_answer(
            question=test["question"],
            research_answer=test["answer"],
            retrieved_context=retrieved_context,
        )

        print(
            result
        )

        actual = result.get(
            "is_valid",
            False,
        )

        print(
            f"Expected: {test['expected']}"
        )

        print(
            f"Actual:   {actual}"
        )

        if actual == test["expected"]:

            print(
                "TEST RESULT: PASS"
            )

            passed += 1

        else:

            print(
                "TEST RESULT: FAIL"
            )

            failed += 1

    print(
        "\n" + "=" * 70
    )

    print(
        "TEST SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Passed: {passed}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Total:  {len(tests)}"
    )

    print(
        "\nResearch Validator Version 7 "
        "testing completed."
    )