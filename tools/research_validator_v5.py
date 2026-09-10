import json
import re
import requests


# ============================================================
# RESEARCH GROUNDING VALIDATOR - VERSION 5
# CLAIM-LEVEL VALIDATION + EXACT-CLAIM CHECKING
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
# BASIC TEXT NORMALIZATION
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
# DETERMINISTIC VALIDATION
# ============================================================


def deterministic_grounding_check(
    research_answer: str,
    retrieved_context: str,
) -> dict | None:
    """
    Fast deterministic checks for clearly unsupported language.

    Returns:
        None if no deterministic problem is found.

        Otherwise:
        {
            "is_valid": False,
            "reason": "...",
            "source": "deterministic_rule"
        }
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

        phrase_lower = phrase.lower()

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

        phrase_lower = phrase.lower()

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
    # Unsupported strengthening qualifiers
    # --------------------------------------------------------

    for qualifier in UNSUPPORTED_QUALIFIERS:

        qualifier_lower = qualifier.lower()

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
    # Satisfaction / dissatisfaction semantic patterns
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
    Extract meaningful claims from the generated research answer.

    This prevents the LLM validator from inventing claims that were
    never present in the answer.

    Sections such as SOURCES and EVIDENCE SUFFICIENCY are excluded.
    """

    if not research_answer.strip():
        return []

    ignored_headers = {
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

        if line in ignored_headers:
            current_section = line
            continue

        # Ignore source filenames.
        if current_section == "SOURCES":
            continue

        # Ignore sufficiency labels.
        if current_section == "EVIDENCE SUFFICIENCY":
            continue

        # Remove bullet prefix.
        if line.startswith("-"):
            line = line[1:].strip()

        if not line:
            continue

        if line.lower() in ignored_values:
            continue

        # Split longer lines into sentence-level claims.
        sentences = re.split(
            r"(?<=[.!?])\s+",
            line,
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            # Avoid duplicates.
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
    Send a grounding-validation prompt to Ollama.
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
    Extract the first valid JSON object from an LLM response.
    """

    if not text:
        return None

    text = text.strip()

    # --------------------------------------------------------
    # Direct JSON parse
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
    # Search for JSON object
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
# CLAIM-LEVEL LLM VALIDATION
# ============================================================


def validate_claim_with_llm(
    question: str,
    claim: str,
    retrieved_context: str,
) -> dict:
    """
    Validate exactly ONE claim against retrieved evidence.

    Important:
    The validator is explicitly forbidden from discussing or rejecting
    claims other than the supplied claim.
    """

    prompt = f"""
You are a strict enterprise grounding validator.

Your job is to validate EXACTLY ONE claim against retrieved evidence.

USER QUESTION:
{question}

CLAIM TO VALIDATE:
{claim}

RETRIEVED EVIDENCE:
{retrieved_context}

CRITICAL RULE:

You are validating ONLY the exact claim shown under CLAIM TO VALIDATE.

Do NOT:
- invent another claim
- quote a sentence that is not in CLAIM TO VALIDATE
- judge something the answer never said
- require the answer to mention every fact in the evidence

A claim is SUPPORTED when it is:
- directly stated in the evidence, or
- a faithful paraphrase that preserves the same meaning.

A claim is UNSUPPORTED when it:
- introduces a new fact
- introduces a new cause
- introduces customer satisfaction or dissatisfaction not stated
- introduces business impact
- introduces recommendations
- strengthens frequency, scale, severity, or certainty
- converts a monitoring plan into a success condition
- adds operational or supply-chain explanations
- makes a conclusion that cannot be traced to the evidence

IMPORTANT EXAMPLES:

Evidence:
"Delivery delays were a recurring source of negative feedback."

Claim:
"Delivery delays were reported as a recurring source of negative feedback."

SUPPORTED.

Claim:
"Delivery delays affected customer satisfaction."

UNSUPPORTED.

Claim:
"Customers were dissatisfied with the company's delivery service."

UNSUPPORTED.

Evidence:
"Management planned to monitor conversion rates and customer response
before expanding the pricing change."

Claim:
"Management planned to monitor conversion rates and customer response
before considering expansion to additional categories."

SUPPORTED.

Claim:
"Management planned to expand the change if it was successful."

UNSUPPORTED.

Before returning your result:

1. Copy the CLAIM TO VALIDATE exactly into the "claim" field.
2. Do not change its wording.
3. Judge only that claim.
4. If you cannot identify supporting evidence, mark it unsupported.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "claim": "<exact claim provided above>",
    "supported": true,
    "supporting_evidence": "<short supporting evidence or empty string>",
    "reason": "<short explanation>"
}}

or:

{{
    "claim": "<exact claim provided above>",
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

    if parsed is None:

        return {
            "claim": claim,
            "supported": False,
            "supporting_evidence": "",
            "reason": (
                "Validator did not return valid structured JSON."
            ),
        }

    returned_claim = str(
        parsed.get(
            "claim",
            "",
        )
    ).strip()

    # --------------------------------------------------------
    # CRITICAL EXACT-CLAIM CHECK
    # --------------------------------------------------------

    if normalize_text(returned_claim) != normalize_text(claim):

        return {
            "claim": claim,
            "supported": False,
            "supporting_evidence": "",
            "reason": (
                "Validator attempted to evaluate a claim different "
                "from the actual research-answer claim."
            ),
        }

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
# MAIN VALIDATOR
# ============================================================


def validate_research_answer(
    question: str,
    research_answer: str,
    retrieved_context: str,
) -> dict:
    """
    Validate a complete research answer against retrieved evidence.

    Version 5 architecture:

    1. Run deterministic guardrails.
    2. Extract only claims actually present in the answer.
    3. Validate each claim independently.
    4. Require exact claim matching.
    5. Fail if any meaningful claim is unsupported.
    """

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
    # STEP 1: DETERMINISTIC CHECK
    # ========================================================

    deterministic_result = deterministic_grounding_check(
        research_answer=research_answer,
        retrieved_context=retrieved_context,
    )

    if deterministic_result is not None:
        return deterministic_result

    # ========================================================
    # STEP 2: EXTRACT ACTUAL ANSWER CLAIMS
    # ========================================================

    claims = extract_answer_claims(
        research_answer
    )

    if not claims:

        return {
            "is_valid": False,
            "reason": (
                "No meaningful claims were found in the research answer."
            ),
            "source": "claim_extraction",
        }

    claim_results = []

    # ========================================================
    # STEP 3: VALIDATE EACH CLAIM
    # ========================================================

    for claim in claims:

        result = validate_claim_with_llm(
            question=question,
            claim=claim,
            retrieved_context=retrieved_context,
        )

        claim_results.append(
            result
        )

        if not result["supported"]:

            return {
                "is_valid": False,
                "reason": (
                    f"Unsupported claim detected: "
                    f"'{claim}'. "
                    f"{result['reason']}"
                ),
                "source": "claim_level_llm_validator",
                "failed_claim": claim,
                "claim_validation": result,
                "claim_results": claim_results,
            }

    # ========================================================
    # STEP 4: ALL CLAIMS SUPPORTED
    # ========================================================

    return {
        "is_valid": True,
        "reason": (
            f"All {len(claims)} meaningful claims in the research "
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
    print("RESEARCH GROUNDING VALIDATOR - VERSION 5")
    print("CLAIM-LEVEL VALIDATION")
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
    ]

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

        else:

            print(
                "TEST RESULT: FAIL"
            )

    print(
        "\n" + "=" * 70
    )

    print(
        "Research Validator Version 5 testing completed."
    )