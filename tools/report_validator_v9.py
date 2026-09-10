import json
import re

from tools.llm_tool import ask_llm


# ============================================================
# REPORT VALIDATOR - VERSION 9
#
# V9 improvements:
# - Preserves V8 count guardrails
# - Preserves V8 revenue/ranking semantic guardrails
# - Preserves canonical entity matching
# - Preserves deterministic ranking validation
# - Preserves deterministic revenue-value validation
# - NEW: context-aware numeric extraction
# - NEW: ignores structural numbers such as:
#       "top 5"
#       "rank 2"
#       "ranked 3"
#       "position 4"
#       "#2"
# ============================================================


# ============================================================
# ABSOLUTE-COUNT GUARDRAILS
# ============================================================

UNSUPPORTED_COUNT_PHRASES = [
    "strong customer loyalty",
    "strong loyalty",
    "high loyalty",
    "loyal customer base",
    "strong customer base",
    "strong base of repeat customers",
    "strong base",
    "level of customer loyalty",
    "certain level of customer loyalty",
    "customer loyalty and retention",
    "retention within the customer base",
    "strong retention",
    "high retention",
    "positive retention",
    "good retention",
    "healthy retention",
    "good foundation for customer retention",
    "positive foundation for customer retention",
    "significant portion",
    "large portion",
    "substantial portion",
    "majority of customers",
    "most customers",
    "large share of customers",
    "significant share of customers",
    "high customer satisfaction",
    "strong customer satisfaction",
    "effective loyalty program",
    "effective loyalty programs",
    "potential effectiveness of loyalty programs",
    "loyalty programs are effective",
]


# ============================================================
# REVENUE / RANKING GUARDRAILS
# ============================================================

UNSUPPORTED_REVENUE_FACTUAL_PHRASES = [
    "high demand",
    "strong demand",
    "customer demand",
    "demand among customers",

    "customer preference",
    "customer preferences",
    "customer interest",
    "customer interests",
    "product interest",
    "product interests",
    "customers prefer",
    "customers favor",

    "high popularity",
    "strong popularity",
    "very popular",
    "most popular",
    "their popularity",

    "high market share",
    "strong market share",
    "market leadership",
    "market leader",
    "strong market position",

    "high profitability",
    "strong profitability",
    "most profitable",
    "highly profitable",

    "complementary product",
    "complementary products",
    "complementary product interests",

    "marketing effectiveness",
    "effective marketing",
    "successful marketing",
    "increased marketing efforts",
]


UNSUPPORTED_REVENUE_RECOMMENDATION_PREMISES = [
    "because demand is high",
    "because of high demand",
    "because demand is strong",
    "based on high demand",
    "based on strong demand",
    "to leverage high demand",

    "because customers prefer",
    "based on customer preference",
    "based on customer preferences",

    "because customers are interested",
    "based on customer interest",
    "to leverage customer interest",
    "to capitalize on customer interest",
    "to capitalize on customer interests",

    "because they are popular",
    "based on their popularity",
    "to leverage their popularity",

    "because these products are complementary",
    "because the products are complementary",
    "based on complementary product interests",
    "to capitalize on complementary product interests",

    "because market share is high",
    "based on strong market share",

    "because profitability is high",
    "based on high profitability",
]


REPORT_SECTION_HEADERS = {
    "EXECUTIVE SUMMARY",
    "KEY RESULTS",
    "BUSINESS IMPLICATION",
    "BUSINESS IMPLICATIONS",
    "RECOMMENDED NEXT STEPS",
    "RECOMMENDATIONS",
}


META_CONTROL_SENTENCES = {
    (
        "the analysis confirms the ranking and values "
        "of the top 5 product categories by total revenue."
    ),
    (
        "the analysis confirms the number of customers "
        "who placed more than one order."
    ),
    (
        "additional context is required to determine "
        "the repeat-customer rate."
    ),
    (
        "unable to produce a fully grounded executive report."
    ),
}


# ============================================================
# TEXT HELPERS
# ============================================================


def normalize_text(text: str) -> str:
    """
    Normalize general text.
    """

    return " ".join(
        str(text).lower().split()
    )


def normalize_entity_name(text: str) -> str:
    """
    Normalize entity/category names into a canonical form.

    Examples:

        health_beauty
        Health & Beauty
        health and beauty

    all become:

        health beauty
    """

    text = str(text).lower()

    text = text.replace(
        "_",
        " ",
    )

    text = text.replace(
        "&",
        " ",
    )

    text = re.sub(
        r"\band\b",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    return " ".join(
        text.split()
    )


def result_to_text(
    analytical_result,
) -> str:
    """
    Convert an analytical result into readable text.
    """

    if hasattr(
        analytical_result,
        "to_string",
    ):
        return analytical_result.to_string(
            index=False
        )

    return str(
        analytical_result
    )


# ============================================================
# RESULT-SHAPE DETECTION
# ============================================================


def result_has_only_absolute_count(
    analytical_result,
) -> bool:
    """
    Detect a one-row, one-column count result.
    """

    if not hasattr(
        analytical_result,
        "shape",
    ):
        return False

    rows, columns = analytical_result.shape

    if rows != 1:
        return False

    if columns != 1:
        return False

    column_name = str(
        analytical_result.columns[0]
    ).lower()

    count_indicators = [
        "count",
        "number",
        "total_customers",
        "customer_count",
    ]

    return any(
        indicator in column_name
        for indicator in count_indicators
    )


def find_revenue_column(
    analytical_result,
):
    """
    Find the revenue-like metric column.
    """

    if not hasattr(
        analytical_result,
        "columns",
    ):
        return None

    revenue_indicators = [
        "revenue",
        "sales",
        "gmv",
        "sales_value",
        "total_value",
    ]

    for column in analytical_result.columns:

        column_name = str(
            column
        ).lower()

        if any(
            indicator in column_name
            for indicator in revenue_indicators
        ):
            return column

    return None


def find_entity_column(
    analytical_result,
    revenue_column,
):
    """
    Find the category/entity column.
    """

    if not hasattr(
        analytical_result,
        "columns",
    ):
        return None

    for column in analytical_result.columns:

        if column == revenue_column:
            continue

        try:

            if str(
                analytical_result[column].dtype
            ) == "object":

                return column

        except Exception:
            pass

    for column in analytical_result.columns:

        if column != revenue_column:
            return column

    return None


def result_is_revenue_ranking(
    analytical_result,
) -> bool:
    """
    Detect a multi-row revenue-ranking result.
    """

    if not hasattr(
        analytical_result,
        "shape",
    ):
        return False

    rows, columns = analytical_result.shape

    if rows < 2:
        return False

    if columns < 2:
        return False

    revenue_column = find_revenue_column(
        analytical_result
    )

    if revenue_column is None:
        return False

    entity_column = find_entity_column(
        analytical_result,
        revenue_column,
    )

    return entity_column is not None


# ============================================================
# REPORT PARSING
# ============================================================


def parse_report_sections(
    executive_report: str,
) -> dict:
    """
    Parse report sections while reconstructing wrapped lines.
    """

    sections = {}

    current_section = None
    current_entry = []

    def flush_current_entry():

        nonlocal current_entry

        if (
            current_section is not None
            and current_entry
        ):

            reconstructed = " ".join(
                current_entry
            ).strip()

            if reconstructed:

                sections.setdefault(
                    current_section,
                    [],
                ).append(
                    reconstructed
                )

        current_entry = []

    for raw_line in executive_report.splitlines():

        line = raw_line.strip()

        if not line:

            flush_current_entry()
            continue

        upper_line = line.upper()

        if upper_line in REPORT_SECTION_HEADERS:

            flush_current_entry()

            current_section = upper_line

            sections.setdefault(
                current_section,
                [],
            )

            continue

        if current_section is None:

            current_section = "UNLABELED"

            sections.setdefault(
                current_section,
                [],
            )

        if re.match(
            r"^[\-\*\u2022]\s+",
            line,
        ):

            flush_current_entry()

            cleaned_bullet = re.sub(
                r"^[\-\*\u2022]\s*",
                "",
                line,
            ).strip()

            if cleaned_bullet:

                current_entry.append(
                    cleaned_bullet
                )

            continue

        current_entry.append(
            line
        )

    flush_current_entry()

    return sections


def split_into_sentences(
    text: str,
) -> list[str]:
    """
    Split reconstructed text into sentence-sized units.
    """

    cleaned = " ".join(
        text.strip().split()
    )

    if not cleaned:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        cleaned,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def is_meta_control_sentence(
    sentence: str,
) -> bool:
    """
    Detect known control/meta language.
    """

    normalized = normalize_text(
        sentence
    )

    if normalized in META_CONTROL_SENTENCES:
        return True

    meta_prefixes = [
        "the analysis confirms",
        "the result provides",
        "additional context is required",
        "additional analysis is required",
        "further analysis is required",
    ]

    return any(
        normalized.startswith(
            prefix
        )
        for prefix in meta_prefixes
    )


def extract_factual_claims(
    executive_report: str,
) -> list[str]:
    """
    Extract factual claims from non-recommendation sections.
    """

    sections = parse_report_sections(
        executive_report
    )

    claims = []

    recommendation_sections = {
        "RECOMMENDED NEXT STEPS",
        "RECOMMENDATIONS",
    }

    for section_name, entries in sections.items():

        if section_name in recommendation_sections:
            continue

        for entry in entries:

            sentences = split_into_sentences(
                entry
            )

            for sentence in sentences:

                if is_meta_control_sentence(
                    sentence
                ):
                    continue

                if sentence not in claims:

                    claims.append(
                        sentence
                    )

    return claims


def extract_recommendations(
    executive_report: str,
) -> list[str]:
    """
    Extract recommendations separately.
    """

    sections = parse_report_sections(
        executive_report
    )

    recommendations = []

    recommendation_sections = {
        "RECOMMENDED NEXT STEPS",
        "RECOMMENDATIONS",
    }

    for section_name in recommendation_sections:

        for entry in sections.get(
            section_name,
            [],
        ):

            cleaned = " ".join(
                entry.split()
            ).strip()

            if cleaned:

                recommendations.append(
                    cleaned
                )

    return recommendations


# ============================================================
# DETERMINISTIC REPORT-LEVEL GUARDRAILS
# ============================================================


def deterministic_count_check(
    analytical_result,
    executive_report: str,
) -> dict:
    """
    Prevent count-only results from supporting unsupported
    qualitative interpretations.
    """

    if not result_has_only_absolute_count(
        analytical_result
    ):

        return {
            "is_valid": True,
            "reason": "Not an absolute-count result.",
            "source": "deterministic_count_rule",
        }

    normalized_report = normalize_text(
        executive_report
    )

    for phrase in UNSUPPORTED_COUNT_PHRASES:

        if phrase in normalized_report:

            return {
                "is_valid": False,
                "reason": (
                    "Unsupported qualitative claim detected "
                    "for an absolute-count result: "
                    f"'{phrase}'."
                ),
                "source": "deterministic_count_rule",
            }

    unsupported_patterns = [
        (
            "loyalty",
            (
                "Unsupported loyalty interpretation detected "
                "for an absolute-count result."
            ),
        ),
        (
            "retention is strong",
            "Unsupported retention-strength claim detected.",
        ),
        (
            "retention is high",
            "Unsupported retention-strength claim detected.",
        ),
        (
            "high retention",
            "Unsupported retention-strength claim detected.",
        ),
        (
            "strong retention",
            "Unsupported retention-strength claim detected.",
        ),
        (
            "strong base",
            (
                "Unsupported strength claim detected for an "
                "absolute-count result."
            ),
        ),
    ]

    for pattern, reason in unsupported_patterns:

        if pattern in normalized_report:

            return {
                "is_valid": False,
                "reason": reason,
                "source": "deterministic_count_rule",
            }

    return {
        "is_valid": True,
        "reason": (
            "No absolute-count grounding violations detected."
        ),
        "source": "deterministic_count_rule",
    }


def deterministic_revenue_ranking_check(
    analytical_result,
    executive_report: str,
) -> dict:
    """
    Prevent revenue rankings from supporting unsupported
    behavior or market interpretations.
    """

    if not result_is_revenue_ranking(
        analytical_result
    ):

        return {
            "is_valid": True,
            "reason": "Not a revenue-ranking result.",
            "source": "deterministic_revenue_rule",
        }

    factual_claims = extract_factual_claims(
        executive_report
    )

    for claim in factual_claims:

        normalized_claim = normalize_text(
            claim
        )

        for phrase in UNSUPPORTED_REVENUE_FACTUAL_PHRASES:

            if phrase in normalized_claim:

                return {
                    "is_valid": False,
                    "reason": (
                        "Unsupported interpretation detected "
                        "for a revenue/ranking result: "
                        f"'{phrase}' in claim '{claim}'. "
                        "Revenue and ranking alone do not "
                        "establish this conclusion."
                    ),
                    "source": "deterministic_revenue_rule",
                }

    return {
        "is_valid": True,
        "reason": (
            "No unsupported revenue/ranking interpretations "
            "were detected."
        ),
        "source": "deterministic_revenue_rule",
    }


def deterministic_grounding_check(
    analytical_result,
    executive_report: str,
) -> dict:
    """
    Run deterministic report-level checks.
    """

    count_result = deterministic_count_check(
        analytical_result=analytical_result,
        executive_report=executive_report,
    )

    if not count_result[
        "is_valid"
    ]:
        return count_result

    revenue_result = deterministic_revenue_ranking_check(
        analytical_result=analytical_result,
        executive_report=executive_report,
    )

    if not revenue_result[
        "is_valid"
    ]:
        return revenue_result

    return {
        "is_valid": True,
        "reason": (
            "No deterministic grounding violations detected."
        ),
        "source": "deterministic_rule",
    }


# ============================================================
# REVENUE RANKING RECORDS
# ============================================================


def build_revenue_ranking_records(
    analytical_result,
) -> list[dict]:
    """
    Build Python-owned revenue ranking records.
    """

    if not result_is_revenue_ranking(
        analytical_result
    ):
        return []

    revenue_column = find_revenue_column(
        analytical_result
    )

    entity_column = find_entity_column(
        analytical_result,
        revenue_column,
    )

    if (
        revenue_column is None
        or entity_column is None
    ):
        return []

    try:

        sorted_result = analytical_result.sort_values(
            by=revenue_column,
            ascending=False,
        ).reset_index(
            drop=True
        )

    except Exception:
        return []

    records = []

    for index, row in sorted_result.iterrows():

        try:

            numeric_value = float(
                row[revenue_column]
            )

        except Exception:
            continue

        entity = str(
            row[entity_column]
        )

        records.append(
            {
                "entity": entity,
                "normalized_entity": normalize_entity_name(
                    entity
                ),
                "value": numeric_value,
                "rank": index + 1,
            }
        )

    return records


def find_entity_record_in_claim(
    claim: str,
    ranking_records: list[dict],
):
    """
    Find the entity mentioned in a factual claim.
    """

    normalized_claim = normalize_entity_name(
        claim
    )

    matches = []

    for record in ranking_records:

        entity_name = record[
            "normalized_entity"
        ]

        if entity_name in normalized_claim:

            matches.append(
                record
            )

    if not matches:
        return None

    matches.sort(
        key=lambda item: len(
            item["normalized_entity"]
        ),
        reverse=True,
    )

    return matches[0]


# ============================================================
# V9 CONTEXT-AWARE NUMERIC EXTRACTION
# ============================================================


def is_structural_number(
    claim: str,
    start_index: int,
    end_index: int,
) -> bool:
    """
    Determine whether a numeric token describes structure,
    ranking, position, or list size rather than a metric value.

    Examples that should return True:

        top 5 categories
        top 10 products
        rank 2
        ranked 3
        ranks 4
        position 5
        #2

    Example that should return False:

        $1,258,681.34
        revenue of 1205005.68
        generated 911954.32 in revenue
    """

    before = claim[
        max(
            0,
            start_index - 30,
        ):start_index
    ].lower()

    token_text = claim[
        start_index:end_index
    ]

    # --------------------------------------------------------
    # Explicit hash ranking:
    #
    # #1
    # #2
    # #5
    # --------------------------------------------------------

    immediate_before = claim[
        max(
            0,
            start_index - 1,
        ):start_index
    ]

    if immediate_before == "#":
        return True

    # --------------------------------------------------------
    # "top 5", "top 10", etc.
    # --------------------------------------------------------

    if re.search(
        r"\btop\s*$",
        before,
    ):
        return True

    # --------------------------------------------------------
    # "rank 2"
    # "ranked 2"
    # "ranks 2"
    # --------------------------------------------------------

    if re.search(
        r"\b(?:rank|ranked|ranks|ranking)\s*$",
        before,
    ):
        return True

    # --------------------------------------------------------
    # "position 2"
    # "place 2"
    # --------------------------------------------------------

    if re.search(
        r"\b(?:position|place)\s*$",
        before,
    ):
        return True

    # --------------------------------------------------------
    # Defensive handling:
    #
    # top-5
    # rank-2
    # position-3
    # --------------------------------------------------------

    if re.search(
        r"\b(?:top|rank|ranked|ranks|position|place)[\s\-]*$",
        before,
    ):
        return True

    # token_text is intentionally retained here so the
    # function can easily be expanded in future versions.
    _ = token_text

    return False


def extract_numeric_values(
    claim: str,
) -> list[float]:
    """
    Extract factual metric values while excluding structural
    ranking/list numbers.

    V8 problem:

        "The top 5 product categories by total revenue are ..."

    previously extracted:

        5.0

    Because the sentence also mentioned "revenue", V8 could
    incorrectly compare 5.0 with an entity's revenue.

    V9 excludes that structural number.
    """

    numeric_pattern = re.compile(
        r"(?<![A-Za-z])\$?(\d[\d,]*(?:\.\d+)?)"
    )

    values = []

    for match in numeric_pattern.finditer(
        claim
    ):

        if is_structural_number(
            claim=claim,
            start_index=match.start(),
            end_index=match.end(),
        ):
            continue

        numeric_text = match.group(
            1
        )

        try:

            numeric_value = float(
                numeric_text.replace(
                    ",",
                    "",
                )
            )

        except ValueError:
            continue

        values.append(
            numeric_value
        )

    return values


def approximately_equal(
    first: float,
    second: float,
    tolerance: float = 0.01,
) -> bool:
    """
    Compare financial values safely.
    """

    return abs(
        first - second
    ) <= tolerance


# ============================================================
# DETERMINISTIC RANK DETECTION
# ============================================================


def detect_rank_claim(
    claim: str,
):
    """
    Detect explicit ordinal ranking claims.
    """

    normalized = normalize_text(
        claim
    )

    rank_patterns = {
        1: [
            "highest",
            "ranks first",
            "rank first",
            "ranked first",
            "comes first",
            "first place",
        ],
        2: [
            "ranks second",
            "rank second",
            "ranked second",
            "comes second",
            "second place",
        ],
        3: [
            "ranks third",
            "rank third",
            "ranked third",
            "comes third",
            "third place",
            "comes in third",
        ],
        4: [
            "ranks fourth",
            "rank fourth",
            "ranked fourth",
            "comes fourth",
            "fourth place",
        ],
        5: [
            "ranks fifth",
            "rank fifth",
            "ranked fifth",
            "comes fifth",
            "fifth place",
        ],
    }

    for rank, patterns in rank_patterns.items():

        for pattern in patterns:

            if pattern in normalized:
                return rank

    return None


def claim_mentions_revenue(
    claim: str,
) -> bool:
    """
    Determine whether the claim discusses revenue.
    """

    normalized = normalize_text(
        claim
    )

    indicators = [
        "revenue",
        "sales",
        "gmv",
    ]

    return any(
        indicator in normalized
        for indicator in indicators
    )


# ============================================================
# DETERMINISTIC CLAIM VALIDATION
# ============================================================


def deterministic_revenue_claim_validation(
    analytical_result,
    claim: str,
) -> dict:
    """
    Validate straightforward revenue and rank claims using
    deterministic Python logic.

    If the claim cannot be safely evaluated structurally,
    return handled=False so semantic validation can be used.
    """

    if not result_is_revenue_ranking(
        analytical_result
    ):

        return {
            "handled": False,
        }

    ranking_records = build_revenue_ranking_records(
        analytical_result
    )

    if not ranking_records:

        return {
            "handled": False,
        }

    entity_record = find_entity_record_in_claim(
        claim,
        ranking_records,
    )

    if entity_record is None:

        return {
            "handled": False,
        }

    expected_rank = detect_rank_claim(
        claim
    )

    numeric_values = extract_numeric_values(
        claim
    )

    mentions_revenue = claim_mentions_revenue(
        claim
    )

    # ========================================================
    # RANK + VALUE CLAIM
    # ========================================================

    if (
        expected_rank is not None
        and numeric_values
        and mentions_revenue
    ):

        if entity_record[
            "rank"
        ] != expected_rank:

            return {
                "handled": True,
                "supported": False,
                "supporting_evidence": "",
                "reason": (
                    f"{entity_record['entity']} is rank "
                    f"{entity_record['rank']} in the validated "
                    f"analytical result, not rank "
                    f"{expected_rank}."
                ),
                "source": "deterministic_numeric_validator",
            }

        claimed_value = numeric_values[-1]

        if not approximately_equal(
            claimed_value,
            entity_record["value"],
        ):

            return {
                "handled": True,
                "supported": False,
                "supporting_evidence": "",
                "reason": (
                    f"The validated revenue for "
                    f"{entity_record['entity']} is "
                    f"{entity_record['value']:.2f}, not "
                    f"{claimed_value:.2f}."
                ),
                "source": "deterministic_numeric_validator",
            }

        return {
            "handled": True,
            "supported": True,
            "supporting_evidence": (
                f"{entity_record['entity']} | "
                f"rank={entity_record['rank']} | "
                f"revenue={entity_record['value']:.2f}"
            ),
            "reason": (
                "Python verified both the ranking and revenue "
                "value directly from the analytical result."
            ),
            "source": "deterministic_numeric_validator",
        }

    # ========================================================
    # PURE RANK CLAIM
    # ========================================================

    if (
        expected_rank is not None
        and mentions_revenue
    ):

        if entity_record[
            "rank"
        ] == expected_rank:

            return {
                "handled": True,
                "supported": True,
                "supporting_evidence": (
                    f"{entity_record['entity']} | "
                    f"rank={entity_record['rank']} | "
                    f"revenue={entity_record['value']:.2f}"
                ),
                "reason": (
                    "Python verified the revenue ranking "
                    "directly from the analytical result."
                ),
                "source": "deterministic_numeric_validator",
            }

        return {
            "handled": True,
            "supported": False,
            "supporting_evidence": "",
            "reason": (
                f"{entity_record['entity']} is rank "
                f"{entity_record['rank']} in the validated "
                f"analytical result, not rank "
                f"{expected_rank}."
            ),
            "source": "deterministic_numeric_validator",
        }

    # ========================================================
    # PURE REVENUE VALUE CLAIM
    # ========================================================

    if (
        numeric_values
        and mentions_revenue
    ):

        claimed_value = numeric_values[-1]

        if approximately_equal(
            claimed_value,
            entity_record["value"],
        ):

            return {
                "handled": True,
                "supported": True,
                "supporting_evidence": (
                    f"{entity_record['entity']} | "
                    f"revenue={entity_record['value']:.2f}"
                ),
                "reason": (
                    "Python verified the revenue value "
                    "directly from the analytical result."
                ),
                "source": "deterministic_numeric_validator",
            }

        return {
            "handled": True,
            "supported": False,
            "supporting_evidence": "",
            "reason": (
                f"The validated revenue for "
                f"{entity_record['entity']} is "
                f"{entity_record['value']:.2f}, not "
                f"{claimed_value:.2f}."
            ),
            "source": "deterministic_numeric_validator",
        }

    # ========================================================
    # IMPORTANT V9 BEHAVIOR
    #
    # Example:
    #
    # "The top 5 product categories by total revenue are
    #  health beauty, watches gifts, ..."
    #
    # "5" has already been removed as a structural number.
    #
    # Therefore numeric_values == [].
    #
    # The claim is NOT incorrectly treated as a revenue-value
    # assertion.
    # ========================================================

    return {
        "handled": False,
    }


# ============================================================
# CLAIM-LEVEL LLM VALIDATION
# ============================================================


def build_claim_validation_prompt(
    question: str,
    analytical_result: str,
    claim: str,
) -> str:
    """
    Build semantic validation prompt.
    """

    return f"""
You are a strict factual grounding validator for an enterprise
analytics system.

Determine whether ONE factual claim is supported by the
validated analytical result.

ORIGINAL BUSINESS QUESTION:
{question}

VALIDATED ANALYTICAL RESULT:
{analytical_result}

FACTUAL CLAIM TO VALIDATE:
{claim}

GROUNDING RULES:

1. The analytical result is authoritative.

2. Validate ONLY the supplied factual claim.

3. Do not invent additional evidence.

4. A faithful paraphrase is supported.

5. Revenue rankings support:
   - numerical revenue values
   - ordering
   - rank position
   - faithful descriptions of the listed top categories

6. Revenue rankings do NOT by themselves support:
   - high demand
   - strong demand
   - customer preference
   - customer interest
   - popularity
   - profitability
   - market share
   - market leadership
   - product complementarity
   - marketing effectiveness

7. An absolute count does NOT establish:
   - retention strength
   - loyalty strength
   - customer-share percentages
   - majority/minority status
   - customer satisfaction

8. Numbers describing list size or structure are not metric
   values.

For example:

"The top 5 categories are A, B, C, D, and E"

uses 5 as the number of listed categories, not as a revenue
value.

9. If a claim contains multiple factual assertions, every
   material assertion must be supported.

10. Human-readable transformations of column names are valid.

Example:

repeat_customer_count = 2997

supports:

"There are 2,997 repeat customers."

11. When uncertain, return false.

Return ONLY valid JSON:

{{
  "supported": true,
  "supporting_evidence": "short evidence from the analytical result",
  "reason": "short explanation"
}}

or:

{{
  "supported": false,
  "supporting_evidence": "",
  "reason": "short explanation"
}}
"""


def parse_json_validation_response(
    response: str,
) -> dict:
    """
    Parse LLM validation JSON safely.
    """

    cleaned = response.strip()

    if cleaned.startswith(
        "```"
    ):

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

    try:

        parsed = json.loads(
            cleaned
        )

    except json.JSONDecodeError:

        return {
            "supported": False,
            "supporting_evidence": "",
            "reason": (
                "Unable to parse claim-level grounding "
                "validation response."
            ),
        }

    supported = parsed.get(
        "supported"
    )

    if not isinstance(
        supported,
        bool,
    ):

        return {
            "supported": False,
            "supporting_evidence": "",
            "reason": (
                "Claim-level validator did not return a "
                "boolean supported field."
            ),
        }

    return {
        "supported": supported,
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


def validate_single_claim(
    question: str,
    analytical_result,
    claim: str,
) -> dict:
    """
    Validate exactly one factual claim.
    """

    deterministic_result = (
        deterministic_revenue_claim_validation(
            analytical_result=analytical_result,
            claim=claim,
        )
    )

    if deterministic_result.get(
        "handled",
        False,
    ):

        return {
            "claim": claim,
            "supported": deterministic_result[
                "supported"
            ],
            "supporting_evidence": (
                deterministic_result[
                    "supporting_evidence"
                ]
            ),
            "reason": deterministic_result[
                "reason"
            ],
            "source": deterministic_result[
                "source"
            ],
        }

    result_text = result_to_text(
        analytical_result
    )

    prompt = build_claim_validation_prompt(
        question=question,
        analytical_result=result_text,
        claim=claim,
    )

    response = ask_llm(
        prompt
    )

    llm_result = parse_json_validation_response(
        response
    )

    return {
        "claim": claim,
        "supported": llm_result[
            "supported"
        ],
        "supporting_evidence": llm_result[
            "supporting_evidence"
        ],
        "reason": llm_result[
            "reason"
        ],
        "source": "claim_level_llm_validator",
    }


# ============================================================
# RECOMMENDATION VALIDATION
# ============================================================


def recommendation_has_unsupported_premise(
    recommendation: str,
    analytical_result,
) -> dict:
    """
    Validate recommendation premises.
    """

    normalized = normalize_text(
        recommendation
    )

    # --------------------------------------------------------
    # Count-based recommendation guardrails
    # --------------------------------------------------------

    if result_has_only_absolute_count(
        analytical_result
    ):

        for phrase in UNSUPPORTED_COUNT_PHRASES:

            if phrase in normalized:

                return {
                    "is_valid": False,
                    "reason": (
                        "Recommendation contains an unsupported "
                        f"factual premise: '{phrase}'."
                    ),
                    "recommendation": recommendation,
                }

        count_premises = [
            "because retention is strong",
            "because retention is high",
            "because loyalty is strong",
            "because customer loyalty is strong",
            "based on strong retention",
            "based on high retention",
            "based on strong customer loyalty",
            "due to strong retention",
            "due to high retention",
        ]

        for pattern in count_premises:

            if pattern in normalized:

                return {
                    "is_valid": False,
                    "reason": (
                        "Recommendation relies on an unsupported "
                        f"premise: '{pattern}'."
                    ),
                    "recommendation": recommendation,
                }

    # --------------------------------------------------------
    # Revenue-based recommendation guardrails
    # --------------------------------------------------------

    if result_is_revenue_ranking(
        analytical_result
    ):

        for pattern in (
            UNSUPPORTED_REVENUE_RECOMMENDATION_PREMISES
        ):

            if pattern in normalized:

                return {
                    "is_valid": False,
                    "reason": (
                        "Recommendation relies on an unsupported "
                        "revenue/ranking premise: "
                        f"'{pattern}'."
                    ),
                    "recommendation": recommendation,
                }

    return {
        "is_valid": True,
        "reason": (
            "Recommendation does not contain an obvious "
            "unsupported factual premise."
        ),
        "recommendation": recommendation,
    }


def validate_recommendations(
    analytical_result,
    recommendations: list[str],
) -> dict:
    """
    Validate recommendations independently.
    """

    results = []

    for recommendation in recommendations:

        result = recommendation_has_unsupported_premise(
            recommendation=recommendation,
            analytical_result=analytical_result,
        )

        results.append(
            result
        )

        if not result[
            "is_valid"
        ]:

            return {
                "is_valid": False,
                "reason": result[
                    "reason"
                ],
                "source": "recommendation_guardrail",
                "recommendation_results": results,
            }

    return {
        "is_valid": True,
        "reason": (
            "Recommendations do not rely on obvious "
            "unsupported factual premises."
        ),
        "source": "recommendation_guardrail",
        "recommendation_results": results,
    }


# ============================================================
# FULL REPORT VALIDATION
# ============================================================


def validate_report(
    question: str,
    analytical_result,
    executive_report: str,
) -> dict:
    """
    Report Validator Version 9.

    Validation flow:

        Executive Report
              |
        Input Validation
              |
        Deterministic Guardrails
              |
        Claim Extraction
              |
        Entity Resolution
              |
        Context-Aware Number Extraction
              |
        Deterministic Numeric/Rank Validation
              |
        Semantic Validation When Needed
              |
        Recommendation Validation
              |
        VALID / INVALID
    """

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    if analytical_result is None:

        return {
            "is_valid": False,
            "reason": (
                "No validated analytical result was provided."
            ),
            "source": "input_validation",
            "claim_results": [],
            "recommendation_results": [],
        }

    if not executive_report:

        return {
            "is_valid": False,
            "reason": (
                "No executive report was provided."
            ),
            "source": "input_validation",
            "claim_results": [],
            "recommendation_results": [],
        }

    # ========================================================
    # REPORT-LEVEL GUARDRAILS
    # ========================================================

    deterministic_result = (
        deterministic_grounding_check(
            analytical_result=analytical_result,
            executive_report=executive_report,
        )
    )

    if not deterministic_result[
        "is_valid"
    ]:

        return {
            **deterministic_result,
            "claim_results": [],
            "recommendation_results": [],
        }

    # ========================================================
    # FACTUAL CLAIM VALIDATION
    # ========================================================

    factual_claims = extract_factual_claims(
        executive_report
    )

    claim_results = []

    for claim in factual_claims:

        claim_result = validate_single_claim(
            question=question,
            analytical_result=analytical_result,
            claim=claim,
        )

        claim_results.append(
            claim_result
        )

        if not claim_result[
            "supported"
        ]:

            return {
                "is_valid": False,
                "reason": (
                    "Unsupported factual claim detected: "
                    f"'{claim}'. "
                    f"{claim_result['reason']}"
                ),
                "source": claim_result.get(
                    "source",
                    "claim_validator",
                ),
                "claim_results": claim_results,
                "recommendation_results": [],
            }

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    recommendations = extract_recommendations(
        executive_report
    )

    recommendation_validation = (
        validate_recommendations(
            analytical_result=analytical_result,
            recommendations=recommendations,
        )
    )

    if not recommendation_validation[
        "is_valid"
    ]:

        return {
            "is_valid": False,
            "reason": recommendation_validation[
                "reason"
            ],
            "source": recommendation_validation[
                "source"
            ],
            "claim_results": claim_results,
            "recommendation_results": (
                recommendation_validation[
                    "recommendation_results"
                ]
            ),
        }

    return {
        "is_valid": True,
        "reason": (
            f"All {len(claim_results)} factual claims are "
            "supported by the validated analytical result, "
            "and recommendations contain no unsupported "
            "factual premises."
        ),
        "source": "hybrid_claim_validator_v9",
        "claim_results": claim_results,
        "recommendation_results": (
            recommendation_validation[
                "recommendation_results"
            ]
        ),
    }


# ============================================================
# LOCAL TESTS - VERSION 9
# ============================================================


if __name__ == "__main__":

    import pandas as pd

    print(
        "Testing Report Validator Version 9"
    )

    print("=" * 70)

    passed_tests = 0
    failed_tests = 0

    # ========================================================
    # COUNT DATA
    # ========================================================

    count_question = (
        "How many repeat customers do we have?"
    )

    count_result = pd.DataFrame(
        {
            "repeat_customer_count": [
                2997
            ]
        }
    )

    # ========================================================
    # REVENUE DATA
    # ========================================================

    revenue_question = (
        "What are the top 5 product categories "
        "by total product revenue?"
    )

    revenue_result = pd.DataFrame(
        {
            "category_english": [
                "health_beauty",
                "watches_gifts",
                "bed_bath_table",
                "sports_leisure",
                "computers_accessories",
            ],
            "total_revenue": [
                1258681.34,
                1205005.68,
                1036988.68,
                988048.97,
                911954.32,
            ],
        }
    )

    # ========================================================
    # DIRECT NUMERIC-EXTRACTION REGRESSION CHECKS
    # ========================================================

    print(
        "\nTesting V9 context-aware numeric extraction"
    )

    numeric_tests = [
        (
            "Top 5 list size",
            "The top 5 product categories by total revenue are listed.",
            [],
        ),
        (
            "Rank 2 structural number",
            "Watches & gifts is rank 2 by total revenue.",
            [],
        ),
        (
            "Hash rank",
            "Watches & gifts is #2 by total revenue.",
            [],
        ),
        (
            "Revenue amount",
            "Health & beauty generated $1,258,681.34 in total revenue.",
            [1258681.34],
        ),
        (
            "Plain decimal revenue",
            "Health & beauty generated 1258681.34 in total revenue.",
            [1258681.34],
        ),
    ]

    for (
        numeric_test_name,
        numeric_claim,
        expected_values,
    ) in numeric_tests:

        actual_values = extract_numeric_values(
            numeric_claim
        )

        print(
            f"\n{numeric_test_name}"
        )

        print(
            "Claim:",
            numeric_claim,
        )

        print(
            "Expected:",
            expected_values,
        )

        print(
            "Actual:  ",
            actual_values,
        )

        if actual_values == expected_values:

            print(
                "NUMERIC TEST: PASS"
            )

        else:

            print(
                "NUMERIC TEST: FAIL"
            )

    # ========================================================
    # REPORT TESTS
    # ========================================================

    test_cases = [
        (
            "Grounded Count Report",
            count_question,
            count_result,
            """
EXECUTIVE SUMMARY
There are 2,997 repeat customers.

KEY RESULTS
- Repeat customer count: 2,997

BUSINESS IMPLICATION
Additional context is required to determine the repeat-customer rate.

RECOMMENDED NEXT STEPS
- Compare repeat customers with total unique customers.
- Calculate the repeat-customer rate.
""",
            True,
        ),
        (
            "Unsupported Strong Loyalty",
            count_question,
            count_result,
            """
EXECUTIVE SUMMARY
There are 2,997 repeat customers.

BUSINESS IMPLICATION
This shows strong customer loyalty.

RECOMMENDED NEXT STEPS
- Analyze purchase frequency.
""",
            False,
        ),
        (
            "Unsupported Retention Strength",
            count_question,
            count_result,
            """
EXECUTIVE SUMMARY
There are 2,997 repeat customers.

BUSINESS IMPLICATION
The company has strong retention.

RECOMMENDED NEXT STEPS
- Analyze purchase frequency.
""",
            False,
        ),
        (
            "Grounded Revenue Ranking",
            revenue_question,
            revenue_result,
            """
EXECUTIVE SUMMARY
Health & beauty has the highest total revenue.

KEY RESULTS
- Health & beauty generated $1,258,681.34 in total revenue.
- Watches & gifts generated $1,205,005.68 in total revenue.

BUSINESS IMPLICATION
The analysis confirms the ranking and values of the top 5 product categories by total revenue.

RECOMMENDED NEXT STEPS
- Analyze unit volume within the top categories.
- Compare category revenue across time periods.
""",
            True,
        ),
        (
            "Correct Second Place Ranking",
            revenue_question,
            revenue_result,
            """
EXECUTIVE SUMMARY
Watches & gifts ranks second by total revenue.

KEY RESULTS
- Watches & gifts generated $1,205,005.68 in total revenue.

BUSINESS IMPLICATION
The analysis confirms the ranking and values of the top 5 product categories by total revenue.

RECOMMENDED NEXT STEPS
- Compare category revenue across time periods.
""",
            True,
        ),
        (
            "Incorrect Highest Ranking",
            revenue_question,
            revenue_result,
            """
EXECUTIVE SUMMARY
Watches & gifts has the highest total revenue.

KEY RESULTS
- Watches & gifts generated $1,205,005.68 in total revenue.

BUSINESS IMPLICATION
The result identifies watches & gifts as the highest-revenue category.

RECOMMENDED NEXT STEPS
- Compare category revenue across time periods.
""",
            False,
        ),
        (
            "Incorrect Revenue Value",
            revenue_question,
            revenue_result,
            """
EXECUTIVE SUMMARY
Health & beauty has the highest total revenue.

KEY RESULTS
- Health & beauty generated $1,500,000.00 in total revenue.

BUSINESS IMPLICATION
The result identifies the highest-revenue category.

RECOMMENDED NEXT STEPS
- Compare category revenue across time periods.
""",
            False,
        ),
        (
            "Unsupported High Demand",
            revenue_question,
            revenue_result,
            """
EXECUTIVE SUMMARY
Health & beauty has the highest total revenue.

KEY RESULTS
- Health & beauty generated $1,258,681.34 in total revenue.

BUSINESS IMPLICATION
The top-performing categories are in high demand among customers.

RECOMMENDED NEXT STEPS
- Analyze category revenue across time periods.
""",
            False,
        ),
        (
            "Unsupported Customer Interest",
            revenue_question,
            revenue_result,
            """
EXECUTIVE SUMMARY
Health & beauty has the highest total revenue.

BUSINESS IMPLICATION
These results indicate strong customer interest in health and beauty products.

RECOMMENDED NEXT STEPS
- Analyze unit volume.
""",
            False,
        ),
        (
            "Unsupported Market Leadership",
            revenue_question,
            revenue_result,
            """
EXECUTIVE SUMMARY
Health & beauty has the highest total revenue.

BUSINESS IMPLICATION
The result demonstrates market leadership in health and beauty.

RECOMMENDED NEXT STEPS
- Compare category revenue over time.
""",
            False,
        ),
        (
            "Valid Cross-Selling Analysis Recommendation",
            revenue_question,
            revenue_result,
            """
EXECUTIVE SUMMARY
Health & beauty has the highest total revenue.

KEY RESULTS
- Health & beauty generated $1,258,681.34 in total revenue.

BUSINESS IMPLICATION
The result identifies health & beauty as the highest-revenue category.

RECOMMENDED NEXT STEPS
- Analyze whether customers purchase products across multiple top categories.
- Evaluate potential cross-selling opportunities using basket analysis.
""",
            True,
        ),
        (
            "Unsupported Cross-Selling Premise",
            revenue_question,
            revenue_result,
            """
EXECUTIVE SUMMARY
Health & beauty has the highest total revenue.

KEY RESULTS
- Health & beauty generated $1,258,681.34 in total revenue.

BUSINESS IMPLICATION
The result identifies health & beauty as the highest-revenue category.

RECOMMENDED NEXT STEPS
- Explore cross-selling opportunities between health & beauty and watches & gifts to capitalize on complementary product interests.
""",
            False,
        ),

        # ====================================================
        # NEW V9 REGRESSION TEST
        #
        # This exact sentence caused Reporting Agent V5 to
        # fail under Validator V8.
        #
        # The "5" must be interpreted as list size, not as a
        # revenue amount.
        # ====================================================

        (
            "Top 5 Enumeration Regression",
            revenue_question,
            revenue_result,
            """
EXECUTIVE SUMMARY
The top 5 product categories by total revenue are health beauty, watches gifts, bed bath table, sports leisure, and computers accessories.

KEY RESULTS
- Health & beauty generated $1,258,681.34 in total revenue.
- Watches & gifts generated $1,205,005.68 in total revenue.
- Bed bath table generated $1,036,988.68 in total revenue.
- Sports leisure generated $988,048.97 in total revenue.
- Computers accessories generated $911,954.32 in total revenue.

BUSINESS IMPLICATION
Additional analysis is required to determine customer demand or preferences.

RECOMMENDED NEXT STEPS
- Compare category revenue across time periods.
- Analyze unit volume within each category.
""",
            True,
        ),
    ]

    # ========================================================
    # RUN REPORT TESTS
    # ========================================================

    for (
        test_name,
        question,
        analytical_result,
        report,
        expected_result,
    ) in test_cases:

        print(
            "\n" + "-" * 70
        )

        print(
            test_name
        )

        print(
            "-" * 70
        )

        validation = validate_report(
            question=question,
            analytical_result=analytical_result,
            executive_report=report,
        )

        actual_result = validation.get(
            "is_valid",
            False,
        )

        print(
            validation
        )

        print(
            f"Expected: {expected_result}"
        )

        print(
            f"Actual:   {actual_result}"
        )

        if actual_result == expected_result:

            print(
                "RESULT: PASS"
            )

            passed_tests += 1

        else:

            print(
                "RESULT: FAIL"
            )

            failed_tests += 1

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

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
        f"Passed: {passed_tests}"
    )

    print(
        f"Failed: {failed_tests}"
    )

    print(
        f"Total:  {len(test_cases)}"
    )

    print(
        "\nReport Validator Version 9 "
        "testing completed."
    )