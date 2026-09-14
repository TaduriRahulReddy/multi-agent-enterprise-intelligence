import json
import re

from tools.llm_tool import ask_llm


# ============================================================
# REPORT VALIDATOR - VERSION 10
#
# V10 improvements over V9:
#
# 1. Preserve all V9 numeric/ranking behavior.
#
# 2. Normalize Markdown section headings:
#
#       ### EXECUTIVE SUMMARY
#       ## KEY RESULTS
#       # RECOMMENDED NEXT STEPS
#
#    are treated exactly like:
#
#       EXECUTIVE SUMMARY
#       KEY RESULTS
#       RECOMMENDED NEXT STEPS
#
# 3. Properly reconstruct numbered lists:
#
#       1. Health & Beauty ...
#       2. Watches & Gifts ...
#
#    instead of creating malformed claims such as:
#
#       "... are: 1."
#
# 4. Recommendations remain separate from factual claims.
#
# 5. Count guardrails now examine factual claims only,
#    instead of scanning the entire report.
#
#    Therefore safe recommendation language such as:
#
#       "Analyze whether loyalty differs by segment."
#
#    is not automatically rejected.
#
# 6. Recommendation premises are still validated separately.
#
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


RECOMMENDATION_SECTIONS = {
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
# V10 REPORT PARSING
# ============================================================


def normalize_section_header(
    line: str,
):
    """
    Normalize plain-text and Markdown section headings.

    Examples:

        EXECUTIVE SUMMARY
        ### EXECUTIVE SUMMARY
        ## Executive Summary
        # Recommended Next Steps

    all map to canonical section names.
    """

    cleaned = line.strip()

    # Remove Markdown heading prefix.
    cleaned = re.sub(
        r"^\s*#{1,6}\s*",
        "",
        cleaned,
    )

    # Remove optional trailing Markdown hashes.
    cleaned = re.sub(
        r"\s*#{1,6}\s*$",
        "",
        cleaned,
    )

    # Remove optional trailing colon.
    cleaned = cleaned.rstrip(
        ":"
    ).strip()

    normalized = cleaned.upper()

    if normalized in REPORT_SECTION_HEADERS:
        return normalized

    return None


def is_bullet_line(
    line: str,
) -> bool:
    """
    Detect unordered bullet lines.
    """

    return bool(
        re.match(
            r"^[\-\*\u2022]\s+",
            line,
        )
    )


def is_numbered_list_line(
    line: str,
) -> bool:
    """
    Detect numbered-list lines.

    Examples:

        1. Health & Beauty...
        2) Watches & Gifts...
    """

    return bool(
        re.match(
            r"^\d+\s*[\.\)]\s+",
            line,
        )
    )


def clean_list_prefix(
    line: str,
) -> str:
    """
    Remove bullet or numbered-list prefix.
    """

    cleaned = re.sub(
        r"^[\-\*\u2022]\s*",
        "",
        line,
    )

    cleaned = re.sub(
        r"^\d+\s*[\.\)]\s*",
        "",
        cleaned,
    )

    return cleaned.strip()


def parse_report_sections(
    executive_report: str,
) -> dict:
    """
    Parse report sections while:

    - normalizing Markdown headings,
    - reconstructing wrapped prose,
    - preserving each bullet as one entry,
    - preserving each numbered-list item as one entry.

    This prevents malformed claims such as:

        "The top 5 categories are: 1."

    which occurred in Reporting Agent V6 integration.
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

        # ----------------------------------------------------
        # Blank line ends current logical entry.
        # ----------------------------------------------------

        if not line:

            flush_current_entry()
            continue

        # ----------------------------------------------------
        # V10: normalize Markdown/plain headings.
        # ----------------------------------------------------

        section_header = normalize_section_header(
            line
        )

        if section_header is not None:

            flush_current_entry()

            current_section = section_header

            sections.setdefault(
                current_section,
                [],
            )

            continue

        # ----------------------------------------------------
        # If text occurs before a recognized heading.
        # ----------------------------------------------------

        if current_section is None:

            current_section = "UNLABELED"

            sections.setdefault(
                current_section,
                [],
            )

        # ----------------------------------------------------
        # Bullet entry
        # ----------------------------------------------------

        if is_bullet_line(
            line
        ):

            flush_current_entry()

            cleaned = clean_list_prefix(
                line
            )

            if cleaned:

                current_entry.append(
                    cleaned
                )

            continue

        # ----------------------------------------------------
        # V10: numbered-list entry
        # ----------------------------------------------------

        if is_numbered_list_line(
            line
        ):

            flush_current_entry()

            cleaned = clean_list_prefix(
                line
            )

            if cleaned:

                current_entry.append(
                    cleaned
                )

            continue

        # ----------------------------------------------------
        # Wrapped continuation line
        # ----------------------------------------------------

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

    NOTE:
    Broad prefix filtering is preserved from V9.
    It can be hardened in a later version.
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
    Extract factual claims only from non-recommendation
    sections.

    V10 guarantees Markdown recommendation headings are
    recognized before claim extraction.
    """

    sections = parse_report_sections(
        executive_report
    )

    claims = []

    for section_name, entries in sections.items():

        if section_name in RECOMMENDATION_SECTIONS:
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
    Extract recommendation entries separately.

    Markdown recommendation headings are supported.
    """

    sections = parse_report_sections(
        executive_report
    )

    recommendations = []

    for section_name in RECOMMENDATION_SECTIONS:

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
# V10 DETERMINISTIC REPORT-LEVEL GUARDRAILS
# ============================================================


def deterministic_count_check(
    analytical_result,
    executive_report: str,
) -> dict:
    """
    Prevent count-only results from supporting unsupported
    qualitative interpretations.

    V10 CHANGE:

    V9 searched the ENTIRE executive report.

    V10 searches only extracted factual claims.

    Therefore a recommendation such as:

        "Analyze whether customer loyalty differs by segment."

    does not automatically become an unsupported factual claim.
    """

    if not result_has_only_absolute_count(
        analytical_result
    ):

        return {
            "is_valid": True,
            "reason": "Not an absolute-count result.",
            "source": "deterministic_count_rule",
        }

    factual_claims = extract_factual_claims(
        executive_report
    )

    for claim in factual_claims:

        normalized_claim = normalize_text(
            claim
        )

        # ----------------------------------------------------
        # Explicit unsupported count interpretations.
        # ----------------------------------------------------

        for phrase in UNSUPPORTED_COUNT_PHRASES:

            if phrase in normalized_claim:

                return {
                    "is_valid": False,
                    "reason": (
                        "Unsupported qualitative claim detected "
                        "for an absolute-count result: "
                        f"'{phrase}' in claim '{claim}'."
                    ),
                    "source": "deterministic_count_rule",
                }

        # ----------------------------------------------------
        # Additional defensive patterns.
        #
        # Unlike V9, these operate only on factual claims.
        # ----------------------------------------------------

        unsupported_patterns = [
            (
                "loyalty is strong",
                (
                    "Unsupported loyalty-strength claim "
                    "detected for an absolute-count result."
                ),
            ),
            (
                "strong loyalty",
                (
                    "Unsupported loyalty-strength claim "
                    "detected for an absolute-count result."
                ),
            ),
            (
                "high loyalty",
                (
                    "Unsupported loyalty-strength claim "
                    "detected for an absolute-count result."
                ),
            ),
            (
                "retention is strong",
                (
                    "Unsupported retention-strength claim "
                    "detected."
                ),
            ),
            (
                "retention is high",
                (
                    "Unsupported retention-strength claim "
                    "detected."
                ),
            ),
            (
                "high retention",
                (
                    "Unsupported retention-strength claim "
                    "detected."
                ),
            ),
            (
                "strong retention",
                (
                    "Unsupported retention-strength claim "
                    "detected."
                ),
            ),
            (
                "strong base",
                (
                    "Unsupported strength claim detected "
                    "for an absolute-count result."
                ),
            ),
        ]

        for pattern, reason in unsupported_patterns:

            if pattern in normalized_claim:

                return {
                    "is_valid": False,
                    "reason": reason,
                    "source": "deterministic_count_rule",
                }

    return {
        "is_valid": True,
        "reason": (
            "No absolute-count grounding violations detected "
            "in factual claims."
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
# CONTEXT-AWARE NUMERIC EXTRACTION
# ============================================================


def is_structural_number(
    claim: str,
    start_index: int,
    end_index: int,
) -> bool:
    """
    Determine whether a numeric token describes structure,
    ranking, position, or list size rather than a metric value.

    Examples:

        top 5
        rank 2
        ranked 3
        position 4
        #2
    """

    before = claim[
        max(
            0,
            start_index - 30,
        ):start_index
    ].lower()

    immediate_before = claim[
        max(
            0,
            start_index - 1,
        ):start_index
    ]

    if immediate_before == "#":
        return True

    if re.search(
        r"\btop\s*$",
        before,
    ):
        return True

    if re.search(
        r"\b(?:rank|ranked|ranks|ranking)\s*$",
        before,
    ):
        return True

    if re.search(
        r"\b(?:position|place)\s*$",
        before,
    ):
        return True

    if re.search(
        (
            r"\b(?:top|rank|ranked|ranks|position|place)"
            r"[\s\-]*$"
        ),
        before,
    ):
        return True

    return False


def extract_numeric_values(
    claim: str,
) -> list[float]:
    """
    Extract factual metric values while excluding structural
    ranking/list numbers.
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
    return handled=False.
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

Example:

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

    Recommendations are proposals, but the reason used to
    justify them must still be grounded.
    """

    normalized = normalize_text(
        recommendation
    )

    # --------------------------------------------------------
    # Count-based recommendation premises
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
    # Revenue-based recommendation premises
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
    Report Validator Version 10.

    Validation flow:

        Executive Report
              |
        Input Validation
              |
        Structural Parsing
              |
        Markdown Header Normalization
              |
        Factual / Recommendation Separation
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
        "source": "hybrid_claim_validator_v10",
        "claim_results": claim_results,
        "recommendation_results": (
            recommendation_validation[
                "recommendation_results"
            ]
        ),
    }


# ============================================================
# LOCAL TESTS - VERSION 10
# ============================================================


if __name__ == "__main__":

    import pandas as pd

    print(
        "Testing Report Validator Version 10"
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
    # NUMERIC EXTRACTION REGRESSION
    # ========================================================

    print(
        "\nTesting V10 numeric extraction"
    )

    numeric_tests = [
        (
            "Top 5 list size",
            (
                "The top 5 product categories by total "
                "revenue are listed."
            ),
            [],
        ),
        (
            "Rank 2 structural number",
            (
                "Watches & gifts is rank 2 by total revenue."
            ),
            [],
        ),
        (
            "Hash rank",
            (
                "Watches & gifts is #2 by total revenue."
            ),
            [],
        ),
        (
            "Revenue amount",
            (
                "Health & beauty generated "
                "$1,258,681.34 in total revenue."
            ),
            [
                1258681.34
            ],
        ),
        (
            "Plain decimal revenue",
            (
                "Health & beauty generated "
                "1258681.34 in total revenue."
            ),
            [
                1258681.34
            ],
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
    # V10 PARSER REGRESSION
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "Testing V10 Markdown / numbered-list parser"
    )

    parser_report = """
### EXECUTIVE SUMMARY
The top 5 product categories by total revenue are:

1. Health & Beauty generated $1,258,681.34.
2. Watches & Gifts generated $1,205,005.68.
3. Bed Bath Table generated $1,036,988.68.
4. Sports Leisure generated $988,048.97.
5. Computers Accessories generated $911,954.32.

### RECOMMENDED NEXT STEPS
1. Conduct basket analysis.
2. Compare category performance over time.
"""

    parsed_sections = parse_report_sections(
        parser_report
    )

    parser_claims = extract_factual_claims(
        parser_report
    )

    parser_recommendations = extract_recommendations(
        parser_report
    )

    print(
        "\nParsed sections:"
    )

    print(
        parsed_sections
    )

    print(
        "\nFactual claims:"
    )

    for claim in parser_claims:
        print(
            "-",
            claim,
        )

    print(
        "\nRecommendations:"
    )

    for recommendation in parser_recommendations:
        print(
            "-",
            recommendation,
        )

    malformed_claim_detected = any(
        claim.strip().endswith(
            "1."
        )
        for claim in parser_claims
    )

    recommendation_leaked = any(
        "basket analysis" in normalize_text(
            claim
        )
        for claim in parser_claims
    )

    if (
        not malformed_claim_detected
        and not recommendation_leaked
        and len(
            parser_recommendations
        ) == 2
    ):

        print(
            "\nPARSER REGRESSION: PASS"
        )

    else:

        print(
            "\nPARSER REGRESSION: FAIL"
        )

    # ========================================================
    # REPORT TEST CASES
    # ========================================================

    test_cases = [
        # ----------------------------------------------------
        # V9 baseline tests
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # NEW V10 regressions
        # ----------------------------------------------------

        (
            "Markdown Revenue Report",
            revenue_question,
            revenue_result,
            """
### EXECUTIVE SUMMARY
Health & beauty has the highest total revenue.

### KEY RESULTS
1. Health & beauty generated $1,258,681.34 in total revenue.
2. Watches & gifts generated $1,205,005.68 in total revenue.
3. Bed bath table generated $1,036,988.68 in total revenue.
4. Sports leisure generated $988,048.97 in total revenue.
5. Computers accessories generated $911,954.32 in total revenue.

### BUSINESS IMPLICATION
Additional analysis is required to determine customer demand or preferences.

### RECOMMENDED NEXT STEPS
1. Conduct basket analysis to evaluate possible cross-selling opportunities.
2. Compare category revenue across time periods.
""",
            True,
        ),

        (
            "Markdown Recommendation Must Not Become Fact",
            revenue_question,
            revenue_result,
            """
### EXECUTIVE SUMMARY
Health & beauty has the highest total revenue.

### KEY RESULTS
- Health & beauty generated $1,258,681.34 in total revenue.

### RECOMMENDED NEXT STEPS
- Conduct basket analysis to evaluate whether cross-selling opportunities exist between health & beauty and watches & gifts.
""",
            True,
        ),

        (
            "Safe Count Recommendation Mentioning Loyalty",
            count_question,
            count_result,
            """
### EXECUTIVE SUMMARY
There are 2,997 repeat customers.

### KEY RESULTS
- Repeat customer count: 2,997

### BUSINESS IMPLICATION
Additional context is required to determine retention or loyalty strength.

### RECOMMENDED NEXT STEPS
- Analyze whether customer loyalty differs by segment.
- Compare repeat customers with total unique customers.
""",
            True,
        ),

        (
            "Unsafe Count Fact Still Rejected",
            count_question,
            count_result,
            """
### EXECUTIVE SUMMARY
There are 2,997 repeat customers.

### BUSINESS IMPLICATION
The repeat-customer count demonstrates strong customer loyalty.

### RECOMMENDED NEXT STEPS
- Analyze purchase frequency.
""",
            False,
        ),

        (
            "Unsafe Revenue Fact With Markdown",
            revenue_question,
            revenue_result,
            """
### EXECUTIVE SUMMARY
Health & beauty has the highest total revenue.

### BUSINESS IMPLICATION
The leading revenue categories demonstrate strong customer interest.

### RECOMMENDED NEXT STEPS
- Compare revenue across time periods.
""",
            False,
        ),

        (
            "Unsafe Revenue Recommendation Premise With Markdown",
            revenue_question,
            revenue_result,
            """
### EXECUTIVE SUMMARY
Health & beauty has the highest total revenue.

### RECOMMENDED NEXT STEPS
- Explore cross-selling opportunities between health & beauty and watches & gifts to capitalize on complementary product interests.
""",
            False,
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
        "REPORT VALIDATOR VERSION 10 TEST SUMMARY"
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
        "\nReport Validator Version 10 "
        "testing completed."
    )