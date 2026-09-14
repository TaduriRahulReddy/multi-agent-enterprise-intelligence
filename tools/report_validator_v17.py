import json
import re

from tools.llm_tool import ask_llm


# ============================================================
# REPORT VALIDATOR - VERSION 17
#
# V17 improvements over V16:
#
# 1. Preserves all V16 deterministic numeric/ranking logic.
#
# 2. Preserves ordered multi-entity revenue validation.
#
# 3. Preserves negation-aware count validation.
#
# 4. Preserves mixed-polarity / contrast-clause detection.
#
# 5. Adds deterministic multi-entity revenue/value validation.
#
# 6. Correctly validates sentences such as:
#
#       "Health & Beauty leads with $1,258,681.34,
#        followed by Watches & Gifts at $1,205,005.68."
#
# 7. Each entity is paired with its corresponding revenue
#    value in textual order.
#
# 8. Prevents the V16 failure where the final number in a
#    multi-entity sentence could be compared against the first
#    entity.
#
# 9. V16 remains preserved as the previous stable validator.
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
    "BUSINESS IMPACT",
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
    Normalize entity names for deterministic matching.
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
    Convert analytical result to readable text.
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
    Detect a one-row / one-column count result.
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
    Find a revenue-like metric column.
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
    Find an entity/category column.
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


def normalize_section_header(
    line: str,
):
    """
    Normalize plain-text and Markdown headings.
    """

    cleaned = line.strip()

    cleaned = re.sub(
        r"^\s*#{1,6}\s*",
        "",
        cleaned,
    ).strip()

    cleaned = re.sub(
        r"\s*#{1,6}\s*$",
        "",
        cleaned,
    ).strip()

    bold_match = re.fullmatch(
        r"\*\*(.+?)\*\*",
        cleaned,
    )

    if bold_match:

        cleaned = bold_match.group(
            1
        ).strip()

    underscore_match = re.fullmatch(
        r"__(.+?)__",
        cleaned,
    )

    if underscore_match:

        cleaned = underscore_match.group(
            1
        ).strip()

    cleaned = cleaned.rstrip(
        ":"
    ).strip()

    normalized = cleaned.upper()

    if normalized in REPORT_SECTION_HEADERS:
        return normalized

    return None


def is_markdown_separator(
    line: str,
) -> bool:
    """
    Detect Markdown horizontal separators.
    """

    cleaned = line.strip()

    return bool(
        re.fullmatch(
            r"(?:-{3,}|\*{3,}|_{3,})",
            cleaned,
        )
    )


def is_bullet_line(
    line: str,
) -> bool:
    """
    Detect bullet lines.
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
    Remove Markdown bullet/number prefix.
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
    Parse report sections and entries.
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

        if is_markdown_separator(
            line
        ):

            flush_current_entry()
            continue

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

        if current_section is None:

            current_section = "UNLABELED"

            sections.setdefault(
                current_section,
                [],
            )

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

        current_entry.append(
            line
        )

    flush_current_entry()

    return sections


def split_into_sentences(
    text: str,
) -> list[str]:
    """
    Split reconstructed text into sentences.
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
    Detect control/meta language.
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
        "additional analysis may be required",
        "further analysis may be required",
        "additional analysis is needed",
        "further analysis is needed",
    ]

    if any(
        normalized.startswith(
            prefix
        )
        for prefix in meta_prefixes
    ):

        return True

    report_meta_patterns = [
        r"^this report provides an overview\b",
        r"^this report provides an analysis\b",
        r"^this report provides a summary\b",
        r"^this report presents an overview\b",
        r"^this report presents an analysis\b",
        r"^this report presents a summary\b",
        r"^this report summarizes\b",

        r"^this executive report provides an overview\b",
        r"^this executive report provides an analysis\b",
        r"^this executive report provides a summary\b",
        r"^this executive report presents an overview\b",
        r"^this executive report presents an analysis\b",
        r"^this executive report presents a summary\b",
        r"^this executive report summarizes\b",
    ]

    for pattern in report_meta_patterns:

        if re.search(
            pattern,
            normalized,
        ):

            return True

    return False


def extract_factual_claims(
    executive_report: str,
) -> list[str]:
    """
    Extract factual claims outside recommendation sections.
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
# NEGATION / POLARITY HELPERS
# ============================================================


def unsupported_phrase_is_negated(
    normalized_claim: str,
    phrase: str,
    phrase_start: int,
) -> bool:
    """
    Determine whether an unsupported phrase is explicitly
    negated within its local clause.

    Prevent negation from leaking across commas or major
    clause boundaries.
    """

    context_before = normalized_claim[
        max(
            0,
            phrase_start - 140,
        ):phrase_start
    ]

    negation_patterns = [
        (
            r"\b(?:does not|do not|did not|cannot|can not|"
            r"doesn't|don't|didn't|can't)\s+"
            r"(?:establish|demonstrate|indicate|show|prove|"
            r"confirm|support|mean|imply|suggest)\b"
            r"[^,.!?;:]{0,100}$"
        ),
        (
            r"\b(?:is not|are not|was not|were not)\s+"
            r"(?:evidence of|proof of|an indication of|"
            r"confirmation of)\s*"
            r"[^,.!?;:]{0,100}$"
        ),
        (
            r"\bnot enough\s+"
            r"(?:to|evidence to)\s*"
            r"(?:establish|demonstrate|indicate|show|prove|"
            r"confirm|support|conclude|infer)\b"
            r"[^,.!?;:]{0,100}$"
        ),
        (
            r"\binsufficient\s+"
            r"(?:to|evidence to)\s*"
            r"(?:establish|demonstrate|indicate|show|prove|"
            r"confirm|support|conclude|infer)\b"
            r"[^,.!?;:]{0,100}$"
        ),
    ]

    return any(
        re.search(
            pattern,
            context_before,
        )
        for pattern in negation_patterns
    )


def contains_non_negated_phrase(
    normalized_claim: str,
    phrase: str,
) -> bool:
    """
    Return True if at least one occurrence is positively
    asserted rather than safely negated.
    """

    search_start = 0

    while True:

        phrase_start = normalized_claim.find(
            phrase,
            search_start,
        )

        if phrase_start == -1:
            return False

        if not unsupported_phrase_is_negated(
            normalized_claim=normalized_claim,
            phrase=phrase,
            phrase_start=phrase_start,
        ):

            return True

        search_start = (
            phrase_start
            + len(
                phrase
            )
        )


def split_contrast_clauses(
    claim: str,
) -> list[str]:
    """
    Split a claim around contrast markers.
    """

    normalized = normalize_text(
        claim
    )

    parts = re.split(
        (
            r"(?:,\s*)?"
            r"\b(?:but|however|yet|although|though|"
            r"nevertheless)\b"
            r"(?:,\s*)?"
        ),
        normalized,
    )

    return [
        part.strip(
            " ,;:-"
        )
        for part in parts
        if part.strip(
            " ,;:-"
        )
    ]


def contrast_clause_has_positive_unsupported_assertion(
    claim: str,
) -> bool:
    """
    Detect unsupported positive assertions after a contrast
    marker.
    """

    clauses = split_contrast_clauses(
        claim
    )

    if len(
        clauses
    ) < 2:

        return False

    direct_patterns = [
        "loyalty is strong",
        "loyalty is high",
        "strong loyalty",
        "high loyalty",

        "customer loyalty is strong",
        "customer loyalty is high",
        "strong customer loyalty",
        "high customer loyalty",

        "retention is strong",
        "retention is high",
        "strong retention",
        "high retention",

        "customer retention is strong",
        "customer retention is high",
        "strong customer retention",
        "high customer retention",

        "engagement is improving",
        "customer engagement is improving",

        "engagement is strong",
        "customer engagement is strong",

        "positive trend",
        "strong trend",

        "customer satisfaction is high",
        "customer satisfaction is strong",
    ]

    for clause in clauses[1:]:

        for pattern in direct_patterns:

            if pattern in clause:

                return True

        for phrase in UNSUPPORTED_COUNT_PHRASES:

            if contains_non_negated_phrase(
                normalized_claim=clause,
                phrase=phrase,
            ):

                return True

    return False


def contains_negated_unsupported_count_phrase(
    claim: str,
) -> bool:
    """
    Return True when the claim contains at least one unsupported
    count phrase and all occurrences are safely negated.
    """

    normalized_claim = normalize_text(
        claim
    )

    found_negated_phrase = False

    for phrase in UNSUPPORTED_COUNT_PHRASES:

        search_start = 0

        while True:

            phrase_start = normalized_claim.find(
                phrase,
                search_start,
            )

            if phrase_start == -1:
                break

            if unsupported_phrase_is_negated(
                normalized_claim=normalized_claim,
                phrase=phrase,
                phrase_start=phrase_start,
            ):

                found_negated_phrase = True

            else:

                return False

            search_start = (
                phrase_start
                + len(
                    phrase
                )
            )

    return found_negated_phrase


def count_claim_has_positive_unsupported_assertion(
    claim: str,
) -> bool:
    """
    Detect unsupported qualitative assertions in a count-only
    claim.
    """

    normalized_claim = normalize_text(
        claim
    )

    for phrase in UNSUPPORTED_COUNT_PHRASES:

        if contains_non_negated_phrase(
            normalized_claim=normalized_claim,
            phrase=phrase,
        ):

            return True

    direct_patterns = [
        "loyalty is strong",
        "loyalty is high",
        "strong loyalty",
        "high loyalty",

        "customer loyalty is strong",
        "customer loyalty is high",

        "retention is strong",
        "retention is high",
        "high retention",
        "strong retention",

        "customer retention is strong",
        "customer retention is high",

        "strong base",

        "engagement is improving",
        "customer engagement is improving",
    ]

    for pattern in direct_patterns:

        if contains_non_negated_phrase(
            normalized_claim=normalized_claim,
            phrase=pattern,
        ):

            return True

    return False


def deterministic_count_claim_validation(
    analytical_result,
    claim: str,
) -> dict:
    """
    Deterministically validate explicit negation of unsupported
    qualitative count conclusions.
    """

    if not result_has_only_absolute_count(
        analytical_result
    ):

        return {
            "handled": False,
        }

    if contrast_clause_has_positive_unsupported_assertion(
        claim
    ):

        return {
            "handled": True,
            "supported": False,
            "supporting_evidence": "",
            "reason": (
                "The claim contains an explicitly negated "
                "statement followed by a positive unsupported "
                "qualitative assertion."
            ),
            "source": (
                "deterministic_count_polarity_validator"
            ),
        }

    if count_claim_has_positive_unsupported_assertion(
        claim
    ):

        return {
            "handled": False,
        }

    if not contains_negated_unsupported_count_phrase(
        claim
    ):

        return {
            "handled": False,
        }

    return {
        "handled": True,
        "supported": True,
        "supporting_evidence": (
            "The analytical result contains only an absolute "
            "count, and the claim explicitly states that the "
            "count does not establish the qualitative "
            "conclusion."
        ),
        "reason": (
            "Python verified that the unsupported qualitative "
            "count conclusion is explicitly negated rather "
            "than asserted."
        ),
        "source": "deterministic_count_negation_validator",
    }


# ============================================================
# DETERMINISTIC REPORT-LEVEL GUARDRAILS
# ============================================================


def deterministic_count_check(
    analytical_result,
    executive_report: str,
) -> dict:
    """
    Reject unsupported qualitative interpretations of an
    absolute count.
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

        if contrast_clause_has_positive_unsupported_assertion(
            claim
        ):

            return {
                "is_valid": False,
                "reason": (
                    "Unsupported positive qualitative assertion "
                    "detected after a contrast clause in claim "
                    f"'{claim}'."
                ),
                "source": (
                    "deterministic_count_polarity_rule"
                ),
            }

        normalized_claim = normalize_text(
            claim
        )

        for phrase in UNSUPPORTED_COUNT_PHRASES:

            if contains_non_negated_phrase(
                normalized_claim=normalized_claim,
                phrase=phrase,
            ):

                return {
                    "is_valid": False,
                    "reason": (
                        "Unsupported qualitative claim detected "
                        "for an absolute-count result: "
                        f"'{phrase}' in claim '{claim}'."
                    ),
                    "source": "deterministic_count_rule",
                }

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

            if contains_non_negated_phrase(
                normalized_claim=normalized_claim,
                phrase=pattern,
            ):

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
    Reject unsupported semantic interpretations of revenue
    rankings.
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
    Find one known ranked entity in a claim.
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
            item[
                "normalized_entity"
            ]
        ),
        reverse=True,
    )

    return matches[0]


# ============================================================
# MULTI-ENTITY ORDER HELPERS
# ============================================================


def find_all_entity_records_in_claim(
    claim: str,
    ranking_records: list[dict],
) -> list[dict]:
    """
    Find all known entities in textual order.
    """

    normalized_claim = normalize_entity_name(
        claim
    )

    matches = []

    for record in ranking_records:

        entity_name = record[
            "normalized_entity"
        ]

        position = normalized_claim.find(
            entity_name
        )

        if position == -1:
            continue

        matches.append(
            {
                **record,
                "_text_position": position,
            }
        )

    matches.sort(
        key=lambda item: item[
            "_text_position"
        ]
    )

    return matches


def looks_like_ordered_revenue_sequence(
    claim: str,
) -> bool:
    """
    Detect ordered ranking language.
    """

    normalized = normalize_text(
        claim
    )

    has_highest_language = any(
        phrase in normalized
        for phrase in [
            "highest",
            "highest-revenue",
            "highest revenue",
            "ranks first",
            "ranked first",
            "comes first",
            "first place",
        ]
    )

    has_followed_language = any(
        phrase in normalized
        for phrase in [
            "followed by",
            "followed closely by",
        ]
    )

    return (
        has_highest_language
        and has_followed_language
        and claim_mentions_revenue(
            claim
        )
    )


def deterministic_ordered_revenue_validation(
    claim: str,
    ranking_records: list[dict],
) -> dict:
    """
    Validate ordered multi-entity revenue statements.
    """

    if not looks_like_ordered_revenue_sequence(
        claim
    ):

        return {
            "handled": False,
        }

    ordered_entities = find_all_entity_records_in_claim(
        claim=claim,
        ranking_records=ranking_records,
    )

    if len(
        ordered_entities
    ) < 2:

        return {
            "handled": False,
        }

    expected_rank = 1

    for record in ordered_entities:

        actual_rank = record[
            "rank"
        ]

        if actual_rank != expected_rank:

            return {
                "handled": True,
                "supported": False,
                "supporting_evidence": "",
                "reason": (
                    f"The ordered revenue claim places "
                    f"{record['entity']} at position "
                    f"{expected_rank}, but it is rank "
                    f"{actual_rank} in the validated "
                    f"analytical result."
                ),
                "source": (
                    "deterministic_ordered_rank_validator"
                ),
            }

        expected_rank += 1

    evidence_parts = []

    for record in ordered_entities:

        evidence_parts.append(
            (
                f"{record['entity']}="
                f"rank {record['rank']}"
            )
        )

    return {
        "handled": True,
        "supported": True,
        "supporting_evidence": " | ".join(
            evidence_parts
        ),
        "reason": (
            "Python verified the complete ordered revenue "
            "sequence directly against the analytical result."
        ),
        "source": (
            "deterministic_ordered_rank_validator"
        ),
    }


# ============================================================
# CONTEXT-AWARE NUMERIC EXTRACTION
# ============================================================


def is_structural_number(
    claim: str,
    start_index: int,
    end_index: int,
) -> bool:
    """
    Determine whether a number is structural rather than a
    metric value.
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
    Extract metric values while excluding rank/list numbers.
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
    Compare metric values safely.
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
    Detect ordinal ranking language.
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
    Determine whether claim discusses revenue.
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
# V17 MULTI-ENTITY REVENUE/VALUE VALIDATION
# ============================================================


def deterministic_multi_entity_revenue_validation(
    claim: str,
    ranking_records: list[dict],
) -> dict:
    """
    Validate claims containing multiple known revenue entities
    and multiple revenue values.

    Example:

        "Health & Beauty leads with a total revenue of
         $1,258,681.34, followed closely by Watches & Gifts
         at $1,205,005.68."

    V16 could incorrectly find Health & Beauty as the primary
    entity and then compare the final numeric value against
    Health & Beauty.

    V17 instead:

        1. Finds every known entity in textual order.
        2. Extracts metric values in textual order.
        3. Pairs entity 1 with value 1.
        4. Pairs entity 2 with value 2.
        5. Validates every pair independently.
    """

    if not claim_mentions_revenue(
        claim
    ):

        return {
            "handled": False,
            "multi_entity": False,
        }

    ordered_entities = find_all_entity_records_in_claim(
        claim=claim,
        ranking_records=ranking_records,
    )

    if len(
        ordered_entities
    ) < 2:

        return {
            "handled": False,
            "multi_entity": False,
        }

    numeric_values = extract_numeric_values(
        claim
    )

    # --------------------------------------------------------
    # This is clearly a multi-entity claim, but Python cannot
    # safely pair entities and values.
    #
    # Do not allow the old single-entity logic to process it.
    # Let claim-level semantic validation evaluate the complete
    # sentence instead.
    # --------------------------------------------------------

    if len(
        numeric_values
    ) != len(
        ordered_entities
    ):

        return {
            "handled": False,
            "multi_entity": True,
        }

    evidence_parts = []

    for record, claimed_value in zip(
        ordered_entities,
        numeric_values,
    ):

        expected_value = record[
            "value"
        ]

        if not approximately_equal(
            claimed_value,
            expected_value,
        ):

            return {
                "handled": True,
                "multi_entity": True,
                "supported": False,
                "supporting_evidence": "",
                "reason": (
                    f"The validated revenue for "
                    f"{record['entity']} is "
                    f"{expected_value:.2f}, not "
                    f"{claimed_value:.2f}."
                ),
                "source": (
                    "deterministic_multi_entity_"
                    "revenue_validator"
                ),
            }

        evidence_parts.append(
            (
                f"{record['entity']} | "
                f"revenue={expected_value:.2f}"
            )
        )

    normalized_claim = normalize_text(
        claim
    )

    first_place_patterns = [
        "leads with",
        "leads at",
        "leads by",
        "highest",
        "highest-revenue",
        "highest revenue",
        "ranks first",
        "ranked first",
        "comes first",
        "first place",
    ]

    implies_first_place = any(
        pattern in normalized_claim
        for pattern in first_place_patterns
    )

    if implies_first_place:

        first_entity = ordered_entities[
            0
        ]

        if first_entity[
            "rank"
        ] != 1:

            return {
                "handled": True,
                "multi_entity": True,
                "supported": False,
                "supporting_evidence": "",
                "reason": (
                    f"The claim presents "
                    f"{first_entity['entity']} as the leading "
                    f"revenue entity, but it is rank "
                    f"{first_entity['rank']} in the validated "
                    f"analytical result."
                ),
                "source": (
                    "deterministic_multi_entity_"
                    "revenue_validator"
                ),
            }

    return {
        "handled": True,
        "multi_entity": True,
        "supported": True,
        "supporting_evidence": " | ".join(
            evidence_parts
        ),
        "reason": (
            "Python paired each referenced entity with its "
            "corresponding revenue value in textual order and "
            "verified every pair against the analytical result."
        ),
        "source": (
            "deterministic_multi_entity_revenue_validator"
        ),
    }


# ============================================================
# DETERMINISTIC REVENUE CLAIM VALIDATION
# ============================================================


def deterministic_revenue_claim_validation(
    analytical_result,
    claim: str,
) -> dict:
    """
    Validate revenue/ranking claims using Python.

    V17 validation order:

    1. Ordered multi-entity ranking sequence
    2. Multi-entity revenue/value pairing
    3. Single-entity rank/value validation
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

    # ========================================================
    # ORDERED MULTI-ENTITY RANKING
    # ========================================================

    ordered_result = (
        deterministic_ordered_revenue_validation(
            claim=claim,
            ranking_records=ranking_records,
        )
    )

    if ordered_result.get(
        "handled",
        False,
    ):

        return ordered_result

    # ========================================================
    # V17 MULTI-ENTITY REVENUE/VALUE VALIDATION
    # ========================================================

    multi_entity_result = (
        deterministic_multi_entity_revenue_validation(
            claim=claim,
            ranking_records=ranking_records,
        )
    )

    if multi_entity_result.get(
        "handled",
        False,
    ):

        return multi_entity_result

    # --------------------------------------------------------
    # Important V17 protection:
    #
    # If this is a recognized multi-entity claim but Python
    # cannot safely pair its values, do not apply the V16
    # single-entity validator.
    # --------------------------------------------------------

    if multi_entity_result.get(
        "multi_entity",
        False,
    ):

        return {
            "handled": False,
        }

    # ========================================================
    # SINGLE-ENTITY VALIDATION
    # ========================================================

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
    # RANK + VALUE
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

        claimed_value = numeric_values[
            -1
        ]

        if not approximately_equal(
            claimed_value,
            entity_record[
                "value"
            ],
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
    # RANK ONLY
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
    # VALUE ONLY
    # ========================================================

    if (
        numeric_values
        and mentions_revenue
    ):

        claimed_value = numeric_values[
            -1
        ]

        if approximately_equal(
            claimed_value,
            entity_record[
                "value"
            ],
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
   - positive or negative trends
   - improving or declining engagement

8. Numbers describing list size or structure are not metric
   values.

9. If a claim contains multiple factual assertions, every
   material assertion must be supported.

10. Human-readable transformations of column names are valid.

11. A single absolute count does not prove a trend.

12. When uncertain, return false.

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
    Parse JSON response safely.
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
    Validate one factual claim.

    V17 validation order:

    1. Deterministic count polarity/negation validation
    2. Deterministic revenue validation
    3. LLM semantic validation
    """

    count_result = (
        deterministic_count_claim_validation(
            analytical_result=analytical_result,
            claim=claim,
        )
    )

    if count_result.get(
        "handled",
        False,
    ):

        return {
            "claim": claim,
            "supported": count_result[
                "supported"
            ],
            "supporting_evidence": count_result[
                "supporting_evidence"
            ],
            "reason": count_result[
                "reason"
            ],
            "source": count_result[
                "source"
            ],
        }

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
    Check recommendation premises.
    """

    normalized = normalize_text(
        recommendation
    )

    if result_has_only_absolute_count(
        analytical_result
    ):

        for phrase in UNSUPPORTED_COUNT_PHRASES:

            if contains_non_negated_phrase(
                normalized_claim=normalized,
                phrase=phrase,
            ):

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
    Validate recommendation entries.
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
    Report Validator Version 17.
    """

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
        "source": "hybrid_claim_validator_v17",
        "claim_results": claim_results,
        "recommendation_results": (
            recommendation_validation[
                "recommendation_results"
            ]
        ),
    }


# ============================================================
# LOCAL TESTS - VERSION 17
# ============================================================


if __name__ == "__main__":

    import pandas as pd

    print(
        "Testing Report Validator Version 17"
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
    # NUMERIC EXTRACTION TESTS
    # ========================================================

    print(
        "\nTesting V17 numeric extraction"
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
    # PARSER TEST
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "Testing V17 Markdown / numbered-list parser"
    )

    parser_report = """
**EXECUTIVE SUMMARY**
The top 5 product categories by total revenue are:

1. Health & Beauty generated $1,258,681.34.
2. Watches & Gifts generated $1,205,005.68.
3. Bed Bath Table generated $1,036,988.68.
4. Sports Leisure generated $988,048.97.
5. Computers Accessories generated $911,954.32.

---

**RECOMMENDED NEXT STEPS**
1. Conduct basket analysis.
2. Compare category performance over time.
"""

    parser_claims = extract_factual_claims(
        parser_report
    )

    parser_recommendations = extract_recommendations(
        parser_report
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

    separator_leaked = any(
        claim.strip() in {
            "---",
            "***",
            "___",
        }
        for claim in parser_claims
    )

    if (
        not malformed_claim_detected
        and not recommendation_leaked
        and not separator_leaked
        and len(
            parser_recommendations
        ) == 2
    ):

        print(
            "PARSER REGRESSION: PASS"
        )

    else:

        print(
            "PARSER REGRESSION: FAIL"
        )

    # ========================================================
    # ORDERED SEQUENCE DIAGNOSTICS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "Testing V17 ordered revenue sequence logic"
    )

    ranking_records = build_revenue_ranking_records(
        revenue_result
    )

    ordered_diagnostics = [
        (
            "Correct Full Sequence",
            (
                "Health & Beauty is the highest-revenue "
                "category, followed by Watches & Gifts, "
                "Bed & Bath & Table, Sports & Leisure, "
                "and Computers & Accessories."
            ),
            True,
        ),
        (
            "Correct Partial Sequence",
            (
                "Health & Beauty is the highest-revenue "
                "category, followed by Watches & Gifts "
                "and Bed & Bath & Table."
            ),
            True,
        ),
        (
            "Incorrect Second Position",
            (
                "Health & Beauty is the highest-revenue "
                "category, followed by Computers & "
                "Accessories and Watches & Gifts."
            ),
            False,
        ),
        (
            "Incorrect Highest Entity",
            (
                "Computers & Accessories is the "
                "highest-revenue category, followed by "
                "Health & Beauty and Watches & Gifts."
            ),
            False,
        ),
    ]

    for (
        diagnostic_name,
        diagnostic_claim,
        expected_supported,
    ) in ordered_diagnostics:

        diagnostic_result = (
            deterministic_ordered_revenue_validation(
                claim=diagnostic_claim,
                ranking_records=ranking_records,
            )
        )

        actual_supported = diagnostic_result.get(
            "supported",
            False,
        )

        print(
            f"\n{diagnostic_name}"
        )

        print(
            diagnostic_result
        )

        print(
            "Expected:",
            expected_supported,
        )

        print(
            "Actual:  ",
            actual_supported,
        )

        if (
            diagnostic_result.get(
                "handled",
                False,
            )
            and actual_supported
            == expected_supported
        ):

            print(
                "ORDERED SEQUENCE TEST: PASS"
            )

        else:

            print(
                "ORDERED SEQUENCE TEST: FAIL"
            )

    # ========================================================
    # V17 MULTI-ENTITY REVENUE DIAGNOSTICS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "Testing V17 multi-entity revenue/value logic"
    )

    multi_entity_diagnostics = [
        (
            "Correct Two-Entity Revenue Pairing",
            (
                "Health & Beauty leads with a total revenue "
                "of $1,258,681.34, followed closely by "
                "Watches & Gifts at $1,205,005.68."
            ),
            True,
        ),
        (
            "Incorrect Second Revenue Pair",
            (
                "Health & Beauty leads with a total revenue "
                "of $1,258,681.34, followed closely by "
                "Watches & Gifts at $999,999.99."
            ),
            False,
        ),
        (
            "Incorrect First Revenue Pair",
            (
                "Health & Beauty leads with a total revenue "
                "of $999,999.99, followed closely by "
                "Watches & Gifts at $1,205,005.68."
            ),
            False,
        ),
    ]

    for (
        diagnostic_name,
        diagnostic_claim,
        expected_supported,
    ) in multi_entity_diagnostics:

        diagnostic_result = (
            deterministic_multi_entity_revenue_validation(
                claim=diagnostic_claim,
                ranking_records=ranking_records,
            )
        )

        actual_supported = diagnostic_result.get(
            "supported",
            False,
        )

        print(
            f"\n{diagnostic_name}"
        )

        print(
            diagnostic_result
        )

        print(
            "Expected:",
            expected_supported,
        )

        print(
            "Actual:  ",
            actual_supported,
        )

        if (
            diagnostic_result.get(
                "handled",
                False,
            )
            and actual_supported
            == expected_supported
        ):

            print(
                "MULTI-ENTITY TEST: PASS"
            )

        else:

            print(
                "MULTI-ENTITY TEST: FAIL"
            )

    # ========================================================
    # REPORT TEST CASES
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

        (
            "Bold Markdown Recommendation Separation",
            revenue_question,
            revenue_result,
            """
**EXECUTIVE SUMMARY**
Health & beauty has the highest total revenue.

---

**KEY RESULTS**
- Health & beauty generated $1,258,681.34 in total revenue.
- Watches & gifts generated $1,205,005.68 in total revenue.

---

**BUSINESS IMPLICATION**
Additional analysis is required to determine customer behavior.

---

**RECOMMENDED NEXT STEPS**
Use basket analysis to determine whether cross-selling opportunities exist between these top categories.
""",
            True,
        ),

        (
            "Pure Report Meta Sentence",
            count_question,
            count_result,
            """
**EXECUTIVE SUMMARY**
This report provides an overview of repeat customer data for the company.

There are 2,997 repeat customers.

**KEY RESULTS**
- Repeat customer count: 2,997

**BUSINESS IMPLICATION**
Additional analysis is required to determine retention strength.

**RECOMMENDED NEXT STEPS**
- Calculate the repeat-customer rate.
""",
            True,
        ),

        (
            "Bold Markdown Unsafe Loyalty Fact",
            count_question,
            count_result,
            """
**EXECUTIVE SUMMARY**
There are 2,997 repeat customers.

**BUSINESS IMPLICATION**
The repeat-customer count demonstrates strong customer loyalty.

**RECOMMENDED NEXT STEPS**
Analyze purchase frequency.
""",
            False,
        ),

        (
            "Bold Recommendation Not Factual Claim",
            revenue_question,
            revenue_result,
            """
**EXECUTIVE SUMMARY**
Health & beauty has the highest total revenue.

**KEY RESULTS**
- Health & beauty generated $1,258,681.34 in total revenue.

**BUSINESS IMPLICATION**
Additional analysis is required to understand purchasing relationships.

**RECOMMENDED NEXT STEPS**
- Conduct basket analysis to evaluate whether cross-selling opportunities exist between health & beauty and watches & gifts.
- Compare category revenue across time periods.
""",
            True,
        ),

        (
            "Executive Report Meta Sentence",
            count_question,
            count_result,
            """
**EXECUTIVE SUMMARY**
This executive report provides an analysis based on validated data regarding repeat customer count for the organization.

There are 2,997 repeat customers.

**KEY RESULTS**
- Repeat customer count: 2,997

**BUSINESS IMPLICATION**
Additional analysis is required to determine retention strength.

**RECOMMENDED NEXT STEPS**
- Calculate the repeat-customer rate.
""",
            True,
        ),

        (
            "Further Analysis May Be Required",
            count_question,
            count_result,
            """
**EXECUTIVE SUMMARY**
There are 2,997 repeat customers.

**BUSINESS IMPLICATION**
Further analysis may be required to understand the factors contributing to customer retention and to explore opportunities for enhancing customer engagement.

**RECOMMENDED NEXT STEPS**
- Analyze purchase frequency.
""",
            True,
        ),

        (
            "Positive Trend Still Rejected",
            count_question,
            count_result,
            """
**EXECUTIVE SUMMARY**
There are 2,997 repeat customers.

**BUSINESS IMPLICATION**
The presence of 2,997 repeat customers highlights a positive trend in repeat business.

**RECOMMENDED NEXT STEPS**
- Analyze purchase frequency.
""",
            False,
        ),

        (
            "Improving Engagement Still Rejected",
            count_question,
            count_result,
            """
**EXECUTIVE SUMMARY**
There are 2,997 repeat customers.

**BUSINESS IMPLICATION**
Customer engagement is improving.

**RECOMMENDED NEXT STEPS**
- Analyze purchase frequency.
""",
            False,
        ),

        (
            "Ordered Full Revenue Sequence",
            revenue_question,
            revenue_result,
            """
EXECUTIVE SUMMARY
The validated analytical result lists Health & Beauty as the highest-revenue category, followed by Watches & Gifts, Bed & Bath & Table, Sports & Leisure, and Computers & Accessories.

KEY RESULTS
- Health & Beauty generated $1,258,681.34 in total revenue.
- Watches & Gifts generated $1,205,005.68 in total revenue.
- Bed & Bath & Table generated $1,036,988.68 in total revenue.
- Sports & Leisure generated $988,048.97 in total revenue.
- Computers & Accessories generated $911,954.32 in total revenue.

BUSINESS IMPLICATION
Additional analysis is required to understand category performance over time.

RECOMMENDED NEXT STEPS
- Compare category revenue across time periods.
""",
            True,
        ),

        (
            "Ordered Partial Revenue Sequence",
            revenue_question,
            revenue_result,
            """
EXECUTIVE SUMMARY
Health & Beauty is the highest-revenue category, followed by Watches & Gifts and Bed & Bath & Table.

RECOMMENDED NEXT STEPS
- Compare category revenue across time periods.
""",
            True,
        ),

        (
            "Incorrect Ordered Revenue Sequence",
            revenue_question,
            revenue_result,
            """
EXECUTIVE SUMMARY
Health & Beauty is the highest-revenue category, followed by Computers & Accessories and Watches & Gifts.

RECOMMENDED NEXT STEPS
- Compare category revenue across time periods.
""",
            False,
        ),

        (
            "Incorrect Ordered Highest Entity",
            revenue_question,
            revenue_result,
            """
EXECUTIVE SUMMARY
Computers & Accessories is the highest-revenue category, followed by Health & Beauty and Watches & Gifts.

RECOMMENDED NEXT STEPS
- Compare category revenue across time periods.
""",
            False,
        ),

        (
            "Negated High Retention",
            count_question,
            count_result,
            """
EXECUTIVE SUMMARY
There are 2,997 repeat customers.

BUSINESS IMPLICATION
However, this alone does not establish strong customer retention or high retention rates.

RECOMMENDED NEXT STEPS
- Calculate the repeat-customer rate.
""",
            True,
        ),

        (
            "Negated Strong Loyalty",
            count_question,
            count_result,
            """
EXECUTIVE SUMMARY
There are 2,997 repeat customers.

BUSINESS IMPLICATION
The repeat-customer count does not prove strong customer loyalty.

RECOMMENDED NEXT STEPS
- Analyze purchase frequency.
""",
            True,
        ),

        (
            "Cannot Establish Strong Retention",
            count_question,
            count_result,
            """
EXECUTIVE SUMMARY
There are 2,997 repeat customers.

BUSINESS IMPLICATION
The available count cannot establish strong retention.

RECOMMENDED NEXT STEPS
- Compare repeat customers with total unique customers.
""",
            True,
        ),

        # ====================================================
        # V17 MULTI-ENTITY REVENUE REGRESSIONS
        # ====================================================

        (
            "Correct Multi-Entity Revenue Values",
            revenue_question,
            revenue_result,
            """
EXECUTIVE SUMMARY
The validated results show that Health & Beauty leads with a total revenue of $1,258,681.34, followed closely by Watches & Gifts at $1,205,005.68.

BUSINESS IMPLICATION
Additional analysis is required to understand category performance over time.

RECOMMENDED NEXT STEPS
- Compare category revenue across time periods.
""",
            True,
        ),

        (
            "Incorrect Second Multi-Entity Revenue Value",
            revenue_question,
            revenue_result,
            """
EXECUTIVE SUMMARY
The validated results show that Health & Beauty leads with a total revenue of $1,258,681.34, followed closely by Watches & Gifts at $999,999.99.

RECOMMENDED NEXT STEPS
- Compare category revenue across time periods.
""",
            False,
        ),

        (
            "Incorrect First Multi-Entity Revenue Value",
            revenue_question,
            revenue_result,
            """
EXECUTIVE SUMMARY
The validated results show that Health & Beauty leads with a total revenue of $999,999.99, followed closely by Watches & Gifts at $1,205,005.68.

RECOMMENDED NEXT STEPS
- Compare category revenue across time periods.
""",
            False,
        ),

        (
            "Correct Three-Entity Revenue Values",
            revenue_question,
            revenue_result,
            """
EXECUTIVE SUMMARY
Health & Beauty generated $1,258,681.34 in total revenue, Watches & Gifts generated $1,205,005.68, and Bed & Bath & Table generated $1,036,988.68.

RECOMMENDED NEXT STEPS
- Compare category revenue across time periods.
""",
            True,
        ),

        (
            "Negation Must Not Hide Positive Assertion",
            count_question,
            count_result,
            """
EXECUTIVE SUMMARY
There are 2,997 repeat customers.

BUSINESS IMPLICATION
The count does not establish high retention, but retention is high.

RECOMMENDED NEXT STEPS
- Analyze purchase frequency.
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
        "REPORT VALIDATOR VERSION 17 TEST SUMMARY"
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
        "\nReport Validator Version 17 "
        "testing completed."
    )