"""
Vietnamese Address Pre-processing Pipeline (Version 2.0)

Architecture: Extract-and-Resolve Model
Author: Refactored for robustness and accuracy
Date: October 2025

This module implements an intelligent address parsing system that prioritizes
exact entity matches over regex-based transformations, dramatically improving
accuracy and maintainability.
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import time

# ============================================================================
# CONFIGURATION / CONSTANTS
# ============================================================================

# Enhanced province abbreviations - now includes "h nội" and other edge cases
PROVINCE_ABBREVIATIONS = {
    "t.t.h": "thừa thiên huế",
    "tth": "thừa thiên huế",
    "hcm": "hồ chí minh",
    "tphcm": "hồ chí minh",
    "tp.hcm": "hồ chí minh",
    "tp. hcm": "hồ chí minh",
    "hn": "hà nội",
    "hnoi": "hà nội",
    "h nội": "hà nội",  # CRITICAL FIX: Handle the split case
    "hnội": "hà nội",
}

# Administrative prefixes for extraction (sorted by specificity)
ADMIN_PREFIXES = {
    "province": [
        "thành phố",
        "tỉnh",
        "t. phố",
        "t.phố",
        "t. p.",
        "t.p.",
        "tp.",
        "tp",
        "t.",
    ],
    "district": [
        "quận",
        "huyện",
        "thị xã",
        "thành phố",
        "q.",
        "h.",
        "t.x.",
        "tx.",
        "tp.",
    ],
    "ward": ["phường", "xã", "thị trấn", "p.", "x.", "t.t.", "tt."],
}

# Expanded blacklist for non-administrative address parts
BLACK_LIST_KEYWORDS = [
    "số",
    "đường",
    "bis",
    "ấp",
    "khu",
    "block",
    "lô",
    "tổ",
    "ngõ",
    "hẻm",
    "street",
    "road",
    "avenue",
    "lane",
    "area",
    "building",
    "floor",
    "apartment",
    "căn hộ",
    "tầng",
    "toà",
    "tòa",
    "nhà",
    "villa",
    "biệt thự",
    "chung cư",
    "kdc",
    "ktx",
    "ccx",
    "c/c",
    "đs",
    "km",
    "ql",
    "tl",
    "mansion",
]

# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class AddressComponent:
    """Represents an extracted address component with metadata"""

    text: str
    component_type: str  # 'province', 'district', 'ward'
    start_pos: int
    end_pos: int
    confidence: float  # 0.0 to 1.0
    prefix: Optional[str] = None
    source: str = "unknown"  # 'exact_match', 'prefix_based', 'inferred'

    def __repr__(self):
        return f"{self.component_type.upper()}('{self.text}', conf={self.confidence:.2f}, src={self.source})"


# ============================================================================
# KNOWN ENTITIES DATABASE
# ============================================================================

# In production, load these from your actual database/CSV files
# This is a sample dataset for demonstration
KNOWN_ENTITIES = {
    "provinces": {
        "hà nội",
        "hồ chí minh",
        "đà nẵng",
        "hải phòng",
        "cần thơ",
        "thừa thiên huế",
        "bình dương",
        "đồng nai",
        "bà rịa vũng tàu",
        "khánh hòa",
        "lâm đồng",
        "nghệ an",
        "thanh hóa",
        "quảng ninh",
        "bắc ninh",
        "hải dương",
        "vĩnh phúc",
        "thái nguyên",
        "nam định",
        "hưng yên",
        "thái bình",
        "hà nam",
        "ninh bình",
        "phú thọ",
    },
    "districts": {
        # Hanoi districts
        "nam từ liêm",
        "bắc từ liêm",
        "cầu giấy",
        "đống đa",
        "hai bà trưng",
        "hoàn kiếm",
        "hoàng mai",
        "long biên",
        "tây hồ",
        "thanh xuân",
        "ba đình",
        "hà đông",
        "sơn tây",
        "đan phượng",
        "hoài đức",
        # HCMC districts
        "quận 1",
        "quận 2",
        "quận 3",
        "quận 4",
        "quận 5",
        "quận 6",
        "quận 7",
        "quận 8",
        "quận 9",
        "quận 10",
        "quận 11",
        "quận 12",
        "thủ đức",
        "bình thạnh",
        "tân bình",
        "tân phú",
        "phú nhuận",
        "gò vấp",
        "bình tân",
        "củ chi",
        "hóc môn",
        "bình chánh",
    },
    "wards": {
        # Sample wards from Hanoi
        "phú đô",
        "cầu giấy",
        "dịch vọng",
        "mai dịch",
        "nghĩa đô",
        "trung hòa",
        "yên hòa",
        "mễ trì",
        "xuân la",
        "phương canh",
        "mỹ đình 1",
        "mỹ đình 2",
        "phú thượng",
        "nhật tân",
        "tứ liên",
        # Sample wards from HCMC
        "bến nghé",
        "bến thành",
        "cầu kho",
        "cầu ông lãnh",
        "cô giang",
        "đa kao",
        "nguyễn cư trinh",
        "nguyễn thái bình",
        "phạm ngũ lão",
    },
}


def load_entities_from_files(
    province_file: str = None, district_file: str = None, ward_file: str = None
) -> Dict[str, Set[str]]:
    """
    PRODUCTION VERSION: Load entities from your actual data files.
    This function should be implemented to read from your classification_layer.py data sources.

    For now, returns the hardcoded sample data.
    """
    # TODO: Implement actual file loading
    # Example:
    # provinces = load_csv_column(province_file, 'name')
    # districts = load_csv_column(district_file, 'name')
    # wards = load_csv_column(ward_file, 'name')

    return KNOWN_ENTITIES


# ============================================================================
# STEP 1: CANONICAL PRE-NORMALIZATION
# ============================================================================


def canonical_normalize(text: str) -> str:
    """
    Minimal, non-destructive normalization.
    Only standardizes basic formatting without making destructive changes.

    Philosophy: Clean the data just enough to enable pattern matching,
    but don't transform it so much that information is lost.
    """
    # Convert to lowercase for case-insensitive matching
    text = text.lower().strip()

    # Normalize whitespace (multiple spaces -> single space)
    text = re.sub(r"\s+", " ", text)

    # Ensure space after commas for consistent splitting
    text = re.sub(r",(?=\S)", ", ", text)

    # Remove trailing punctuation that adds no value
    text = text.strip(".,;")

    # Remove hyphens (often inconsistently used)
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)  # Re-normalize after hyphen removal

    return text


def resolve_province_abbreviations(text: str) -> str:
    """
    Resolve known province abbreviations FIRST, before any other processing.

    This is safe and critical because:
    1. Province abbreviations are highly specific and unambiguous
    2. Resolving them early prevents misinterpretation by later steps
    3. It immediately fixes cases like "H Nội" -> "Hà Nội"
    """
    text_lower = text.lower()

    # Sort by length (longest first) to avoid partial matches
    # e.g., "tp.hcm" should match before "tp"
    sorted_abbrevs = sorted(
        PROVINCE_ABBREVIATIONS.items(), key=lambda x: len(x[0]), reverse=True
    )

    for abbrev, full_name in sorted_abbrevs:
        # Use word boundaries to avoid false matches
        # But also handle cases where abbreviation might be adjacent to punctuation
        pattern = r"\b" + re.escape(abbrev) + r"\b"
        text_lower = re.sub(pattern, full_name, text_lower, flags=re.IGNORECASE)

    return text_lower


# ============================================================================
# STEP 2: HIGH-CONFIDENCE ENTITY EXTRACTION
# ============================================================================


def build_entity_patterns(
    entities: Dict[str, Set[str]],
) -> Dict[str, List[Tuple[re.Pattern, str]]]:
    """
    Pre-compile regex patterns for all known entities.
    Returns patterns sorted by length (longest first) for greedy matching.

    This is done once at module initialization for performance.
    """
    patterns = {}

    for entity_type, entity_set in entities.items():
        type_patterns = []
        # Sort by length (longest first) to match longer names first
        # This prevents "Hà" from matching before "Hà Nội"
        sorted_entities = sorted(entity_set, key=len, reverse=True)

        for entity in sorted_entities:
            # Create pattern with word boundaries for exact matching
            pattern = re.compile(r"\b" + re.escape(entity) + r"\b", re.IGNORECASE)
            type_patterns.append((pattern, entity))

        patterns[entity_type] = type_patterns

    return patterns


# Initialize patterns at module load time
_ENTITY_PATTERNS = build_entity_patterns(KNOWN_ENTITIES)


def extract_known_entities(text: str, debug: bool = False) -> List[AddressComponent]:
    """
    Extract all known administrative entities from the text.
    Returns high-confidence matches with positions.

    This is the CORE of the new approach: we actively search for known
    good data instead of blindly applying transformations.
    """
    components = []

    # Map entity types to component types
    type_mapping = {"provinces": "province", "districts": "district", "wards": "ward"}

    for entity_type, patterns in _ENTITY_PATTERNS.items():
        component_type = type_mapping[entity_type]

        for pattern, entity_name in patterns:
            for match in pattern.finditer(text):
                component = AddressComponent(
                    text=entity_name,
                    component_type=component_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=1.0,  # Exact match = highest confidence
                    source="exact_match",
                )
                components.append(component)

                if debug:
                    print(f"  Exact match: {component}")

    return components


def remove_overlapping_components(
    components: List[AddressComponent],
) -> List[AddressComponent]:
    """
    When multiple components overlap, keep the longer/higher confidence one.

    Example: If both "nam" and "nam từ liêm" match, keep "nam từ liêm"
    """
    if not components:
        return []

    # Sort by start position, then by length (longer first), then by confidence
    components.sort(
        key=lambda c: (c.start_pos, -(c.end_pos - c.start_pos), -c.confidence)
    )

    filtered = []
    last_end = -1

    for component in components:
        # If this component doesn't overlap with the last kept component
        if component.start_pos >= last_end:
            filtered.append(component)
            last_end = component.end_pos

    return filtered


# ============================================================================
# STEP 3: HEURISTIC-BASED COMPONENT PARSING
# ============================================================================


def extract_prefix(text: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    Extract administrative prefix from text.
    Returns: (prefix, component_type, remaining_text)
    """
    text = text.strip()

    # Check each prefix type in order of specificity
    for component_type, prefixes in ADMIN_PREFIXES.items():
        # Sort prefixes by length (longest first) to match specific ones first
        sorted_prefixes = sorted(prefixes, key=len, reverse=True)

        for prefix in sorted_prefixes:
            # Check if text starts with this prefix
            if text.lower().startswith(prefix):
                # Ensure there's a space or end after prefix
                prefix_len = len(prefix)
                if prefix_len >= len(text) or text[prefix_len].isspace():
                    remaining = text[prefix_len:].strip()
                    return prefix, component_type, remaining

    return None, None, text


def contains_blacklist_keywords(text: str) -> bool:
    """Check if text contains non-administrative keywords"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in BLACK_LIST_KEYWORDS)


def is_valid_component(text: str, min_length: int = 2) -> bool:
    """Validate that text could be a legitimate administrative component"""
    if not text or len(text) < min_length:
        return False

    text_stripped = text.strip()

    # Pure numbers are not valid components
    if text_stripped.isdigit():
        return False

    # Must contain at least some letters
    if not re.search(r"[a-zà-ỹ]", text_stripped, re.IGNORECASE):
        return False

    # Check for blacklist keywords
    if contains_blacklist_keywords(text_stripped):
        return False

    return True


def parse_unmatched_parts(
    text: str, found_components: Dict[str, List[str]], debug: bool = False
) -> List[AddressComponent]:
    """
    Parse parts that weren't matched in high-confidence extraction.
    Uses prefixes and context to make educated guesses.

    This is the fallback mechanism when exact matching doesn't find everything.
    """
    components = []

    # Split by commas (standard Vietnamese address format)
    parts = [p.strip() for p in text.split(",") if p.strip()]

    if debug and parts:
        print(f"  Parsing unmatched parts: {parts}")

    for part in parts:
        # Skip blacklisted parts
        if contains_blacklist_keywords(part):
            if debug:
                print(f"    Skipped (blacklist): '{part}'")
            continue

        # Validate component
        if not is_valid_component(part):
            if debug:
                print(f"    Skipped (invalid): '{part}'")
            continue

        # Extract prefix if present
        prefix, component_type, remaining = extract_prefix(part)

        if component_type and remaining:
            # We found a prefix, so we know the type
            component = AddressComponent(
                text=remaining,
                component_type=component_type,
                start_pos=-1,  # Unknown position in heuristic parsing
                end_pos=-1,
                confidence=0.7,  # Lower confidence than exact match
                prefix=prefix,
                source="prefix_based",
            )
            components.append(component)

            if debug:
                print(f"    Prefix-based: {component}")
        else:
            # No prefix - try to infer from context
            # Fill in the first missing slot (province -> district -> ward)
            inferred_type = None

            if not found_components.get("province"):
                inferred_type = "province"
            elif not found_components.get("district"):
                inferred_type = "district"
            elif not found_components.get("ward"):
                inferred_type = "ward"

            if inferred_type:
                component = AddressComponent(
                    text=part,
                    component_type=inferred_type,
                    start_pos=-1,
                    end_pos=-1,
                    confidence=0.3,  # Low confidence guess
                    source="inferred",
                )
                components.append(component)

                # Update found components for next iteration
                if inferred_type not in found_components:
                    found_components[inferred_type] = []
                found_components[inferred_type].append(part)

                if debug:
                    print(f"    Inferred: {component}")

    return components


# ============================================================================
# STEP 4: FINAL ASSEMBLY
# ============================================================================


def assemble_results(
    high_confidence: List[AddressComponent],
    heuristic: List[AddressComponent],
    original_input: str,
) -> Dict:
    """
    Combine high-confidence and heuristic results.
    Prioritize high-confidence matches.
    """
    result = {
        "province": [],
        "district": [],
        "ward": [],
        "input": original_input,
        "remaining": None,
        "remaining_parts": [],
        "debug_info": {
            "high_confidence_matches": len(high_confidence),
            "heuristic_matches": len(heuristic),
        },
    }

    # First, add all high-confidence matches
    for component in high_confidence:
        if component.text not in result[component.component_type]:
            result[component.component_type].append(component.text)

    # Then add heuristic matches for empty slots only
    for component in heuristic:
        if not result[component.component_type]:  # Only if slot is empty
            if component.text not in result[component.component_type]:
                result[component.component_type].append(component.text)

    # Remove debug_info if all components were found with high confidence
    if result["province"] and result["district"] and result["ward"]:
        if all(c.confidence == 1.0 for c in high_confidence):
            del result["debug_info"]

    return result


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def preprocess_address(address: str, debug: bool = False) -> Dict:
    """
    Main preprocessing pipeline using the extract-and-resolve approach.

    Steps:
    1. Canonical pre-normalization (minimal, non-destructive)
    2. Province abbreviation resolution (specific, safe)
    3. High-confidence entity extraction (exact matches)
    4. Heuristic-based parsing of remaining parts (fallback)
    5. Final assembly (prioritize high-confidence)

    Args:
        address: Raw address string
        debug: If True, print detailed processing steps

    Returns:
        Dictionary with extracted components:
        {
            'province': [list of province names],
            'district': [list of district names],
            'ward': [list of ward names],
            'input': original input,
            'remaining': unparsed parts,
            'remaining_parts': list of unparsed parts
        }
    """
    if debug:
        print(f"\n{'=' * 70}")
        print(f"RAW INPUT: '{address}'")
        print(f"{'=' * 70}")

    # STEP 1: Canonical Pre-Normalization
    if debug:
        print("\n[STEP 1] Canonical Pre-Normalization")

    normalized = canonical_normalize(address)

    if debug:
        print(f"  After basic normalization: '{normalized}'")

    # STEP 1.5: Province Abbreviation Resolution
    if debug:
        print("\n[STEP 1.5] Province Abbreviation Resolution")

    normalized = resolve_province_abbreviations(normalized)

    if debug:
        print(f"  After abbreviation resolution: '{normalized}'")

    # STEP 2: High-Confidence Entity Extraction
    if debug:
        print("\n[STEP 2] High-Confidence Entity Extraction")

    high_conf_components = extract_known_entities(normalized, debug=debug)
    high_conf_components = remove_overlapping_components(high_conf_components)

    if debug:
        print(f"  Found {len(high_conf_components)} high-confidence matches:")
        for comp in high_conf_components:
            print(f"    {comp}")

    # Build dict of found components
    found = {
        "province": [
            c.text for c in high_conf_components if c.component_type == "province"
        ],
        "district": [
            c.text for c in high_conf_components if c.component_type == "district"
        ],
        "ward": [c.text for c in high_conf_components if c.component_type == "ward"],
    }

    # STEP 3: Heuristic Parsing of Unmatched Parts
    if debug:
        print("\n[STEP 3] Heuristic Parsing of Remaining Parts")

    # Remove high-confidence matches from text to see what's left
    remaining_text = normalized
    for component in sorted(
        high_conf_components, key=lambda c: c.start_pos, reverse=True
    ):
        remaining_text = (
            remaining_text[: component.start_pos] + remaining_text[component.end_pos :]
        )

    remaining_text = re.sub(r"\s+", " ", remaining_text).strip(" ,")

    if debug:
        print(f"  Remaining text after removing exact matches: '{remaining_text}'")

    heuristic_components = []
    if remaining_text:
        heuristic_components = parse_unmatched_parts(remaining_text, found, debug=debug)

    # STEP 4: Final Assembly
    if debug:
        print("\n[STEP 4] Final Assembly")

    result = assemble_results(high_conf_components, heuristic_components, address)

    if debug:
        print(f"\n{'=' * 70}")
        print(f"FINAL RESULT:")
        print(f"  Province: {result['province']}")
        print(f"  District: {result['district']}")
        print(f"  Ward: {result['ward']}")
        if "debug_info" in result:
            print(f"  Debug Info: {result['debug_info']}")
        print(f"{'=' * 70}\n")

    return result


# ============================================================================
# PERFORMANCE MEASUREMENT
# ============================================================================


def time_performance(func):
    """Decorator to measure function execution time"""

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000
        print(f"Function '{func.__name__}' took {elapsed_ms:.4f} ms")
        return result

    return wrapper


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Vietnamese Address Pre-processing Pipeline v2.0")
    print("=" * 70)

    test_cases = [
        "Phường Phú Đô, Quận Nam Từ Liêm, Thành phố H Nội",  # The problematic case
        "Phú Đô, Nam Từ Liêm, Hà Nội",  # Without prefixes
        "P. Phú Đô, Q. Nam Từ Liêm, TP. Hà Nội",  # Abbreviated prefixes
        "phường phú đô, quận nam từ liêm, tp.hcm",  # Wrong province (should still parse)
        "Yên Hòa, Cầu Giấy, HN",  # Abbreviation test
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n\nTEST CASE {i}:")
        result = preprocess_address(test, debug=True)
