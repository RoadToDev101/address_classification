import re

from utils.time_perf import time_performance_measure

# ============================================================================
# CONFIGURATION / CONSTANTS
# ============================================================================

ADMINISTRATION_PREFIX_PATTERNS = {
    "province": [
        r"^(thành phố|t\.\s*phố|tỉnh|tp|tp\.|t\.)\s+",
        r"^(thanh pho|tinh)\s+",
    ],
    "district": [
        r"^(huyện|quận|thị xã|thành phố|q\.|h\.|tx\.|tp\.)\s+",
        r"^(huyen|quan|thi xa|thanh pho)\s+",
    ],
    "ward": [
        r"^(phường|xã|thị trấn|p\.|x\.|tt\.)\s+",
        r"^(phuong|xa|thi tran)\s+",
    ],
}

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

PROVINCE_ABBREVIATIONS = {
    "t.t.h": "thừa thiên huế",
    "tth": "thừa thiên huế",
    "hcm": "hồ chí minh",
    "tphcm": "hồ chí minh",
    "tp.hcm": "hồ chí minh",
    "hn": "hà nội",
    "hnoi": "hà nội",
}

ADMIN_ABBREVIATION_MAP = {
    "t.ph": "thành phố",
    "t. ph": "thành phố",
    "tp.": "thành phố",
    "tp": "thành phố",
    "t.x.": "thị xã",
    "t.x": "thị xã",
    "tx.": "thị xã",
    "tx": "thị xã",
    "tt.": "thị trấn",
    "tt": "thị trấn",
    "p.": "phường",
    "q.": "quận",
    "h.": "huyện",
    "x.": "xã",
    "t.": "tỉnh",
}

SINGLE_LETTER_PREFIXES = {"x", "h", "q", "t", "p"}

# ============================================================================
# PRE-COMPILED PATTERNS
# ============================================================================

_PROVINCE_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in PROVINCE_ABBREVIATIONS.keys()) + r")\b",
    re.IGNORECASE,
)

_SINGLE_LETTER_PREFIX_PATTERN = re.compile(r"\b([xhqtp])\s+(?=[a-zà-ỹ])")
_PREFIX_DIGIT_PATTERN1 = re.compile(r"([pq])(\d{1,2})([pq]\d{1,2})")
_PREFIX_DIGIT_PATTERN2 = re.compile(r"\b([pq])(\d{1,2})\b")
_PREFIX_DIGIT_PATTERN3 = re.compile(r"([pq])(\d{1,2})([a-zà-ỹ])")
_CONCAT_ADMIN_PATTERN = re.compile(
    r"\b(huyện|quận|phường|xã|thị xã|thành phố|tỉnh|thị trấn)([a-zà-ỹ])"
)
_ABBREVIATED_PREFIX_PATTERN = re.compile(r"\b(tp|tx|tt)([a-zà-ỹ])")
_MERGE_TPH_PATTERN = re.compile(r"t\.\s+(phố|ph)(?=\s|$|[0-9])", re.IGNORECASE)
_MERGE_TP_PATTERN = re.compile(r"\bt\.\s*p\.?\s*(?=\s|$)", re.IGNORECASE)
_MERGE_TX_PATTERN = re.compile(r"t\.\s+x\.")
_MERGE_TT_PATTERN = re.compile(r"t\.\s+t\.")
_ENSURE_SPACING_PATTERN = re.compile(r"(tp\.|tx\.|tt\.)([0-9])")

_ADMIN_PREFIX_PATTERN = re.compile(
    r"\b(huyện|quận|phường|xã|thị xã|thành phố|tỉnh|thị trấn|tp\.|tx\.|tt\.|p\.|q\.|h\.|x\.|t\.)\s+",
    re.IGNORECASE,
)

_COMMA_AFTER_DOT_PATTERN = re.compile(r",(?=\S)")
_SPACE_AFTER_DOT_PATTERN = re.compile(r"\.(?=\S)(?![a-z]\.)")
_DOUBLE_SPACE_PATTERN = re.compile(r"\s+")
_DOUBLE_COMMA_PATTERN = re.compile(r",\s*,")
_TRAILING_COMMA_PATTERN = re.compile(r",\s*$")

# ============================================================================
# PHASE 1: PROVINCE ABBREVIATION RESOLUTION
# ============================================================================


def resolve_province_abbreviations(text: str) -> str:
    text_lower = text.lower().strip()

    def replace_abbrev(match):
        return PROVINCE_ABBREVIATIONS.get(match.group(1).lower(), match.group(1))

    return _PROVINCE_ABBREV_PATTERN.sub(replace_abbrev, text_lower)


# ============================================================================
# PHASE 2: TEXT NORMALIZATION (COMBINED OPERATIONS)
# ============================================================================


@time_performance_measure
def normalize_input(address_input: str, debug: bool = False) -> str:
    """Normalization pipeline combining multiple operations."""
    if debug:
        print(f"RAW INPUT: '{address_input}'")
    # Phase 1: Province abbreviations
    text = resolve_province_abbreviations(address_input)

    if debug:
        print(f"AFTER PROVINCE ABBREV RESOLUTION: '{text}'")

    # Phase 2: Basic normalization (combined)
    text = " ".join(text.lower().split())
    text = text.replace("-", "").rstrip(".")

    if debug:
        print(f"AFTER BASIC NORMALIZATION: '{text}'")

    # Phase 3: Punctuation spacing
    text = _COMMA_AFTER_DOT_PATTERN.sub(", ", text)
    text = _SPACE_AFTER_DOT_PATTERN.sub(". ", text)
    if debug:
        print(f"AFTER PUNCTUATION SPACING: '{text}'")

    # Phase 4: Prefix merging (MOVE TP PATTERN HERE, BEFORE SINGLE LETTER)
    text = _MERGE_TP_PATTERN.sub("tp.", text)  # Match "t. p" or "t.p" or "t. p."
    text = _MERGE_TPH_PATTERN.sub("tp.", text)
    text = _MERGE_TX_PATTERN.sub("tx.", text)
    text = _MERGE_TT_PATTERN.sub("tt.", text)
    text = _ENSURE_SPACING_PATTERN.sub(r"\1 \2", text)
    if debug:
        print(f"AFTER PREFIX MERGING: '{text}'")

    # NOW apply single-letter patterns AFTER merging is done
    text = _SINGLE_LETTER_PREFIX_PATTERN.sub(r"\1. ", text)
    text = _PREFIX_DIGIT_PATTERN1.sub(r"\1. \2 \3", text)
    text = _PREFIX_DIGIT_PATTERN2.sub(r"\1. \2", text)
    text = _PREFIX_DIGIT_PATTERN3.sub(r"\1. \2 \3", text)
    if debug:
        print(f"AFTER PREFIX DIGIT HANDLING: '{text}'")

    text = _CONCAT_ADMIN_PATTERN.sub(r"\1 \2", text)
    text = _ABBREVIATED_PREFIX_PATTERN.sub(r"\1. \2", text)

    if debug:
        print(f"AFTER CONCAT ADMIN HANDLING: '{text}'")

    # Phase 5: Add commas
    text = add_commas_before_admin_prefixes(text)

    if debug:
        print(f"AFTER ADDING COMMAS: '{text}'")

    # Phase 6: Final cleanup
    text = _DOUBLE_SPACE_PATTERN.sub(" ", text).strip()
    text = _DOUBLE_COMMA_PATTERN.sub(",", text)
    text = _TRAILING_COMMA_PATTERN.sub("", text)

    if debug:
        print(f"NORMALIZED: '{text}'")

    return text


def add_commas_before_admin_prefixes(text: str) -> str:
    matches = list(_ADMIN_PREFIX_PATTERN.finditer(text))

    if not matches:
        return text

    # Build list of segments to join (avoid repeated string concatenation)
    segments = []
    last_end = 0

    for match in matches:
        start_pos = match.start()

        if start_pos > 0:
            char_before = text[start_pos - 1]
            matched_prefix = text[start_pos : match.end()].lower()

            # Check for t. followed by p./phố/x.
            is_after_t = start_pos >= 2 and text[start_pos - 2 : start_pos] == "t."
            is_follow_token = matched_prefix.startswith(("p. ", "phố ", "x. "))

            if is_after_t and is_follow_token:
                continue

            # Add comma if needed
            if char_before == " " and (
                start_pos < 2 or text[start_pos - 2 : start_pos] != ", "
            ):
                segments.append(text[last_end : start_pos - 1])
                segments.append(", ")
                last_end = start_pos

    segments.append(text[last_end:])
    return "".join(segments)


# ============================================================================
# PHASE 3: ABBREVIATION EXPANSION
# ============================================================================

# Pre-compile expansion patterns
_EXPANSION_PATTERNS = []
for abbrev, full in sorted(ADMIN_ABBREVIATION_MAP.items(), key=lambda x: -len(x[0])):
    if "." in abbrev:
        pattern = re.compile(re.escape(abbrev) + r"(?=\s|$|\d)", re.IGNORECASE)
    else:
        pattern = re.compile(r"\b" + re.escape(abbrev) + r"\b", re.IGNORECASE)
    _EXPANSION_PATTERNS.append((pattern, full + " "))


@time_performance_measure
def expand_abbreviations(normalized: str) -> str:
    for pattern, replacement in _EXPANSION_PATTERNS:
        normalized = pattern.sub(replacement, normalized)

    return _DOUBLE_SPACE_PATTERN.sub(" ", normalized).strip()


# ============================================================================
# HELPER FUNCTIONS FOR COMPONENT EXTRACTION
# ============================================================================


def clean_component(text: str, component_type: str) -> str:
    if not text:
        return None
    text = text.strip()
    for pattern in ADMINISTRATION_PREFIX_PATTERNS[component_type]:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    return text if text else None


def check_single_letter_prefix(word: str):
    if re.match(r"^(tp|tx|tt|q|h|p|x)\.$", word, flags=re.IGNORECASE):
        return False, None, word

    if len(word) > 2 and word[0] in SINGLE_LETTER_PREFIXES:
        full_admin_prefixes = ["thành", "thị", "huyện", "quận", "phường", "tỉnh"]
        for prefix in full_admin_prefixes:
            if word.lower().startswith(prefix):
                return False, None, word

        remaining = word[1:]
        if not re.search(r"[à-ỹ\s]", remaining, flags=re.IGNORECASE):
            return False, None, word

        if len(remaining) < 2:
            return False, None, word

        return True, word[0], remaining

    return False, None, word


def contains_black_list_keywords(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in BLACK_LIST_KEYWORDS)


def is_valid_admin_component(text: str, min_length: int = 3) -> bool:
    if not text or len(text) < min_length:
        return False
    text_stripped = text.strip()
    if text_stripped.isdigit():
        return False
    if re.search(r"[à-ỹ]", text_stripped, flags=re.IGNORECASE):
        return True
    if len(text_stripped) >= 3 and re.search(
        r"[a-z]{3,}", text_stripped, flags=re.IGNORECASE
    ):
        return True
    return False


# ============================================================================
# COMPONENT EXTRACTION
# ============================================================================


@time_performance_measure
def suggest_address_components(address_input: str):
    def add_component(bucket, key, value):
        if value is None:
            return
        value = value.strip()
        if not value:
            return
        if value not in bucket[key]:
            bucket[key].append(value)

    def take_remaining_words(words, start_idx, used_indices):
        return " ".join(
            [
                w
                for j, w in enumerate(words[start_idx:], start=start_idx)
                if j not in used_indices
            ]
        )

    def mark_used_range(used_indices, start_idx, end_idx):
        for j in range(start_idx, end_idx):
            used_indices.add(j)

    def classify_and_add_by_letter(prefix_letter, text):
        nonlocal province_assigned
        if prefix_letter == "t":
            add_component(result, "province", clean_component(text, "province"))
            province_assigned = True
        elif prefix_letter in ["q", "h"]:
            add_component(result, "district", clean_component(text, "district"))
        elif prefix_letter in ["p", "x"]:
            add_component(result, "ward", clean_component(text, "ward"))

    def word_matches_any(patterns, word):
        return any(re.match(pat, word, flags=re.IGNORECASE) for pat in patterns)

    result = {
        "province": [],
        "district": [],
        "ward": [],
        "input": address_input,
        "remaining": None,
        "remaining_parts": [],
    }
    province_assigned = False

    if "," not in address_input:
        words = address_input.split()
        used_indices = set()

        for i in reversed(range(len(words))):
            if i in used_indices:
                continue

            word = words[i]

            if (
                i + 1 < len(words)
                and words[i] == "t."
                and words[i + 1] in {"p.", "phố", "x."}
            ):
                remaining_text = (
                    take_remaining_words(words, i + 2, used_indices)
                    if i + 2 < len(words)
                    else ""
                )
                if words[i + 1] in {"p.", "phố"}:
                    if not province_assigned:
                        add_component(
                            result,
                            "province",
                            clean_component(remaining_text, "province"),
                        )
                        province_assigned = True
                    else:
                        add_component(
                            result,
                            "district",
                            clean_component(remaining_text, "district"),
                        )
                else:
                    add_component(
                        result, "district", clean_component(remaining_text, "district")
                    )
                mark_used_range(used_indices, i, len(words))
                continue

            has_prefix, prefix_letter, remaining_word = check_single_letter_prefix(word)
            if has_prefix:
                component_text = " ".join(
                    [remaining_word]
                    + [
                        words[j]
                        for j in range(i + 1, len(words))
                        if j not in used_indices
                    ]
                )
                classify_and_add_by_letter(prefix_letter, component_text)
                mark_used_range(used_indices, i, len(words))
                continue

            if len(word) == 1 and word in SINGLE_LETTER_PREFIXES and i < len(words) - 1:
                remaining_text = take_remaining_words(words, i, used_indices)
                if word == "t":
                    add_component(
                        result, "province", clean_component(remaining_text, "province")
                    )
                    province_assigned = True
                elif word in ["q", "h"]:
                    add_component(
                        result, "district", clean_component(remaining_text, "district")
                    )
                elif word in ["p", "x"]:
                    add_component(
                        result, "ward", clean_component(remaining_text, "ward")
                    )
                mark_used_range(used_indices, i, len(words))
                continue

            province_tokens = [r"^(thành phố|tỉnh|tp|tp\.|t\.)$"]
            district_tokens = [r"^(huyện|quận|thị xã|thành phố|q\.|h\.|tx\.|tp\.)$"]
            ward_tokens = [r"^(phường|xã|thị trấn|p\.|x\.|tt\.)$"]

            if word_matches_any(province_tokens, word):
                remaining_text = take_remaining_words(words, i, used_indices)
                add_component(
                    result, "province", clean_component(remaining_text, "province")
                )
                province_assigned = True
                mark_used_range(used_indices, i, len(words))
                continue

            if i not in used_indices and word_matches_any(district_tokens, word):
                remaining_text = take_remaining_words(words, i, used_indices)
                add_component(
                    result, "district", clean_component(remaining_text, "district")
                )
                mark_used_range(used_indices, i, len(words))
                continue

            if i not in used_indices and word_matches_any(ward_tokens, word):
                remaining_text = take_remaining_words(words, i, used_indices)
                add_component(result, "ward", clean_component(remaining_text, "ward"))
                mark_used_range(used_indices, i, len(words))
                continue

        remaining_words = [
            words[idx] for idx in range(len(words)) if idx not in used_indices
        ]
        remaining_str = " ".join(remaining_words) if remaining_words else None
        result["remaining"] = remaining_str
        result["remaining_parts"] = [remaining_str] if remaining_str else []
        return result

    # Comma-separated path
    parts = [part.strip() for part in address_input.split(",")]
    parts = [part for part in parts if part]

    matched_parts = []
    unmatched_parts = []

    for i, part in enumerate(reversed(parts)):
        matched = False

        if re.match(r"^\s*t\.\s*(p\.|phố)\s+", part, flags=re.IGNORECASE):
            component_text = re.sub(
                r"^\s*t\.\s*(p\.|phố)\s+", "", part, flags=re.IGNORECASE
            )
            if not province_assigned:
                add_component(
                    result, "province", clean_component(component_text, "province")
                )
                province_assigned = True
            else:
                add_component(
                    result, "district", clean_component(component_text, "district")
                )
            matched = True
        elif re.match(r"^\s*t\.\s*x\.\s+", part, flags=re.IGNORECASE):
            component_text = re.sub(r"^\s*t\.\s*x\.\s+", "", part, flags=re.IGNORECASE)
            add_component(
                result, "district", clean_component(component_text, "district")
            )
            matched = True
        elif re.match(
            r"^\s*(thành phố|t\. phố|t\. p\.|tp\.?)[\s]+", part, flags=re.IGNORECASE
        ):
            if not province_assigned:
                add_component(result, "province", clean_component(part, "province"))
                province_assigned = True
            else:
                add_component(result, "district", clean_component(part, "district"))
            matched = True
        elif any(
            re.match(p, part, flags=re.IGNORECASE)
            for p in ADMINISTRATION_PREFIX_PATTERNS["province"]
        ):
            add_component(result, "province", clean_component(part, "province"))
            province_assigned = True
            matched = True
        elif any(
            re.match(p, part, flags=re.IGNORECASE)
            for p in ADMINISTRATION_PREFIX_PATTERNS["district"]
        ):
            add_component(result, "district", clean_component(part, "district"))
            matched = True
        elif any(
            re.match(p, part, flags=re.IGNORECASE)
            for p in ADMINISTRATION_PREFIX_PATTERNS["ward"]
        ):
            add_component(result, "ward", clean_component(part, "ward"))
            matched = True

        if matched:
            matched_parts.append(part)
        else:
            unmatched_parts.append(part)

    if unmatched_parts:
        for part in unmatched_parts:
            if contains_black_list_keywords(part):
                result["remaining_parts"].append(part)
                continue

            if not is_valid_admin_component(part):
                result["remaining_parts"].append(part)
                continue

            if not result["province"]:
                add_component(result, "province", part)
            elif not result["district"]:
                add_component(result, "district", part)
            elif not result["ward"]:
                add_component(result, "ward", part)
            else:
                result["remaining_parts"].append(part)

    if not result["remaining_parts"]:
        used_parts = matched_parts.copy()
        for i, part in enumerate(unmatched_parts):
            if not contains_black_list_keywords(part) and i < 3:
                used_parts.append(part)
        remaining_parts = [p for p in parts if p not in used_parts]
        result["remaining_parts"] = remaining_parts

    result["remaining"] = (
        ", ".join(result["remaining_parts"]) if result["remaining_parts"] else None
    )
    return result


# ============================================================================
# MAIN PIPELINE
# ============================================================================

if __name__ == "__main__":
    import time

    from segment import Trie
    from spelling_correction import (
        build_spelling_correction_trie,
        correct_address_spelling,
        load_vietnamese_dictionary,
    )

    # Load dictionary and build tries once
    segment_dictionary = load_vietnamese_dictionary()
    spelling_trie = build_spelling_correction_trie()
    segment_trie = Trie()
    for word in segment_dictionary:
        segment_trie.insert(word.lower())

    tests = [
        # "X Tây Yên, H.An Biên, TKiên Giang",
        # "Thanh Long, Yên Mỹ Hưng Yên",
        # "X.Nga Thanh hyện Nga son TỉnhThanhQ Hóa",
        # " Duy Phú,  Duy Xuyen,  Quang Nam",
        # " Đức Lĩnh,Vũ Quang,",
        # "Thi trấ Ea. Knốp,H. Ea Kar,",
        # ", Nam Đông,T. T.T.H",
        # "P4 T.Ph9ốĐông Hà ",
        # "Xã Thịnh Sơn H., Đô dương T. Nghệ An",
        # "Phưng Khâm Thiên Quận Đ.Đa T.Phố HàNội",
        # "Tiểu khu 3, thị trấn Ba Hàng, huyện Phổ Yên, tỉnh Thái Nguyên.",
        "PhườngNguyễn Trãi, T.P Kon Tum, T Kon Tum",
        "285 B/1A Bình Gĩa Phường 8,Vũng Tàu,Bà Rịa - Vũng Tàu",
        "Khu phố Nam Tân, TT Thuận Nam, Hàm Thuận Bắc, Bình Thuận.",
        "A:12A.21BlockA C/c BCA,P.AnKhánh,TP.Thủ Đức, TP. HCM",
        "14.5 Block A2, The Mansion, ĐS 7,KDC 13E,Ấp 5,PP,BC, TP.HCM",
    ]

    # Test the spelling correction
    for test_address in tests:
        print(f"\nTesting address: {test_address}")
        start_time = time.perf_counter()
        normalized = normalize_input(test_address, debug=False)
        print(f"NORMALIZED: {normalized}")
        # segmented = segment_text_using_common_vn_words(test_address, segment_trie)
        corrected_address, corrections = correct_address_spelling(
            address=normalized,
            spelling_trie=spelling_trie,
            segment_trie=segment_trie,
            debug=True,
        )
        expanded = expand_abbreviations(corrected_address)
        result = suggest_address_components(expanded)
        end_time = time.perf_counter()
        elapsed_time_ms = (end_time - start_time) * 1000
        if elapsed_time_ms > 10:
            print(
                f"WARNING: Pre-processing took {elapsed_time_ms:.4f} ms, exceeding 10ms limit."
            )
        else:
            print(f"Pre-processing time: {elapsed_time_ms:.4f} ms")
        print(f"SUGGESTION RESULT: {result}")
