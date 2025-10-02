import re
from typing import List, Tuple

from bk_tree import BKTree, levenshtein_distance
from pre_processing import (
    ADMIN_ABBREVIATION_MAP,
    BLACK_LIST_KEYWORDS,
    PROVINCE_ABBREVIATIONS,
    normalize_input,
    expand_abbreviations,
)
from segment import segment, Trie
from utils.time_perf import time_performance_measure


def load_vietnamese_dictionary(
    dict_path: str = "data/vietnamese_address_dictionary.txt",
) -> List[str]:
    """Load Vietnamese dictionary from file"""
    try:
        with open(dict_path, "r", encoding="utf-8") as f:
            words = [line.strip().lower() for line in f if line.strip()]
        print(f"Loaded {len(words)} words from Vietnamese dictionary")
        return words
    except FileNotFoundError:
        print(f"Dictionary file {dict_path} not found")
        return []


def build_spelling_correction_trie() -> BKTree:
    """Build BK-Tree for spelling correction"""
    dictionary_words = load_vietnamese_dictionary()
    dictionary_words = dictionary_words + [
        abbr.lower() for abbr in PROVINCE_ABBREVIATIONS.keys()
    ]
    dictionary_words = dictionary_words + [
        abbr.lower() for abbr in ADMIN_ABBREVIATION_MAP.keys()
    ]
    dictionary_words = dictionary_words + [word.lower() for word in BLACK_LIST_KEYWORDS]
    dictionary_words = list(set(dictionary_words))
    if not dictionary_words:
        return BKTree()

    spelling_trie = BKTree(dictionary_words)
    print(f"Built spelling correction trie with {spelling_trie.word_count} words")
    return spelling_trie


def is_valid_vietnamese_word(word: str, spelling_trie: BKTree) -> bool:
    """Check if a word exists in the Vietnamese dictionary"""
    if not word or not word.strip():
        return True
    exact_match = spelling_trie.get_exact_match(word.lower())
    return exact_match is not None


def calculate_word_similarity_score(original: str, candidate: str) -> float:
    """Calculate similarity score between original and candidate word"""
    if not original or not candidate:
        return 0.0
    length_bonus = min(len(candidate), len(original)) / max(
        len(candidate), len(original)
    )
    common_prefix = sum(1 for c1, c2 in zip(original, candidate) if c1 == c2)
    prefix_score = common_prefix / max(len(original), len(candidate))
    edit_distance = levenshtein_distance(original, candidate)
    max_length = max(len(original), len(candidate))
    distance_score = 1 - (edit_distance / max_length) if max_length > 0 else 0
    return (distance_score * 0.6) + (prefix_score * 0.3) + (length_bonus * 0.1)


def correct_word(
    word: str, spelling_trie: BKTree, max_distance: int = 2
) -> Tuple[str, bool, int]:
    """Correct a single word using the spelling trie"""
    if not word or not word.strip():
        return word, False, 0
    word_lower = word.lower().strip()
    if (
        word_lower.isdigit()
        or len(word_lower) < 2
        or word_lower in {"-", ".", ",", "/", "\\", "(", ")", "[", "]", "{", "}"}
    ):
        return word, False, 0
    if is_valid_vietnamese_word(word_lower, spelling_trie):
        return word, False, 0
    matches = spelling_trie.search(word_lower, max_distance)
    if matches:
        scored_matches = [
            (
                candidate,
                distance,
                calculate_word_similarity_score(word_lower, candidate),
                calculate_word_similarity_score(word_lower, candidate)
                - (distance * 0.1),
            )
            for candidate, distance in matches
        ]
        scored_matches.sort(key=lambda x: x[3], reverse=True)
        best_match, distance, similarity, _ = scored_matches[0]
        if distance <= max_distance and similarity > 0.3:
            return best_match, True, distance
    return word, False, float("inf")


def pre_correct_administrative_words(
    text: str, spelling_trie: BKTree, max_distance: int = 1
) -> str:
    from pre_processing import BLACK_LIST_KEYWORDS  # Import at top

    admin_keywords = [
        "huyện",
        "quận",
        "xã",
        "phường",
        "thị",
        "trấn",
        "tỉnh",
        "thành",
        "phố",
    ]

    parts = text.split(",")
    corrected_parts = []

    for part in parts:
        tokens = part.strip().split()
        corrected_tokens = []

        for token in tokens:
            clean_token = re.sub(
                r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
                "",
                token.lower(),
            )

            # NEW: Skip blacklisted words
            if clean_token.lower() in BLACK_LIST_KEYWORDS or len(clean_token) < 3:
                corrected_tokens.append(token)
                continue

            # Check if valid in dictionary
            if is_valid_vietnamese_word(clean_token, spelling_trie):
                corrected_tokens.append(token)
                continue

            # Only correct if similar to admin keyword
            best_match = None
            best_distance = float("inf")

            for admin_word in admin_keywords:
                distance = levenshtein_distance(clean_token, admin_word)
                if distance <= max_distance and distance < best_distance:
                    best_match = admin_word
                    best_distance = distance

            if best_match:
                corrected_tokens.append(best_match)
            else:
                corrected_tokens.append(token)

        corrected_parts.append(" ".join(corrected_tokens))

    return ", ".join(corrected_parts)


@time_performance_measure
def correct_address_spelling(
    address: str,
    spelling_trie: BKTree,
    segment_trie: Trie,
    max_distance: int = 2,
    debug: bool = False,
) -> Tuple[str, List[dict]]:
    """
    Correct spelling errors in Vietnamese address text

    Args:
        address: Input address string
        spelling_trie: BK-Trie for spell checking
        max_distance: Maximum edit distance for corrections
        debug: Whether to print debug information

    Returns:
        Tuple of (corrected_address, corrections_made)
    """
    if not address or not address.strip():
        return address, []

    # PRE-CORRECTION: Fix administrative terms before segmentation
    pre_corrected = pre_correct_administrative_words(
        address.lower(), spelling_trie, max_distance=1
    )

    if debug:
        print(f"Pre-corrected: {pre_corrected}")

    # Now segment the pre-corrected text
    segmented_text = segment(pre_corrected, segment_trie)

    # Continue with existing token-level correction...
    tokens = segmented_text.split()

    corrected_tokens = []
    corrections_made = []

    for i, token in enumerate(tokens):
        # Clean token of punctuation for spell checking
        clean_token = re.sub(
            r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]",
            "",
            token,
        )

        if clean_token:
            corrected_word, was_corrected, distance = correct_word(
                clean_token, spelling_trie, max_distance
            )
            if was_corrected:
                corrected_token = token.replace(clean_token, corrected_word)
                corrected_tokens.append(corrected_token)
                correction_info = {
                    "position": i,
                    "original": clean_token,
                    "corrected": corrected_word,
                    "distance": distance,
                    "full_token_original": token,
                    "full_token_corrected": corrected_token,
                }
                corrections_made.append(correction_info)
                if debug:
                    print(
                        f"Corrected: '{clean_token}' -> '{corrected_word}' (distance: {distance})"
                    )
            else:
                corrected_tokens.append(token)
        else:
            corrected_tokens.append(token)

    corrected_address = " ".join(corrected_tokens)
    if debug and corrections_made:
        print(f"Original: {address}")
        print(f"Segmented: {segmented_text}")
        print(f"Corrected: {corrected_address}")
        print(f"Made {len(corrections_made)} corrections")
        for correction in corrections_made:
            print(
                f" - At token {correction['position']}: '{correction['original']}' -> '{correction['corrected']}'"
            )
    return corrected_address, corrections_made


if __name__ == "__main__":
    import time

    from segment import Trie

    # Load dictionary and build tries once
    segment_dictionary = load_vietnamese_dictionary()
    spelling_trie = build_spelling_correction_trie()
    segment_trie = Trie()
    for word in segment_dictionary:
        segment_trie.insert(word.lower())

    tests = [
        "X Tây Yên, H.An Biên, TKiên Giang",
        "Thanh Long, Yên Mỹ Hưng Yên",
        "X.Nga Thanh hyện Nga son TỉnhThanhQ Hóa",
        " Duy Phú,  Duy Xuyen,  Quang Nam",
        " Đức Lĩnh,Vũ Quang,",
        "Thi trấ Ea. Knốp,H. Ea Kar,",
        ", Nam Đông,T. T.T.H",
        "P4 T.Ph9ốĐông Hà ",
        "Xã Thịnh Sơn H., Đô dương T. Nghệ An",
        "Phưng Khâm Thiên Quận Đ.Đa T.Phố HàNội",
        "Tiểu khu 3, thị trấn Ba Hàng, huyện Phổ Yên, tỉnh Thái Nguyên.",
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
        normalized = normalize_input(test_address)
        print(f"Normalized: {normalized}")
        # segmented = segment_text_using_common_vn_words(test_address, segment_trie)
        corrected_address, corrections = correct_address_spelling(
            address=normalized,
            spelling_trie=spelling_trie,
            segment_trie=segment_trie,
            debug=True,
        )
        expanded = expand_abbreviations(corrected_address)

        end_time = time.perf_counter()
        elapsed_time_ms = (end_time - start_time) * 1000
        if elapsed_time_ms > 10:
            print(
                f"WARNING: Pre-processing took {elapsed_time_ms:.4f} ms, exceeding 10ms limit."
            )
        else:
            print(f"Pre-processing time: {elapsed_time_ms:.4f} ms")
        print(f"Finish Spell Check: {expanded}")
