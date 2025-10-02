from typing import List, Tuple
from bk_tree import BKTree
from segment import segment
from pre_processing import (
    PROVINCE_ABBREVIATIONS,
    ADMIN_ABBREVIATION_MAP,
    BLACK_LIST_KEYWORDS,
    expand_abbreviations,
    normalize_input,
)
import re
from bk_tree import levenshtein_distance


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
    """Build BK-Trie for spelling correction using Vietnamese dictionary"""
    dictionary_words = load_vietnamese_dictionary()
    dictionary_words = dictionary_words + [
        abbr.lower() for abbr in PROVINCE_ABBREVIATIONS.keys()
    ]
    dictionary_words = dictionary_words + [
        abbr.lower() for abbr in ADMIN_ABBREVIATION_MAP.keys()
    ]
    dictionary_words = dictionary_words + [word.lower() for word in BLACK_LIST_KEYWORDS]
    # Add common street keywords
    dictionary_words = list(set(dictionary_words))
    if not dictionary_words:
        return BKTree()

    spelling_trie = BKTree(dictionary_words)
    print(f"Built spelling correction trie with {spelling_trie.word_count} words")
    return spelling_trie


def is_valid_vietnamese_word(word: str, spelling_trie: BKTree) -> bool:
    """Check if a word exists in the Vietnamese dictionary"""
    if not word or not word.strip():
        return True  # Consider empty words as valid (no correction needed)

    # Check for exact match (case-insensitive)
    exact_match = spelling_trie.get_exact_match(word.lower())
    return exact_match is not None


def calculate_word_similarity_score(original: str, candidate: str) -> float:
    """
    Calculate similarity score between original and candidate word.
    Higher score means better match.
    """
    if not original or not candidate:
        return 0.0

    # Prefer longer matches
    length_bonus = min(len(candidate), len(original)) / max(
        len(candidate), len(original)
    )

    # Prefer words that start with the same characters
    common_prefix = 0
    for i, (c1, c2) in enumerate(zip(original, candidate)):
        if c1 == c2:
            common_prefix += 1
        else:
            break
    prefix_score = common_prefix / max(len(original), len(candidate))

    # Calculate overall similarity
    edit_distance = levenshtein_distance(original, candidate)
    max_length = max(len(original), len(candidate))
    distance_score = 1 - (edit_distance / max_length) if max_length > 0 else 0

    # Combine scores (weighted)
    final_score = (distance_score * 0.6) + (prefix_score * 0.3) + (length_bonus * 0.1)
    return final_score


def correct_word(
    word: str, spelling_trie: BKTree, max_distance: int = 2
) -> Tuple[str, bool, int]:
    """
    Correct a single word using the spelling trie with improved scoring

    Args:
        word: Word to correct
        spelling_trie: BK-Trie containing dictionary words
        max_distance: Maximum edit distance for suggestions

    Returns:
        Tuple of (corrected_word, was_corrected, edit_distance)
    """
    if not word or not word.strip():
        return word, False, 0

    word_lower = word.lower().strip()

    # Skip numbers and very short words
    if word_lower.isdigit() or len(word_lower) < 2:
        return word, False, 0

    # Skip common punctuation and special characters
    if word_lower in {"-", ".", ",", "/", "\\", "(", ")", "[", "]", "{", "}"}:
        return word, False, 0

    # Check if word is already correct
    if is_valid_vietnamese_word(word_lower, spelling_trie):
        return word, False, 0

    # Find spelling corrections
    matches = spelling_trie.search(word_lower, max_distance)

    if matches:
        # Score all matches and pick the best one
        scored_matches = []
        for candidate, distance in matches:
            similarity_score = calculate_word_similarity_score(word_lower, candidate)
            # Combine distance and similarity (lower distance is better, higher similarity is better)
            combined_score = similarity_score - (
                distance * 0.1
            )  # Penalize distance slightly
            scored_matches.append(
                (candidate, distance, similarity_score, combined_score)
            )

        # Sort by combined score (descending)
        scored_matches.sort(key=lambda x: x[3], reverse=True)

        best_match, distance, similarity, combined = scored_matches[0]

        # Only suggest correction if it's reasonable
        if distance <= max_distance and similarity > 0.3:
            return best_match, True, distance

    # No good correction found
    return word, False, float("inf")


def correct_address_spelling(
    address: str,
    spelling_trie: BKTree,
    max_distance: int = 2,
    debug: bool = False,
    dictionary: List[str] = None,
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

    # Segment the address first
    if dictionary is None:
        # Fallback: use spelling_trie words if no dictionary provided
        if hasattr(spelling_trie, "root") and hasattr(spelling_trie.root, "word"):
            # Collect all words from BKTree (not efficient, but fallback)
            def collect_words(node):
                words = []
                if node.word:
                    words.append(node.word)
                for child in getattr(node, "children", {}).values():
                    words.extend(collect_words(child))
                return words

            dictionary = collect_words(spelling_trie.root)
        else:
            dictionary = []

    segmented_text = segment(address.lower(), dictionary)
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

    # Example usage
    spelling_trie = build_spelling_correction_trie()

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
        # corrected_address, corrections = correct_address_spelling(
        #     test_address, spelling_trie, debug=True
        # )
        start_time = time.perf_counter()
        normalized = normalize_input(test_address)
        expanded = expand_abbreviations(normalized)
        corrected_address, corrections = correct_address_spelling(
            expanded, spelling_trie, debug=True
        )
        end_time = time.perf_counter()
        elapsed_time_ms = (end_time - start_time) * 1000
        if elapsed_time_ms > 10:
            print(
                f"WARNING: Pre-processing took {elapsed_time_ms:.4f} ms, exceeding 10ms limit."
            )
        print(f"Pre-processing time: {elapsed_time_ms:.4f} ms")
        print(f"Final Corrected Address: {corrected_address}")
