import re
import time
import unicodedata
import json
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter

############################################################
# CSV loading helpers - Data preprocessing
############################################################


def load_csv_with_encoding(file_path: str, location_type: str = None) -> List[str]:
    """
    Load data from CSV file with automatic encoding detection and prefix removal

    ALGORITHM: Multi-encoding fallback with administrative prefix removal
    - Try multiple encodings (UTF-8-sig first to handle BOM)
    - Remove Vietnamese administrative prefixes (Thành phố, Tỉnh, Huyện, etc.)
    - This preprocessing normalizes data for better matching
    """
    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1"]

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                data = [line.strip() for line in f if line.strip()]

            # Remove prefixes based on location type
            if location_type:
                data = [remove_prefix(name, location_type) for name in data]
            return data
        except (UnicodeDecodeError, FileNotFoundError):
            continue

    raise Exception(f"Could not read file {file_path} with any supported encoding")


# Administrative prefix removal functions
def remove_prefix(name: str, location_type: str) -> str:
    """Remove administrative prefixes from Vietnamese location names"""
    prefixes = {
        "province": ["Thành phố ", "Tỉnh "],
        "district": ["Huyện ", "Quận ", "Thị xã ", "Thành phố "],
        "ward": ["Phường ", "Xã ", "Thị trấn "],
    }

    for prefix in prefixes.get(location_type, []):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


############################################################
# Address classifier - Main Algorithm
############################################################


class AddressClassifier:
    """
    MAIN ALGORITHM: Multi-stage fuzzy string matching with Dynamic Programming

    APPROACH:
    1. Text Normalization - Handle Unicode, remove diacritics, normalize spacing
    2. Alias Expansion - Handle common abbreviations (Q1 -> Quan 1, HCM -> Ho Chi Minh)
    3. N-gram Generation - Create all possible substrings (1-3 words)
    4. Fuzzy Matching - Use DP edit distance for spell error tolerance
    5. Candidate Scoring - Score candidates based on similarity + frequency penalty
    6. Combination Selection - Find best non-overlapping province/district/ward combo

    KEY OPTIMIZATIONS:
    - Bucketing by first character for O(1) candidate lookup
    - LRU cache for normalization (hot path optimization)
    - Edit distance cache with early cutoff for performance
    - Frequency-based penalty to prefer common locations
    """

    def __init__(self, provinces: List[str], districts: List[str], wards: List[str]):
        """
        Initialize classifier with preprocessing for fast lookup

        DATA STRUCTURES:
        - norm_* : normalized_name -> original_name mapping
        - freq_* : frequency counters for penalty calculation
        - bucket_* : first_char -> [candidates] for O(1) lookup
        """
        self.provinces = provinces
        self.districts = districts
        self.wards = wards

        # Create normalized mappings for reverse lookup
        self.norm_provinces = {self._normalize(p): p for p in provinces}
        self.norm_districts = {self._normalize(d): d for d in districts}
        self.norm_wards = {self._normalize(w): w for w in wards}

        # Frequency counters for penalty calculation (common names get lower penalty)
        self.freq_provinces = Counter(self._normalize(p) for p in provinces)
        self.freq_districts = Counter(self._normalize(d) for d in districts)
        self.freq_wards = Counter(self._normalize(w) for w in wards)

        # Bucketing by first alphanumeric character for fast candidate lookup
        # This reduces search space from O(n) to O(candidates_per_bucket)
        self.bucket_provinces = defaultdict(list)
        self.bucket_districts = defaultdict(list)
        self.bucket_wards = defaultdict(list)
        for n in self.norm_provinces:
            self.bucket_provinces[self._first_alnum(n)].append(n)
        for n in self.norm_districts:
            self.bucket_districts[self._first_alnum(n)].append(n)
        for n in self.norm_wards:
            self.bucket_wards[self._first_alnum(n)].append(n)

        # Alias mapping for common abbreviations and alternative names
        self.alias_map = {
            "hcm": "ho chi minh",
            "sai gon": "ho chi minh",
            "saigon": "ho chi minh",
            "sg": "ho chi minh",
            "hn": "ha noi",
            "hnoi": "ha noi",
            "tp": "thanh pho",
            "t": "tinh",
            "tt": "thi tran",
        }

        # Cache for edit distance calculations (performance optimization)
        self._edit_distance_cache: Dict[Tuple[str, str], int] = {}

    @lru_cache(maxsize=20000)
    def _normalize(self, text: str) -> str:
        """
        TEXT NORMALIZATION ALGORITHM:
        1. Convert to lowercase
        2. Remove Vietnamese diacritics using Unicode NFD decomposition
        3. Replace punctuation with spaces
        4. Normalize whitespace

        This handles spelling variations and diacritic inconsistencies
        """
        if not text:
            return ""
        text = text.lower().strip()
        # Unicode NFD decomposition separates base chars from diacritics
        normalized = unicodedata.normalize("NFD", text)
        # Remove diacritic marks (category 'Mn' = nonspacing marks)
        normalized = "".join(
            ch for ch in normalized if unicodedata.category(ch) != "Mn"
        )
        # Replace punctuation with spaces for consistent tokenization
        normalized = re.sub(r"[\,\.;/\\|\-]", " ", normalized)
        # Normalize whitespace
        normalized = " ".join(normalized.split())
        return normalized

    def _first_alnum(self, s: str) -> str:
        """Get first alphanumeric character for bucketing"""
        for ch in s:
            if ch.isalnum():
                return ch
        return "#"

    def _apply_aliases(self, text_norm: str) -> str:
        """
        ALIAS EXPANSION ALGORITHM:
        1. Expand abbreviated administrative units (Q1 -> quan 1)
        2. Replace common abbreviations (HCM -> ho chi minh)

        This handles common Vietnamese address abbreviations
        """

        # Expand Q1/P5/H7 -> quan 1 / phuong 5 / huyen 7
        def repl_q(m):
            return f"quan {int(m.group(1))}"

        def repl_p(m):
            return f"phuong {int(m.group(1))}"

        def repl_h(m):
            return f"huyen {int(m.group(1))}"

        text_norm = re.sub(r"\bq\.?\s*(\d{1,2})\b", repl_q, text_norm)
        text_norm = re.sub(r"\bp\.?\s*(\d{1,2})\b", repl_p, text_norm)
        text_norm = re.sub(r"\bh\.?\s*(\d{1,2})\b", repl_h, text_norm)

        # Word-level aliases
        words = text_norm.split()
        for i, w in enumerate(words):
            if w in self.alias_map:
                words[i] = self.alias_map[w]
        return " ".join(words)

    def _generate_tokens(self, address: str) -> List[str]:
        """Generate normalized tokens from input address"""
        norm_full = self._normalize(address)
        norm_full = self._apply_aliases(norm_full)
        return norm_full.split()

    def _ngrams(self, tokens: List[str], max_n: int = 3) -> List[Tuple[int, int, str]]:
        """
        N-GRAM GENERATION ALGORITHM:
        Generate all possible contiguous subsequences of 1-3 words
        Returns (start_idx, end_idx, text) tuples

        This allows matching location names that span multiple words
        """
        out = []
        n = len(tokens)
        for i in range(n):
            for L in range(1, max_n + 1):
                j = i + L
                if j <= n:
                    out.append((i, j, " ".join(tokens[i:j])))
        return out

    # ---------------- Dynamic Programming Edit Distance ----------------
    def _edit_distance_dp(self, s1: str, s2: str, max_dist: int) -> int:
        """
        DYNAMIC PROGRAMMING EDIT DISTANCE with early cutoff

        ALGORITHM: Wagner-Fischer DP with optimizations
        1. Early termination if length difference > max_dist
        2. Row-wise DP with space optimization (O(min(m,n)) space)
        3. Early cutoff if minimum in current row > max_dist
        4. Caching for repeated calculations

        This handles spelling errors within reasonable edit distance
        """
        key = (s1, s2) if s1 <= s2 else (s2, s1)
        if key in self._edit_distance_cache:
            return self._edit_distance_cache[key]

        len1, len2 = len(s1), len(s2)
        # Early termination: if length difference > max_dist, impossible to match
        if abs(len1 - len2) > max_dist:
            return float("inf")

        # DP table: prev[j] = edit distance between s1[:i-1] and s2[:j]
        prev = list(range(len2 + 1))
        curr = [0] * (len2 + 1)

        for i in range(1, len1 + 1):
            curr[0] = i
            min_val = float("inf")
            ch1 = s1[i - 1]

            for j in range(1, len2 + 1):
                cost = 0 if ch1 == s2[j - 1] else 1
                # Three operations: insert, delete, substitute
                a = prev[j] + 1  # deletion
                b = curr[j - 1] + 1  # insertion
                c = prev[j - 1] + cost  # substitution
                curr[j] = min(a, b, c)
                min_val = min(min_val, curr[j])

            # Early cutoff: if all values in current row > max_dist, impossible
            if min_val > max_dist:
                self._edit_distance_cache[key] = float("inf")
                return float("inf")

            prev, curr = curr, prev

        d = prev[len2]
        self._edit_distance_cache[key] = d
        return d

    def _similarity(self, a: str, b: str, max_dist_cap: int = 4) -> float:
        """
        SIMILARITY SCORING:
        Convert edit distance to similarity score [0,1]
        - Early exit for exact matches
        - Adaptive max_dist based on string length (30% of max length)
        - Normalize by max string length
        """
        if a == b:
            return 1.0
        max_len = max(len(a), len(b))
        max_dist = min(max_dist_cap, max(2, int(0.3 * max_len)))
        d = self._edit_distance_dp(a, b, max_dist)
        if d == float("inf"):
            return 0.0
        return 1.0 - (d / max_len)

    # ---------------- Candidate Search + Scoring ----------------
    def _best_match_for_type(
        self, text_norm: str, loc_type: str
    ) -> Tuple[Optional[str], float, int, float]:
        """
        CANDIDATE MATCHING ALGORITHM:
        1. Bucket lookup by first character (O(1) filtering)
        2. Fuzzy similarity calculation for each candidate
        3. Frequency-based penalty (prefer common locations)
        4. Prefix bonus (first word match bonus)
        5. Exact match bonus
        6. Threshold filtering

        SCORING FORMULA:
        score = similarity * frequency_penalty + prefix_bonus + exact_bonus
        """
        if not text_norm:
            return None, 0.0, 0, 0.0

        # Get candidates from bucket (O(1) lookup vs O(n) linear search)
        first = self._first_alnum(text_norm)
        if loc_type == "province":
            cand_bucket = self.bucket_provinces.get(first, [])
            norm_map, freq_map = self.norm_provinces, self.freq_provinces
            base_thr, exact_bonus = 0.78, 0.20  # Province thresholds
        elif loc_type == "district":
            cand_bucket = self.bucket_districts.get(first, [])
            norm_map, freq_map = self.norm_districts, self.freq_districts
            base_thr, exact_bonus = 0.76, 0.16  # District thresholds
        else:  # ward
            cand_bucket = self.bucket_wards.get(first, [])
            norm_map, freq_map = self.norm_wards, self.freq_wards
            base_thr, exact_bonus = 0.83, 0.14  # Ward thresholds (stricter)

        t_first = text_norm.split()[0]
        best_name, best_score, best_freq, best_raw = None, 0.0, 0, 0.0

        for cand_norm in cand_bucket:
            # Calculate base similarity using edit distance
            sim = self._similarity(text_norm, cand_norm)
            if sim <= 0:
                continue

            # Frequency penalty: less common locations get slight penalty
            freq = freq_map[cand_norm]
            penalty = 1.0 / (1.0 + (0.6 * (freq - 1)))

            # Prefix bonus: if first word matches (helps distinguish similar names)
            prefix_bonus = 0.06 if cand_norm.split()[0] == t_first else 0.0

            # Calculate final score
            score = sim * penalty + prefix_bonus
            if sim == 1.0:
                score += exact_bonus  # Strong bonus for exact matches

            if score > best_score:
                best_score = score
                best_name = norm_map[cand_norm]  # Return original name
                best_freq = freq
                best_raw = sim

        # Apply threshold filtering
        if best_name is None:
            return None, 0.0, 0, 0.0
        if best_raw < base_thr:
            return None, 0.0, 0, best_raw
        return best_name, best_score, best_freq, best_raw

    def classify(self, address: str) -> Dict[str, Optional[str]]:
        """
        MAIN CLASSIFICATION ALGORITHM:

        STEPS:
        1. Tokenization & N-gram generation
        2. Candidate search for each location type
        3. Deduplication & top-K selection
        4. Combination optimization (find best non-overlapping combo)

        OVERLAP PREVENTION: Ensures province/district/ward don't overlap in text
        WARD STRICTNESS: Only accept wards with exact match OR ward hint words
        """
        # Cache management for memory efficiency
        if len(self._edit_distance_cache) > 50000:
            self._edit_distance_cache.clear()

        # Step 1: Tokenization and preprocessing
        tokens = self._generate_tokens(address)
        if not tokens:
            return {"province": None, "district": None, "ward": None}

        # Generate all possible n-grams (substrings)
        spans = self._ngrams(tokens, max_n=3)
        norm_full = " ".join(tokens)

        # Ward hint detection: only accept ward candidates if we see ward keywords
        # This prevents false ward matches (stricter ward matching)
        has_ward_hint = bool(re.search(r"\b(phuong|xa|thi tran)\b", norm_full))

        # Step 2: Candidate search
        topK = 3
        cand_by_type = {"province": [], "district": [], "ward": []}

        for i, j, text in spans:
            # Search provinces and districts normally
            for t in ("province", "district"):
                name, score, freq, raw = self._best_match_for_type(text, t)
                if name:
                    cand_by_type[t].append(((i, j), name, score, freq, raw))

            # Ward search with stricter criteria
            name, score, freq, raw = self._best_match_for_type(text, "ward")
            if name and (raw == 1.0 or has_ward_hint):  # Exact match OR ward hint
                cand_by_type["ward"].append(((i, j), name, score, freq, raw))

        # Step 3: Deduplication and top-K selection
        for t in ("province", "district", "ward"):
            dedup = {}
            for span, name, score, freq, raw in cand_by_type[t]:
                # Keep best score for each location name
                if name not in dedup or score > dedup[name][1]:
                    dedup[name] = (span, score, freq, raw)
            # Sort by score (desc) then by span length (desc) and take top-K
            cand_by_type[t] = sorted(
                [
                    (span, name, score, freq, raw)
                    for name, (span, score, freq, raw) in dedup.items()
                ],
                key=lambda x: (-x[2], -(x[0][1] - x[0][0])),
            )[:topK]

        # Add None option for each type (allows missing components)
        for t in ("province", "district", "ward"):
            cand_by_type[t].append(((None, None), None, 0.0, 0, 0.0))

        # Step 4: Combination optimization
        def overlap(a, b):
            """Check if two spans overlap in the text"""
            (i1, j1) = a
            (i2, j2) = b
            if None in (i1, j1, i2, j2):
                return False
            return not (j1 <= i2 or j2 <= i1)

        # Find best non-overlapping combination
        best_combo = (None, None, None)
        best_total = -1.0

        for p_span, p_name, p_score, *_ in cand_by_type["province"]:
            for d_span, d_name, d_score, *_ in cand_by_type["district"]:
                # Skip if province and district overlap
                if p_name and d_name and overlap(p_span, d_span):
                    continue
                for w_span, w_name, w_score, *_ in cand_by_type["ward"]:
                    # Skip if ward overlaps with province or district
                    if (w_name and p_name and overlap(w_span, p_span)) or (
                        w_name and d_name and overlap(w_span, d_span)
                    ):
                        continue
                    # Calculate total score
                    total = p_score + d_score + w_score
                    # Bonus for complete addresses (all three components)
                    if p_name and d_name and w_name:
                        total += 0.05
                    if total > best_total:
                        best_total = total
                        best_combo = (p_name, d_name, w_name)

        return {
            "province": best_combo[0],
            "district": best_combo[1],
            "ward": best_combo[2],
        }

    def classify_batch(self, addresses: List[str]) -> List[Dict[str, Optional[str]]]:
        """Batch classification with cache management"""
        if len(self._edit_distance_cache) > 50000:
            self._edit_distance_cache.clear()
        return [self.classify(a) for a in addresses]


############################################################
# Testing Infrastructure
############################################################
def test_classifier():
    """
    Test the address classifier with test cases loaded from JSON file

    PERFORMANCE MONITORING:
    - Tracks individual and average timing
    - Validates against performance constraints (≤10ms avg, ≤100ms max)
    - Measures accuracy against expected results
    """
    print("Loading data from CSV files...")
    classifier = load_from_csv()

    # Load test cases from JSON file
    print("Loading test cases from JSON file...")
    try:
        with open("test.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print("Error: test.json file not found")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        return

    # Convert JSON data to test cases format
    tests = []
    for item in test_data:
        test_case = (
            item["input"],
            {
                "province": item["expected_province"],
                "district": item["expected_district"],
                "ward": item["expected_ward"],
            },
        )
        tests.append(test_case)

    print(f"Loaded {len(tests)} test cases")
    print("Testing Address Classifier")
    print("=" * 80)

    # Performance and accuracy tracking
    correct_count = 0
    total_time = 0
    max_time = 0

    for i, (address, expected) in enumerate(tests, 1):
        # Time each classification
        t0 = time.perf_counter()
        got = classifier.classify(address)
        dt = (time.perf_counter() - t0) * 1000  # Convert to milliseconds
        total_time += dt
        max_time = max(max_time, dt)

        # Check accuracy
        ok = all(got.get(k) == v for k, v in expected.items())
        if ok:
            correct_count += 1

        print(f"\nTest {i}:")
        print(f"Input: {address}")
        print(f"Normalized: {classifier._normalize(address)}")
        print(f"Expected: {expected}")
        print(f"Got:      {got}")
        print(f"{'PASS' if ok else 'FAIL'} in {dt:.2f} ms")

    # Summary statistics
    avg_time = total_time / len(tests)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(tests)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {correct_count / len(tests) * 100:.1f}%")
    print(f"Average time: {avg_time:.2f} ms")
    print(f"Max time: {max_time:.2f} ms")
    print(f"Total time: {total_time:.1f} ms")

    # Performance constraint validation
    print("\n" + "=" * 80)
    print("PERFORMANCE CONSTRAINTS")
    print("=" * 80)
    print(f"Average time ≤ 10ms: {'✓' if avg_time <= 10 else '✗'} ({avg_time:.2f} ms)")
    print(f"Max time ≤ 100ms: {'✓' if max_time <= 100 else '✗'} ({max_time:.2f} ms)")


def load_from_csv(data_dir: str = "data") -> AddressClassifier:
    """Load classifier from CSV files with proper encoding handling"""
    provinces = load_csv_with_encoding(f"{data_dir}/provinces.csv", "province")
    districts = load_csv_with_encoding(f"{data_dir}/districts.csv", "district")
    wards = load_csv_with_encoding(f"{data_dir}/wards.csv", "ward")
    print(
        f"Loaded {len(provinces)} provinces, {len(districts)} districts, {len(wards)} wards"
    )
    return AddressClassifier(provinces, districts, wards)


if __name__ == "__main__":
    test_classifier()
