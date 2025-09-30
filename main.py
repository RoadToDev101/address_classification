# Vietnam Address Classification Solution
# Combined approach from both algorithms with performance optimizations
# Author: AI Assistant
# Performance Requirements: ≤0.1s max, ≤0.01s average per request

import re
import unicodedata
import time
import json
import csv
import os
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Optional, Tuple


class TrieNode:
    """Trie node for efficient prefix-based searching"""

    def __init__(self):
        self.children = {}
        self.is_end = False
        self.original_names = []


class Trie:
    """Trie data structure for fast prefix matching"""

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, original_name):
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        if original_name not in node.original_names:
            node.original_names.append(original_name)

    def search_prefix(self, prefix, max_results=5):
        node = self.root
        prefix = prefix.lower()

        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        results = []
        self._collect_words(node, results, max_results)
        return results

    def _collect_words(self, node, results, max_results):
        if len(results) >= max_results:
            return
        if node.is_end:
            results.extend(node.original_names[: max_results - len(results)])
        for child in node.children.values():
            if len(results) < max_results:
                self._collect_words(child, results, max_results)


class AddressClassifier:
    """
    Vietnam address classifier combining best features from both algorithms:
    - Trie for prefix matching
    - Hash tables for O(1) exact lookups
    - Simplified fuzzy matching with Levenshtein distance
    - Better text normalization and preprocessing
    - Performance optimizations for speed
    """

    def __init__(self, provinces: List[str], districts: List[str], wards: List[str]):
        self.provinces = provinces
        self.districts = districts
        self.wards = wards

        # Data structures for fast lookup
        self.province_hash = {}
        self.district_hash = {}
        self.ward_hash = {}

        # Trie structures for prefix matching
        self.province_trie = Trie()
        self.district_trie = Trie()
        self.ward_trie = Trie()

        # N-gram tables for fuzzy matching
        self.province_ngrams = defaultdict(set)
        self.district_ngrams = defaultdict(set)
        self.ward_ngrams = defaultdict(set)

        # Caching for performance
        self.normalization_cache = {}
        self.match_cache = {}
        self.edit_distance_cache = {}

        # Common Vietnamese address patterns and abbreviations
        self.address_patterns = {
            "province_prefixes": [r"\b(tỉnh|tnh|t\.)\s*"],
            "district_prefixes": [
                r"\b(quận|qun|q\.?|huyện|huyn|h\s|h\.|thị xã|tx\.?|thành phố|thnh phố|tp\.?)\s*"
            ],
            "ward_prefixes": [r"\b(phường|phng|p\.?|xã|x\.?|thị trần|tt\.?)\s*"],
        }

        # Alias mapping for common abbreviations
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

        self._build_data_structures()

    @lru_cache(maxsize=20000)
    def _normalize_vietnamese_text(self, text: str) -> str:
        """
        Advanced Vietnamese text normalization with caching
        Handles diacritics, case, spacing, and common abbreviations
        """
        if not text:
            return ""

        # Check cache first
        if text in self.normalization_cache:
            return self.normalization_cache[text]

        # Normalize text
        result = text.lower().strip()

        # Remove diacritics using Unicode normalization
        normalized = unicodedata.normalize("NFD", result)
        without_diacritics = "".join(
            c for c in normalized if unicodedata.category(c) != "Mn"
        )

        # Clean up spaces and punctuation
        cleaned = re.sub(r"[^\w\s]", " ", without_diacritics)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Apply aliases
        words = cleaned.split()
        for i, word in enumerate(words):
            if word in self.alias_map:
                words[i] = self.alias_map[word]
        cleaned = " ".join(words)

        # Cache result
        self.normalization_cache[text] = cleaned
        return cleaned

    def _generate_ngrams(self, text: str, n: int = 2) -> List[str]:
        """Generate n-grams for fuzzy matching - now includes 1-grams"""
        if len(text) < n:
            return list(text)  # Return individual characters
        ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]
        # Also add individual characters for better typo tolerance
        if n > 1:
            ngrams.extend(list(text))
        return ngrams

    def _build_data_structures(self):
        """Build all data structures with better n-gram indexing"""
        print("Building data structures...")

        # Build province structures
        for province in self.provinces:
            variants = self._generate_text_variants(province)
            for variant in variants:
                # Hash table
                self.province_hash[variant] = province
                # Trie
                self.province_trie.insert(variant, province)
                # N-grams - now includes 1-grams for typo tolerance
                ngrams = self._generate_ngrams(variant, 2)
                for ngram in ngrams:
                    self.province_ngrams[ngram].add(province)

        # Build district structures
        for district in self.districts:
            variants = self._generate_text_variants(district)
            for variant in variants:
                self.district_hash[variant] = district
                self.district_trie.insert(variant, district)
                ngrams = self._generate_ngrams(variant, 2)
                for ngram in ngrams:
                    self.district_ngrams[ngram].add(district)

        # Build ward structures
        for ward in self.wards:
            variants = self._generate_text_variants(ward)
            for variant in variants:
                self.ward_hash[variant] = ward
                self.ward_trie.insert(variant, ward)
                ngrams = self._generate_ngrams(variant, 2)
                for ngram in ngrams:
                    self.ward_ngrams[ngram].add(ward)

        print("Data structures built successfully!")

    def _generate_text_variants(self, text: str) -> List[str]:
        """Generate variants of text for better matching"""
        variants = set()

        # Original and normalized
        variants.add(text)
        normalized = self._normalize_vietnamese_text(text)
        variants.add(normalized)

        # Without spaces
        variants.add(normalized.replace(" ", ""))

        # Individual significant words
        words = normalized.split()
        for word in words:
            if len(word) >= 3:  # Only meaningful words
                variants.add(word)

        return [v for v in variants if v]

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Levenshtein distance calculation with caching
        """
        if s1 == s2:
            return 0

        # Check cache
        key = (s1, s2) if s1 <= s2 else (s2, s1)
        if key in self.edit_distance_cache:
            return self.edit_distance_cache[key]

        len1, len2 = len(s1), len(s2)
        if len1 == 0:
            return len2
        if len2 == 0:
            return len1

        # Use only two rows for space optimization
        prev_row = list(range(len2 + 1))
        curr_row = [0] * (len2 + 1)

        for i in range(1, len1 + 1):
            curr_row[0] = i
            for j in range(1, len2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                curr_row[j] = min(
                    curr_row[j - 1] + 1,  # insertion
                    prev_row[j] + 1,  # deletion
                    prev_row[j - 1] + cost,  # substitution
                )
            prev_row, curr_row = curr_row, prev_row

        result = prev_row[len2]
        self.edit_distance_cache[key] = result
        return result

    def _fuzzy_match(
        self,
        query: str,
        hash_table: dict,
        trie: Trie,
        ngram_table: dict,
        threshold: float = 0.7,
    ) -> Optional[str]:
        """
        Improved fuzzy matching with better candidate selection
        """
        if not query:
            return None

        query_norm = self._normalize_vietnamese_text(query)

        # Algorithm 1: Direct hash lookup (fastest)
        if query_norm in hash_table:
            return hash_table[query_norm]

        # Algorithm 2: Trie prefix matching
        trie_matches = trie.search_prefix(query_norm, max_results=3)
        if trie_matches:
            return trie_matches[0]

        # Algorithm 3: IMPROVED - Direct Levenshtein for small errors
        # For queries with likely typos (1-2 char errors), check ALL candidates
        candidates = set()

        # Collect candidates more aggressively
        if len(query_norm) >= 2:
            # Use single character n-grams for better typo tolerance
            for i in range(len(query_norm)):
                char = query_norm[i]
                if char in ngram_table:
                    candidates.update(ngram_table[char])

            # Also use 2-grams
            query_ngrams = set(self._generate_ngrams(query_norm, 2))
            for ngram in query_ngrams:
                if ngram in ngram_table:
                    candidates.update(ngram_table[ngram])

        # For short queries, also check by first few characters
        if len(query_norm) >= 3:
            prefix_key = query_norm[:3]
            for key in hash_table:
                if key.startswith(prefix_key):
                    candidates.add(hash_table[key])

        # Dynamic candidate limit based on query length
        max_candidates = min(100, len(query_norm) * 10)
        if len(candidates) > max_candidates:
            # Prioritize candidates with similar length
            candidates = sorted(
                candidates,
                key=lambda x: abs(
                    len(self._normalize_vietnamese_text(x)) - len(query_norm)
                ),
            )
            candidates = candidates[:max_candidates]

        # Score candidates using Levenshtein distance
        best_match = None
        best_score = 0.0

        for candidate in candidates:
            candidate_norm = self._normalize_vietnamese_text(candidate)

            # Quick length check - skip if too different
            if abs(len(candidate_norm) - len(query_norm)) > 3:
                continue

            distance = self._levenshtein_distance(query_norm, candidate_norm)
            max_len = max(len(query_norm), len(candidate_norm))

            if max_len > 0:
                similarity = 1.0 - (distance / max_len)

                # Bonus for exact prefix match
                if candidate_norm.startswith(query_norm[: min(3, len(query_norm))]):
                    similarity += 0.1

                if similarity > best_score and similarity >= threshold:
                    best_score = similarity
                    best_match = candidate

        return best_match

    def _extract_address_components(
        self, address: str
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Intelligent address component extraction using regex and context
        Now handles ambiguous names that appear in multiple administrative levels
        """
        if not address:
            return [], [], []

        # Normalize address
        address = re.sub(r"\s+", " ", address.strip())

        # Split by common delimiters
        parts = re.split(r"[,\-\n]", address)
        parts = [part.strip() for part in parts if part.strip()]

        potential_provinces = []
        potential_districts = []
        potential_wards = []

        for i, part in enumerate(parts):
            original_part = part
            part_lower = part.lower()

            # Detect component type by prefixes
            has_province_prefix = any(
                re.search(prefix, part_lower)
                for prefix in self.address_patterns["province_prefixes"]
            )
            has_district_prefix = any(
                re.search(prefix, part_lower)
                for prefix in self.address_patterns["district_prefixes"]
            )
            has_ward_prefix = any(
                re.search(prefix, part_lower)
                for prefix in self.address_patterns["ward_prefixes"]
            )

            # Remove prefixes more carefully - only remove the specific prefix that was detected
            cleaned_part = original_part
            if has_province_prefix:
                for prefix in self.address_patterns["province_prefixes"]:
                    cleaned_part = re.sub(
                        "^" + prefix, "", cleaned_part, flags=re.IGNORECASE
                    )
            elif has_district_prefix:
                for prefix in self.address_patterns["district_prefixes"]:
                    cleaned_part = re.sub(
                        "^" + prefix, "", cleaned_part, flags=re.IGNORECASE
                    )
            elif has_ward_prefix:
                for prefix in self.address_patterns["ward_prefixes"]:
                    cleaned_part = re.sub(
                        "^" + prefix, "", cleaned_part, flags=re.IGNORECASE
                    )

            cleaned_part = cleaned_part.strip()
            if not cleaned_part:
                continue

            # Smart categorization based on prefixes, position, and context
            if has_province_prefix or self._is_major_city(cleaned_part):
                potential_provinces.append(cleaned_part)
            elif has_district_prefix:
                potential_districts.append(cleaned_part)
            elif has_ward_prefix:
                potential_wards.append(cleaned_part)
            else:
                # For ambiguous components, use context-aware categorization
                self._categorize_ambiguous_component(
                    cleaned_part,
                    i,
                    len(parts),
                    potential_provinces,
                    potential_districts,
                    potential_wards,
                )

            # Special handling for "thành phố" - can be both province and district
            if has_district_prefix and any(
                prefix in part_lower for prefix in ["thành phố", "thnh phố", "tp"]
            ):
                # Check if this thành phố exists as a province (major cities)
                if self._is_major_city(cleaned_part):
                    potential_provinces.append(cleaned_part)

        return potential_provinces, potential_districts, potential_wards

    def _categorize_ambiguous_component(
        self,
        component: str,
        position: int,
        total_parts: int,
        potential_provinces: List[str],
        potential_districts: List[str],
        potential_wards: List[str],
    ):
        """
        Categorize components that could belong to multiple administrative levels
        Uses position, context, and actual data presence to make smart decisions
        """
        # Check which administrative levels this component actually exists in
        exists_in_provinces = self._component_exists_in_level(component, "province")
        exists_in_districts = self._component_exists_in_level(component, "district")
        exists_in_wards = self._component_exists_in_level(component, "ward")

        # Position-based hints (Vietnamese addresses typically go: ward, district, province)
        is_likely_ward = position == 0  # First part is usually ward
        is_likely_district = position == 1 or (
            position == 0 and total_parts == 2
        )  # Second part or only part
        is_likely_province = (
            position == total_parts - 1
        )  # Last part is usually province

        # Special case: If component exists in both districts and wards,
        # use more sophisticated context analysis
        if exists_in_districts and exists_in_wards and not exists_in_provinces:
            # Check surrounding context for hints
            if self._has_district_context_hints(component, position, total_parts):
                potential_districts.append(component)
            elif self._has_ward_context_hints(component, position, total_parts):
                potential_wards.append(component)
            else:
                # Default to position-based categorization
                if is_likely_district:
                    potential_districts.append(component)
                elif is_likely_ward:
                    potential_wards.append(component)
                else:
                    # Add to both and let confidence scoring decide
                    potential_districts.append(component)
                    potential_wards.append(component)
            return

        # Add to categories based on existence and position
        if exists_in_provinces and (
            is_likely_province or not exists_in_districts and not exists_in_wards
        ):
            potential_provinces.append(component)

        if exists_in_districts and (
            is_likely_district or not exists_in_provinces and not exists_in_wards
        ):
            potential_districts.append(component)

        if exists_in_wards and (
            is_likely_ward or not exists_in_provinces and not exists_in_districts
        ):
            potential_wards.append(component)

        # If component exists in multiple levels and position is unclear, add to all relevant levels
        # but with lower priority (will be handled by confidence scoring later)
        if exists_in_provinces and exists_in_districts and exists_in_wards:
            # Add to the most likely category based on position
            if is_likely_province:
                potential_provinces.append(component)
            elif is_likely_district:
                potential_districts.append(component)
            elif is_likely_ward:
                potential_wards.append(component)
            else:
                # Default fallback - add to all but let confidence scoring decide
                potential_provinces.append(component)
                potential_districts.append(component)
                potential_wards.append(component)
        elif exists_in_provinces and exists_in_districts:
            if is_likely_province:
                potential_provinces.append(component)
            elif is_likely_district:
                potential_districts.append(component)
            else:
                potential_provinces.append(component)
                potential_districts.append(component)
        elif exists_in_provinces and exists_in_wards:
            if is_likely_province:
                potential_provinces.append(component)
            elif is_likely_ward:
                potential_wards.append(component)
            else:
                potential_provinces.append(component)
                potential_wards.append(component)

    def _has_district_context_hints(
        self, component: str, position: int, total_parts: int
    ) -> bool:
        """Check if context suggests this component should be a district"""
        # If it's in the middle position of a 3-part address, likely district
        if total_parts == 3 and position == 1:
            return True

        # If it's the only component in a 2-part address, likely district
        if total_parts == 2 and position == 0:
            return True

        # If it's the second-to-last component, likely district
        if position == total_parts - 2:
            return True

        return False

    def _has_ward_context_hints(
        self, component: str, position: int, total_parts: int
    ) -> bool:
        """Check if context suggests this component should be a ward"""
        # If it's the first component, likely ward
        if position == 0:
            return True

        # If it's followed by a district-like component, likely ward
        # (This would need more context analysis in a real implementation)
        return False

    def _component_exists_in_level(self, component: str, level: str) -> bool:
        """Check if a component exists in a specific administrative level"""
        if level == "province":
            hash_table = self.province_hash
        elif level == "district":
            hash_table = self.district_hash
        else:  # ward
            hash_table = self.ward_hash

        normalized = self._normalize_vietnamese_text(component)
        return normalized in hash_table

    def _is_major_city(self, text: str) -> bool:
        """Check if text contains major Vietnamese cities"""
        text_lower = text.lower()
        major_cities = ["hà nội", "hồ chí minh", "đà nẵng", "cần thơ", "hải phòng"]
        return any(city in text_lower for city in major_cities)

    def classify_address(self, address: str) -> Dict[str, Optional[str]]:
        """
        Main classification method for performance
        Returns: Dictionary with 'province', 'district', 'ward' keys
        """
        if not address or not address.strip():
            return {"province": None, "district": None, "ward": None}

        # Check cache
        cache_key = address.strip()
        if cache_key in self.match_cache:
            return self.match_cache[cache_key]

        # Extract components
        province_candidates, district_candidates, ward_candidates = (
            self._extract_address_components(address)
        )

        # Find best matches using multi-algorithm approach
        province = self._find_best_match(province_candidates, "province")
        district = self._find_best_match(district_candidates, "district")
        ward = self._find_best_match(ward_candidates, "ward")

        result = {"province": province, "district": district, "ward": ward}

        # Cache result
        self.match_cache[cache_key] = result
        return result

    def _find_best_match(
        self, candidates: List[str], component_type: str
    ) -> Optional[str]:
        """Find the best match with adjusted thresholds"""
        if not candidates:
            return None

        # Lower thresholds to handle more spelling errors
        if component_type == "province":
            hash_table, trie, ngram_table = (
                self.province_hash,
                self.province_trie,
                self.province_ngrams,
            )
            threshold = 0.65  # Lowered from 0.75
        elif component_type == "district":
            hash_table, trie, ngram_table = (
                self.district_hash,
                self.district_trie,
                self.district_ngrams,
            )
            threshold = 0.60  # Lowered from 0.70
        else:  # ward
            hash_table, trie, ngram_table = (
                self.ward_hash,
                self.ward_trie,
                self.ward_ngrams,
            )
            threshold = 0.65  # Lowered from 0.80

        # Score all candidates and return the best one
        best_match = None
        best_confidence = 0.0

        for candidate in candidates:
            match = self._fuzzy_match(
                candidate, hash_table, trie, ngram_table, threshold
            )
            if match:
                # Calculate confidence score based on match quality and context
                confidence = self._calculate_match_confidence(
                    candidate, match, component_type, hash_table
                )
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = match

        return best_match

    def _calculate_match_confidence(
        self, candidate: str, match: str, component_type: str, hash_table: dict
    ) -> float:
        """Calculate confidence score for a match"""
        base_confidence = 1.0

        # Exact match gets highest confidence
        normalized_candidate = self._normalize_vietnamese_text(candidate)
        if normalized_candidate in hash_table:
            return base_confidence

        # Trie match gets high confidence
        if component_type == "province":
            trie_matches = self.province_trie.search_prefix(
                normalized_candidate, max_results=1
            )
        elif component_type == "district":
            trie_matches = self.district_trie.search_prefix(
                normalized_candidate, max_results=1
            )
        else:
            trie_matches = self.ward_trie.search_prefix(
                normalized_candidate, max_results=1
            )

        if trie_matches and match in trie_matches:
            return base_confidence * 0.9

        # Fuzzy match gets lower confidence based on similarity
        distance = self._levenshtein_distance(
            normalized_candidate, self._normalize_vietnamese_text(match)
        )
        max_len = max(
            len(normalized_candidate), len(self._normalize_vietnamese_text(match))
        )
        if max_len > 0:
            similarity = 1.0 - (distance / max_len)
            return base_confidence * similarity

        return base_confidence * 0.5  # Default confidence for any match


def load_txt_with_encoding(file_path: str) -> List[str]:
    """Load data from TXT file with automatic encoding detection"""
    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1"]

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                data = [line.strip() for line in f if line.strip()]
            return data
        except (UnicodeDecodeError, FileNotFoundError):
            continue

    raise Exception(f"Could not read file {file_path} with any supported encoding")


def load_test_cases(filename: str) -> List[Dict]:
    """Load test cases from JSON file"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filename} not found, using empty list")
        return []


def main():
    """Main function demonstrating the classifier usage with comprehensive testing"""
    print("Loading Vietnamese administrative data...")

    # Load data from TXT files
    provinces = load_txt_with_encoding("data/provinces.txt")
    districts = load_txt_with_encoding("data/districts.txt")
    wards = load_txt_with_encoding("data/wards.txt")

    print(
        f"Loaded {len(provinces)} provinces, {len(districts)} districts, {len(wards)} wards"
    )

    # Initialize classifier
    print("Initializing classifier...")
    classifier = AddressClassifier(provinces, districts, wards)

    # Load test cases from JSON
    print("Loading test cases from test.json...")
    test_cases = load_test_cases("test.json")

    print(f"\nTesting address classification with {len(test_cases)} test cases:")
    print("=" * 80)

    correct_predictions = 0
    total_time = 0
    max_time = 0
    failed_cases = []
    all_test_results = []

    for i, test_case in enumerate(test_cases):
        address = test_case["text"]
        expected = test_case["result"]
        notes = test_case.get("notes", "")

        start_time = time.perf_counter()
        result = classifier.classify_address(address)
        end_time = time.perf_counter()

        processing_time = (end_time - start_time) * 1000
        total_time += processing_time
        max_time = max(max_time, processing_time)

        # Check if prediction matches expected result
        is_correct = (
            result["province"] == expected["province"]
            and result["district"] == expected["district"]
            and result["ward"] == expected["ward"]
        )

        if is_correct:
            correct_predictions += 1

        # Prepare result data
        reasons = []
        if result["province"] != expected["province"]:
            reasons.append("province mismatch")
        if result["district"] != expected["district"]:
            reasons.append("district mismatch")
        if result["ward"] != expected["ward"]:
            reasons.append("ward mismatch")
        if processing_time > 100:
            reasons.append(f"time_exceeded ({processing_time:.2f} ms > 100 ms)")

        fail_reason = "; ".join(reasons) if reasons else ""

        test_result = {
            "index": i + 1,
            "input": address,
            "normalized": classifier._normalize_vietnamese_text(address),
            "notes": notes,
            "expected_province": expected["province"],
            "expected_district": expected["district"],
            "expected_ward": expected["ward"],
            "got_province": result["province"],
            "got_district": result["district"],
            "got_ward": result["ward"],
            "time_ms": round(processing_time, 3),
            "status": "PASS" if is_correct else "FAIL",
            "fail_reason": fail_reason,
        }

        all_test_results.append(test_result)

        if not is_correct:
            failed_cases.append(test_result)

        # Show results for first 10 test cases or if incorrect
        if i < 10 or not is_correct:
            print(f"\nTest {i + 1}: {address}")
            print(f"Expected: {expected}")
            print(f"Predicted: {result}")
            print(f"Correct: {'✅' if is_correct else '❌'}")
            print(f"Processing time: {processing_time:.2f}ms")
            if notes:
                print(f"Notes: {notes}")

        # Check performance requirements
        if processing_time > 100:  # 0.1s = 100ms
            print("⚠️  WARNING: Exceeds maximum time requirement!")
        elif processing_time > 10:  # 0.01s = 10ms
            print("⚠️  WARNING: Exceeds average time requirement!")

    # Summary statistics
    accuracy = (correct_predictions / len(test_cases)) * 100
    avg_time = total_time / len(test_cases)

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"Total test cases: {len(test_cases)}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average processing time: {avg_time:.2f}ms")
    print(f"Maximum processing time: {max_time:.2f}ms")

    print("\n" + "=" * 80)
    print("PERFORMANCE CONSTRAINTS")
    print("=" * 80)
    print(f"Average time ≤ 10ms: {'✓' if avg_time <= 10 else '✗'} ({avg_time:.2f} ms)")
    print(f"Max time ≤ 100ms: {'✓' if max_time <= 100 else '✗'} ({max_time:.2f} ms)")

    if avg_time <= 10 and max_time <= 100:
        print("✅ All performance requirements met!")
    else:
        print("⚠️  Performance requirements not met!")

    # Export test results
    print("\n" + "=" * 80)
    print("EXPORTING TEST RESULTS")
    print("=" * 80)
    try:
        # CSV headers
        csv_headers = [
            "index",
            "input",
            "normalized",
            "notes",
            "expected_province",
            "expected_district",
            "expected_ward",
            "got_province",
            "got_district",
            "got_ward",
            "time_ms",
            "status",
            "fail_reason",
        ]

        # Create output directory if it doesn't exist
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        # Export full test report
        test_report_path = os.path.join(output_dir, "_test_report.csv")
        with open(test_report_path, "w", encoding="utf-8-sig", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=csv_headers)
            writer.writeheader()
            for row in all_test_results:
                writer.writerow(row)
        print(
            f"Wrote test report CSV: {test_report_path} ({len(all_test_results)} rows)"
        )

        # Export failed cases only
        failed_cases_path = os.path.join(output_dir, "_failed_cases.csv")
        with open(failed_cases_path, "w", encoding="utf-8-sig", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=csv_headers)
            writer.writeheader()
            for row in failed_cases:
                writer.writerow(row)
        print(f"Wrote failed cases CSV: {failed_cases_path} ({len(failed_cases)} rows)")

        # Export JSON format for full report
        json_report_path = os.path.join(output_dir, "_test_report.json")
        with open(json_report_path, "w", encoding="utf-8") as jf:
            json.dump(all_test_results, jf, ensure_ascii=False, indent=2)
        print(
            f"Wrote test report JSON: {json_report_path} ({len(all_test_results)} rows)"
        )

    except Exception as e:
        print(f"Failed to export test results: {e}")


if __name__ == "__main__":
    main()
