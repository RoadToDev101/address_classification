# classification_layer.py
"""
Vietnamese Address Classification Layer
Handles the final step of mapping pre-processed suggestions to actual administrative units.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
from bk_tree import BKTree


@dataclass
class ClassificationResult:
    """Result of address classification"""

    province: Optional[str] = None
    district: Optional[str] = None
    ward: Optional[str] = None
    confidence_scores: Dict[str, float] = None
    processing_time_ms: float = 0.0

    def __post_init__(self):
        if self.confidence_scores is None:
            self.confidence_scores = {}


class AddressClassifier:
    """Main classifier using 3 BK-Trees for administrative units"""

    def __init__(self, provinces_file: str, districts_file: str, wards_file: str):
        """Initialize classifier with administrative data files"""
        self.province_tree = self._load_bk_tree(provinces_file)
        self.district_tree = self._load_bk_tree(districts_file)
        self.ward_tree = self._load_bk_tree(wards_file)

        # Cache for performance optimization
        self._exact_match_cache: Dict[
            str, Tuple[str, str]
        ] = {}  # query -> (result, type)

        print(f"Loaded {self.province_tree.word_count} provinces")
        print(f"Loaded {self.district_tree.word_count} districts")
        print(f"Loaded {self.ward_tree.word_count} wards")

    def _load_bk_tree(self, filepath: str) -> BKTree:
        """Load administrative units from file into BK-Tree"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                # Remove prefixes and clean names
                names = []
                for line in f:
                    name = line.lower().strip()
                    if name:
                        # Remove common prefixes: Tỉnh, Thành phố, Huyện, Quận, Xã, Phường, etc.
                        cleaned = self._clean_administrative_name(name)
                        names.append(cleaned)

                return BKTree(names)
        except FileNotFoundError:
            print(f"Warning: File {filepath} not found")
            return BKTree([])

    def _clean_administrative_name(self, name: str) -> str:
        """Remove administrative prefixes from names"""
        import re

        # Common prefixes to remove
        prefixes = [
            r"^tỉnh\s+",
            r"^thành phố\s+",
            r"^tp\.?\s*",
            r"^huyện\s+",
            r"^quận\s+",
            r"^thị xã\s+",
            r"^tx\.?\s*",
            r"^xã\s+",
            r"^phường\s+",
            r"^thị trấn\s+",
            r"^tt\.?\s*",
        ]

        cleaned = name.lower().strip()
        for prefix_pattern in prefixes:
            cleaned = re.sub(prefix_pattern, "", cleaned, flags=re.IGNORECASE).strip()

        return cleaned

    def _calculate_match_score(
        self,
        query: str,
        candidate: str,
        edit_distance: int,
        expected_type: str,
        found_type: str,
    ) -> float:
        """Calculate comprehensive matching score"""
        if not query or not candidate:
            return 0.0

        # Base score from edit distance (higher is better)
        max_len = max(len(query), len(candidate))
        distance_score = 1.0 - (edit_distance / max_len) if max_len > 0 else 0.0

        # Length similarity bonus
        length_similarity = min(len(query), len(candidate)) / max(
            len(query), len(candidate)
        )

        # Type matching bonus/penalty
        type_bonus = 1.0 if expected_type == found_type else 0.7

        # Prefix similarity bonus
        prefix_len = min(3, min(len(query), len(candidate)))
        prefix_match = (
            sum(1 for i in range(prefix_len) if query[i] == candidate[i]) / prefix_len
        )

        # Exact match bonus
        exact_bonus = 1.2 if edit_distance == 0 else 1.0

        # Combine scores
        final_score = (
            distance_score * 0.4
            + length_similarity * 0.2
            + prefix_match * 0.2
            + type_bonus * 0.2
        ) * exact_bonus

        return min(final_score, 1.0)

    def _search_in_tree(
        self, query: str, tree: BKTree, admin_type: str, max_distance: int = 2
    ) -> List[Tuple[str, float]]:
        """Search for matches in a specific BK-Tree with scoring"""
        if not query or not query.strip():
            return []

        query_clean = query.lower().strip()

        # Check exact match cache first
        cache_key = f"{query_clean}_{admin_type}"
        if cache_key in self._exact_match_cache:
            cached_result, cached_type = self._exact_match_cache[cache_key]
            if cached_type == admin_type:
                return [(cached_result, 1.0)]

        # Search in BK-Tree
        matches = tree.search(query_clean, max_distance)

        if not matches:
            return []

        # Score and rank matches
        scored_matches = []
        for candidate, distance in matches:
            score = self._calculate_match_score(
                query_clean, candidate.lower(), distance, admin_type, admin_type
            )
            if score > 0.3:  # Minimum threshold
                scored_matches.append((candidate, score))

        # Sort by score (descending)
        scored_matches.sort(key=lambda x: x[1], reverse=True)

        # Cache the best exact match
        if scored_matches and scored_matches[0][1] >= 0.9:
            self._exact_match_cache[cache_key] = (scored_matches[0][0], admin_type)

        return scored_matches

    def _find_best_match_across_types(
        self, query: str, expected_type: str
    ) -> Tuple[Optional[str], str, float]:
        """Find best match across all administrative types"""
        all_matches = []

        # Search in expected type first (with bonus)
        expected_tree = getattr(self, f"{expected_type}_tree")
        matches = self._search_in_tree(query, expected_tree, expected_type)
        for candidate, score in matches:
            # Bonus for correct type
            adjusted_score = score * 1.1
            all_matches.append((candidate, expected_type, adjusted_score))

        # If no good matches in expected type, search other types
        if not all_matches or all_matches[0][2] < 0.8:
            other_types = {"province", "district", "ward"} - {expected_type}

            for other_type in other_types:
                other_tree = getattr(self, f"{other_type}_tree")
                matches = self._search_in_tree(query, other_tree, other_type)
                for candidate, score in matches[:3]:  # Limit to top 3 per type
                    # Penalty for wrong type
                    adjusted_score = score * 0.8
                    all_matches.append((candidate, other_type, adjusted_score))

        if not all_matches:
            return None, expected_type, 0.0

        # Sort by adjusted score
        all_matches.sort(key=lambda x: x[2], reverse=True)
        best_match, best_type, best_score = all_matches[0]

        return best_match, best_type, best_score

    def classify_address(
        self, suggestions: Dict[str, List[str]]
    ) -> ClassificationResult:
        """
        Main classification method that maps suggestions to actual administrative units

        Args:
            suggestions: Output from suggest_address_components() containing:
                        {'province': [...], 'district': [...], 'ward': [...]}

        Returns:
            ClassificationResult with matched administrative units
        """
        start_time = time.perf_counter()

        result = ClassificationResult()
        confidence_scores = {}

        # Process each type of administrative unit
        for admin_type in ["province", "district", "ward"]:
            suggestions_list = suggestions.get(admin_type, [])

            best_match = None
            best_score = 0.0
            best_actual_type = admin_type

            # Try each suggestion for this type
            for suggestion in suggestions_list:
                if not suggestion or not suggestion.strip():
                    continue

                match, actual_type, score = self._find_best_match_across_types(
                    suggestion, admin_type
                )

                if score > best_score:
                    best_match = match
                    best_score = score
                    best_actual_type = actual_type

            # Store result if confidence is sufficient
            if best_match and best_score >= 0.5:  # Minimum confidence threshold
                setattr(result, admin_type, best_match)
                confidence_scores[admin_type] = best_score

                # Log type mismatches for debugging
                if best_actual_type != admin_type:
                    print(
                        f"Type mismatch: {admin_type} suggestion '{suggestions_list[0] if suggestions_list else 'N/A'}' "
                        f"matched in {best_actual_type} with score {best_score:.3f}"
                    )

        # Calculate processing time
        end_time = time.perf_counter()
        result.processing_time_ms = (end_time - start_time) * 1000
        result.confidence_scores = confidence_scores

        return result

    def clear_cache(self):
        """Clear the exact match cache to free memory"""
        self._exact_match_cache.clear()
