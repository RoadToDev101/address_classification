from typing import Optional, Tuple, List


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalize_for_comparison(text: str) -> str:
    """Normalize text for fuzzy matching by converting to lowercase"""
    return text.lower().strip()


class BKTreeNode:
    """Node for BK-Tree (Burkhard-Keller Tree) structure"""

    def __init__(self, word: str = None):
        self.word = word
        self.normalized_word = normalize_for_comparison(word) if word else None
        self.children = {}  # distance -> BKTreeNode (not a list)
        self.is_terminal = word is not None

    def add_word(self, word: str):
        """Add a word to the BK-Tree"""
        if self.word is None:
            # This is root node, make it the first word
            self.word = word
            self.normalized_word = normalize_for_comparison(word)
            self.is_terminal = True
            return

        # Calculate distance from current node's word
        distance = levenshtein_distance(
            self.normalized_word, normalize_for_comparison(word)
        )

        # Word already exists
        if distance == 0:
            return

        # If we already have a child at this distance, recurse into it
        if distance in self.children:
            self.children[distance].add_word(word)
        else:
            # Create a new node at this distance
            self.children[distance] = BKTreeNode(word)

    def search(self, query: str, max_distance: int = 2) -> List[Tuple[str, int]]:
        """Search for words within max_distance of query"""
        results = []
        normalized_query = normalize_for_comparison(query)

        if self.word is not None:
            distance = levenshtein_distance(self.normalized_word, normalized_query)
            if distance <= max_distance:
                results.append((self.word, distance))

        # Search children within distance range
        for child_distance, child_node in self.children.items():
            # Triangle inequality: |d(query, node) - d(node, child)| <= d(query, child) <= d(query, node) + d(node, child)
            if abs(distance - child_distance) <= max_distance:
                results.extend(child_node.search(query, max_distance))

        return results


class BKTree:
    """BK-Tree for efficient fuzzy string matching"""

    def __init__(self, words: List[str] = None):
        self.root = BKTreeNode()
        self.word_count = 0

        if words:
            for word in words:
                self.add_word(word)

    def add_word(self, word: str):
        """Add a word to the trie"""
        if word and word.strip():
            self.root.add_word(word.strip())
            self.word_count += 1

    def search(self, query: str, max_distance: int = 2) -> List[Tuple[str, int]]:
        """Search for words within max_distance of query"""
        if not query or not query.strip():
            return []

        results = self.root.search(query.strip(), max_distance)
        # Sort by distance (closest matches first)
        return sorted(results, key=lambda x: x[1])

    def get_best_match(
        self, query: str, max_distance: int = 2
    ) -> Optional[Tuple[str, int]]:
        """Get the best (closest) match for query"""
        results = self.search(query, max_distance)
        return results[0] if results else None

    def get_exact_match(self, query: str) -> Optional[str]:
        """Get exact match if exists"""
        results = self.search(query, max_distance=0)
        return results[0][0] if results else None
