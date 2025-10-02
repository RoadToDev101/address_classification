import re
from pre_processing import normalize_input, expand_abbreviations


class TrieNode:
    def __init__(self):
        self.children, self.is_word, self.word = {}, False, None


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word, node.word = True, word

    def search_prefixes(self, text, start_pos):
        matches, node = [], self.root
        for i in range(start_pos, len(text)):
            char = text[i]
            if char not in node.children:
                break
            node = node.children[char]
            if node.is_word:
                matches.append((i + 1, node.word))
        return matches


def segment(text: str, trie: Trie) -> str:
    from pre_processing import BLACK_LIST_KEYWORDS

    # Normalize input using existing pre_processing function
    text = normalize_input(text)
    text = expand_abbreviations(text)
    # Split on commas to handle address components separately
    parts = [part.strip() for part in text.split(",")]
    result_parts = []

    for part in parts:
        # NEW: Skip blacklisted parts entirely
        if any(keyword in part.lower() for keyword in BLACK_LIST_KEYWORDS):
            result_parts.append(part)  # Keep as-is
            continue

        clean_part = re.sub(r"[,.]", "", part).strip()
        if not clean_part:
            result_parts.append(part)
            continue

        n = len(clean_part)
        dp = [(-float("inf"), -1, None)] * (n + 1)
        dp[0] = (0, -1, None)

        # Dynamic programming for word segmentation
        for i in range(n):
            if dp[i][0] == -float("inf"):
                continue
            # Try dictionary matches
            for end_pos, word in trie.search_prefixes(clean_part, i):
                score = dp[i][0] + len(word) * 2  # Prefer longer words
                if score > dp[end_pos][0]:
                    dp[end_pos] = (score, i, word)
            # Fallback: single char (with heavy penalty)
            if i + 1 <= n and dp[i + 1][0] < dp[i][0] - 100:
                dp[i + 1] = (dp[i][0] - 100, i, clean_part[i])

        # Reconstruct words
        words, pos = [], n
        while pos > 0:
            _, prev_pos, word = dp[pos]
            if word:
                words.append(word)
            pos = prev_pos
        segmented = " ".join(reversed(words))

        # Reattach punctuation
        result_parts.append(segmented)

    # Join parts with commas
    result = ", ".join(result_parts)
    return result


def segment_text_using_common_vn_words(text: str, trie: Trie) -> str:
    segmented = segment(text, trie)
    # Clean multiple spaces
    segmented = re.sub(r"\s+", " ", segmented).strip()
    return segmented
