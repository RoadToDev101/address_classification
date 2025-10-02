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


def segment(text, dictionary):
    trie = Trie()
    for word in dictionary:
        trie.insert(word.lower())
    n, dp = len(text), [(-float("inf"), -1, None)] * (len(text) + 1)
    dp[0] = (0, -1, None)
    for i in range(n):
        if dp[i][0] == -float("inf"):
            continue
        for end_pos, word in trie.search_prefixes(text, i):
            score = dp[i][0] + len(word) * 2  # Prefer longer words
            if score > dp[end_pos][0]:
                dp[end_pos] = (score, i, word)
        # fallback: treat char as word (with penalty)
        if dp[i + 1][0] < dp[i][0] - 10:
            dp[i + 1] = (dp[i][0] - 10, i, text[i])
    words, pos = [], n
    while pos > 0:
        _, prev_pos, word = dp[pos]
        if word:
            words.append(word)
        pos = prev_pos
    return " ".join(reversed(words))
