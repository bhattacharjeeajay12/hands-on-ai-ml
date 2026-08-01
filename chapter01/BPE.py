import re
import string
from collections import defaultdict

import requests


def fetch_wikipedia_text(title: str, lang: str = "en") -> str:
    """
    Download the plain-text extract of a Wikipedia article.

    Example: fetch_wikipedia_text("Byte-pair encoding") -> the article's
    plain text (infoboxes/markup stripped, references left as bracketed
    numbers in the API's plaintext extract).
    """
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    pages = response.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if "extract" not in page:
        raise ValueError(f"No article found for title: {title!r}")
    return page["extract"]


def text_to_word_freqs(text: str) -> dict[str, int]:
    """
    Turn raw article text into a word -> frequency dict, ready to hand
    to BPE.train(). Lowercases and keeps only alphabetic words (drops
    numbers/punctuation), same shape as the toy corpora used earlier:
        {"low": 5, "lower": 2, ...}
    """
    words = re.findall(r"[a-zA-Z']+", text.lower())
    word_freqs: dict[str, int] = {}
    for word in words:
        word_freqs[word] = word_freqs.get(word, 0) + 1
    return word_freqs

class BPE:
    def __init__(self):
        self.merges = []    # it contains (symbol_a, symbol_b), in learned order
        self.vocab = set()  # all symbols: base chars + merged symbols

    def word_to_symbols(self, word: str) -> tuple[str, ...]:
        token_tuple = tuple(word) + ("</w>",)
        return token_tuple

    def get_pair_stats(self, word_freqs) -> dict[tuple[str,str], int]:
        """
        :param word_freqs: looks like this:
        {
          ("l","o","w","</w>"): 5,
          ("l","o","w","e","r","</w>"): 2,
          ("n","e","w","e","s","t","</w>"): 6,
          ("w","i","d","e","s","t","</w>"): 3,
        }
        :return:
        {
          ("l","o"): 7, ("o","w"): 7, ("w","</w>"): 5,
          ("w","e"): 8, ("e","r"): 2, ("r","</w>"): 2,
          ("n","e"): 6, ("e","w"): 6, ("e","s"): 9,
          ("s","t"): 9, ("t","</w>"): 9,
          ("w","i"): 3, ("i","d"): 3, ("d","e"): 3,
        }
        """
        pair_counts = defaultdict(int)
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += freq
        return dict(pair_counts)

    def merge_pair_in_single_word(self, pair: tuple[str, str], word: tuple[str, ...]) -> tuple[str, ...]:
        # Input: pair = ("e", "s"), word = ("n", "e", "w", "e", "s", "t", "</w>")
        # Output: ("n", "e", "w", "es", "t", "</w>")
        first, second = pair
        new_word = []
        i = 0

        while i < len(word):
            # Check if current and next tokens match the target pair
            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                new_word.append(first + second)
                i += 2  # Skip both merged elements
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def merge_pair_in_words(self, pair: tuple[str, str], word_freqs: dict) -> dict:
        """
            pair = ("e", "s")
            word_freqs = {
                ("l", "o", "w", "</w>"): 5,
                ("l", "o", "w", "e", "r", "</w>"): 2,
                ("n", "e", "w", "e", "s", "t", "</w>"): 6,
                ("w", "i", "d", "e", "s", "t", "</w>"): 3,
            }

            output = {
              ("l","o","w","</w>"): 5,
              ("l","o","w","e","r","</w>"): 2,
              ("n","e","w","es","t","</w>"): 6,
              ("w","i","d","es","t","</w>"): 3,
            }
        """
        merged_word_freqs = {}
        for word, freq in word_freqs.items():
            # Call the single-word method we fixed earlier
            new_word = self.merge_pair_in_single_word(pair, word)
            merged_word_freqs[new_word] = freq
        return merged_word_freqs

    def train(self, corpus, num_merges):
        """
        corpus: either
            - list[str]: raw words (frequencies will be counted here), or
            - dict[str, int]: word -> frequency, already counted
        num_merges: max number of merge operations to learn
        """
        # normalize corpus into a word -> frequency dict
        if isinstance(corpus, dict):
            raw_word_freqs = corpus
        else:
            raw_word_freqs = {}
            for word in corpus:
                raw_word_freqs[word] = raw_word_freqs.get(word, 0) + 1

        # convert each word into its starting symbols (chars + </w>)
        word_freqs = {}
        for word, freq in raw_word_freqs.items():
            symbols = self.word_to_symbols(word)
            word_freqs[symbols] = word_freqs.get(symbols, 0) + freq

        # seed vocab with every symbol currently in use
        self.vocab = set()
        for symbols in word_freqs:
            self.vocab.update(symbols)

        self.merges = []

        for _ in range(num_merges):
            pair_stats = self.get_pair_stats(word_freqs)

            if not pair_stats:
                break  # nothing left to merge, every word is a single symbol

            # max() returns the first key with the highest value when there's
            # a tie, since dict iteration follows insertion order in Python —
            # this is exactly the "first pair encountered" tie-break rule
            best_pair = max(pair_stats, key=pair_stats.get)

            word_freqs = self.merge_pair_in_words(best_pair, word_freqs)

            self.merges.append(best_pair)
            self.vocab.add(best_pair[0] + best_pair[1])

        return self.merges, self.vocab

    def apply(self, word: str) -> list[str]:
        """
        Tokenize a (possibly unseen) word using the merges learned during
        training. Unlike train(), this does NOT recompute frequencies —
        there's no corpus at encode time, just this one word. It simply
        replays self.merges in the exact order they were learned.

        Example (after training on low/lower/newest/widest with 5 merges,
        i.e. merges = [("e","s"), ("es","t"), ("est","</w>"), ("l","o"), ("lo","w")]):
            apply("lowest") -> ["low", "est</w>"]
        """
        symbols = self.word_to_symbols(word)
        for pair in self.merges:
            symbols = self.merge_pair_in_single_word(pair, symbols)
        return list(symbols)


if __name__ == "__main__":
    bpe = BPE()

    test_word = "low"
    symbols = bpe.word_to_symbols(test_word)
    # symbols: ('l', 'o', 'w', '</w>')

    word_freqs={
        ("l", "o", "w", "</w>"): 5,
        ("l", "o", "w", "e", "r", "</w>"): 2,
        ("n", "e", "w", "e", "s", "t", "</w>"): 6,
        ("w", "i", "d", "e", "s", "t", "</w>"): 3,
    }
    pair_stats = bpe.get_pair_stats(word_freqs)
    m_pair = bpe.merge_pair_in_single_word(("e", "s"),("n", "e", "w", "e", "s", "t", "</w>"))
    word_freqs = bpe.merge_pair_in_words(("e", "s"), word_freqs)
    chk = 1

    # --- train + apply check ---
    corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
    merges, vocab = bpe.train(corpus, num_merges=5)
    print("merges:", merges)
    # expected: [("e","s"), ("es","t"), ("est","</w>"), ("l","o"), ("lo","w")]

    tokens = bpe.apply("lowest")
    print("apply('lowest'):", tokens)
    # expected: ["low", "est</w>"]

    # --- train BPE on a real Wikipedia page ---
    page_title = "Byte-pair encoding"
    print(f"\nFetching Wikipedia page: {page_title!r} ...")
    wiki_text = fetch_wikipedia_text(page_title)
    print(f"Fetched {len(wiki_text)} characters")

    wiki_word_freqs = text_to_word_freqs(wiki_text)
    print(f"Unique words: {len(wiki_word_freqs)}")

    wiki_bpe = BPE()
    wiki_merges, wiki_vocab = wiki_bpe.train(wiki_word_freqs, num_merges=200)
    print(f"Learned {len(wiki_merges)} merges -> vocab size {len(wiki_vocab)}")
    print("First 10 merges:", wiki_merges[:10])

    for sample_word in ["tokenization", "encoding", "algorithm"]:
        print(f"apply({sample_word!r}):", wiki_bpe.apply(sample_word))

