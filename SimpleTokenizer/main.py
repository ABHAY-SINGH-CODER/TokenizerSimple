import re
from typing import List, Tuple, Set

def common_prefix(a: str, b: str) -> str:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return a[:i]

def similarity_score(prefix: str, word: str) -> float:
    return len(prefix) / len(word)

def tokenizer(text: str, threshold: float = 0.5) -> List[str]:
    words = re.findall(r'\b\w+\b', text.lower())
    unique_words = list(set(words))

    pairs: List[Tuple[str, str, str, float]] = []

    for w1 in unique_words:
        for w2 in unique_words:
            if len(w2) > len(w1):
                prefix = common_prefix(w1, w2)
                if prefix:
                    score = similarity_score(prefix, w1)
                    pairs.append((w1, w2, prefix, score))

    pairs.sort(key=lambda x: x[3], reverse=True)

    base_tokens: Set[str] = set()
    suffix_tokens: Set[str] = set()
    matched_words: Set[str] = set()
    split_words: Set[str] = set()

    for w1, w2, prefix, score in pairs:
        if score >= threshold:
            if w2 in split_words:
                continue

            base_tokens.add(prefix)
            matched_words.add(w1)
            matched_words.add(w2)
            split_words.add(w2)

            suffix = w2[len(prefix):]
            if suffix:
                suffix_tokens.add(suffix)

    for word in unique_words:
        if word not in matched_words:
            base_tokens.add(word)

    return list(base_tokens) + list(suffix_tokens)


text = input("Enter your text: ")
tokens = tokenizer(text)

print("Tokens:", tokens)