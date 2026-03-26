import re

def common_prefix(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return a[:i]

def similarity_score(prefix, word):
    return len(prefix) / len(word)

def tokenizer(text, threshold=0.5):
    words = re.findall(r'\b\w+\b', text.lower())
    unique_words = list(set(words))

    pairs = []

    for w1 in unique_words:
        for w2 in unique_words:
            if len(w2) > len(w1):
                prefix = common_prefix(w1, w2)
                if prefix:
                    score = similarity_score(prefix, w1)
                    pairs.append((w1, w2, prefix, score))

    pairs.sort(key=lambda x: x[3], reverse=True)

    top_k = max(1, int(len(pairs) * 0.4))
    best_pairs = pairs[:top_k]

    base_tokens = set()
    suffix_tokens = set()
    matched_words = set()

    for w1, w2, prefix, score in best_pairs:
        if score >= threshold:
            base_tokens.add(prefix)
            matched_words.add(w1)
            matched_words.add(w2)

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