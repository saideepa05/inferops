"""Deterministic prompt datasets with controlled input lengths.

Input length drives prefill cost (TTFT), so benchmarks must say which mix they
used. Approximate sizes (Qwen tokenizer, ~1.3 tokens/word):

    short   ~20 tokens     chat questions
    medium  ~350 tokens    summarize a short document
    long    ~2000 tokens   answer a question about a long document
    mixed   60% short / 30% medium / 10% long, closer to real traffic
"""

import random

SHORT_QUESTIONS = [
    "What is the capital of Australia?",
    "Explain recursion in one sentence.",
    "Give me three tips for writing clean Python code.",
    "What is the difference between a list and a tuple?",
    "Why is the sky blue?",
    "Write a haiku about databases.",
    "What does HTTP status code 429 mean?",
    "How do vaccines train the immune system?",
    "Name two sorting algorithms and their time complexity.",
    "What is a GPU used for in machine learning?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "What is the difference between latency and throughput?",
    "How does compound interest work?",
    "Suggest a name for a coffee shop run by robots.",
    "What is a hash map?",
    "Explain what an API is to a ten-year-old.",
    "What causes the seasons on Earth?",
    "Give an example of a SQL JOIN.",
    "What is overfitting in machine learning?",
    "How do I reverse a string in Python?",
]

_SUBJECTS = ["The warehouse team", "Our regional office", "The finance department", "The support desk",
             "The platform group", "The logistics partner", "The quality team", "The data engineering team"]
_VERBS = ["reported", "reviewed", "reduced", "migrated", "audited", "delayed", "expanded", "automated"]
_OBJECTS = ["the quarterly shipping backlog", "customer refund requests", "the nightly batch jobs",
            "the vendor onboarding process", "inventory reconciliation", "the incident response runbook",
            "on-call escalations", "the legacy billing service"]
_DETAILS = ["after the March outage", "to meet the new compliance rules", "ahead of the holiday peak",
            "because error rates kept climbing", "with help from two contractors", "under a tight budget",
            "following customer complaints", "as part of the cost-saving initiative"]


def _document(n_words: int, rng: random.Random) -> str:
    sentences, words = [], 0
    while words < n_words:
        s = f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)} {rng.choice(_DETAILS)}."
        sentences.append(s)
        words += len(s.split())
    return " ".join(sentences)


def _short(rng: random.Random) -> list[dict]:
    return [{"role": "user", "content": rng.choice(SHORT_QUESTIONS)}]


def _medium(rng: random.Random) -> list[dict]:
    doc = _document(250, rng)
    return [{"role": "user", "content": f"{doc}\n\nSummarize the document above in three bullet points."}]


def _long(rng: random.Random) -> list[dict]:
    doc = _document(1500, rng)
    return [{"role": "user", "content": f"{doc}\n\nWhich teams are mentioned most often above, and what did they work on?"}]


def build_dataset(kind: str, n: int = 50, seed: int = 0) -> list[list[dict]]:
    rng = random.Random(seed)
    makers = {"short": _short, "medium": _medium, "long": _long}
    if kind in makers:
        return [makers[kind](rng) for _ in range(n)]
    if kind == "mixed":
        return [rng.choices([_short, _medium, _long], weights=[6, 3, 1])[0](rng) for _ in range(n)]
    raise ValueError(f"Unknown dataset {kind!r}; choose short, medium, long or mixed")
