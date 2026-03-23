# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

# Emojis and text emoticons → sentinel tokens inserted before tokenizing.
# Sentinel tokens survive punctuation-stripping and are checked in score_text.
_EMOJI_MAP: Dict[str, str] = {
    "😊": "emoji_positive",
    "✨": "emoji_positive",
    "🎵": "emoji_positive",
    "😂": "emoji_positive",
    "🔥": "emoji_positive",   # fire = hype / strong positive in slang context
    ":)":  "emoji_positive",
    ":-)": "emoji_positive",
    "💀": "emoji_negative",   # used as "I'm dead 💀" / chaos signal
    "😭": "emoji_negative",
    "🙃": "emoji_negative",   # upside-down = sarcastic / done
    ":(":  "emoji_negative",
    ":-(": "emoji_negative",
}

# Words that flip the sentiment of the next sentiment word.
_NEGATION_WORDS = frozenset({
    "not", "never", "no", "don't", "dont",
    "doesn't", "doesnt", "can't", "cant",
    "won't", "wont", "barely", "hardly",
})


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        Steps:
          1. Replace emojis / text emoticons with sentinel tokens so they
             survive punctuation-stripping and carry sentiment signal.
          2. Lowercase everything.
          3. Split on whitespace.
          4. Strip leading / trailing punctuation from each token so
             "tired," matches "tired" and "love!" matches "love".
        """
        # Step 1 — swap emojis for sentinel tokens before any other change.
        for symbol, sentinel in _EMOJI_MAP.items():
            text = text.replace(symbol, f" {sentinel} ")

        # Step 2 — lowercase.
        text = text.lower()

        # Step 3 — split.
        raw_tokens = text.split()

        # Step 4 — strip surrounding punctuation, but leave sentinel tokens alone.
        tokens: List[str] = []
        for tok in raw_tokens:
            if tok.startswith("emoji_"):
                tokens.append(tok)
            else:
                cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", tok)
                if cleaned:
                    tokens.append(cleaned)

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Base rule:
          - Positive word / emoji_positive → +1
          - Negative word / emoji_negative → -1

        Enhancement — negation handling:
          When a negation word ("not", "never", "can't", …) appears, the
          *next* sentiment word has its delta flipped.  This lets the model
          score "not happy" as -1 instead of +1.
        """
        tokens = self.preprocess(text)
        score = 0
        negated = False
        negation_budget = 0  # how many more tokens the negation can travel

        for token in tokens:
            if token in _NEGATION_WORDS:
                negated = True
                negation_budget = 2  # only flip the next sentiment word within 2 tokens
                continue

            if negated:
                negation_budget -= 1
                if negation_budget < 0:
                    negated = False  # negation expired without hitting a sentiment word

            if token in self.positive_words or token == "emoji_positive":
                delta = 1
            elif token in self.negative_words or token == "emoji_negative":
                delta = -1
            else:
                continue

            if negated:
                delta = -delta
                negated = False

            score += delta

        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        Mapping:
          - Both positive AND negative hits present, |score| <= 1 → "mixed"
          - score > 0  → "positive"
          - score < 0  → "negative"
          - score == 0 → "neutral"

        The "mixed" check fires before the sign check so that a post like
        "tired but hopeful" (score = 0) gets "mixed" rather than "neutral"
        when both signal types fired.
        """
        tokens = self.preprocess(text)
        score = self.score_text(text)

        pos_hits = sum(1 for t in tokens if t in self.positive_words or t == "emoji_positive")
        neg_hits = sum(1 for t in tokens if t in self.negative_words or t == "emoji_negative")

        if pos_hits > 0 and neg_hits > 0 and abs(score) <= 1:
            return "mixed"
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )


# ---------------------------------------------------------------------
# Quick smoke-test — run with: python mood_analyzer.py
# ---------------------------------------------------------------------

if __name__ == "__main__":
    analyzer = MoodAnalyzer()

    test_cases = [
        # text                                          expected
        ("I love this class so much",                  "positive"),
        ("Today was a terrible day",                   "negative"),
        ("Feeling tired but kind of hopeful",          "mixed"),
        ("This is fine",                               "neutral"),
        ("I am not happy about this",                  "negative"),   # negation
        ("lowkey obsessed with this song rn 🎵",       "positive"),   # emoji
        ("I absolutely love when my wifi cuts out 🙃", "mixed"),      # sarcasm / emoji
        ("exhausted but we actually pulled it off",    "mixed"),
    ]

    print(f"{'TEXT':<48} {'TOKENS'}")
    print("-" * 90)
    for text, expected in test_cases:
        tokens = analyzer.preprocess(text)
        score  = analyzer.score_text(text)
        label  = analyzer.predict_label(text)
        match  = "✓" if label == expected else f"✗ (expected {expected})"
        print(f"{text:<48} tokens={tokens}")
        print(f"  score={score:+d}  label={label}  {match}")
        print()
