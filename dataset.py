"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    "hopeful",
    "proud",
    "peace",
    "obsessed",
    "sick",    # slang: "that's sick" = impressive/great (tradeoff: "I feel sick" would score positive)
    "fire",    # slang: "that's fire" = excellent
    "wicked",  # slang: "wicked fun" = very / intensely good
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    "exhausted",
    "chaos",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    # --- new posts ---
    "lowkey obsessed with this song rn 🎵",
    "woke up late, missed the bus, spilled coffee... it's giving chaos 💀",
    "I absolutely love when my wifi cuts out mid-presentation 🙃",
    "ngl today hit different, feeling really at peace ✨",
    "exhausted but we actually pulled it off no cap",
    "this week has been A LOT but somehow I'm still here lol",
    "can't tell if I'm happy or just caffeinated",
    "bro what even was today 💀",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    # --- new labels ---
    "positive",  # "lowkey obsessed with this song rn 🎵"
    "negative",  # "woke up late, missed the bus, spilled coffee... it's giving chaos 💀"
    "negative",  # "I absolutely love when my wifi cuts out mid-presentation 🙃" (sarcasm)
    "positive",  # "ngl today hit different, feeling really at peace ✨"
    "mixed",     # "exhausted but we actually pulled it off no cap"
    "mixed",     # "this week has been A LOT but somehow I'm still here lol"
    "mixed",     # "can't tell if I'm happy or just caffeinated"
    "neutral",   # "bro what even was today 💀" (vague/ambiguous — edge case)
]

# TODO: Add 5-10 more posts and labels.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
