# Model Card: Mood Machine

This model card covers both versions of the Mood Machine mood classifier:

1. A **rule-based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit-learn

---

## 1. Model Overview

**Model type:** Both models were built and compared.

**Intended purpose:** Classify short text messages (social media style posts) into one of four mood labels: `positive`, `negative`, `neutral`, or `mixed`.

**How it works (brief):**

- *Rule-based:* Each post is tokenized and scanned for words from a curated positive/negative word list. A numeric score is computed (+1 per positive hit, −1 per negative hit), with negation handling to flip scores after words like "not" or "can't". Emojis are mapped to sentiment tokens before tokenizing. The final score is mapped to a label.
- *ML model:* Posts are converted to bag-of-words vectors using `CountVectorizer`, then a classifier (e.g. Naive Bayes or Logistic Regression) is trained on the `SAMPLE_POSTS` / `TRUE_LABELS` pairs. The model learns which word combinations correlate with each label from the training data directly.

---

## 2. Data

**Dataset description:** The dataset contains 14 posts in `SAMPLE_POSTS`. The original 6 came with the starter code. 8 new posts were added, covering slang, emojis, sarcasm, and ambiguous language.

**Labeling process:** Labels were assigned by reading each post and asking: what mood would a human infer? Posts with words pointing in opposite directions (e.g. "tired but hopeful") were labeled `mixed`. Posts that were vague or expressionless (e.g. "This is fine") were labeled `neutral`. A few posts were genuinely hard to label:

- `"bro what even was today 💀"` — labeled `neutral` (bewildered, not clearly negative), but the 💀 emoji points negative. A classmate might reasonably disagree.
- `"can't tell if I'm happy or just caffeinated"` — labeled `mixed` because both positive and uncertain signals are present, but the uncertainty could also read as neutral.

**Important characteristics:**
- Contains slang: `"lowkey"`, `"no cap"`, `"ngl"`, `"hit different"`, `"it's giving"`
- Contains emojis: 🎵 ✨ 💀 😭 🙃 🔥
- Includes sarcasm: `"I absolutely love when my wifi cuts out mid-presentation 🙃"`
- Several posts express mixed feelings
- All posts are short (1–2 sentences), social-media-style

**Possible issues:**
- 14 examples is very small — results are not statistically meaningful.
- Label distribution is uneven: more `positive` and `negative` than `neutral` or `mixed`.
- All text is informal English with Western internet slang. The model would not generalize to formal writing, other languages, or cultural dialects not represented here.

---

## 3. How the Rule-Based Model Works

**Scoring rules:**

1. **Preprocessing:** Emojis and text emoticons are swapped for sentinel tokens (`emoji_positive`, `emoji_negative`) before tokenizing. Text is lowercased, then split on whitespace. Leading/trailing punctuation is stripped from each token using regex, so `"tired,"` matches `"tired"`.

2. **Scoring loop:** Tokens are scanned left to right. Each token matched in `POSITIVE_WORDS` or equal to `emoji_positive` adds +1. Each token in `NEGATIVE_WORDS` or equal to `emoji_negative` subtracts −1.

3. **Negation handling:** If a negation word (`not`, `never`, `can't`, `won't`, etc.) is seen, a budget of 2 is set. The next sentiment word within that window has its delta flipped. The budget expires after 2 neutral tokens, preventing negation from traveling too far.

4. **Label thresholds:**
   - Both positive and negative hits present, `|score| ≤ 1` → `"mixed"`
   - `score > 0` → `"positive"`
   - `score < 0` → `"negative"`
   - `score == 0` → `"neutral"`

**Strengths:**
- Transparent and inspectable — you can trace exactly which tokens affected the score.
- Handles negation: `"not happy"` correctly scores as negative.
- Emoji signals work: `🔥` and `✨` boost scores; `💀` and `🙃` reduce them.
- No training data needed — runs immediately.

**Weaknesses:**
- Cannot detect sarcasm. `"I absolutely love getting stuck in traffic"` → `positive` because `love` fires with no counterweight.
- Slang outside the vocabulary is silently ignored. `"it's giving chaos"` only scores if `chaos` is in the word list.
- The `mixed` label requires both positive and negative word-list hits in the same post. Posts that express complexity through tone alone (e.g. `"this week has been A LOT"`) score as neutral because no listed words matched.

---

## 4. How the ML Model Works

**Features used:** Bag-of-words representation via `CountVectorizer`. Each post becomes a vector of word counts. No emoji or negation preprocessing is applied at this stage.

**Training data:** Trained directly on `SAMPLE_POSTS` and `TRUE_LABELS` from `dataset.py`.

**Training behavior:** Because the model is trained and evaluated on the same 14 examples, it achieves 100% accuracy on the training set. This is overfitting — not a sign of good generalization. Adding more examples or holding out a test set would give a more honest picture.

**Strengths:**
- Learns patterns automatically without hand-coded rules.
- Correctly classified the sarcasm case (`"I absolutely love when my wifi cuts out 🙃"` → `negative`) that the rule-based model missed, because it saw the `🙃` sentinel token co-occurring with a `negative` label.
- Also correctly predicted `mixed` for posts the rule-based model got wrong (e.g. `"exhausted but we actually pulled it off"`), likely because those exact strings were in the training set.

**Weaknesses:**
- 100% training accuracy almost certainly means the model memorized the data rather than learned a generalizable pattern.
- It would fail on any new post with words it has never seen.
- Small dataset means the model is highly sensitive to individual label choices — relabeling one post could shift accuracy significantly.

---

## 5. Evaluation

**Accuracy observed:**
- Rule-based model: **64%** (9/14 correct)
- ML model: **100%** — but this is training accuracy on the same data used to train it, so it is not a fair comparison.

**Examples of correct predictions (rule-based):**

| Post | Label | Why it worked |
|---|---|---|
| `"I am not happy about this"` | negative | Negation handling flipped `happy` → −1 |
| `"Feeling tired but kind of hopeful"` | mixed | Both `tired` (negative) and `hopeful` (positive) hit, score = 0, mixed detected |
| `"not bad at all"` | positive | `not` negated `bad` → +1 |

**Examples of incorrect predictions (rule-based):**

| Post | Predicted | True | Why it failed |
|---|---|---|---|
| `"I absolutely love when my wifi cuts out 🙃"` | mixed | negative | `love` (+1) and `emoji_negative` (−1) cancel to 0 — model can't detect sarcasm, calls it mixed |
| `"this week has been A LOT but somehow I'm still here lol"` | neutral | mixed | No vocabulary matches at all; tone words like "A LOT" aren't in the word lists |
| `"can't tell if I'm happy or just caffeinated"` | positive | mixed | Negation budget expired before reaching `happy`, so `happy` scored +1 without flip |

---

## 6. Limitations

- **Tiny dataset.** 14 labeled examples is not enough to train or meaningfully evaluate any model. Patterns observed here may not hold on new data.
- **No train/test split.** The ML model's 100% accuracy is training accuracy — not a measure of real performance.
- **Sarcasm is unsolvable with these rules.** Any sentence containing a positive word for ironic effect will be misclassified. Example: `"Oh great, another Monday"` → positive.
- **Vocabulary coverage is the rule-based model's ceiling.** Posts that use emotional language outside the word lists (e.g. `"A LOT"`, `"lol"`, `"no cap"` without a sentiment word) return neutral regardless of actual tone.
- **Negation window is approximate.** A window of 2 neutral tokens is an engineering estimate — it works on most cases here but is not linguistically principled.
- **Short-post assumption.** Both models were designed for 1–2 sentence social posts. Longer text with topic shifts would require different logic.

---

## 7. Ethical Considerations

- **Misclassifying distress.** A post like `"I'm fine 😭"` scores negative (correctly here), but rule-based models can easily flip this on small wording changes. In a welfare-monitoring context, missing a cry for help is a serious failure mode.
- **Language bias.** The dataset is entirely informal Western English internet slang. The model has no ability to interpret mood in formal writing, non-English text, or slang from communities not represented (e.g. AAVE, regional dialects). Deploying it as a general-purpose sentiment tool would produce systematically worse results for those groups.
- **Label subjectivity.** Several labels in this dataset were edge cases where reasonable people would disagree. Any downstream conclusions based on model predictions inherit that subjectivity.
- **Privacy.** Mood analysis applied to real personal messages without consent is a privacy concern, even when the model is simple.

---

## 8. Ideas for Improvement

- **Add more labeled data** — at minimum 50–100 diverse examples before drawing any conclusions.
- **Add a real test set** — hold out 20% of labeled posts before training; never evaluate on training data.
- **Use TF-IDF** instead of raw counts to down-weight common words (`"I"`, `"was"`) and up-weight distinctive ones.
- **Better slang coverage** — maintain a slang lexicon (e.g. `sick`, `fire`, `lowkey`, `bussin`) separate from the general word lists, so context-dependent words can be handled more carefully.
- **Sarcasm signals** — patterns like `"I love [negative situation]"` or `"great, just great"` could be detected with phrase-level rules rather than single-word matching.
- **Use a small transformer model** (e.g. a fine-tuned DistilBERT) once the dataset is large enough — these handle context, negation, and tone far better than bag-of-words approaches.
