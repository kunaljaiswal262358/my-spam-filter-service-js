# Spam Classification Service: Report

## Task 1: Data & Lightweight EDA

### 1.1 Statistics

- **Total Messages:** 5572
- **Class Balance:**
  - Ham: 4825 (86.59%)
  - Spam: 747 (13.41%)
- **Average Message Length:** 80.19 characters

### 1.2 Text Preprocessing Plan

Our preprocessing pipeline is designed to be fast, simple, and effective for classic ML models. It involves the following steps, applied in order, before tokenization:

1.  **Case Normalization:** Convert all text to lowercase (`"WINNER!"` -> `"winner!"`). This is crucial for reducing the feature vocabulary (e.g., "free" and "Free" are treated as the same token).
2.  **URL Removal:** Replace all URLs (e.g., `http://...`, `www...com`) with a generic `<URL>` token. URLs are a strong signal but their specific content is noise.
3.  **Number/Digit Removal:** Remove all standalone numbers and digits. We will keep words that _contain_ numbers (e.g., `3g`) as they can be significant, but will remove `1000` or `123`.
4.  **Punctuation Removal:** Remove all punctuation (e.g., `!`, `?`, `.`, `,`, `£`, `$`) and replace them with a space to ensure words are separated.
5.  **Tokenization:** Split the cleaned text into individual words (tokens) based on whitespace.
6.  **Stop Word Policy:** We will use a _minimal_ stop-word list. We will **keep** common words like "to", "you", "a", "your" because their high frequency in _spam_ (e.g., "call you", "your prize") makes them informative for a Naive Bayes classifier. We will only remove extremely common English articles/prepositions like "the", "is", "at".
