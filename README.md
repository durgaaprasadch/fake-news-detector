# AI Fake News Detector 📰

A Streamlit-based **dual-engine** app that helps you evaluate misinformation in two ways:

1. **Analyze Content (ML Patterns):** Uses TF‑IDF + Logistic Regression to predict whether an article *looks like* fake news based on language patterns.
2. **Live Fact Check (Web):** Takes a short claim and tries to verify it using **recent web sources** (via web search + LLM reasoning).

> **Disclaimer:** This project is experimental. Results are not guaranteed and should not be used for critical decisions.

---

## Demo (UI Screens)

### 1) Live Fact Check — Debunked example

When a claim is not supported by sources, the app returns **DEBUNKED (FALSE)** and shows the reasoning + the sources it checked.

![Live fact check – debunked](assets/demo-live-debunked.png)

### 2) Live Fact Check — Verified example

When a claim is supported by sources, the app returns **VERIFIED (TRUE)** and shows the reasoning + sources.

![Live fact check – verified](assets/demo-live-verified.png)

### 3) Live Fact Check — Another debunked example

A second example of a claim that the app could not verify.

![Live fact check – debunked (example 2)](assets/demo-live-debunked-2.png)

### 4) Analyze Content (ML Patterns)

Paste a news article (or a snippet) and the model predicts **FAKE / REAL-like** with a confidence score.

![Analyze content – fake prediction](assets/demo-analyze-content.png)

---

## How it works (high-level)

### A) Analyze Content (ML Patterns)

1. Input text is cleaned and vectorized using **TF‑IDF**.
2. A **Logistic Regression** model predicts the label.
3. The UI shows the prediction and confidence.

### B) Live Fact Check (Web)

1. You enter a short factual claim.
2. The app searches the web for recent context.
3. An LLM (Google Gemini) summarizes the evidence and returns **VERIFIED / DEBUNKED**.
4. Sources used for the decision are displayed in the UI.

---

## Getting started (run locally)

### 1) Clone

```bash
git clone https://github.com/durgaaprasadch/fake-news-detector.git
cd fake-news-detector
```

### 2) Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run the app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

---

## Project structure

- `app.py` — Streamlit UI + app logic
- `model/` — saved ML artifacts
- `data/` — dataset / training data
- `utils/` — helper utilities

---

## Limitations

- **ML model is basic:** TF‑IDF + Logistic Regression is a strong baseline, but it’s not a deep semantic fact checker.
- **Depends on data & sources:** ML predictions depend on training data; live checks depend on search results.
- **Sarcasm/satire:** Satirical or sarcastic text may be misclassified.

---

## Roadmap / Improvements

- Add a transformer model (e.g., BERT) for deeper language understanding.
- Improve dataset coverage with more recent articles.
- Better claim decomposition + source ranking for fact checking.

---

*Built by Durga Prasad | v1 – experimental*