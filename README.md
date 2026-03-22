# AI Fake News Detector 📰

A Streamlit-based **dual-engine** app that helps you evaluate misinformation in two ways:

1. **Analyze Content (ML Patterns):** Uses TF‑IDF + Logistic Regression to predict whether an article *looks like* fake news based on language patterns.
2. **Live Fact Check (Web):** Takes a short claim and tries to verify it using **recent web sources** (via web search + LLM reasoning).

> **Disclaimer:** This project is experimental. Results are not guaranteed and should not be used for critical decisions.

---

## Demo (UI Screens)

### 1) Live Fact Check — Debunked example

When a claim is not supported by sources, the app returns **DEBUNKED (FALSE)** and shows the reasoning + the sources it checked.

<img width="1919" height="1014" alt="image" src="https://github.com/user-attachments/assets/d41c8273-b6c9-4bb5-864f-666eec8e20f6" />


### 2) Live Fact Check — Verified example

When a claim is supported by sources, the app returns **VERIFIED (TRUE)** and shows the reasoning + sources.

<img width="1919" height="1021" alt="Screenshot 2026-03-22 173653" src="https://github.com/user-attachments/assets/9f77e4dc-f192-45b9-b04f-23272221c2cf" />


### 3) Live Fact Check — Another debunked example

A second example of a claim that the app could not verify.
<img width="1919" height="1015" alt="Screenshot 2026-03-22 180147" src="https://github.com/user-attachments/assets/d5f49b41-54ba-4c59-9d2e-3cecc3473670" />


### 4) Analyze Content (ML Patterns)

Paste a news article (or a snippet) and the model predicts **FAKE / REAL-like** with a confidence score.
<img width="1919" height="1014" alt="image" src="https://github.com/user-attachments/assets/d0a2a1c5-f7d4-4806-b7ce-02dd6790580e" />



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
