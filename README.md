# AI Fake News Detector 📰

This is a small project I built to understand how machine learning can be used to detect fake news.

## Why I chose this problem
The internet is full of claims and news articles, and it's becoming incredibly hard to tell what's real and what's fake. I wanted to see if I could build a simple tool that uses Natural Language Processing (NLP) to catch patterns of misinformation and verify claims against live data.

## What I built
A dual-engine web application that handles two types of misinformation:
1. **Statistical NLP:** A standard Logistic Regression model trained to recognize the linguistic style of fake news.
2. **Live Fact-Checking:** A Web-scraper attached to a Large Language Model (Google Gemini) that actively searches the internet to verify if a specific claim is true.

## What worked well
- Using TF-IDF vectorization was surprisingly effective at catching emotionally manipulative "clickbait" words.
- The Streamlit framework made it really easy to deploy a clean UI without writing a ton of frontend code.

## What I struggled with
- Web scraping DuckDuckGo was a nightmare! The API constantly rate-limited me or returned empty results, so I had to build a fallback protocol.
- Sarcasm. The TF-IDF model has absolutely no idea what sarcasm is and will flag true satirical articles as "FAKE". 

## What I learned while building this
I learned that traditional Machine Learning (like Logistic Regression) is great for recognizing patterns in text structure, but it doesn't actually "know" anything to be true or false. To build a true fact-checker, you have to connect an AI to the live internet so it can read recent news context.

## Limitations
- **Model is basic:** It's just a simple TF-IDF regression model, not a deep neural network yet.
- **Depends on dataset quality:** If the training data is biased, the model's predictions will be biased.
- **Not for critical use:** This is an experimental project and is *not reliable for real-world critical decisions!*

## What I would improve
- I'd love to swap out the TF-IDF model for a modern transformer model like BERT.
- Improve the dataset by mixing in more recent news articles.

---
*Built by Durga Prasad | v1 – experimental*
