import streamlit as st
import joblib
import os
import json
import time
from datetime import datetime

from utils.preprocess import clean_text
from utils.explain import get_prediction_explainability
from utils.fact_checker import fact_check_claim

# Must be the first Streamlit command
st.set_page_config(page_title="AI Truth Checker", page_icon="🔍", layout="centered")

# Custom CSS for SaaS-like feel
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Card CSS for clean, modern look */
    .result-card-real {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-left: 6px solid #22c55e; /* Green */
        margin-bottom: 10px;
    }
    
    .result-card-fake {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-left: 6px solid #ef4444; /* Red */
        margin-bottom: 10px;
    }
    
    .result-card-unsure {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-left: 6px solid #eab308; /* Yellow */
        margin-bottom: 10px;
    }

    /* Tags */
    .tag-real {
        background-color: #dcfce7;
        color: #166534;
        padding: 6px 12px;
        border-radius: 9999px;
        font-weight: 500;
        font-size: 0.875rem;
        margin-right: 8px;
        display: inline-block;
        margin-bottom: 8px;
        border: 1px solid #bbf7d0;
    }
    
    .tag-fake {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 6px 12px;
        border-radius: 9999px;
        font-weight: 500;
        font-size: 0.875rem;
        margin-right: 8px;
        display: inline-block;
        margin-bottom: 8px;
        border: 1px solid #fecaca;
    }
    
    h1, h2, h3 {
        color: #0f172a;
    }
    
    /* Button Hover Effect */
    .stButton>button {
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Loads and caches the ML model and vectorizer."""
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'model.pkl')
    vec_path = os.path.join(os.path.dirname(__file__), 'model', 'vectorizer.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        return None, None
        
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer

model, vectorizer = load_models()

def log_prediction(text, prediction, confidence):
    """Saves predictions to a JSONL log file."""
    log_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'predictions.jsonl')
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "text_snippet": text[:100] + "..." if len(text) > 100 else text,
        "prediction": prediction,
        "confidence": round(confidence, 2)
    }
    
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def predict_news(text: str) -> dict:
    # Could refactor later into API
    if not model or not vectorizer:
        raise ValueError("Models are not loaded.")
        
    # Basic preprocessing, can improve later
    preprocessed = clean_text(text)
    if not preprocessed.strip():
        return None
        
    tfidf_features = vectorizer.transform([preprocessed])
    
    # Tried different models, LR worked best for now
    prediction = model.predict(tfidf_features)[0]
    probabilities = model.predict_proba(tfidf_features)[0]
    
    classes = model.classes_
    pred_idx = list(classes).index(prediction)
    confidence = probabilities[pred_idx] * 100
    
    prob_dist = {classes[i]: probs * 100 for i, probs in enumerate(probabilities)}
    explanation = get_prediction_explainability(preprocessed, model, vectorizer, top_n=5)
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "prob_dist": prob_dist,
        "top_words_fake": explanation.get("FAKE", []),
        "top_words_real": explanation.get("REAL", []),
        "preprocessed": preprocessed
    }

# =======================
# Sidebar Controls
# =======================
st.sidebar.title("AI Truth Checker")
st.sidebar.markdown("**Real-time News Verification**")

app_mode = st.sidebar.radio("Select Analysis Engine", [
    "🌐 Fact Check Claim (Live Web)",
    "🧠 Analyze Article (ML Patterns)" 
])

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Status:** Running locally 🟢  
""")
st.sidebar.markdown("---")
st.sidebar.markdown("*Built by Durga Prasad | v1 – experimental*")

# TODO: try transformer models (BERT) if time permits
# TODO: better dataset instead of synthetic

if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ""

def set_sample_fake():
    st.session_state['input_text'] = "Breaking! NASA confirms that aliens have completely taken over the US government in a secret deep state coup! The shocking truth revealed in new leaked emails."

def set_sample_real():
    st.session_state['input_text'] = "The Federal Reserve announced on Wednesday that it will be keeping interest rates steady for the time being, citing cooling inflation metrics."

def clear_input():
    st.session_state['input_text'] = ""


st.markdown("Quickly check whether a piece of news or a claim is likely real or misinformation.")
st.markdown("---")

if app_mode == "🧠 Analyze Article (ML Patterns)":
    
    st.title("🔍 Analyze Content")
    st.markdown("Submit text excerpts for statistical pattern classification.")
    # Model struggles with sarcasm here!

    if model is None or vectorizer is None:
        st.warning("Warning: Model files not found. Please run `python model/train.py`.")
    else:
        user_input = st.text_area("Paste article text here:", value=st.session_state['input_text'], height=200, placeholder="Enter a long paragraph or full news article to analyze...")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn1:
            if st.button("Analyze Article", type="primary", use_container_width=True):
                analyze_triggered = True
            else:
                analyze_triggered = False
        with col_btn2:
            st.button("Load Fake Example", on_click=set_sample_fake, use_container_width=True)
        with col_btn3:
            st.button("Clear Input", on_click=clear_input, use_container_width=True)

        if analyze_triggered:
            if not user_input.strip():
                st.error("Input cannot be empty.")
            else:
                with st.spinner("Analyzing patterns in text..."):
                    # honestly this part took longer than expected 😅
                    time.sleep(0.5) # UX delay
                    result = predict_news(user_input)
                    
                if result:
                    # This part felt messy but works for now
                    prediction = result['prediction']
                    confidence = result['confidence']
                    prob_dist = result['prob_dist']
                    
                    log_prediction(user_input, prediction, confidence)
                    
                    st.markdown("### 📊 Results")
                    st.caption("Model Output")
                    
                    if prediction == "REAL":
                        st.markdown(f"""
                        <div class="result-card-real">
                            <h2 style='margin-top:0; color: #166534;'>✅ This appears to be REAL news</h2>
                            <p style='font-size: 1.1rem; color: #475569;'><strong>Confidence:</strong> {confidence:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-card-fake">
                            <h2 style='margin-top:0; color: #991b1b;'>⚠️ This appears to be FAKE news</h2>
                            <p style='font-size: 1.1rem; color: #475569;'><strong>Confidence:</strong> {confidence:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.caption("Last checked just now")
                    st.info("This is based on patterns learned from training data — not a guarantee.")
                    
                    st.markdown("#### Confidence breakdown")
                    st.progress(prob_dist.get('REAL', 0) / 100, text=f"Real Probability ({prob_dist.get('REAL', 0):.0f}%)")
                    st.progress(prob_dist.get('FAKE', 0) / 100, text=f"Fake Probability ({prob_dist.get('FAKE', 0):.0f}%)")
                    
                    st.markdown("### 🧠 Why the model thinks this")
                    real_words = result["top_words_real"]
                    fake_words = result["top_words_fake"]
                    
                    st.write("The model identified the following terms as strong indicators:")
                    
                    if real_words:
                        tags = " ".join([f"<span class='tag-real'>{w}</span>" for w in real_words])
                        st.markdown(tags, unsafe_allow_html=True)
                        st.write("")
                    if fake_words:
                        tags = " ".join([f"<span class='tag-fake'>{w}</span>" for w in fake_words])
                        st.markdown(tags, unsafe_allow_html=True)
                else:
                    st.warning("Insufficient valid vocabulary for statistical mapping.")

elif app_mode == "🌐 Fact Check Claim (Live Web)":
    # API key completely hidden from the Web UI
    api_key = "API_KEY"
    
    st.title("🌐 Live Fact Check")
    st.markdown("Enter a claim and I’ll try to verify it using recent sources.")
    
    claim_input = st.text_input("Claim to verify:", placeholder="e.g. 'Federal interest rates increased today'")
    
    if st.button("Verify Claim", type="primary"):
        if not api_key:
            st.error("Authentication required. Missing hidden API key allocation.")
        elif not claim_input.strip():
            st.error("Input missing valid claim.")
        else:
            with st.spinner("Checking live sources..."):
                result = fact_check_claim(claim_input, api_key)
                
            if "error" in result:
                st.error(f"Something went wrong while checking this: {result['error']}")
            else:
                verdict = result["verdict"].upper()
                
                st.markdown("### 📊 Results")
                st.caption("Model Output")
                
                if "TRUE" in verdict or "REAL" in verdict:
                    st.markdown(f"""
                    <div class="result-card-real">
                        <h2 style='margin-top:0; color: #166534;'>✅ VERIFIED (TRUE)</h2>
                    </div>
                    """, unsafe_allow_html=True)
                elif "FALSE" in verdict or "FAKE" in verdict:
                    st.markdown(f"""
                    <div class="result-card-fake">
                        <h2 style='margin-top:0; color: #991b1b;'>❌ DEBUNKED (FALSE)</h2>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card-unsure">
                        <h2 style='margin-top:0; color: #854d0e;'>⚠️ INCONCLUSIVE</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.caption("Last checked just now")
                st.info("This is based on patterns learned from training data — not a guarantee.")
                
                st.markdown("### 🧠 Why this result?")
                st.write(result['reasoning'])
                
                st.markdown("### 🌐 Sources")
                with st.expander("Expand to view raw web search data used"):
                    st.text(result["context"])
