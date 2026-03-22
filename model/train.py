import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocess import clean_text

def generate_synthetic_data() -> pd.DataFrame:
    """
    Downloads a real Fake News dataset consisting of ~6300 articles 
    so the model has a large enough vocabulary to accurately predict real-world inputs.
    """
    print("Downloading authentic fake/real news dataset (~30MB) from GitHub...")
    url = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"
    try:
        df = pd.read_csv(url)
        # Check columns
        if 'text' in df.columns and 'label' in df.columns:
            # Drop unnecessary columns and handle NAs
            df = df[['text', 'label']].dropna()
            
            # Subsample to speed up training locally (e.g., 2000 random samples)
            # You can remove `.sample(2000)` to train on all 6300 items for maximum accuracy!
            df = df.sample(n=2000, random_state=42).reset_index(drop=True)
            return df
        else:
            raise Exception("Dataset structure is incorrect.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Falling back to small synthetic data...")
        # Fallback to a very small synthetic dataset if no internet
        data = [
            {"text": "The local city council approved the new budget.", "label": "REAL"},
            {"text": "Scientists discover a new species of frog.", "label": "REAL"},
            {"text": "Aliens have landed in New York!", "label": "FAKE"},
            {"text": "Eating chocolate cures all diseases instantly.", "label": "FAKE"}
        ] * 50
        return pd.DataFrame(data).sample(frac=1).reset_index(drop=True)

def train_model():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'news.csv')
    
    if os.path.exists(csv_path):
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
    else:
        print("Dataset not found. Generating synthetic dataset...")
        df = generate_synthetic_data()
        df.to_csv(csv_path, index=False)
        print(f"Synthetic data saved to {csv_path}")
        
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
        
    print("Preprocessing text data...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Train / Test Split
    print("Splitting dataset (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    print("Vectorizing text (TF-IDF)...")
    # TF-IDF Vectorizer with max_features=5000 and ngram_range=(1,2)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("Training Logistic Regression model...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train_vec, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test_vec)
    
    # Metrics calculation
    # Pos_label depends on what we consider 'positive'. We will evaluate on 'REAL' being positive.
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="REAL")
    rec = recall_score(y_test, y_pred, pos_label="REAL")
    f1 = f1_score(y_test, y_pred, pos_label="REAL")
    cm = confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"])
    
    print("\n" + "="*40)
    print("          MODEL EVALUATION")
    print("="*40)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print("\nConfusion Matrix:")
    print("             Predicted FAKE | Predicted REAL")
    print(f"Actual FAKE: {cm[0][0]:<14} | {cm[0][1]}")
    print(f"Actual REAL: {cm[1][0]:<14} | {cm[1][1]}")
    print("="*40 + "\n")
    
    # Saving models
    model_dir = os.path.dirname(__file__)
    model_path = os.path.join(model_dir, 'model.pkl')
    vec_path = os.path.join(model_dir, 'vectorizer.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vec_path}")

if __name__ == "__main__":
    train_model()
