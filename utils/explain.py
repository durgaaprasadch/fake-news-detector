import numpy as np

def get_prediction_explainability(preprocessed_text: str, model, vectorizer, top_n: int = 5) -> dict:
    """
    Extracts top contributing words pushing the prediction towards FAKE or REAL.
    Uses TF-IDF weights multiplied by Logistic Regression coefficients.
    """
    tfidf_vec = vectorizer.transform([preprocessed_text])
    feature_names = vectorizer.get_feature_names_out()
    
    indices = tfidf_vec.nonzero()[1]
    
    if len(indices) == 0:
        return {"REAL": [], "FAKE": []}
        
    coefs = model.coef_[0]
    
    word_scores = []
    for idx in indices:
        word = feature_names[idx]
        score = coefs[idx] * tfidf_vec[0, idx] # Weight * TFIDF value
        word_scores.append((word, score))
        
    # Get model classes
    classes = model.classes_
    class_0 = classes[0]
    class_1 = classes[1]
    
    # For Logistic Regression, class_0 is negative class, class_1 is positive class
    # If word score < 0, it pushes towards class_0
    # If word score > 0, it pushes towards class_1
    
    # Sort terms by their negative influence
    top_class_0_words = [w[0] for w in sorted(word_scores, key=lambda x: x[1]) if w[1] < 0][:top_n]
    # Sort terms by their positive influence
    top_class_1_words = [w[0] for w in sorted(word_scores, key=lambda x: x[1], reverse=True) if w[1] > 0][:top_n]
    
    result = {
        class_0: top_class_0_words,
        class_1: top_class_1_words
    }
    
    # Ensure keys FAKE and REAL exist to prevent UI KeyError
    if "FAKE" not in result:
        result["FAKE"] = []
    if "REAL" not in result:
        result["REAL"] = []
        
    return result
