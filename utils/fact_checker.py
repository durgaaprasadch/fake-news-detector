from duckduckgo_search import DDGS
import google.generativeai as genai

def fact_check_claim(claim: str, api_key: str) -> dict:
    """
    Fact checks a claim by searching the live web and summarizing the findings using Gemini LLM.
    """
    if not api_key:
        return {"error": "API Key is missing."}
        
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        return {"error": f"Failed to configure Gemini API: {e}"}
        
    # 1. Search the live web for the claim using DuckDuckGo
    try:
        ddgs = DDGS()
        # Prioritize Live News Search for breaking global events
        results = list(ddgs.news(claim, max_results=4))
        
        # Fallback to general text search if no news pops up
        if not results:
            results = list(ddgs.text(claim, max_results=4))
            
        if not results:
            context = "No recent web results found for this claim."
        else:
            context = "\n\n".join([f"Title: {r.get('title', '')}\nSnippet: {r.get('body', '')}" for r in results])
    except Exception as e:
        return {"error": f"Web Search failed to retrieve context: {e}"}
        
    # 2. Evaluate with Gemini LLM
    try:
        # Dynamically find an available model for their specific API key to avoid 404s
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
                
        if not available_models:
            return {"error": "Your API key does not have access to any Generation models. Ensure the Generative Language API is enabled on your Google Account."}
            
        prompt = f"""
Evaluate the following claim based on your knowledge and the provided web context.
You MUST follow this exact two-line format. Do not add conversational filler before the verdict.

Format:
Verdict: [TRUE or FALSE or UNVERIFIED]
Reasoning: [1-2 short paragraphs explaining why]

Claim: "{claim}"
Context: {context}
"""
        
        # Try evaluating with available models, effectively bypassing quota/404 issues on specific models
        last_error = None
        response_text = ""
        for model_name in available_models:
            # Skip explicitly non-text models
            if "tts" in model_name.lower() or "embedding" in model_name.lower() or "vision" in model_name.lower():
                continue
                
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                response_text = response.text
                break # Success! Break the loop
            except Exception as e:
                last_error = e
                continue # Try the next model, don't break on 400s or 429s
                    
        if not response_text:
            return {"error": f"LLM Quota Exceeded. Please try again! Error: {last_error}"}
            
        verdict = "UNVERIFIED"
        reasoning = response_text
        
        response_upper = response_text.upper()
        
        # Look for the absolute verdict classification anywhere in the response
        if "VERDICT: FALSE" in response_upper or "VERDICT: FAKE" in response_upper:
            verdict = "FALSE"
        elif "VERDICT: TRUE" in response_upper or "VERDICT: REAL" in response_upper:
            verdict = "TRUE"
        elif "FALSE" in response_upper.split('\n')[0] or "FAKE" in response_upper.split('\n')[0]:
            # Fallback if it completely ignores the "Verdict:" tag but puts False on Line 1
            verdict = "FALSE"
        elif "TRUE" in response_upper.split('\n')[0]:
            verdict = "TRUE"
        else:
            verdict = "UNVERIFIED"
            
        # Try to safely extract just the reasoning portion if possible
        reasoning_idx = response_upper.find("REASONING:")
        if reasoning_idx != -1:
            reasoning = response_text[reasoning_idx + 10:].strip()
        else:
            # If no reasoning tag, just show the whole response minus the verdict line
            lines = response_text.split('\n')
            reasoning = "\n".join(lines[1:]).strip() if len(lines) > 1 else response_text
            
        return {
            "verdict": verdict,
            "reasoning": reasoning,
            "context": context
        }
        
    except Exception as e:
        return {"error": f"LLM Fact-Checking failed: {e}"}
