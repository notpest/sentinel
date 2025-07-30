import os
import re
import google.generativeai as genai
from duckduckgo_search import DDGS

# --- API Key Configuration ---
try:
    # Load the API key from environment variables for better security
    api_key = "AIzaSyAIWcChy_lSnXmorjKIX_E7ClMC4iVTP7U"
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    print("✅ Web verifier (Gemini API) configured successfully.")
except Exception as e:
    print(f"❌ Error configuring Web Verifier: {e}")
    # Set model to None if configuration fails
    llm_model = None
else:
    # Initialize the model only if configuration is successful
    llm_model = genai.GenerativeModel('gemini-2.5-pro')

def _extract_key_claims(article_text: str) -> list[str]:
    """
    Uses the LLM to perform a preliminary pass to extract the main, verifiable claims.
    """
    if not llm_model:
        return []
    
    print("-> Step 1: Extracting key claims from article...")
    
    prompt = f"""
    You are an analytical assistant. Read the following news article and identify up to 3 primary, verifiable claims it is making. List them concisely as a numbered list.

    Article:
    ---
    {article_text}
    ---

    Claims:
    1. [First claim]
    2. [Second claim]
    ...
    """
    
    try:
        response = llm_model.generate_content(prompt)
        # Use regex to find all numbered list items
        claims = re.findall(r'^\s*\d+\.\s*(.*)', response.text, re.MULTILINE)
        if not claims: # Fallback if regex fails
            claims = response.text.strip().split('\n')
        print(f"   Extracted {len(claims)} claims.")
        return claims
    except Exception:
        # Fallback to using the first 50 words if claim extraction fails
        print("   Claim extraction failed. Falling back to simple query.")
        return [" ".join(article_text.split()[:50])]


def _search_duckduckgo(query: str, max_results: int = 3) -> str:
    """Performs a web search and formats results for the LLM."""
    print(f"-> Performing web search for: '{query[:75]}...'")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return f"No search results found for the query: '{query}'.\n\n"
        
        context = f"Evidence found for claim '{query}':\n"
        for i, result in enumerate(results):
            context += f"  Source {i+1} (Title: {result['title']}):\n"
            context += f"    URL: {result['href']}\n"
            context += f"    Snippet: {result['body']}\n\n"
        return context
    except Exception as e:
        return f"An error occurred during web search for '{query}': {e}\n\n"


def _synthesize_with_llm(aggregated_context: str, article_text: str) -> str:
    """Uses the LLM to perform a final, detailed analysis."""
    if not llm_model:
        return "LLM model not available due to configuration error."

    print("-> Step 3: Synthesizing all evidence for a final verdict...")
    prompt = f"""
    You are a meticulous, impartial fact-checker. Your task is to provide a final, reasoned analysis of an original article based on aggregated web evidence.

    **Original Article:**
    ---
    {article_text}
    ---

    **Aggregated Web Evidence (from multiple targeted searches):**
    ---
    {aggregated_context}
    ---

    **Your Analysis Task (Think Step-by-Step):**
    1.  **Evaluate Sources:** Briefly comment on the likely reliability of the retrieved sources (e.g., "Source 1 is a major news outlet, Source 3 appears to be a personal blog.").
    2.  **Analyze Evidence vs. Article:** Compare the claims in the **Original Article** against the **Aggregated Web Evidence**. Does the evidence support, contradict, or is it unrelated to the article's claims?
    3.  **Formulate Final Verdict:** Provide a clear, final verdict based on your analysis.

    **Final Output Format:**
    - **Evidence Reliability Assessment:** [Your brief assessment of the sources' credibility.]
    - **Verdict:** [**Corroborated** / **Strongly Contradicted** / **Partially Contradicted** / **Insufficient Information**]
    - **Reasoning:** [A concise, one-paragraph explanation of your conclusion. Reference specific sources (e.g., "Source 1," "Source 4") to support your reasoning.]
    """
    
    try:
        response = llm_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the LLM response: {e}"

# --- Main Public Function ---
def verify_with_web(article_text: str):
    """
    Orchestrates a multi-step RAG process for more accurate fact-checking.
    """
    # 1. Use the LLM to extract key claims first
    key_claims = _extract_key_claims(article_text)
    if not key_claims:
        return "Could not extract verifiable claims from the article."

    # 2. Search the web for each specific claim and aggregate the evidence
    print("-> Step 2: Aggregating web evidence for all claims...")
    aggregated_context = ""
    for claim in key_claims:
        aggregated_context += _search_duckduckgo(claim)
    
    # 3. Synthesize all gathered evidence into a final report
    final_report = _synthesize_with_llm(aggregated_context, article_text)
    
    return final_report