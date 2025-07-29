import os
import google.generativeai as genai
from duckduckgo_search import DDGS

# --- API Key Configuration ---
try:
    # Load the API key from environment variables for better security
    api_key = os.getenv('GOOGLE_API_KEY')
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

# --- Helper Functions ---
def _search_duckduckgo(query: str, max_results: int = 5) -> str:
    """Performs a web search and formats results for the LLM."""
    print("-> Performing web search...")
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
        
    if not results:
        return "No search results found."
    
    context = "Search Results:\n"
    for i, result in enumerate(results):
        context += f"Source {i+1} (Title: {result['title']}):\n"
        context += f"URL: {result['href']}\n"
        context += f"Snippet: {result['body']}\n\n"
        
    return context

def _synthesize_with_llm(context: str, article_text: str) -> str:
    """Uses the LLM to analyze search context against an article."""
    if not llm_model:
        return "LLM model not available due to configuration error."

    print("-> Synthesizing evidence with LLM...")
    prompt = f"""
    You are a professional, impartial fact-checker. Your task is to analyze an original news article against a set of retrieved web search results and determine if the article's main claims are corroborated.

    **Original Article:**
    ---
    {article_text}
    ---

    **Retrieved Web Evidence:**
    ---
    {context}
    ---

    **Your Analysis Task:**
    Provide a final verdict and a concise, one-paragraph explanation of your conclusion, citing the sources from the evidence by number (e.g., "Source 1," "Source 3").
    - **Verdict:** [Corroborated / Contradicted / Insufficient Information]
    - **Reasoning:** [Your explanation here.]
    """
    
    try:
        response = llm_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the LLM response: {e}"

# --- Main Public Function ---
def verify_with_web(article_text: str):
    """
    Orchestrates the RAG process for verifying a news article.
    This is the main function to be called from outside this module.
    """
    # Use the first 50 words as the search query for simplicity
    query = " ".join(article_text.split()[:50])
    
    retrieved_context = _search_duckduckgo(query)
    final_report = _synthesize_with_llm(retrieved_context, article_text)
    
    return final_report

# --- Self-Test Block ---
if __name__ == '__main__':
    # This block allows you to test this file directly
    print("\n--- Running self-test for web_verifier.py ---")
    test_article = "BREAKING: Sources say a new secret tax will be implemented by the government tomorrow morning without any public announcement."
    report = verify_with_web(test_article)
    print("\n--- TEST REPORT ---")
    print(report)