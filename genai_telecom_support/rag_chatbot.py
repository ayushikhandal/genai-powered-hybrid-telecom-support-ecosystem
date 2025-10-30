"""
RAG Chatbot for Telecom Support
Uses synthetic data as a knowledge base and LLM for response generation.
"""

import json
from typing import List, Dict

# Load synthetic data as knowledge base
with open("synthetic_telecom_data.json", "r", encoding="utf-8") as f:
    KB = json.load(f)

def retrieve_relevant_entries(query: str, kb: List[Dict], top_k=3) -> List[Dict]:
    """Simple keyword-based retrieval from the knowledge base."""
    results = []
    for entry in kb:
        if any(word.lower() in entry["query"].lower() for word in query.split()):
            results.append(entry)
    # Fallback: return top_k random if nothing found
    if not results:
        import random
        results = random.sample(kb, min(top_k, len(kb)))
    return results[:top_k]

def generate_response(query: str, kb: List[Dict]) -> str:
    """Generate a response using retrieved knowledge base entries."""
    entries = retrieve_relevant_entries(query, kb)
    # For demo, concatenate responses (replace with LLM call in real system)
    response = "\n---\n".join([f"Q: {e['query']}\nA: {e['response']}" for e in entries])
    return response

if __name__ == "__main__":
    print("Welcome to the GenAI-Powered Telecom Support Chatbot!")
    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        answer = generate_response(user_query, KB)
        print(f"Bot:\n{answer}\n")
