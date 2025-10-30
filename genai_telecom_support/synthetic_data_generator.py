"""
Synthetic Telecom Data Generator
Generates synthetic customer queries and responses for telecom support scenarios.
"""

import random

# Define telecom issue categories and sample templates
CATEGORIES = [
    "Billing",
    "Network Outage",
    "Plan Change",
    "Device Problem"
]

BILLING_QUERIES = [
    "Why is my bill higher this month?",
    "I was charged twice for my plan.",
    "Can you explain the taxes on my bill?",
    "I need a refund for an incorrect charge."
]

NETWORK_OUTAGE_QUERIES = [
    "My internet is not working.",
    "There is no network in my area.",
    "Why is my 4G so slow today?",
    "When will the outage be fixed?"
]

PLAN_CHANGE_QUERIES = [
    "How do I upgrade my plan?",
    "Can I switch to a cheaper plan?",
    "What are the benefits of the premium plan?",
    "I want to add more data to my plan."
]

DEVICE_PROBLEM_QUERIES = [
    "My SIM card is not detected.",
    "Phone shows 'No Service'.",
    "How do I activate my new SIM?",
    "My device can't connect to WiFi."
]

# Sample responses (to be replaced by LLM in real system)
RESPONSES = {
    "Billing": "Please check your bill details in the app. If you see an incorrect charge, you can raise a dispute. For refunds, our team will process it within 3-5 business days.",
    "Network Outage": "We are aware of the outage in your area and our engineers are working to resolve it. Estimated resolution time is 2 hours.",
    "Plan Change": "You can upgrade or downgrade your plan anytime from the app. For more data, select 'Add-ons' in your plan settings.",
    "Device Problem": "Try restarting your device and reinserting the SIM card. If the issue persists, visit the nearest service center or contact support."
}

def generate_synthetic_data(num_samples=20):
    data = []
    for _ in range(num_samples):
        category = random.choice(CATEGORIES)
        if category == "Billing":
            query = random.choice(BILLING_QUERIES)
        elif category == "Network Outage":
            query = random.choice(NETWORK_OUTAGE_QUERIES)
        elif category == "Plan Change":
            query = random.choice(PLAN_CHANGE_QUERIES)
        else:
            query = random.choice(DEVICE_PROBLEM_QUERIES)
        response = RESPONSES[category]
        data.append({
            "category": category,
            "query": query,
            "response": response
        })
    return data

if __name__ == "__main__":
    samples = generate_synthetic_data(30)
    for s in samples:
        print(f"[{s['category']}] Q: {s['query']}\nA: {s['response']}\n")
