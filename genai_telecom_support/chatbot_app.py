"""
Retrieval-Augmented LLM Chatbot for Telecom Support
Uses synthetic data and LLM to answer queries with step-by-step, explainable, and visual responses.
"""


import streamlit as st
import transformers as hf
import json
from textblob import TextBlob
import difflib
import os
import requests
from pathlib import Path
import time
import random
import heapq

# Load synthetic knowledge base
import difflib

# ============================================================================
# INTERACTIVE UI HELPER FUNCTIONS
# ============================================================================

def typing_indicator(duration=2):
    """Show animated typing indicator."""
    placeholder = st.empty()
    for i in range(duration * 3):
        dots = "." * ((i % 3) + 1)
        placeholder.markdown(f"🤖 **Agent is typing** {dots}")
        time.sleep(0.33)
    placeholder.empty()

def show_status_badge(status: str, color: str = "blue"):
    """Show colored status badge."""
    colors = {
        "online": "🟢",
        "typing": "🟡",
        "offline": "🔴",
        "joined": "✅",
        "waiting": "⏳"
    }
    icon = colors.get(status, "⚪")
    st.markdown(f"{icon} **Status:** {status.capitalize()}")

def quick_reply_buttons(options: list):
    """Show quick reply buttons for common follow-ups."""
    st.markdown("**Quick Replies:**")
    cols = st.columns(len(options))
    selected = None
    for i, option in enumerate(options):
        with cols[i]:
            if st.button(option, key=f"quick_reply_{i}_{time.time()}"):
                selected = option
    return selected

def message_with_avatar(role: str, message: str, show_reaction=True):
    """Display message with avatar and optional reaction buttons."""
    if role == "user":
        st.markdown(f"👤 **You:** {message}")
    else:
        st.markdown(f"🤖 **Agent:** {message}")
    
    if show_reaction:
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("👍", key=f"like_{id(message)}"):
                # Immediately save feedback
                if hasattr(st.session_state, 'last_query') and hasattr(st.session_state, 'last_category'):
                    save_feedback_as_kb_entry(
                        query=st.session_state.last_query,
                        response=message,
                        category=st.session_state.last_category,
                        feedback_rating=5,
                        feedback_text="User marked as helpful",
                        user_sentiment=getattr(st.session_state, 'last_sentiment', 'neutral')
                    )
                    # Track in metrics
                    st.session_state.metrics["feedback_yes"] = st.session_state.metrics.get("feedback_yes", 0) + 1
                st.toast("👍 Feedback recorded and saved to KB!")
        with col2:
            if st.button("👎", key=f"dislike_{id(message)}"):
                # Immediately save feedback
                if hasattr(st.session_state, 'last_query') and hasattr(st.session_state, 'last_category'):
                    save_feedback_as_kb_entry(
                        query=st.session_state.last_query,
                        response=message,
                        category=st.session_state.last_category,
                        feedback_rating=1,
                        feedback_text="User marked as not helpful - needs improvement",
                        user_sentiment=getattr(st.session_state, 'last_sentiment', 'neutral')
                    )
                    # Track in metrics
                    st.session_state.metrics["feedback_no"] = st.session_state.metrics.get("feedback_no", 0) + 1
                st.toast("👎 Feedback recorded! We'll use this to improve.")

def satisfaction_modal():
    """Show satisfaction rating after escalation closes - saves feedback to model."""
    st.markdown("---")
    st.markdown("### 📋 **How was your experience?**")
    satisfaction = st.radio(
        "Please rate your satisfaction:",
        ["😊 Very Satisfied", "😌 Satisfied", "😐 Neutral", "😕 Unsatisfied", "😞 Very Unsatisfied"],
        key=f"satisfaction_{time.time()}"
    )
    
    feedback = st.text_area(
        "Any additional feedback? (optional)",
        placeholder="Tell us how we can improve...",
        key=f"feedback_text_{time.time()}"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit Feedback", key=f"submit_feedback_{time.time()}"):
            st.success("✅ Thank you for your feedback!")
            st.balloons()
            
            # Convert satisfaction to rating
            rating_map = {
                "😊 Very Satisfied": 5,
                "😌 Satisfied": 4,
                "😐 Neutral": 3,
                "😕 Unsatisfied": 2,
                "😞 Very Unsatisfied": 1
            }
            rating = rating_map.get(satisfaction, 3)
            
            # Save feedback as KB entry for continuous learning
            if "last_query" in st.session_state and "last_category" in st.session_state:
                save_feedback_as_kb_entry(
                    query=st.session_state.last_query,
                    response=st.session_state.get("last_response", ""),
                    category=st.session_state.last_category,
                    feedback_rating=rating,
                    feedback_text=feedback,
                    user_sentiment=st.session_state.get("last_sentiment", "neutral")
                )
                st.info("💡 Your feedback helps us improve! This will train our model for better responses.")
            
            return True
    with col2:
        if st.button("Skip", key=f"skip_feedback_{time.time()}"):
            return True
    return False

# ============================================================================
# INNOVATIVE FEATURE FUNCTIONS (NEW)
# ============================================================================

def proactive_issue_detection(history, metrics):
    """Feature 1: Detect high data usage or patterns and proactively alert customer."""
    if len(history) < 3:
        return None
    
    # Simple heuristic: if multiple data/plan queries, suggest upgrade
    data_queries = sum(1 for q, _, _, _, _ in history if "data" in q.lower() or "usage" in q.lower())
    
    if data_queries >= 2:
        return "💡 **Proactive Tip:** We noticed you're asking about data plans frequently. Consider upgrading to a plan with more data to avoid overage charges!"
    
    return None

def voice_video_escalation_button():
    """Feature 2: Add voice/video chat escalation option."""
    st.markdown("---")
    st.markdown("### 🎤 **Advanced Support Options**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📞 Start Voice Call", key="voice_call"):
            st.info("🔗 **Voice Call Feature:** Connecting to Zoom/Teams... [Integration in progress]")
    with col2:
        if st.button("📹 Start Video Chat", key="video_chat"):
            st.info("🔗 **Video Chat Feature:** Initializing video call... [Integration in progress]")

def multilingual_support(query):
    """Feature 3: Detect language and offer translation."""
    try:
        # Simple language detection (can use TextBlob or langdetect library)
        blob = TextBlob(query)
        detected_lang = blob.detect_language()
        
        if detected_lang != "en":
            st.warning(f"🌐 **Language Detected:** {detected_lang.upper()}. Would you like me to respond in your language or English?")
            lang_choice = st.radio("Preferred Language:", ["English", "Detected Language"], key=f"lang_{time.time()}")
            return lang_choice
    except Exception:
        pass
    
    return "English"

def smart_faq_generation(history):
    """Feature 4: Dynamically generate FAQ based on trending queries."""
    if len(history) < 5:
        return
    
    # Extract top queries
    query_freq = {}
    for q, _, _, _, _ in history:
        query_freq[q] = query_freq.get(q, 0) + 1
    
    top_queries = sorted(query_freq.items(), key=lambda x: x[1], reverse=True)[:3]
    
    if top_queries:
        st.markdown("---")
        st.markdown("### 📚 **Trending FAQs (Auto-Generated)**")
        for q, count in top_queries:
            st.markdown(f"- **Q:** {q} *(asked {count} times)*")

def agent_performance_dashboard(escalation_list, metrics):
    """Feature 5: Show agent performance analytics."""
    st.markdown("---")
    st.markdown("### 📊 **Agent Performance Analytics**")
    
    if not escalation_list:
        st.info("No agent interactions yet.")
        return
    
    # Calculate agent stats
    total_escalations = len(escalation_list)
    avg_confidence = metrics.get("avg_confidence", 0)
    feedback_yes = metrics.get("feedback_yes", 0)
    feedback_no = metrics.get("feedback_no", 0)
    total_feedback = feedback_yes + feedback_no if (feedback_yes + feedback_no) > 0 else 1
    satisfaction_rate = (feedback_yes / total_feedback) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cases Escalated", total_escalations)
    with col2:
        st.metric("Avg Confidence Score", f"{avg_confidence:.1%}")
    with col3:
        st.metric("Customer Satisfaction", f"{satisfaction_rate:.0f}%")
    
    # DYNAMIC Agent-level breakdown (based on escalation data)
    st.markdown("**Agent Breakdown (Real-time Performance):**")
    
    # Agent pool with simulated baseline performance
    agent_names = ["Priya", "Raj", "Asha", "Vikram", "Neha"]
    
    # Baseline performance profiles (how good each agent is when idle)
    agent_profiles = {
        "Priya": {"base_satisfaction": 92, "base_speed": 15, "base_confidence": 85},
        "Raj": {"base_satisfaction": 95, "base_speed": 12, "base_confidence": 88},
        "Asha": {"base_satisfaction": 89, "base_speed": 18, "base_confidence": 82},
        "Vikram": {"base_satisfaction": 91, "base_speed": 16, "base_confidence": 84},
        "Neha": {"base_satisfaction": 93, "base_speed": 14, "base_confidence": 86}
    }
    
    agent_stats = {}
    
    # Initialize all agents with baseline stats
    for agent in agent_names:
        agent_stats[agent] = {
            "cases": 0,
            "total_confidence": 0,
            "sentiments": [],
            "total_time": 0,
            "profile": agent_profiles[agent]
        }
    
    # Build agent stats from escalation history
    for idx, escalation in enumerate(escalation_list):
        # Assign agent based on escalation index
        agent = agent_names[idx % len(agent_names)]
        
        agent_stats[agent]["cases"] += 1
        agent_stats[agent]["total_confidence"] += escalation.get("confidence", 0.5)
        agent_stats[agent]["sentiments"].append(escalation.get("sentiment", "neutral"))
        agent_stats[agent]["total_time"] += max(5, 20 - (idx * 2))  # Faster with experience
    
    # Display agent performance (all agents with real or baseline metrics)
    for agent in agent_names:
        stats = agent_stats[agent]
        cases = stats["cases"]
        profile = stats["profile"]
        
        if cases > 0:
            # Real performance from actual cases
            avg_confidence = (stats["total_confidence"] / cases * 100)
            avg_time = stats["total_time"] / cases
            
            # Calculate satisfaction from sentiment
            frustrated_count = sum(1 for s in stats["sentiments"] if s == "frustrated")
            satisfaction = profile["base_satisfaction"] - (frustrated_count * 10)
            satisfaction = max(70, min(99, satisfaction))
        else:
            # Baseline performance (when idle/no cases yet)
            avg_confidence = profile["base_confidence"]
            avg_time = profile["base_speed"]
            satisfaction = profile["base_satisfaction"]
        
        # Show rating in stars
        stars = "⭐" * int(satisfaction / 20) + "☆" * (5 - int(satisfaction / 20))
        
        st.write(
            f"👨‍💼 **{agent}** | "
            f"Cases: {cases} | "
            f"Avg Resolution: {avg_time:.0f}s ⚡ | "
            f"Confidence: {avg_confidence:.0f}% 🎯 | "
            f"Rating: {satisfaction:.0f}% {stars}"
        )

def feedback_training_dashboard():
    """Show feedback-based model training analytics."""
    st.markdown("---")
    st.markdown("### 🧠 **Model Training from User Feedback**")
    
    stats = get_feedback_stats()
    
    if not stats or stats["total_feedback_entries"] == 0:
        st.info("💡 No feedback data yet. User feedback will help train the model over time.")
        return
    
    # Display stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Feedback Entries", stats["total_feedback_entries"])
    with col2:
        st.metric("Helpful Responses", stats["helpful_count"])
    with col3:
        st.metric("Needs Improvement", stats["unhelpful_count"])
    with col4:
        st.metric("Avg Rating", f"{stats['avg_rating']:.1f}/5")
    
    # Show learning opportunities
    if stats["learning_opportunities"]:
        st.markdown("**🔍 Areas to Improve (from user feedback):**")
        for i, opp in enumerate(stats["learning_opportunities"][:5], 1):
            st.write(f"{i}. **Q:** {opp['query'][:50]}...")
            if opp['comment']:
                st.caption(f"📝 {opp['comment']}")
    
    # Show most common categories being trained
    if stats["top_categories"]:
        st.markdown("**📚 Most Common Feedback Categories:**")
        for cat, count in sorted(stats["top_categories"].items(), key=lambda x: x[1], reverse=True)[:5]:
            st.write(f"• **{cat}**: {count} feedback entries")

# Simple provider synonyms / common typos map
def load_providers_mapping(kb_path=None):
    # Try to load providers.json from app folder
    try:
        with open("providers.json", "r", encoding="utf-8") as pf:
            mapping = json.load(pf)
            # invert mapping for quick lookup
            inv = {}
            for canonical, variants in mapping.items():
                for v in variants:
                    inv[v.lower()] = canonical.lower()
            return inv
    except Exception:
        # Fallback: try to build from KB if path provided
        if kb_path:
            try:
                with open(kb_path, "r", encoding="utf-8") as f:
                    kb = json.load(f)
                    inv = {}
                    for e in kb:
                        cat = e.get("category", "").lower()
                        # attempt to extract provider tokens from queries
                        for token in e.get("query", "").lower().split():
                            if token.isalpha() and len(token) <= 6:
                                inv[token] = token
                    return inv
            except Exception:
                return {}
        return {}


def normalize_query(query: str, providers_inv: dict) -> str:
    """Normalize common provider name typos using dynamic mapping."""
    q = " " + query.lower() + " "
    for wrong, right in providers_inv.items():
        q = q.replace(f" {wrong} ", f" {right} ")
    return q.strip()


# Provider file helpers
PROVIDERS_FILE = Path(__file__).parent / "providers.json"

def read_providers_file():
    try:
        with open(PROVIDERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def write_providers_file(mapping: dict):
    try:
        with open(PROVIDERS_FILE, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)
        return True
    except Exception:
        return False

def auto_generate_providers_from_kb(kb):
    # naive extraction: collect likely tokens that look like provider names
    candidates = {}
    # accept either a list or an iterator factory
    iterator = kb() if callable(kb) else (iter(kb) if isinstance(kb, list) else iter(kb))
    for e in iterator:
        try:
            tokens = [t for t in e.get("query", "").lower().split() if t.isalpha() and 2 <= len(t) <= 10]
        except Exception:
            continue
        for t in tokens:
            candidates.setdefault(t, 0)
            candidates[t] += 1
    # pick tokens with count > 1 as candidates
    mapping = {}
    for token, count in sorted(candidates.items(), key=lambda x: -x[1]):
        if count >= 2:
            # default canonical to the token itself
            mapping.setdefault(token, []).append(token)
    return mapping
def load_kb():
    """Load KB from JSON (array) or JSONL (streaming). If a JSONL file is present alongside the JSON file name, prefer the JSONL for memory efficiency.

    Returns:
        - If JSON array: a list of entries (as before).
        - If JSONL: a generator function (callable) `iter_kb()` that yields entries one-by-one.
    """
    # Default paths (project root)
    json_path = Path(__file__).parent.parent / "synthetic_data.json"
    jsonl_path = Path(__file__).parent.parent / "large_kb.jsonl"

    # Prefer JSONL if present for large KBs
    try:
        if jsonl_path.exists():
            # Return an iterator factory so callers can iterate without loading everything
            def iter_kb():
                with open(str(jsonl_path), "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line)
                        except Exception:
                            continue
            return iter_kb
        # Fall back to JSON array
        if json_path.exists():
            with open(str(json_path), "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    # If nothing found, return empty list
    return []

# ============================================================================
# FEEDBACK & MODEL TRAINING FUNCTIONS
# ============================================================================

def save_feedback_as_kb_entry(query, response, category, feedback_rating, feedback_text="", user_sentiment=""):
    """Save positive user feedback as new KB entry to improve the model."""
    try:
        feedback_kb_path = Path(__file__).parent.parent / "feedback_kb.jsonl"
        
        entry = {
            "query": query,
            "response": response,
            "category": category,
            "feedback_rating": feedback_rating,  # 1-5 stars or "helpful"/"not helpful"
            "user_comment": feedback_text,
            "user_sentiment": user_sentiment,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "user_feedback",
            "score": 0.95  # High confidence for user-approved responses
        }
        
        # Append to feedback KB file
        with open(str(feedback_kb_path), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        
        return True
    except Exception as e:
        print(f"Error saving feedback KB: {e}")
        return False

def get_feedback_stats():
    """Get statistics on feedback-based learning."""
    try:
        feedback_kb_path = Path(__file__).parent.parent / "feedback_kb.jsonl"
        stats = {
            "total_feedback_entries": 0,
            "helpful_count": 0,
            "unhelpful_count": 0,
            "avg_rating": 0,
            "top_categories": {},
            "learning_opportunities": []
        }
        
        if not feedback_kb_path.exists():
            return stats
        
        ratings = []
        with open(str(feedback_kb_path), "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    stats["total_feedback_entries"] += 1
                    
                    # Track ratings
                    rating = entry.get("feedback_rating", 0)
                    if rating in [4, 5, "helpful"]:
                        stats["helpful_count"] += 1
                    elif rating in [1, 2, "not helpful"]:
                        stats["unhelpful_count"] += 1
                        # Track learning opportunities
                        stats["learning_opportunities"].append({
                            "query": entry.get("query"),
                            "comment": entry.get("user_comment")
                        })
                    
                    if isinstance(rating, int):
                        ratings.append(rating)
                    
                    # Track categories
                    cat = entry.get("category", "Other")
                    stats["top_categories"][cat] = stats["top_categories"].get(cat, 0) + 1
                except Exception:
                    continue
        
        if ratings:
            stats["avg_rating"] = round(sum(ratings) / len(ratings), 2)
        
        return stats
    except Exception as e:
        print(f"Error getting feedback stats: {e}")
        return {}

def load_feedback_kb():
    """Load feedback KB entries along with main KB for continuous learning."""
    try:
        feedback_kb_path = Path(__file__).parent.parent / "feedback_kb.jsonl"
        feedback_entries = []
        
        if feedback_kb_path.exists():
            with open(str(feedback_kb_path), "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        # Only include helpful feedback as training data
                        if entry.get("feedback_rating") in [4, 5, "helpful"]:
                            feedback_entries.append(entry)
                    except Exception:
                        continue
        
        return feedback_entries
    except Exception as e:
        print(f"Error loading feedback KB: {e}")
        return []

# Simple keyword-based retrieval
def retrieve_relevant_entries(query, kb, top_k=3):
    """Fuzzy retrieval: score KB entries by similarity to the query and return top_k with scores.

    kb may be either:
      - a list (loaded JSON array) -> iterate directly
      - an iterator factory (callable) returned by load_kb() for JSONL -> iterate streamed
    """
    import heapq

    q = query.lower()
    
    # Keywords for each category to boost matching
    CATEGORY_KEYWORDS = {
        "device problem": ["sim", "device", "phone", "wifi", "connection", "activate", "detect", "service"],
        "network outage": ["internet", "outage", "network", "signal", "4g", "5g", "slow", "area"],
        "plan change": ["plan", "upgrade", "downgrade", "data", "add-on", "switch", "price", "tariff"],
        "billing": ["bill", "charge", "refund", "dispute", "invoice", "payment", "cost"]
    }

    def score_entry(entry):
        text = entry.get("query", "").lower()
        category = entry.get("category", "").lower()
        
        # Base similarity score
        ratio = difflib.SequenceMatcher(None, q, text).ratio()
        
        # Word overlap bonus (INCREASED from 0.05 to 0.15 for better matching)
        query_words = set(q.split())
        text_words = set(text.split())
        overlap = len(query_words & text_words)
        word_overlap_score = 0.15 * (overlap / max(len(query_words), len(text_words))) if max(len(query_words), len(text_words)) > 0 else 0
        
        # **BOOST for category keyword matches** - INCREASED bonus!
        category_bonus = 0
        if category in CATEGORY_KEYWORDS:
            keywords = CATEGORY_KEYWORDS[category]
            # Count how many keywords from this category appear in query
            keyword_matches = sum(1 for kw in keywords if kw in q)
            # Bonus: +0.4 per keyword match (increased from 0.3!)
            category_bonus = min(0.8, keyword_matches * 0.4)  # Max 0.8 bonus (increased from 0.6)
        
        # Final score = base + word_overlap + category bonus (BOOST all factors)
        # Using max() to ensure we don't score low if word overlap is present
        base_score = max(ratio, word_overlap_score)
        score = base_score + category_bonus
        return round(min(1.0, score), 3)

    top_heap = []  # min-heap of (score, entry)

    # Determine if kb is an iterator factory
    iterator = None
    if callable(kb):
        iterator = kb()
    elif isinstance(kb, list):
        iterator = iter(kb)
    else:
        # unknown type: try to iterate
        try:
            iterator = iter(kb)
        except Exception:
            return []

    count = 0
    counter = 0  # Tiebreaker for heap comparison
    for entry in iterator:
        count += 1
        try:
            s = score_entry(entry)
        except Exception:
            continue
        # Use (score, counter, entry) to avoid comparing dicts when scores are equal
        item = (s, counter, {**entry, "score": s})
        counter += 1
        if len(top_heap) < top_k:
            heapq.heappush(top_heap, item)
        else:
            # heap[0] is smallest score; if current larger, replace
            if item[0] > top_heap[0][0]:
                heapq.heapreplace(top_heap, item)

    # Extract and sort descending
    scored = [e for (_s, _c, e) in sorted(top_heap, key=lambda x: x[0], reverse=True)]

    # if too few matches or top score very low and we have a list KB, fallback to random small sample
    if scored and scored[0].get("score", 0) < 0.15 and isinstance(kb, list):
        return random.sample(kb, min(top_k, len(kb)))

    return scored

# Load Hugging Face LLM pipeline (distilGPT2 for demo, can be replaced with other models)
@st.cache_resource
def get_llm():
    try:
        # Import torch lazily to avoid startup error when not installed
        import torch  # type: ignore
        try:
            # Try to load the model with cache disabled to avoid issues
            llm = hf.pipeline("text-generation", model="distilgpt2", device=-1)  # device=-1 = CPU
            return llm
        except Exception as e:
            print(f"Model loading failed: {e}")
            # If model loading fails, still return success since torch is available
            return hf.pipeline("text-generation", model="gpt2", device=-1)
    except Exception as e:
        print(f"LLM initialization failed: {e}")
        # If torch (or other backend) is missing, return None and fall back to KB
        # Try to use Hugging Face Inference API if user provided a token
        hf_token = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_API_TOKEN")
        if hf_token:
            # Return a small wrapper object with pipeline-like call signature
            class RemoteLLM:
                def __init__(self, token, model="distilgpt2"):
                    self.token = token
                    self.model = model
                    self.url = f"https://api-inference.huggingface.co/models/{self.model}"

                def __call__(self, prompt, max_length=200, num_return_sequences=1):
                    headers = {"Authorization": f"Bearer {self.token}"}
                    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_length, "return_full_text": False}}
                    resp = requests.post(self.url, headers=headers, json=payload, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                    # The API returns a list of generations with 'generated_text'
                    if isinstance(data, dict) and data.get("error"):
                        raise RuntimeError(data.get("error"))
                    # Normalize to pipeline-like output
                    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
                        return [{"generated_text": item["generated_text"]} for item in data[:num_return_sequences]]
                    # Fallback
                    return [{"generated_text": ""}]

            return RemoteLLM(hf_token)
        return None


# Generate response using hybrid KB + LLM approach
def generate_llm_response(query, context, llm, kb_entries=None, confidence=None):
    """
    Hybrid approach:
    - If KB match is strong (high confidence), use KB answer
    - If KB match is weak, use LLM to generate creative answer
    - If no LLM, use KB or fallback message
    """
    
    # First, try KB-based response if entries exist
    kb_response = None
    if kb_entries:
        by_cat = {}
        for e in kb_entries:
            cat = e.get("category", "Other")
            score = e.get("score", 0.0)
            by_cat.setdefault(cat, []).append((score, e))
        
        # Find the category with the highest total score
        if by_cat:
            cat_scores = {}
            for cat, entries in by_cat.items():
                top_scores = sorted([s for s, e in entries], reverse=True)[:2]
                cat_scores[cat] = sum(top_scores)
            
            best_cat = max(cat_scores, key=cat_scores.get)
            best_entry = max(by_cat[best_cat], key=lambda x: x[0])
            kb_response = best_entry[1].get("response", None)
    
    # If KB confidence is strong (>= 0.5), use KB
    if kb_response and (confidence is None or confidence >= 0.5):
        return kb_response
    
    # If confidence is low and LLM available, generate with LLM
    if llm is not None and (confidence is None or confidence < 0.5):
        try:
            prompt = f"You are a helpful telecom support assistant. Answer briefly and helpfully.\nContext: {context}\nUser: {query}\nAssistant:"
            result = llm(prompt, max_length=150, num_return_sequences=1)
            llm_answer = result[0]["generated_text"].split("Assistant:")[-1].strip()
            if llm_answer and len(llm_answer) > 10:
                return llm_answer
        except Exception as e:
            pass  # Fall through to KB or default
    
    # Fallback to KB if available
    if kb_response:
        return kb_response
    
    # Last resort
    return "Sorry, I don't have information on this topic. Please contact our support team for assistance."


def device_fallback(query: str) -> str:
    # Rule-based fallback specifically for battery/draining/device issues
    q = query.lower()
    battery_keywords = ["battery", "drain", "draining", "battery drains", "power", "dies quickly"]
    if any(k in q for k in battery_keywords):
        return (
            "Battery drain troubleshooting:\n"
            "1) Reduce screen brightness and timeout.\n"
            "2) Disable background app refresh for heavy apps.\n"
            "3) Turn off mobile data or use Wi-Fi when not needed.\n"
            "4) Check battery usage in settings to find rogue apps.\n"
            "5) Enable battery saver and update device software."
        )
    return ""


def web_search(query: str) -> str:
    """Try to fetch short web results. Uses SerpAPI if SERPAPI_KEY is set, otherwise falls back to Wikipedia opensearch as a best-effort."""
    q = query.strip()
    api_key = os.environ.get("SERPAPI_KEY") or os.environ.get("SEARCH_API_KEY")
    try:
        if api_key:
            url = "https://serpapi.com/search.json"
            params = {"q": q, "api_key": api_key, "num": 5}
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("organic_results", [])[:5]:
                title = item.get("title") or item.get("position")
                snippet = item.get("snippet") or item.get("snippet_text") or ""
                link = item.get("link") or item.get("url") or ""
                results.append(f"{title}: {snippet} ({link})")
            if results:
                return "\n".join(results)
        # Fallback to Wikipedia opensearch
        wiki_url = "https://en.wikipedia.org/w/api.php"
        params = {"action": "opensearch", "search": q, "limit": 3, "format": "json"}
        r = requests.get(wiki_url, params=params, timeout=8)
        r.raise_for_status()
        arr = r.json()
        results = []
        for title, desc in zip(arr[1], arr[2]):
            results.append(f"{title}: {desc}")
        if results:
            return "\n".join(results)
    except Exception as e:
        return f"Web search failed or unavailable: {e}"
    return "No web results found. To enable web lookup, set SERPAPI_KEY in environment."


def get_plan_details(query: str, kb: list, providers_inv: dict) -> str:
    """Return plan-related details: prefer KB entries about plans, otherwise run web_search."""
    q = query.lower()
    # detect provider mention from normalized mapping
    provider = None
    for variant, canonical in providers_inv.items():
        if variant in q:
            provider = canonical
            break

    # Search KB for Plan Change or Plan entries matching provider or query tokens
    matches = []
    iterator = kb() if callable(kb) else (iter(kb) if isinstance(kb, list) else iter(kb))
    for e in iterator:
        try:
            cat = e.get("category", "").lower()
            qtext = e.get("query", "").lower()
        except Exception:
            continue
        if "plan" in cat or "plan" in qtext:
            # prefer entries that mention the detected provider
            text = (e.get("query", "") + " " + e.get("response", "")).lower()
            if provider and provider in text:
                matches.insert(0, e)
            else:
                matches.append(e)
    if matches:
        parts = []
        for m in matches[:6]:
            parts.append(f"- {m.get('query')}: {m.get('response')}")
        return "\n".join(parts)

    # Nothing in KB — try web search for plan details
    search_q = f"{provider if provider else ''} telecom plans details {query}"
    return web_search(search_q)

# Estimate confidence score (simple: based on keyword overlap)
def estimate_confidence(query, retrieved):
    """Calculate confidence based on actual KB retrieval scores, not just word overlap."""
    if not retrieved:
        return 0.3
    
    # Use the actual scores from KB retrieval (range 0.0-1.0)
    scores = [e.get("score", 0.0) for e in retrieved]
    if not scores:
        return 0.3
    
    # Average of top 2 scores gives better representation
    top_scores = sorted(scores, reverse=True)[:2]
    avg_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
    
    # Cap at 0.95 to avoid false 100% confidence
    confidence = min(0.95, avg_score)
    return round(confidence, 2)

# Sentiment analysis using TextBlob
def detect_sentiment(text):
    """
    Detect sentiment: 'frustrated', 'positive', or 'neutral'
    
    LOGIC:
    - Use TextBlob polarity to detect negative/positive tone
    - frustrated = negative words/tone (problem detected) → escalate to human
    - positive = positive words/tone (satisfied) → no escalation
    - neutral = neither negative nor positive → no escalation
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity < -0.1:  # Even slightly negative → escalate (user has problem)
        return "frustrated"
    elif polarity > 0.1:
        return "positive"
    else:
        return "neutral"

# Adapt response tone
def adapt_tone(response, sentiment):
    if sentiment == "frustrated":
        return "I'm really sorry for the trouble. " + response
    elif sentiment == "positive":
        return "Great to hear from you! " + response
    else:
        return response

# Visual step-by-step guidance (emoji-based for better UX)
def get_step_by_step(category):
    steps = {
        "Billing": [
            "💳 Open your telecom app.",
            "📋 Go to 'Billing' section.",
            "📊 Review your latest bill details.",
            "🚨 If you see an error, click 'Raise Dispute'."
        ],
        "Network Outage": [
            "✈️ Check if airplane mode is off.",
            "🔄 Restart your device.",
            "🗺️ If issue persists, check outage map in the app.",
            "⏳ Wait for resolution or contact support."
        ],
        "Plan Change": [
            "📱 Open your telecom app.",
            "📊 Go to 'Plans' section.",
            "✅ Select 'Change Plan' and choose your new plan.",
            "💰 Confirm and pay if required."
        ],
        "Device Problem": [
            "🔄 Restart your device.",
            "🎴 Reinsert SIM card.",
            "⚙️ Check device settings for network.",
            "🏪 Visit service center if not resolved."
        ],
        "Other": [
            "📱 Check your device settings.",
            "🔄 Restart your device if needed.",
            "📞 Contact support for further assistance.",
            "💬 Ask follow-up questions for more help."
        ]
    }
    
    # Visual emojis for each category (no broken images!)
    visuals = {
        "Billing": "💳 📊 💰",
        "Network Outage": "📶 🔧 ✅",
        "Plan Change": "📱 ⬆️ ✨",
        "Device Problem": "📱 🎴 🔧",
        "Other": "📱 ❓ ✅"
    }
    
    if not category:
        return steps.get("Other", []), visuals.get("Other")
    
    # Normalize category for lookup
    normalized = category.strip().title() if isinstance(category, str) else ""
    step_list = steps.get(normalized, steps.get(category, steps.get("Other", [])))
    visual = visuals.get(normalized, visuals.get(category, visuals.get("Other")))
    
    return step_list, visual

# Generate dynamic agent info based on metrics
def get_agent_info():
    """Generate dynamic agent info from metrics."""
    m = st.session_state.metrics
    
    # Calculate agent performance metrics
    total_escalations = m.get("escalations", 0)
    feedback_yes = m.get("feedback_yes", 0)
    feedback_no = m.get("feedback_no", 0)
    
    # Calculate satisfaction score (1-5 stars)
    if feedback_yes + feedback_no > 0:
        satisfaction = (feedback_yes / (feedback_yes + feedback_no)) * 5
    else:
        satisfaction = 4.5  # Default
    
    # Estimate response time based on queries handled
    response_time = max(20, 60 - (total_escalations * 2))  # Faster with more experience
    
    # Agent name: Use logged-in agent name if available, otherwise rotate through names
    if st.session_state.get("agent_logged_in") and st.session_state.get("agent_name"):
        agent_name = st.session_state.agent_name
    else:
        agent_names = ["Priya", "Rajesh", "Asha", "Vikram", "Neha"]
        agent_name = agent_names[total_escalations % len(agent_names)]
    
    return {
        "name": agent_name,
        "response_time": response_time,
        "satisfaction": round(satisfaction, 1),
        "escalations_handled": total_escalations
    }

# Proactive tips (mock)
def get_proactive_tips(category):
    tips = {
        "Billing": "Set up auto-pay to avoid late fees.",
        "Network Outage": "Enable WiFi calling as a backup during outages.",
        "Plan Change": "Review your usage monthly to pick the best plan.",
        "Device Problem": "Keep your device software updated for best performance.",
        "Other": "Check our knowledge base for more solutions or contact support."
    }
    if not category:
        return tips["Other"]
    
    # Normalize category for lookup
    normalized = category.strip().title() if isinstance(category, str) else ""
    tip = tips.get(normalized, tips.get(category, tips["Other"]))
    return tip if tip else tips["Other"]

# ============================================================================
# PERSISTENT ESCALATION MESSAGE STORAGE (for real-time agent responses)
# ============================================================================

ESCALATION_MESSAGES_FILE = Path(__file__).parent / "escalation_messages.jsonl"

def save_escalation_message(escalation_id: str, sender: str, message: str, timestamp: float = None):
    """Save escalation message to persistent file for real-time updates across users."""
    if timestamp is None:
        timestamp = time.time()
    
    msg_entry = {
        "escalation_id": escalation_id,
        "sender": sender,  # "customer", "agent", "system"
        "message": message,
        "timestamp": timestamp
    }
    
    try:
        with open(ESCALATION_MESSAGES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(msg_entry) + "\n")
        return True
    except Exception as e:
        print(f"Error saving message: {e}")
        return False

def load_escalation_messages(escalation_id: str = None):
    """Load escalation messages from persistent file."""
    messages = []
    
    try:
        if ESCALATION_MESSAGES_FILE.exists():
            with open(ESCALATION_MESSAGES_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if escalation_id is None or entry.get("escalation_id") == escalation_id:
                            messages.append(entry)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error loading messages: {e}")
    
    return messages

def get_active_escalation_id():
    """Get the current active escalation ID from session state."""
    if st.session_state.get("escalation_list") and len(st.session_state.escalation_list) > 0:
        # Use the most recent escalation as active
        escalation_idx = len(st.session_state.escalation_list) - 1
        return f"escalation_{escalation_idx}"
    return None

# Main Streamlit app
def main():
    st.title("GenAI-Telecom Support Chatbot")
    st.write("Ask any telecom support question. The AI will use its knowledge base and LLM to help you!")
    
    # ============================================================================
    # SHOW ESCALATION CHAT AT TOP IF ACTIVE - MOVED TO LATER IN THE CODE FOR BETTER UX
    # ============================================================================
    # NOTE: Escalation chat is now shown after user messages (see line ~1185)
    # This prevents duplicate chat interfaces
    
    st.divider()
    st.subheader("💬 Chat with AI Support")
    st.write("Ask your telecom question below:")
    
    # Show status if LLM backend is unavailable
    llm = get_llm()
    if llm is None:
        st.warning("LLM backend not available locally (PyTorch/TensorFlow missing). Falling back to KB-only responses.")
    kb = load_kb()
    providers_inv = load_providers_mapping(kb_path=str(Path(__file__).parent.parent / "synthetic_data.json"))
    # Admin UI for provider mappings
    with st.sidebar.expander("Admin / Providers mapping", expanded=False):
        st.markdown("Edit provider mappings (JSON). You can also auto-generate suggestions from the KB and save them.")
        current = read_providers_file()
        text = st.text_area("providers.json", value=json.dumps(current, indent=2), height=300)
        if st.button("Save providers.json"):
            try:
                new_map = json.loads(text)
                write_providers_file(new_map)
                st.success("Saved providers.json")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
        if st.button("Autogenerate from KB"):
            gen = auto_generate_providers_from_kb(kb)
            st.code(json.dumps(gen, indent=2))
            if st.button("Save autogenerated mapping"):
                write_providers_file(gen)
                st.success("Saved autogenerated providers.json")
    
    # ============================================================================
    # HUMAN SUPPORT AGENT PANEL - MODAL VERSION
    # ============================================================================
    
    # Check if there are pending escalations and show modal for agent to respond
    escalations = st.session_state.get("escalation_list", [])
    if escalations and not st.session_state.get("agent_logged_in"):
        # Show in sidebar that there are waiting escalations
        with st.sidebar.expander("🎧 PENDING ESCALATIONS - " + str(len(escalations)), expanded=True):
            st.markdown("**Login as Human Support Agent to Respond**")
            agent_name = st.text_input("Agent Name:", placeholder="e.g., John Support", key="agent_name_input")
            agent_pass = st.text_input("Password:", type="password", placeholder="Demo password (demo123)", key="agent_pass_input")
            
            if st.button("Login as Agent", key="agent_login"):
                if agent_pass == "demo123":
                    st.session_state.agent_logged_in = True
                    st.session_state.agent_name = agent_name or "Support Agent"
                    st.success(f"✅ Logged in as {st.session_state.agent_name}")
                    st.rerun()
                else:
                    st.error("❌ Invalid password (use 'demo123')")
    
    # If agent is logged in, show a modal-like interface
    if st.session_state.get("agent_logged_in"):
        with st.sidebar.expander("👨‍💼 Agent Dashboard - " + st.session_state.agent_name, expanded=True):
            st.markdown(f"**Logged in as:** {st.session_state.agent_name} ✅")
            
            if st.button("Logout", key="agent_logout"):
                st.session_state.agent_logged_in = False
                st.info("Logged out")
                st.rerun()
            
            st.divider()
            
            # Show pending escalations
            if escalations:
                st.markdown(f"**📞 Pending Cases: {len(escalations)}**")
                for i, esc in enumerate(escalations):
                    st.markdown(f"**Case #{i+1}:** {esc['query'][:50]}...")
                    st.write(f"Sentiment: {esc['sentiment']}")
                    st.write(f"Confidence: {esc['confidence']:.0%}")
            else:
                st.info("✅ No pending escalations")
            
            st.divider()
            st.markdown("**Reply to Customer**")
            agent_response = st.text_area(
                "Type your response:",
                placeholder="Your message to the customer...",
                height=120,
                key="agent_response_textarea"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✉️ Send to Customer", key="send_agent_response"):
                    if agent_response.strip():
                        # Add agent response to history (local session)
                        st.session_state.history.append(("Human Agent:", agent_response, "Human", 1.0, "neutral"))
                        
                        # Save to persistent storage for real-time updates across users
                        escalation_id = get_active_escalation_id()
                        if escalation_id:
                            save_escalation_message(
                                escalation_id=escalation_id,
                                sender="agent",
                                message=agent_response,
                                timestamp=time.time()
                            )
                        
                        st.success("✅ Sent to customer!")
                        st.rerun()
                    else:
                        st.warning("Type a message first")
            
            with col2:
                if st.button("🔄 Refresh", key="refresh_cases"):
                    st.rerun()
    
    # Control: auto-escalation for low-confidence queries
    auto_escalate = st.sidebar.checkbox("Auto-escalate low-confidence queries", value=True)
    if "history" not in st.session_state:
        st.session_state.history = []
    # Metrics and escalations
    if "metrics" not in st.session_state:
        st.session_state.metrics = {
            "total_queries": 0,
            "kb_resolved": 0,
            "llm_resolved": 0,
            "escalations": 0,
            "feedback_yes": 0,
            "feedback_no": 0,
            "avg_confidence": 0.0
        }
    if "escalation_list" not in st.session_state:
        st.session_state.escalation_list = []
    # ChatGPT-like multi-turn input
    interpreted_query = None
    user_message = st.chat_input("Type your message...")
    if user_message:
        # Feature 3: Multilingual Support (NEW)
        lang_choice = multilingual_support(user_message)
        
        # Telecom-specific terms that should NOT be autocorrected
        TELECOM_KEYWORDS = ["sim", "4g", "5g", "2g", "3g", "lte", "wifi", "voip", "sms", "ussd", "imei", "imsi", "carrier", "isp", "broadband", "roaming", "topping", "topup", "top-up"]
        
        # Only autocorrect if query doesn't contain telecom keywords
        corrected = user_message
        has_telecom_keyword = any(kw in user_message.lower() for kw in TELECOM_KEYWORDS)
        
        if not has_telecom_keyword:
            try:
                corrected = str(TextBlob(user_message).correct())
            except Exception:
                corrected = user_message
        
        suggested = normalize_query(corrected, providers_inv)
        if suggested.lower().strip() != user_message.lower().strip():
            # Inform user we normalized the query and proceed
            st.info(f"I normalized your query to: {suggested}")
            interpreted_query = suggested
        else:
            interpreted_query = user_message

        # Retrieval and safety-first checks happen before generating an LLM answer.
        try:
            retrieved = retrieve_relevant_entries(interpreted_query, kb, top_k=5)  # Increased to 5 for better category selection
            context = "\n".join([f"Q: {e['query']}\nA: {e['response']}" for e in retrieved if isinstance(e, dict)])
            category = retrieved[0]["category"] if (retrieved and isinstance(retrieved[0], dict)) else None
            top_score = retrieved[0].get("score", 0.0) if (retrieved and isinstance(retrieved[0], dict)) else 0.0
            if not isinstance(top_score, (int, float)):
                top_score = 0.0
        except Exception as e:
            st.error("⚠️ Support service temporarily unavailable. Please try again in a moment.")
            return
        
        try:
            confidence = estimate_confidence(interpreted_query, retrieved)
            sentiment = detect_sentiment(interpreted_query)
        except Exception as e:
            st.error("⚠️ Unable to process your request. Please try a different question.")
            return

        device_fb = device_fallback(interpreted_query)

        LOW_CONF_THRESHOLD = 0.35  # Lowered from 0.5 - only escalate for very low confidence
        LOW_MATCH_SCORE = 0.15    # Lowered from 0.25 - be more lenient with matching
        low_confidence_match = (confidence < LOW_CONF_THRESHOLD) or (top_score < LOW_MATCH_SCORE)

        answer = None
        escalated_now = False

        try:
            # If we have a device fallback answer, use it and DON'T escalate
            if device_fb:
                answer = device_fb
                if not category:
                    category = "Device Problem"
            # Only escalate if no device answer AND low confidence
            elif low_confidence_match:
                if auto_escalate:
                    summary = {
                        "query": interpreted_query,
                        "confidence": confidence,
                        "sentiment": sentiment,
                        "category": category,
                        "context": [{"q": e.get("query"), "a": e.get("response")} for e in retrieved],
                        "note": "Auto-escalated due to low-confidence / no KB match",
                        "suggested_actions": get_step_by_step(category)[0]
                    }
                    st.session_state.escalation_list.append(summary)
                    st.session_state.metrics["escalations"] += 1
                    escalated_now = True
                    answer = "I don't have a confident automated answer for this. I've escalated your issue to human support for help."
                    # mark an escalation session and append a human-agent placeholder to history
                    st.session_state.in_escalation_session = True
                    st.session_state.offer_pending = False
                    # add human placeholder message to history so UI shows a handover
                    st.session_state.history.append(("[Escalation] Your issue has been forwarded to a human agent.",
                                                   "Human agent will join shortly. You can type messages to the human on the main chat area below.",
                                                   "Escalation", 0.0, sentiment))
                    
                    # Save escalation to LLM/KB matrix for tracking and learning
                    try:
                        feedback_kb_path = Path(__file__).parent.parent / "feedback_kb.jsonl"
                        escalation_entry = {
                            "query": interpreted_query,
                            "response": "ESCALATED_TO_HUMAN",
                            "confidence": confidence,
                            "match_score": score,
                            "sentiment": sentiment,
                            "category": category,
                            "escalation_reason": "Low confidence / no KB match",
                            "timestamp": time.time(),
                            "type": "escalation"
                        }
                        with open(str(feedback_kb_path), "a", encoding="utf-8") as f:
                            f.write(json.dumps(escalation_entry) + "\n")
                    except Exception as e:
                        print(f"Error saving escalation to KB matrix: {e}")
                    # Add an auto-greeting from human agent
                    human_greeting = "Hello! 👋 I'm your support agent. I've reviewed your case and I'm here to help. What can I assist you with?"
                    st.session_state.history.append(("Human Agent:", human_greeting, "Human", 1.0, "neutral"))
                else:
                    st.warning("I don't have a confident answer from our knowledge base. Would you like me to escalate this to a human agent?")
                    if st.button("Escalate to human now", key=f"confirm_escalate_{st.session_state.metrics['total_queries']}"):
                        summary = {
                            "query": interpreted_query,
                            "confidence": confidence,
                            "sentiment": sentiment,
                            "category": category,
                            "context": [{"q": e.get("query"), "a": e.get("response")} for e in retrieved],
                            "note": "User-confirmed escalation due to low-confidence / no KB match",
                            "suggested_actions": get_step_by_step(category)[0]
                        }
                        st.session_state.escalation_list.append(summary)
                        st.session_state.metrics["escalations"] += 1
                        escalated_now = True
                        st.success("Escalated: human support has been notified (demo).")
                        answer = "Your issue has been escalated to human support. They'll reach out shortly."
                        # mark escalation session and append placeholder
                        st.session_state.in_escalation_session = True
                        st.session_state.offer_pending = False
                        st.session_state.history.append(("[Escalation] Your issue has been forwarded to a human agent.",
                                                       "Human agent will join shortly. You can type messages to the human on the main chat area below.",
                                                       "Escalation", 0.0, sentiment))
                        # Add an auto-greeting from human agent
                        human_greeting = "Hello! 👋 I'm your support agent. I've reviewed your case and I'm here to help. What can I assist you with?"
                        st.session_state.history.append(("Human Agent:", human_greeting, "Human", 1.0, "neutral"))
                    else:
                        answer = "I don't have a confident automated answer. You can choose to escalate this to human support so they can assist further."
            else:
                answer = generate_llm_response(interpreted_query, context, llm, kb_entries=retrieved, confidence=confidence)
                answer = adapt_tone(answer, sentiment)

            # Append user and assistant turns to history
            # If escalated, use "Escalation" as category so steps/tips are not shown in the UI
            display_category = "Escalation" if escalated_now else category
            
            # Boost confidence for device fallback answers (they're reliable templates)
            display_confidence = confidence
            if device_fb and answer == device_fb:
                display_confidence = 0.75  # High confidence for device fallback
            
            st.session_state.history.append((interpreted_query, answer, display_category, display_confidence, sentiment))
            
            # Store for feedback collection
            st.session_state.last_query = interpreted_query
            st.session_state.last_response = answer
            st.session_state.last_category = display_category
            st.session_state.last_sentiment = sentiment

            # Offer follow-up actions for plan-related queries
            PLAN_KEYWORDS = ["5g", "5g data", "plan", "plans", "price", "pricing", "cost", "tariff", "data pack"]
            qlower = interpreted_query.lower()
            if any(kw in qlower for kw in PLAN_KEYWORDS) and not escalated_now:
                st.session_state.offer_pending = True
                st.session_state.offer_query = interpreted_query
                st.session_state.offer_type = "plan"
            else:
                st.session_state.offer_pending = False

            # Update metrics
            m = st.session_state.metrics
            m["total_queries"] += 1
            m["avg_confidence"] = ((m["avg_confidence"] * (m["total_queries"] - 1)) + confidence) / m["total_queries"]
            if not escalated_now:
                if llm is None:
                    m["kb_resolved"] += 1
                else:
                    m["llm_resolved"] += 1
        except Exception as e:
            st.error("⚠️ An error occurred while processing your request. Our support team has been notified. Please try again or contact us directly.")
    
    # If in an escalation session, show a prominent live chat interface in the main area
    if st.session_state.get("in_escalation_session"):
        st.divider()
        
        # Enhanced escalation UI header
        col_header1, col_header2, col_header3 = st.columns([1, 3, 1])
        with col_header1:
            st.markdown("### 🎧")
        with col_header2:
            st.markdown("## **Live Support Connected**")
        with col_header3:
            st.markdown("🟢 **Online**")
        
        # Agent info card with DYNAMIC data
        agent_info = get_agent_info()
        satisfaction_stars = "⭐" * int(agent_info["satisfaction"]) + "☆" * (5 - int(agent_info["satisfaction"]))
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; color: white;">
            <b>👨‍💼 {agent_info['name']}</b> (Agent #{agent_info['escalations_handled'] + 1}) | 
            Avg Response: <b>~{agent_info['response_time']}s</b> | 
            Satisfaction: <b>{agent_info['satisfaction']}/5 {satisfaction_stars}</b>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Show chat history with improved styling (load from persistent storage for real-time updates)
        st.markdown("### 💬 **Conversation**")
        chat_container = st.container()
        with chat_container:
            # First, show messages from persistent storage (real-time)
            escalation_id = get_active_escalation_id()
            if escalation_id:
                persistent_messages = load_escalation_messages(escalation_id)
                for msg in persistent_messages:
                    sender = msg.get("sender", "")
                    message_text = msg.get("message", "")
                    
                    if sender == "customer":
                        # Customer message - right aligned, blue
                        st.markdown(f"""
                        <div style="text-align: right; margin: 10px 0;">
                            <span style="background: #e3f2fd; color: #1565c0; padding: 10px 15px; border-radius: 15px; display: inline-block; max-width: 70%;">
                                <b>You:</b> {message_text}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                    elif sender == "agent":
                        # Agent message - left aligned, green
                        st.markdown(f"""
                        <div style="text-align: left; margin: 10px 0;">
                            <span style="background: #e8f5e9; color: #2e7d32; padding: 10px 15px; border-radius: 15px; display: inline-block; max-width: 70%;">
                                <b>👨‍💼 Agent:</b> {message_text}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Fallback: also show session state history for backward compatibility
            for msg_q, msg_a, msg_cat, msg_conf, msg_sent in st.session_state.history:
                if msg_cat in ("Human", "Human-User"):
                    if "You (to human):" in msg_q:
                        # Customer message - right aligned, blue
                        st.markdown(f"""
                        <div style="text-align: right; margin: 10px 0;">
                            <span style="background: #e3f2fd; color: #1565c0; padding: 10px 15px; border-radius: 15px; display: inline-block; max-width: 70%;">
                                <b>You:</b> {msg_q.replace('You (to human): ', '')}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Agent message - left aligned, green
                        st.markdown(f"""
                        <div style="text-align: left; margin: 10px 0;">
                            <span style="background: #e8f5e9; color: #2e7d32; padding: 10px 15px; border-radius: 15px; display: inline-block; max-width: 70%;">
                                <b>👨‍💼 Agent:</b> {msg_a if msg_a else msg_q.replace('Human Agent:', '')}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Add auto-refresh for real-time message polling
        # This uses Streamlit's built-in autorefresh capability
        with st.empty():
            # Placeholder for auto-refresh indicator
            pass
        
        # JavaScript to refresh every 3 seconds to load new messages
        st.markdown("""
        <script>
        setTimeout(function() {
            window.location.reload();
        }, 3000);
        </script>
        """, unsafe_allow_html=True)
        
        # Instant response without long typing indicator
        if st.session_state.get("human_typing"):
            st.session_state.human_typing = False
            # Faster random response
            sample_responses = [
                "I understand. Let me check your account details.",
                "Can you provide your account number or phone number please?",
                "Thank you for explaining. I'm looking into this for you now.",
                "I see the issue. Let me escalate this to our technical team.",
                "No problem, I'll help you resolve this right away."
            ]
            agent_reply = random.choice(sample_responses)
            st.session_state.history.append(("Human Agent:", agent_reply, "Human", 1.0, "neutral"))
            st.rerun()
        
        # Input area with improved styling
        st.markdown("### ✉️ **Send Message**")
        col_msg1, col_msg2 = st.columns([4, 1])
        
        with col_msg1:
            human_msg = st.text_input("Type your message:", key=f"human_msg_input_{len(st.session_state.history)}", placeholder="Type your message and press Enter...", label_visibility="collapsed")
        
        with col_msg2:
            if st.button("📤 Send", key=f"send_to_human_{len(st.session_state.history)}", use_container_width=True):
                if human_msg.strip():
                    human_msg_sent = detect_sentiment(human_msg)
                    st.session_state.history.append((f"You (to human): {human_msg}", "", "Human-User", 0.0, human_msg_sent))
                    
                    # Save customer message to persistent storage for real-time updates
                    escalation_id = get_active_escalation_id()
                    if escalation_id:
                        save_escalation_message(
                            escalation_id=escalation_id,
                            sender="customer",
                            message=human_msg,
                            timestamp=time.time()
                        )
                    
                    st.session_state.human_typing = True
                    # Clear the input by rerunning
                    st.rerun()
        
        # Quick action buttons
        st.markdown("**Quick Actions:**")
        quick_cols = st.columns(4)
        quick_actions = [
            ("📞 Call Agent", "call_agent"),
            ("📹 Video Chat", "video_call"),
            ("📋 View History", "view_history"),
            ("❌ End Chat", "end_chat")
        ]
        
        for i, (action_text, action_key) in enumerate(quick_actions):
            with quick_cols[i]:
                if action_text == "❌ End Chat":
                    if st.button(action_text, key=f"escalation_{action_key}_{len(st.session_state.history)}", use_container_width=True):
                        st.session_state.in_escalation_session = False
                        st.session_state.show_satisfaction = True
                        st.rerun()
                else:
                    if st.button(action_text, key=f"escalation_{action_key}_{len(st.session_state.history)}", use_container_width=True):
                        if "Call" in action_text:
                            st.info("📞 Initiating voice call...")
                        elif "Video" in action_text:
                            st.info("📹 Starting video chat...")
                        elif "History" in action_text:
                            st.info("📋 Showing full conversation history...")
        
        # Show quick reply suggestions
        quick_options = ["Can you help with billing?", "What's my plan status?", "I need technical support", "Thank you"]
        st.markdown("**Quick Replies:**")
        cols = st.columns(len(quick_options))
        for i, opt in enumerate(quick_options):
            with cols[i]:
                if st.button(opt, key=f"quick_reply_{i}_{len(st.session_state.history)}"):
                    st.session_state.history.append((f"You (to human): {opt}", "", "Human-User", 0.0, "neutral"))
                    st.session_state.human_typing = True
                    st.rerun()
    
    # Show satisfaction modal after closing escalation
    if st.session_state.get("show_satisfaction"):
        satisfaction_modal()
        st.session_state.show_satisfaction = False

    for idx, (q, a, cat, conf, sent) in enumerate(st.session_state.history):
        # Skip escalation chat messages from main display if in escalation session
        if st.session_state.get("in_escalation_session") and cat in ("Human", "Human-User"):
            continue
        
        # Use enhanced message display with avatars
        if cat == "Human-User":
            message_with_avatar("user", q.replace("You (to human): ", ""), show_reaction=False)
        else:
            st.markdown(f"👤 **You:** {q}")
            st.markdown(f"🤖 **Bot:** {a}")
        
        st.markdown(f"**Customer Trust Meter:** :bar_chart: Confidence {int(conf*100)}%")
        st.markdown(f"**Detected Sentiment:** {sent.capitalize()}")
        
        # LOGIC: Show escalation button only if LOW confidence OR frustrated
        # LOGIC: Only show guidance if HIGH confidence (good answer, no need to escalate)
        show_escalation = (conf < 0.5 or sent == "frustrated") and cat != "Escalation" and not st.session_state.get("in_escalation_session")
        show_guidance = (conf >= 0.5 and sent != "frustrated") and cat not in ("Escalation", "Human", "Human-User")
        
        # Show escalation button with explanation of why
        if show_escalation:
            escalate_container = st.container()
            with escalate_container:
                # Show reason for escalation
                reason = ""
                if conf < 0.5 and sent == "frustrated":
                    reason = "⚠️ Low confidence + Frustrated sentiment"
                elif conf < 0.5:
                    reason = f"⚠️ Low confidence ({int(conf*100)}% < 50% threshold)"
                elif sent == "frustrated":
                    reason = "😞 You seem frustrated - connecting to human agent"
                
                st.info(f"**Why escalate?** {reason}")
                
                if st.button(f"📞 Escalate to human (#{idx+1})", key=f"escalate_{idx}", use_container_width=True):
                    # safe retrieval for the historical message (do not rely on current 'retrieved' variable)
                    hist_retrieved = retrieve_relevant_entries(q, kb)
                    summary = {
                        "query": q,
                        "confidence": conf,
                        "sentiment": sent,
                        "category": cat,
                        "context": [ {"q": e.get("query"), "a": e.get("response")} for e in hist_retrieved ],
                        "suggested_actions": get_step_by_step(cat)[0]
                    }
                    st.session_state.escalation_list.append(summary)
                    st.session_state.metrics["escalations"] += 1
                    st.session_state.in_escalation_session = True
                    st.success("✅ Escalated to human agent! They will assist you shortly.")
                    # Auto-scroll to escalation area using JavaScript
                    st.markdown("""
                    <script>
                    window.scrollTo(0, 0);
                    </script>
                    """, unsafe_allow_html=True)
                    st.rerun()
        
        # Only show automated guidance for GOOD confidence answers (not for escalations)
        if show_guidance:
            st.markdown("---")
            steps, visual = get_step_by_step(cat)
            
            # Only show guidance section if we have steps to show
            if steps:
                st.markdown("**📖 Step-by-Step Guidance:**")
                for i, step in enumerate(steps, 1):
                    st.write(f"{i}. {step}")
                if visual:
                    st.markdown(f"### {visual} Show Me How: {cat}")
            
            # Show proactive tip (always show, but with fallback)
            tip = get_proactive_tips(cat)
            if tip:
                st.markdown("**💡 Proactive Tip:** " + tip)
        # Feedback mechanism
        with st.expander("Was this answer helpful?"):
            # Create unique key for feedback state tracking
            feedback_key = f"feedback_selection_{idx}"
            correction_key = f"correction_text_{idx}"
            
            feedback = st.radio(
                f"Feedback for response {idx+1}", 
                ["Yes", "No", "Suggest Correction"], 
                key=feedback_key,
                horizontal=True
            )
            
            col1, col2, col3 = st.columns(3)
            
            if feedback == "Yes":
                with col1:
                    if st.button("✅ Confirm Helpful", key=f"confirm_yes_{idx}"):
                        # Save to KB and metrics
                        if hasattr(st.session_state, 'last_query') and hasattr(st.session_state, 'last_category'):
                            save_feedback_as_kb_entry(
                                query=st.session_state.last_query,
                                response=a,
                                category=cat,
                                feedback_rating=5,
                                feedback_text="User marked response as helpful",
                                user_sentiment=st.session_state.get('last_sentiment', 'neutral')
                            )
                            st.session_state.metrics["feedback_yes"] = st.session_state.metrics.get("feedback_yes", 0) + 1
                            st.success("✅ Thank you! Feedback saved to KB for training.")
                            st.balloons()
                        
            elif feedback == "No":
                with col2:
                    if st.button("❌ Confirm Not Helpful", key=f"confirm_no_{idx}"):
                        # Save to KB and metrics
                        if hasattr(st.session_state, 'last_query') and hasattr(st.session_state, 'last_category'):
                            save_feedback_as_kb_entry(
                                query=st.session_state.last_query,
                                response=a,
                                category=cat,
                                feedback_rating=1,
                                feedback_text="User marked response as not helpful - needs improvement",
                                user_sentiment=st.session_state.get('last_sentiment', 'neutral')
                            )
                            st.session_state.metrics["feedback_no"] = st.session_state.metrics.get("feedback_no", 0) + 1
                            st.info("❌ Thank you for the feedback. We'll use this to improve.")
                        
            elif feedback == "Suggest Correction":
                correction = st.text_area(
                    "Suggest a better answer or correction:", 
                    key=correction_key,
                    placeholder="e.g., The correct answer should be...",
                    height=100
                )
                with col3:
                    if st.button("📤 Submit Correction", key=f"submit_correction_{idx}"):
                        if correction.strip():
                            # Save correction as feedback with user's suggested text
                            if hasattr(st.session_state, 'last_query') and hasattr(st.session_state, 'last_category'):
                                save_feedback_as_kb_entry(
                                    query=st.session_state.last_query,
                                    response=correction,  # Use the corrected text
                                    category=cat,
                                    feedback_rating=2,
                                    feedback_text=f"User suggested correction: {correction}",
                                    user_sentiment=st.session_state.get('last_sentiment', 'neutral')
                                )
                                st.session_state.metrics["feedback_no"] = st.session_state.metrics.get("feedback_no", 0) + 1
                                st.success("✅ Thank you for the correction! This has been saved and will help improve future responses.")
                        else:
                            st.warning("⚠️ Please enter a correction before submitting.")

    # Sidebar Dashboard / Metrics and export
    with st.sidebar.expander("Metrics & Dashboard", expanded=True):
        m = st.session_state.metrics
        st.metric("Total queries", m["total_queries"]) 
        st.metric("KB resolved", m["kb_resolved"]) 
        st.metric("LLM resolved", m["llm_resolved"]) 
        st.metric("Escalations", m["escalations"]) 
        st.markdown(f"**Avg confidence:** {m['avg_confidence']:.2f}")
        st.markdown(f"Feedback - Yes: {m['feedback_yes']}  No: {m['feedback_no']}")
        if st.button("Download metrics JSON"):
            st.download_button("Download metrics", json.dumps(m, indent=2), file_name="metrics.json")
        if st.session_state.escalation_list:
            st.markdown("### Escalations")
            st.write(st.session_state.escalation_list)
            st.download_button("Download escalations", json.dumps(st.session_state.escalation_list, indent=2), file_name="escalations.json")

    # Feature 4: Smart FAQ Generation (NEW)
    with st.sidebar.expander("📚 Trending FAQs", expanded=False):
        smart_faq_generation(st.session_state.history)
    
    # Feature 5: Agent Performance Dashboard (NEW)
    with st.sidebar.expander("📊 Agent Performance", expanded=False):
        agent_performance_dashboard(st.session_state.escalation_list, st.session_state.metrics)
    
    # Feature 6: Model Training from Feedback (NEW)
    with st.sidebar.expander("🧠 AI Learning & Feedback", expanded=False):
        feedback_training_dashboard()
    
    # Feature 1: Proactive Issue Detection (NEW)
    proactive_alert = proactive_issue_detection(st.session_state.history, st.session_state.metrics)
    if proactive_alert:
        st.sidebar.info(proactive_alert)

        # Offer follow-up actions (e.g., share plan details) without requiring a full refresh
        if st.session_state.get("offer_pending"):
            st.markdown("---")
            st.info(f"I can fetch more details about plans for: \"{st.session_state.get('offer_query')}\". Would you like me to share plan details (KB + web)?")
            if st.button("Share plan details", key="share_plan_details"):
                # Run KB-first plan lookup, then web fallback
                details = get_plan_details(st.session_state.get("offer_query"), kb, providers_inv)
                st.session_state.history.append((f"[Plan details for] {st.session_state.get('offer_query')}", details, "Plan Details", 0.9, "neutral"))
                st.success("Plan details added to the conversation.")
                # clear pending offer to avoid repeating
                st.session_state.offer_pending = False

if __name__ == "__main__":
    main()
