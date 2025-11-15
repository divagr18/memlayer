from typing import List, Optional
import re
import numpy as np

# --- Hardcoded Semantic Prototypes ---
# These represent the core concepts of "salient" vs. "non-salient" information.
# Expanded to cover more diverse conversation types.
SALIENT_PROTOTYPES = [
    # Personal Identity & Background
    "My name is Sarah and I work in the Marketing department.",
    "I graduated from Stanford University in 2019.",
    "I live in San Francisco with my two cats.",
    "My email is sarah@company.com.",
    
    # Factual Statements & Data
    "The user's API key is sk-12345.",
    "My flight number is BA2490.",
    "The project deadline is next Friday, November 22nd.",
    "The server IP address is 192.168.1.101.",
    "The bug occurs on line 234 of the authentication module.",
    
    # User Preferences & Instructions
    "I prefer all reports to be in PDF format.",
    "Please remember to CC me on all future emails about this topic.",
    "My favorite color is blue.",
    "Never share my personal contact information.",
    "I prefer morning meetings and coffee without sugar.",
    "I like to receive notifications via email, not SMS.",
    
    # Work & Projects
    "I'm leading the Project Phoenix initiative.",
    "Our team consists of Alice, Bob, and Charlie.",
    "We're using Python and FastAPI for the backend.",
    "The client requested a mobile-first design.",
    "Alice is responsible for the database architecture.",
    
    # Relationships & People
    "Dr. Emma Watson is my primary care physician.",
    "John from accounting helped me with the expense report.",
    "My manager's name is David Chen.",
    "I collaborate closely with the design team.",
    
    # Decisions & Plans
    "We have decided to approve the budget for the Alpha phase.",
    "The meeting is scheduled for 3 PM tomorrow.",
    "Let's proceed with option B.",
    "The final plan is to launch on the first Monday of next month.",
    "I'll be on vacation from December 15th to January 2nd.",
    
    # Events & Activities
    "I attended the AI conference in Boston last week.",
    "The workshop starts at 9 AM on Thursday.",
    "We deployed version 2.3 to production yesterday.",
]

NON_SALIENT_PROTOTYPES = [
    # Greetings & Pleasantries
    "Hello, how are you doing today?",
    "Good morning!",
    "Nice to meet you.",
    "Hi there!",
    "Hey, what's up?",
    
    # Acknowledgements & Agreements
    "Okay, that sounds good.",
    "I understand.",
    "Got it, thanks.",
    "Perfect.",
    "Sure thing.",
    "Makes sense.",
    "Alright.",
    
    # Gratitude & Closings
    "Thank you for your help!",
    "That's all for now, goodbye.",
    "Appreciate it.",
    "Thanks!",
    "Bye!",
    
    # Conversational Filler
    "Hmm, let me think about that for a moment.",
    "That's an interesting question.",
    "One second.",
    "Give me a minute.",
    "Let me see.",
    
    # Meta-conversation (talking about the conversation itself)
    "Can you repeat that?",
    "What did you just say?",
    "I didn't catch that.",
    "Could you clarify?",
    
    # Simple Responses
    "Yes.",
    "No.",
    "Maybe.",
    "I'm not sure.",
    "I don't know.",
]

# --- Fast Heuristic Patterns ---
# Quick regex patterns to catch obvious salient/non-salient content
# without needing expensive embedding computation.

# Patterns that indicate salient content
SALIENT_PATTERNS = [
    r'\b(?:my|our|the)\s+(?:name|email|phone|address|username|password)\s+(?:is|:)',
    r'\b(?:I|we)\s+(?:work|study|live|graduated|studied)\s+(?:at|in|for)\b',
    r'\b(?:prefer|like|love|hate|dislike|want|need)\s+(?:to|the|my|our)',
    r'\b(?:deadline|due date|scheduled|meeting|appointment)\s+(?:is|on|at)',
    r'\b(?:project|team|client|manager|colleague|doctor|professor)\s+(?:is|are|named)',
    r'\b(?:never|always|remember to|make sure to|don\'t forget)\b',
    r'\b(?:version|ip address|port|server|database|api key|token)\b',
    r'\b(?:\d{1,4}[-/]\d{1,2}[-/]\d{1,4})\b',  # Dates
    r'\b(?:\d+:\d+\s*(?:AM|PM|am|pm))\b',  # Times
    r'\b(?:[A-Z][a-z]+\s+[A-Z][a-z]+)\b',  # Proper names (e.g., "John Smith")
]

# Patterns that indicate non-salient content
NON_SALIENT_PATTERNS = [
    r'^\s*(?:hi|hey|hello|good morning|good afternoon|good evening)\s*[!.?]*\s*$',
    r'^\s*(?:thanks?|thank you|thx|ty)\s*[!.?]*\s*$',
    r'^\s*(?:bye|goodbye|see you|see ya|cya)\s*[!.?]*\s*$',
    r'^\s*(?:ok|okay|sure|alright|got it|understood)\s*[!.?]*\s*$',
    r'^\s*(?:yes|no|maybe|perhaps|possibly)\s*[!.?]*\s*$',
    r'^\s*(?:hmm|uh|um|er|ah)\s*[!.?]*\s*$',
    r'^\s*(?:what|huh|pardon|sorry)\s*\??\s*$',
]

class SalienceGate:
    """
    A hybrid salience classifier using fast heuristics + semantic similarity.
    
    Two-stage filtering:
    1. Fast heuristic filter: Regex patterns catch obvious cases (< 1ms)
    2. Semantic classifier: Embedding similarity for borderline cases (~10ms)
    
    The gate compares text similarity to salient vs non-salient prototypes.
    Text is saved if: salient_score > (non_salient_score + threshold)
    
    Threshold guidelines:
    - 0.1: Very strict (only saves clear facts/preferences)
    - 0.0: Balanced (saves if salient score is higher)
    - -0.05: Permissive (saves most content except clear greetings/filler)
    """
    def __init__(self, threshold: float = 0.0):
        """
        Initializes the SalienceGate.

        Args:
            threshold (float): The required cosine similarity margin for a text
                               to be considered salient. Default 0.0 (balanced).
        """
        try:
            from sentence_transformers import SentenceTransformer
            from sentence_transformers.util import cos_sim
        except ImportError:
            raise ImportError(
                "The 'ml-gate' feature requires 'sentence-transformers'. "
                "Please install it with: pip install memory-bank[ml-gate]"
            )
        
        self.cos_sim = cos_sim
        self.threshold = threshold
        
        # Compile regex patterns for fast filtering
        self.salient_patterns = [re.compile(p, re.IGNORECASE) for p in SALIENT_PATTERNS]
        self.non_salient_patterns = [re.compile(p, re.IGNORECASE) for p in NON_SALIENT_PATTERNS]
        
        # Load the lightweight, local sentence transformer model.
        # This will be downloaded from Hugging Face Hub on the first run.
        print("Initializing SalienceGate: Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("SalienceGate model loaded.")

        # Pre-compute the embeddings for our hardcoded prototypes.
        # This is a one-time cost during initialization.
        self.salient_embeddings = self.model.encode(SALIENT_PROTOTYPES)
        self.non_salient_embeddings = self.model.encode(NON_SALIENT_PROTOTYPES)
    
    def _quick_heuristic_check(self, text: str) -> Optional[bool]:
        """
        Fast pattern-based check. Returns:
        - True: Definitely salient (matched salient pattern)
        - False: Definitely not salient (matched non-salient pattern)
        - None: Uncertain, needs semantic check
        """
        # Check non-salient patterns first (faster rejection)
        for pattern in self.non_salient_patterns:
            if pattern.search(text):
                return False
        
        # Check salient patterns
        for pattern in self.salient_patterns:
            if pattern.search(text):
                return True
        
        # Additional simple heuristics
        text_lower = text.lower().strip()
        
        # Very short responses are usually not salient
        if len(text_lower) < 10:
            return False
        
        # Contains named entities (capitalized words)
        if re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text):
            return True
        
        # Contains numbers or dates (often factual)
        if re.search(r'\b\d+\b', text):
            return True
        
        # Uncertain - proceed to semantic check
        return None

    def is_worth_saving(self, text: str, verbose: bool = False) -> bool:
        """
        Determines if a given text is salient enough to be worth saving to memory.

        Args:
            text (str): The text to analyze.
            verbose (bool): If True, prints detailed decision process.

        Returns:
            bool: True if the text is deemed salient, False otherwise.
        """
        if not text or not text.strip():
            return False

        # Stage 1: Fast heuristic check
        quick_result = self._quick_heuristic_check(text)
        
        if quick_result is True:
            if verbose:
                print(f"Salience Check: '{text[:50]}...' -> SAVE (heuristic match)")
            return True
        
        if quick_result is False:
            if verbose:
                print(f"Salience Check: '{text[:50]}...' -> SKIP (heuristic match)")
            return False
        
        # Stage 2: Semantic similarity check (for uncertain cases)
        # 1. Compute the embedding for the input text.
        text_embedding = self.model.encode([text])

        # 2. Calculate cosine similarity against all salient prototypes.
        # Use max similarity (best match) rather than average
        salient_scores = self.cos_sim(text_embedding, self.salient_embeddings)
        max_salient_score = float(salient_scores.max())
        avg_salient_score = float(salient_scores.mean())

        # 3. Calculate cosine similarity against all non-salient prototypes.
        non_salient_scores = self.cos_sim(text_embedding, self.non_salient_embeddings)
        max_non_salient_score = float(non_salient_scores.max())
        avg_non_salient_score = float(non_salient_scores.mean())

        # 4. Apply the decision logic using max scores (more robust)
        # The text is salient if its similarity to the "salient" class
        # is greater than its similarity to the "non-salient" class
        # by at least the defined threshold.
        is_salient = max_salient_score > (max_non_salient_score + self.threshold)
        
        if verbose:
            print(
                f"Salience Check: '{text[:50]}...' -> "
                f"Salient: {max_salient_score:.2f} (avg: {avg_salient_score:.2f}), "
                f"Non-Salient: {max_non_salient_score:.2f} (avg: {avg_non_salient_score:.2f}), "
                f"Threshold: {self.threshold}, "
                f"Result: {'SAVE' if is_salient else 'SKIP'} (semantic)"
            )
        
        return is_salient