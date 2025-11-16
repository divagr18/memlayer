from typing import List, Optional, Literal
import re
import numpy as np
from enum import Enum

class SalienceMode(str, Enum):
    """Salience gate operating modes."""
    LOCAL = "local"  # Local sentence-transformers for salience + storage (default, slow startup)
    ONLINE = "online"  # OpenAI embeddings API for salience + storage (fast startup, API cost)
    LIGHTWEIGHT = "lightweight"  # No embeddings: keyword salience + graph-only storage (instant startup)

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
    "I didn't catch that?",
    "Could you clarify?",
    
    # Simple Queries (asking for already-stated info)
    "What's my name?",
    "What did I say my name was?",
    "What's my email?",
    "What was that number?",
    "What's the status?",
    "What time is it?",
    "What day is today?",
    
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
    r"(?:what'?s|what is|what was)\s+(?:my|the|your)\s+\w+\?",  # "What's my name?", "What is the status?"
]

# --- TF-IDF Keywords for Lightweight Mode ---
# Keyword-based classification without embeddings
SALIENT_KEYWORDS = [
    # Personal identifiers
    'name', 'email', 'phone', 'address', 'username', 'password',
    'born', 'birthday', 'age', 'ssn', 'id',
    
    # Work/Education
    'work', 'job', 'company', 'employer', 'colleague', 'manager',
    'study', 'university', 'degree', 'major', 'graduated',
    
    # Preferences
    'prefer', 'like', 'love', 'hate', 'dislike', 'favorite',
    'always', 'never', 'want', 'need',
    
    # Facts & Data
    'deadline', 'due', 'scheduled', 'meeting', 'appointment',
    'project', 'team', 'client', 'api', 'key', 'token',
    'version', 'ip', 'server', 'database', 'port',
    
    # People & Relationships
    'doctor', 'physician', 'lawyer', 'friend', 'family',
    'spouse', 'partner', 'child', 'parent',
    
    # Locations
    'live', 'lives', 'address', 'city', 'state', 'country',
    'office', 'home',
    
    # Instructions
    'remember', 'remind', 'note', 'important', 'critical',
    'must', 'should', 'don\'t', 'do not',
]

NON_SALIENT_KEYWORDS = [
    # Greetings
    'hello', 'hi', 'hey', 'morning', 'afternoon', 'evening',
    
    # Acknowledgements
    'ok', 'okay', 'sure', 'alright', 'got', 'understood',
    'yes', 'no', 'maybe', 'perhaps',
    
    # Gratitude
    'thanks', 'thank', 'thx', 'ty', 'appreciate',
    
    # Farewells
    'bye', 'goodbye', 'see', 'cya',
    
    # Filler
    'hmm', 'uh', 'um', 'er', 'ah', 'well',
    
    # Meta
    'what', 'huh', 'pardon', 'sorry', 'repeat',
]

class SalienceGate:
    """
    A hybrid salience classifier with multiple operating modes.
    
    Modes:
    1. LOCAL (default): Uses local sentence-transformers for salience filtering
       - Pros: High accuracy, no API costs, works offline
       - Cons: Slow startup (~11s model load), requires disk space (~500MB)
       - Storage: Vector (ChromaDB) + Graph (NetworkX)
    
    2. ONLINE: Uses OpenAI embeddings API for salience filtering
       - Pros: Fast startup (~2s), no local model needed
       - Cons: API costs (~$0.0001 per check), requires internet
       - Storage: Vector (ChromaDB) + Graph (NetworkX)
    
    3. LIGHTWEIGHT: Uses keyword matching for salience filtering (no embeddings)
       - Pros: Instant startup (<1s), no dependencies, no API costs
       - Cons: Lower accuracy (rule-based), no semantic search
       - Storage: Graph-only (NetworkX) - no vector storage
    
    Two-stage filtering (LOCAL/ONLINE modes):
    1. Fast heuristic filter: Regex patterns catch obvious cases (< 1ms)
    2. Semantic classifier: Embedding similarity for borderline cases (~10ms)
    
    LIGHTWEIGHT mode: Only uses heuristic patterns + keyword matching
    
    The gate compares text similarity to salient vs non-salient prototypes.
    Text is saved if: salient_score > (non_salient_score + threshold)
    
    Threshold guidelines:
    - 0.1: Very strict (only saves clear facts/preferences)
    - 0.0: Balanced (saves if salient score is higher)
    - -0.05: Permissive (saves most content except clear greetings/filler)
    """
    def __init__(
        self, 
        threshold: float = 0.0, 
        embedding_model=None,
        mode: SalienceMode = SalienceMode.LOCAL,
        openai_api_key: Optional[str] = None
    ):
        """
        Initializes the SalienceGate.

        Args:
            threshold (float): The required cosine similarity margin for a text
                               to be considered salient. Default 0.0 (balanced).
            embedding_model: Optional pre-initialized embedding model to reuse.
                           Only used in LOCAL mode. If None, will create its own.
            mode (SalienceMode): Operating mode - LOCAL, ONLINE, or LIGHTWEIGHT.
            openai_api_key (str): OpenAI API key for ONLINE mode. If None, will
                                 try to read from OPENAI_API_KEY env variable.
        """
        self.threshold = threshold
        self.mode = mode
        
        # Compile regex patterns for fast filtering (used in all modes)
        self.salient_patterns = [re.compile(p, re.IGNORECASE) for p in SALIENT_PATTERNS]
        self.non_salient_patterns = [re.compile(p, re.IGNORECASE) for p in NON_SALIENT_PATTERNS]
        
        # Mode-specific initialization
        if mode == SalienceMode.LOCAL:
            self._init_local_mode(embedding_model)
        elif mode == SalienceMode.ONLINE:
            self._init_online_mode(openai_api_key)
        elif mode == SalienceMode.LIGHTWEIGHT:
            self._init_lightweight_mode()
        else:
            raise ValueError(f"Unknown salience mode: {mode}")
    
    def _init_local_mode(self, embedding_model=None):
        """Initialize LOCAL mode with sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            from sentence_transformers.util import cos_sim
        except ImportError:
            raise ImportError(
                "The 'ml-gate' feature requires 'sentence-transformers'. "
                "Please install it with: pip install memlayer[ml-gate]"
            )
        
        self.cos_sim = cos_sim
        
        # Reuse existing model or create new one
        if embedding_model is not None:
            print("SalienceGate (LOCAL): Reusing shared embedding model.")
            # Use the embedding_model interface (get_embeddings method)
            self.embedding_model = embedding_model
        else:
            # Load the lightweight, local embedding model
            print("SalienceGate (LOCAL): Loading embedding model...")
            from .embedding_models import LocalEmbeddingModel
            self.embedding_model = LocalEmbeddingModel()
            print("SalienceGate model loaded.")

        # Pre-compute the embeddings for our hardcoded prototypes.
        self.salient_embeddings = self.embedding_model.get_embeddings(SALIENT_PROTOTYPES)
        self.non_salient_embeddings = self.embedding_model.get_embeddings(NON_SALIENT_PROTOTYPES)
    
    def _init_online_mode(self, api_key: Optional[str] = None):
        """Initialize ONLINE mode with OpenAI embeddings API."""
        import os
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "ONLINE mode requires 'openai' package. "
                "Please install it with: pip install openai"
            )
        
        # Get API key
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "ONLINE mode requires OpenAI API key. "
                "Provide it via openai_api_key parameter or OPENAI_API_KEY env variable."
            )
        
        self.openai_client = OpenAI(api_key=api_key)
        print("SalienceGate (ONLINE): Using OpenAI embeddings API.")
        
        # Pre-compute embeddings for prototypes using OpenAI API
        print("SalienceGate (ONLINE): Pre-computing prototype embeddings...")
        self.salient_embeddings = self._get_openai_embeddings(SALIENT_PROTOTYPES)
        self.non_salient_embeddings = self._get_openai_embeddings(NON_SALIENT_PROTOTYPES)
        print("SalienceGate (ONLINE): Ready.")
    
    def _init_lightweight_mode(self):
        """Initialize LIGHTWEIGHT mode with TF-IDF keyword matching."""
        print("SalienceGate (LIGHTWEIGHT): Using keyword-based classification.")
        
        # Pre-compile keyword patterns for faster matching
        self.salient_keyword_patterns = [
            re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
            for kw in SALIENT_KEYWORDS
        ]
        self.non_salient_keyword_patterns = [
            re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
            for kw in NON_SALIENT_KEYWORDS
        ]
        print("SalienceGate (LIGHTWEIGHT): Ready.")
    
    def _get_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from OpenAI API."""
        response = self.openai_client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"  # Cheaper and faster than ada-002
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between vectors (for ONLINE mode)."""
        # Normalize vectors
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        # Compute similarity
        return np.dot(a_norm, b_norm.T)
    
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
    
    def _lightweight_check(self, text: str) -> float:
        """
        Lightweight keyword-based scoring (no embeddings).
        Returns a salience score based on keyword matching.
        
        Returns:
            float: Score where positive = salient, negative = non-salient
        """
        text_lower = text.lower()
        
        # Count keyword matches
        salient_matches = sum(
            1 for pattern in self.salient_keyword_patterns
            if pattern.search(text_lower)
        )
        
        non_salient_matches = sum(
            1 for pattern in self.non_salient_keyword_patterns
            if pattern.search(text_lower)
        )
        
        # Weight by text length (longer text with keywords is more significant)
        word_count = len(text.split())
        length_weight = min(word_count / 10.0, 2.0)  # Cap at 2x boost
        
        # Calculate weighted score
        salient_score = salient_matches * length_weight
        non_salient_score = non_salient_matches * length_weight
        
        return salient_score - non_salient_score

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

        # Stage 1: Fast heuristic check (used in all modes)
        quick_result = self._quick_heuristic_check(text)
        
        if quick_result is True:
            if verbose:
                print(f"Salience Check [{self.mode}]: '{text[:50]}...' -> SAVE (heuristic match)")
            return True
        
        if quick_result is False:
            if verbose:
                print(f"Salience Check [{self.mode}]: '{text[:50]}...' -> SKIP (heuristic match)")
            return False
        
        # Stage 2: Mode-specific semantic/keyword check
        if self.mode == SalienceMode.LIGHTWEIGHT:
            return self._lightweight_is_salient(text, verbose)
        elif self.mode == SalienceMode.LOCAL:
            return self._local_is_salient(text, verbose)
        elif self.mode == SalienceMode.ONLINE:
            return self._online_is_salient(text, verbose)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _lightweight_is_salient(self, text: str, verbose: bool) -> bool:
        """LIGHTWEIGHT mode: Keyword-based classification."""
        score = self._lightweight_check(text)
        
        # Apply threshold
        is_salient = score > self.threshold
        
        if verbose:
            print(
                f"Salience Check [LIGHTWEIGHT]: '{text[:50]}...' -> "
                f"Score: {score:.2f}, Threshold: {self.threshold}, "
                f"Result: {'SAVE' if is_salient else 'SKIP'}"
            )
        
        return is_salient
    
    def _local_is_salient(self, text: str, verbose: bool) -> bool:
        """LOCAL mode: Sentence-transformers embeddings."""
        # Compute the embedding for the input text.
        text_embedding = self.embedding_model.get_embeddings([text])

        # Calculate cosine similarity against all salient prototypes.
        salient_scores = self.cos_sim(text_embedding, self.salient_embeddings)
        max_salient_score = float(salient_scores.max())
        avg_salient_score = float(salient_scores.mean())

        # Calculate cosine similarity against all non-salient prototypes.
        non_salient_scores = self.cos_sim(text_embedding, self.non_salient_embeddings)
        max_non_salient_score = float(non_salient_scores.max())
        avg_non_salient_score = float(non_salient_scores.mean())

        # Apply the decision logic using max scores
        is_salient = max_salient_score > (max_non_salient_score + self.threshold)
        
        if verbose:
            print(
                f"Salience Check [LOCAL]: '{text[:50]}...' -> "
                f"Salient: {max_salient_score:.2f} (avg: {avg_salient_score:.2f}), "
                f"Non-Salient: {max_non_salient_score:.2f} (avg: {avg_non_salient_score:.2f}), "
                f"Threshold: {self.threshold}, "
                f"Result: {'SAVE' if is_salient else 'SKIP'}"
            )
        
        return is_salient
    
    def _online_is_salient(self, text: str, verbose: bool) -> bool:
        """ONLINE mode: OpenAI embeddings API."""
        # Get embedding from OpenAI API
        text_embedding = self._get_openai_embeddings([text])

        # Calculate cosine similarity against prototypes
        salient_scores = self._cosine_similarity(text_embedding, self.salient_embeddings)
        max_salient_score = float(salient_scores.max())
        avg_salient_score = float(salient_scores.mean())

        non_salient_scores = self._cosine_similarity(text_embedding, self.non_salient_embeddings)
        max_non_salient_score = float(non_salient_scores.max())
        avg_non_salient_score = float(non_salient_scores.mean())

        # Apply decision logic
        is_salient = max_salient_score > (max_non_salient_score + self.threshold)
        
        if verbose:
            print(
                f"Salience Check [ONLINE]: '{text[:50]}...' -> "
                f"Salient: {max_salient_score:.2f} (avg: {avg_salient_score:.2f}), "
                f"Non-Salient: {max_non_salient_score:.2f} (avg: {avg_non_salient_score:.2f}), "
                f"Threshold: {self.threshold}, "
                f"Result: {'SAVE' if is_salient else 'SKIP'}"
            )
        
        return is_salient