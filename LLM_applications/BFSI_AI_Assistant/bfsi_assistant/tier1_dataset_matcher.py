"""
Tier 1: Dataset Similarity Matcher
Fast responses from curated dataset (~70% query coverage)
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, Optional
import config

class Tier1DatasetMatcher:
    """
    Dataset-based similarity matching for common queries
    
    Features:
    - Fast similarity search (~15ms response time)
    - Normalized embeddings for accurate matching
    - Out-of-domain query filtering
    - Confidence scoring
    """
    
    def __init__(self, dataset_path: str = None, threshold: float = None):
        """
        Initialize the dataset matcher
        
        Args:
            dataset_path: Path to Alpaca-format dataset
            threshold: Similarity threshold (default from config)
        """
        self.dataset_path = dataset_path or config.DATASET_PATH
        self.threshold = threshold or config.TIER1_THRESHOLD
        
        self.model = None
        self.embeddings = None
        self.queries = []
        self.responses = []
        self.dataset = []
        
        print(f"🔧 Initializing Tier 1 (Dataset Matcher)...")
        print(f"   Threshold: {self.threshold}")
    
    def load_dataset(self):
        """Load and parse the Alpaca dataset"""
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        
        # Extract queries and responses
        self.queries = [
            (record["instruction"] + " " + record.get("input", "")).strip()
            for record in self.dataset
        ]
        self.responses = [record["output"] for record in self.dataset]
        
        print(f"   ✅ Loaded {len(self.dataset)} records")
    
    def create_embeddings(self):
        """Create normalized embeddings for fast similarity search"""
        print(f"   📊 Loading model: {config.TIER1_MODEL}")
        self.model = SentenceTransformer(config.TIER1_MODEL)
        
        print(f"   🔄 Creating embeddings...")
        self.embeddings = self.model.encode(
            self.queries,
            normalize_embeddings=True,  # For faster cosine similarity
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        print(f"   ✅ Embeddings created: {self.embeddings.shape}")
    
    def initialize(self):
        """Initialize the matcher (load dataset + create embeddings)"""
        self.load_dataset()
        self.create_embeddings()
        print(f"✅ Tier 1 ready")
    
    def is_out_of_domain(self, query: str) -> bool:
        """Check if query is out of domain"""
        query_lower = query.lower()
        return any(
            keyword in query_lower 
            for keyword in config.OUT_OF_DOMAIN_KEYWORDS
        )
    
    def search(self, query: str, top_k: int = None) -> list:
        """
        Search for similar queries in the dataset
        
        Args:
            query: User query
            top_k: Number of top results to return
            
        Returns:
            List of top matches with similarity scores
        """
        top_k = top_k or config.TIER1_TOP_K
        
        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Fast dot product similarity (works because embeddings are normalized)
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            results.append({
                "query": self.queries[idx],
                "response": self.responses[idx],
                "similarity": float(similarities[idx]),
                "index": int(idx)
            })
        
        return results
    
    def query(self, user_query: str) -> Optional[Dict]:
        """
        Process a user query
        
        Args:
            user_query: User's question
            
        Returns:
            Response dict if match found, None otherwise
        """
        # Out-of-domain filter
        if self.is_out_of_domain(user_query):
            return None
        
        # Search for matches
        matches = self.search(user_query, top_k=1)
        
        if not matches:
            return None
        
        best_match = matches[0]
        similarity = best_match["similarity"]
        
        # Check threshold
        if similarity < self.threshold:
            return None
        
        # Determine confidence
        if similarity >= 0.90:
            confidence = "high"
        elif similarity >= 0.80:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "tier": "tier_1",
            "answer": best_match["response"],
            "matched_query": best_match["query"],
            "similarity": similarity,
            "confidence": confidence,
            "response_time_estimate": "~15ms"
        }
    
    def get_stats(self) -> Dict:
        """Get matcher statistics"""
        return {
            "total_records": len(self.dataset),
            "threshold": self.threshold,
            "model": config.TIER1_MODEL,
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0
        }

# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("TESTING TIER 1 MODULE")
    print("="*60)
    
    # Initialize
    matcher = Tier1DatasetMatcher()
    matcher.initialize()
    
    # Test queries
    test_queries = [
        "What is the interest rate for personal loans?",
        "How do I check my loan status?",
        "What is the weather today?",  # Out of domain
    ]
    
    print("\n🧪 Testing queries...\n")
    for query in test_queries:
        result = matcher.query(query)
        
        if result:
            print(f"✅ [{result['similarity']:.3f}] {query}")
            print(f"   → {result['answer'][:100]}...")
        else:
            print(f"❌ [No match] {query}")
        print()
    
    # Print stats
    print("📊 Stats:", matcher.get_stats())
