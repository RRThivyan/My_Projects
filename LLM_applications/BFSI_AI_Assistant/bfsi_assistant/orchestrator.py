"""
Orchestrator - Main Query Router
Routes queries through Tier 1 → Tier 2 → Tier 3 as per PRD
"""

import re
from typing import Dict
import config
from tier1_dataset_matcher import Tier1DatasetMatcher
from tier2_slm import Tier2FineTunedSLM
from tier3_rag import Tier3RAGSystem

class BFSIOrchestrator:
    """
    Main orchestrator for BFSI Call Center AI
    
    Routing Priority:
    1. Tier 1: Dataset Match (70% coverage, ~15ms)
    2. Tier 2: Fine-tuned SLM (20% coverage, ~500ms)
    3. Tier 3: RAG System (10% coverage, ~2s)
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the orchestrator
        
        Args:
            verbose: Print detailed routing information
        """
        self.verbose = verbose
        
        self.tier1 = None
        self.tier2 = None
        self.tier3 = None
        
        self.stats = {
            'tier_1_hits': 0,
            'tier_2_hits': 0,
            'tier_3_hits': 0,
            'total_queries': 0
        }
        
        print("="*60)
        print("INITIALIZING BFSI ORCHESTRATOR")
        print("="*60)
    
    def initialize(self):
        """Initialize all tiers"""
        # Tier 1
        print("\n" + "="*60)
        self.tier1 = Tier1DatasetMatcher()
        self.tier1.initialize()
        
        # Tier 2
        print("\n" + "="*60)
        self.tier2 = Tier2FineTunedSLM()
        self.tier2.initialize()
        
        # Tier 3
        print("\n" + "="*60)
        self.tier3 = Tier3RAGSystem()
        self.tier3.initialize()
        
        print("\n" + "="*60)
        print("✅ ALL TIERS INITIALIZED")
        print("="*60)
        
        self._print_system_info()
    
    def _print_system_info(self):
        """Print system information"""
        print("\n📊 System Configuration:")
        print(f"   • Tier 1: {self.tier1.get_stats()['total_records']} dataset records")
        print(f"   • Tier 2: {self.tier2.get_stats()['status']}")
        print(f"   • Tier 3: {self.tier3.get_stats()['total_chunks']} chunks, {self.tier3.get_stats()['table_chunks']} tables")
        print(f"\n🎯 Expected Coverage:")
        print(f"   • Tier 1: ~70% (instant)")
        print(f"   • Tier 2: ~20% (fast)")
        print(f"   • Tier 3: ~10% (comprehensive)")
    
    def query(self, user_query: str, return_metadata: bool = False) -> Dict:
        """
        Process a user query through the 3-tier system
        
        Args:
            user_query: User's question
            return_metadata: Include routing metadata
            
        Returns:
            Response dictionary
        """
        self.stats['total_queries'] += 1
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🔍 QUERY: {user_query}")
            print(f"{'='*60}")
        
        # TIER 1: Dataset Similarity Match
        if self.verbose:
            print(f"\n1️⃣  Tier 1: Checking dataset...")
        
        tier1_result = self.tier1.query(user_query)
        
        if tier1_result is not None:
            self.stats['tier_1_hits'] += 1
            
            if self.verbose:
                print(f"   ✅ Match found! Similarity: {tier1_result['similarity']:.3f}")
                print(f"   ⚡ {tier1_result['response_time_estimate']}")
            
            if return_metadata:
                tier1_result['routing'] = {
                    'tier_attempted': ['tier_1'],
                    'tier_used': 'tier_1'
                }
            
            return tier1_result
        
        if self.verbose:
            print(f"   ⏭️  No match (threshold: {config.TIER1_THRESHOLD})")
        
        # TIER 2: Fine-tuned SLM
        if self.verbose:
            print(f"\n2️⃣  Tier 2: Trying fine-tuned SLM...")
        
        if self.tier2.is_available():
            tier2_result = self.tier2.query(user_query)
            
            if tier2_result is not None:
                self.stats['tier_2_hits'] += 1
                
                if self.verbose:
                    print(f"   ✅ Response generated")
                    print(f"   ⚡ {tier2_result['response_time_estimate']}")
                
                if return_metadata:
                    tier2_result['routing'] = {
                        'tier_attempted': ['tier_1', 'tier_2'],
                        'tier_used': 'tier_2'
                    }
                
                return tier2_result
        else:
            if self.verbose:
                print(f"   ⏭️  SLM not available")
        
        # TIER 3: RAG System
        if self.verbose:
            print(f"\n3️⃣  Tier 3: Using RAG system...")
        
        self.stats['tier_3_hits'] += 1
        
        tier3_result = self.tier3.query(user_query)
        
        # If RAG returned context, generate answer using Tier 2
        if tier3_result and 'results' in tier3_result and self.tier2.is_available():
            # Clean and prepare context
            context_parts = []
            for r in tier3_result['results'][:3]:
                text = r['text']
                text = re.sub(r'\[.*?\]', '', text)
                text = re.sub(r'---.*?---', '', text)
                text = text.strip()[:500]
                if text:
                    context_parts.append(text)
            
            context = " ".join(context_parts)
            
            # Generate with context
            generation_result = self.tier2.query(user_query, context=context)
            
            if generation_result:
                # Combine generation with RAG sources
                tier3_result['answer'] = generation_result['answer']
            else:
                # Fallback to context summary
                tier3_result['answer'] = f"Based on the documents: {context[:300]}..."
        elif tier3_result and 'context' in tier3_result:
            # Already has answer from context_only mode
            pass
        else:
            # Low confidence
            tier3_result = {
                'tier': 'tier_3',
                'answer': "I could not find sufficient information in the official documents to answer this question accurately.",
                'confidence': 'low',
                'sources': []
            }
        
        if self.verbose:
            print(f"   ✅ Retrieved and generated response")
            print(f"   ⚡ ~2s")
        
        if return_metadata:
            tier3_result['routing'] = {
                'tier_attempted': ['tier_1', 'tier_2', 'tier_3'],
                'tier_used': 'tier_3'
            }
        
        return tier3_result
    
    def get_stats(self) -> Dict:
        """Get usage statistics"""
        total = self.stats['total_queries']
        
        if total == 0:
            return {
                'total_queries': 0,
                'tier_1_percentage': 0,
                'tier_2_percentage': 0,
                'tier_3_percentage': 0
            }
        
        return {
            'total_queries': total,
            'tier_1_hits': self.stats['tier_1_hits'],
            'tier_2_hits': self.stats['tier_2_hits'],
            'tier_3_hits': self.stats['tier_3_hits'],
            'tier_1_percentage': (self.stats['tier_1_hits'] / total) * 100,
            'tier_2_percentage': (self.stats['tier_2_hits'] / total) * 100,
            'tier_3_percentage': (self.stats['tier_3_hits'] / total) * 100
        }
    
    def print_stats(self):
        """Print usage statistics"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("📊 USAGE STATISTICS")
        print("="*60)
        print(f"Total Queries: {stats['total_queries']}")
        print(f"\nTier Distribution:")
        print(f"   • Tier 1: {stats['tier_1_hits']} ({stats['tier_1_percentage']:.1f}%)")
        print(f"   • Tier 2: {stats['tier_2_hits']} ({stats['tier_2_percentage']:.1f}%)")
        print(f"   • Tier 3: {stats['tier_3_hits']} ({stats['tier_3_percentage']:.1f}%)")
        print("="*60)

# ============================================================
# SIMPLE QUERY FUNCTION
# ============================================================

def ask(orchestrator, query: str):
    """
    Simple interface for querying the system
    
    Args:
        orchestrator: BFSIOrchestrator instance
        query: User's question
    """
    result = orchestrator.query(query, return_metadata=False)
    
    print(f"\n{'='*60}")
    print(f"📝 RESPONSE")
    print(f"{'='*60}")
    print(f"Tier: {result['tier'].upper()}")
    print(f"Confidence: {result.get('confidence', 'N/A').upper()}")
    
    print(f"\nAnswer:")
    print(result['answer'])
    
    if 'sources' in result and result['sources']:
        print(f"\nSources:")
        for i, src in enumerate(result['sources'][:3], 1):
            table_marker = " [TABLE]" if src.get('contains_table') else ""
            print(f"   {i}. {src['source']} (Page {src['page']}){table_marker}")
    
    if 'matched_query' in result:
        print(f"\nMatched Query: {result['matched_query']}")
    
    if 'similarity' in result:
        print(f"Similarity Score: {result['similarity']:.3f}")
    
    print(f"{'='*60}")
    
    return result

# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("TESTING ORCHESTRATOR")
    print("="*60)
    
    # Initialize
    orchestrator = BFSIOrchestrator(verbose=True)
    orchestrator.initialize()
    
    # Test queries
    test_queries = [
        "What is the interest rate for personal loans?",  # Should hit Tier 1
        "How do I check my loan eligibility?",  # Should hit Tier 1
        "What is the LTV ratio for home loans above 75 lakhs?",  # Should hit Tier 3
    ]
    
    print("\n" + "="*60)
    print("RUNNING TEST QUERIES")
    print("="*60)
    
    for query in test_queries:
        result = orchestrator.query(query, return_metadata=False)
        
        print(f"\n✅ [{result['tier']}] {query}")
        print(f"   → {result['answer'][:100]}...")
    
    # Print stats
    orchestrator.print_stats()
