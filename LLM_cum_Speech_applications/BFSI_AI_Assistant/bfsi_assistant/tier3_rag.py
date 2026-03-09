# """
# Tier 3: RAG System
# Advanced retrieval for complex queries (~10% coverage)
# """

# import numpy as np
# import chromadb
# from chromadb.config import Settings
# from sentence_transformers import SentenceTransformer, CrossEncoder
# from rank_bm25 import BM25Okapi
# from typing import List, Dict, Optional
# import re
# import config

# class Tier3RAGSystem:
#     """
#     Retrieval-Augmented Generation system
    
#     Features:
#     - Hybrid search (BM25 + semantic)
#     - Reciprocal Rank Fusion
#     - Cross-encoder reranking
#     - Confidence scoring
#     """
    
#     def __init__(self):
#         """Initialize the RAG system"""
#         self.embedding_model = None
#         self.reranker = None
#         self.collection = None
#         self.bm25 = None
#         self.chunks = []
        
#         print(f"🔧 Initializing Tier 3 (RAG System)...")
    
#     def load_vector_db(self):
#         """Load existing vector database"""
#         print(f"   📂 Loading vector database...")
        
#         # Connect to ChromaDB
#         chroma_client = chromadb.Client(Settings(
#             persist_directory=config.INDEX_PATH,
#             anonymized_telemetry=False
#         ))
        
#         # Get collection
#         try:
#             self.collection = chroma_client.get_collection(
#                 name=config.CHROMA_COLLECTION_NAME
#             )
            
#             # Get all documents to build BM25 index
#             results = self.collection.get()
            
#             self.chunks = [
#                 {
#                     'text': doc,
#                     'metadata': meta,
#                     'id': doc_id
#                 }
#                 for doc, meta, doc_id in zip(
#                     results['documents'],
#                     results['metadatas'],
#                     results['ids']
#                 )
#             ]
            
#             print(f"   ✅ Loaded {len(self.chunks)} chunks")
            
#         except Exception as e:
#             print(f"   ❌ Failed to load vector DB: {e}")
#             print(f"   ⚠️  Run vector_db_creator.py first!")
#             raise
    
#     def create_bm25_index(self):
#         """Create BM25 index for keyword search"""
#         print(f"   🔄 Creating BM25 index...")
        
#         # Simple tokenization
#         tokenized_chunks = [
#             chunk['text'].lower().split() 
#             for chunk in self.chunks
#         ]
        
#         self.bm25 = BM25Okapi(tokenized_chunks)
        
#         print(f"   ✅ BM25 index created")
    
#     def load_models(self):
#         """Load embedding and reranking models"""
#         print(f"   📊 Loading models...")
        
#         # Embedding model
#         self.embedding_model = SentenceTransformer(
#             config.TIER3_EMBEDDING_MODEL
#         )
        
#         # Reranker
#         self.reranker = CrossEncoder(
#             config.TIER3_RERANKER_MODEL
#         )
        
#         print(f"   ✅ Models loaded")
    
#     def initialize(self):
#         """Initialize the RAG system"""
#         self.load_vector_db()
#         self.create_bm25_index()
#         self.load_models()
        
#         print(f"✅ Tier 3 ready")
    
#     def dense_search(self, query: str, top_k: int = 10) -> List[Dict]:
#         """Semantic search using embeddings"""
#         # Encode query
#         query_embedding = self.embedding_model.encode(
#             [query],
#             normalize_embeddings=True
#         )[0]
        
#         # Search in vector DB
#         results = self.collection.query(
#             query_embeddings=[query_embedding.tolist()],
#             n_results=top_k
#         )
        
#         # Format results
#         formatted_results = []
#         for i in range(len(results['ids'][0])):
#             formatted_results.append({
#                 'id': results['ids'][0][i],
#                 'text': results['documents'][0][i],
#                 'metadata': results['metadatas'][0][i],
#                 'score': 1 - results['distances'][0][i],
#                 'source': 'dense'
#             })
        
#         return formatted_results
    
#     def bm25_search(self, query: str, top_k: int = 10) -> List[Dict]:
#         """Keyword search using BM25"""
#         # Tokenize query
#         tokenized_query = query.lower().split()
        
#         # Get scores
#         scores = self.bm25.get_scores(tokenized_query)
        
#         # Get top k indices
#         top_indices = np.argsort(scores)[-top_k:][::-1]
        
#         # Format results
#         formatted_results = []
#         for idx in top_indices:
#             formatted_results.append({
#                 'id': self.chunks[idx]['id'],
#                 'text': self.chunks[idx]['text'],
#                 'metadata': self.chunks[idx]['metadata'],
#                 'score': scores[idx],
#                 'source': 'bm25'
#             })
        
#         return formatted_results
    
#     def reciprocal_rank_fusion(
#         self,
#         dense_results: List[Dict],
#         bm25_results: List[Dict],
#         k: int = 60
#     ) -> List[Dict]:
#         """
#         Combine results using Reciprocal Rank Fusion
        
#         RRF score = sum(1 / (k + rank))
#         """
#         rrf_scores = {}
        
#         # Add dense results
#         for rank, result in enumerate(dense_results, 1):
#             doc_id = result['id']
#             rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank))
        
#         # Add BM25 results
#         for rank, result in enumerate(bm25_results, 1):
#             doc_id = result['id']
#             rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank))
        
#         # Combine all documents
#         all_docs = {r['id']: r for r in dense_results + bm25_results}
        
#         # Sort by RRF score
#         sorted_docs = sorted(
#             rrf_scores.items(),
#             key=lambda x: x[1],
#             reverse=True
#         )
        
#         # Format results
#         results = []
#         for doc_id, score in sorted_docs:
#             if doc_id in all_docs:
#                 result = all_docs[doc_id].copy()
#                 result['rrf_score'] = score
#                 results.append(result)
        
#         return results
    
#     def rerank(
#         self,
#         query: str,
#         results: List[Dict],
#         top_k: int = None
#     ) -> tuple[List[Dict], float]:
#         """
#         Rerank results using cross-encoder
        
#         Returns:
#             Reranked results and average confidence score
#         """
#         top_k = top_k or config.TIER3_TOP_K
        
#         if len(results) == 0:
#             return [], 0.0
        
#         # Create query-document pairs
#         pairs = [[query, r['text']] for r in results]
        
#         # Get reranking scores
#         scores = self.reranker.predict(pairs)
        
#         # Add scores to results
#         for i, result in enumerate(results):
#             result['rerank_score'] = float(scores[i])
        
#         # Sort by rerank score
#         reranked = sorted(
#             results,
#             key=lambda x: x['rerank_score'],
#             reverse=True
#         )[:top_k]
        
#         # Calculate average confidence
#         avg_confidence = np.mean([r['rerank_score'] for r in reranked])
        
#         return reranked, avg_confidence
    
#     def hybrid_search(
#         self,
#         query: str,
#         dense_k: int = 10,
#         bm25_k: int = 10,
#         fusion_top_k: int = 20,
#         final_k: int = None
#     ) -> tuple[List[Dict], float]:
#         """
#         Complete hybrid search pipeline
        
#         Steps:
#         1. Dense search
#         2. BM25 search
#         3. Reciprocal Rank Fusion
#         4. Cross-encoder reranking
        
#         Returns:
#             Top results and confidence score
#         """
#         final_k = final_k or config.TIER3_TOP_K
        
#         # Step 1: Dense search
#         dense_results = self.dense_search(query, top_k=dense_k)
        
#         # Step 2: BM25 search
#         bm25_results = self.bm25_search(query, top_k=bm25_k)
        
#         # Step 3: RRF fusion
#         fused_results = self.reciprocal_rank_fusion(
#             dense_results,
#             bm25_results
#         )[:fusion_top_k]
        
#         # Step 4: Reranking
#         final_results, confidence = self.rerank(
#             query,
#             fused_results,
#             top_k=final_k
#         )
        
#         return final_results, confidence
    
#     def query(
#         self,
#         user_query: str,
#         context_only: bool = False
#     ) -> Optional[Dict]:
#         """
#         Process a user query
        
#         Args:
#             user_query: User's question
#             context_only: Return context without generation
            
#         Returns:
#             Response dict with answer and sources
#         """
#         # Search
#         results, confidence = self.hybrid_search(user_query)
        
#         # Check confidence threshold
#         if confidence < config.TIER3_CONFIDENCE_THRESHOLD:
#             return {
#                 "tier": "tier_3",
#                 "answer": "I could not find sufficient information in the official documents to answer this question accurately.",
#                 "confidence": "low",
#                 "confidence_score": confidence,
#                 "sources": []
#             }
        
#         # Format sources
#         sources = [
#             {
#                 'source': r['metadata']['source'],
#                 'page': r['metadata']['page'],
#                 'contains_table': r['metadata'].get('contains_table', False),
#                 'score': r['rerank_score']
#             }
#             for r in results
#         ]
        
#         # If context_only, return retrieved context
#         if context_only:
#             context_text = "\n\n".join([
#                 self._clean_text(r['text'])
#                 for r in results[:3]
#             ])
            
#             return {
#                 "tier": "tier_3",
#                 "answer": f"Retrieved context:\n\n{context_text}",
#                 "confidence": self._get_confidence_level(confidence),
#                 "confidence_score": confidence,
#                 "sources": sources,
#                 "context": context_text
#             }
        
#         # Otherwise, return with sources for generation
#         # (Generation will be handled by Tier 2 SLM in orchestrator)
#         return {
#             "tier": "tier_3",
#             "confidence": self._get_confidence_level(confidence),
#             "confidence_score": confidence,
#             "sources": sources,
#             "results": results  # For generation
#         }
    
#     def _clean_text(self, text: str) -> str:
#         """Remove markers and clean text"""
#         text = re.sub(r'\[.*?\]', '', text)
#         text = re.sub(r'---.*?---', '', text)
#         return text.strip()
    
#     def _get_confidence_level(self, score: float) -> str:
#         """Convert score to confidence level"""
#         if score > 0.6:
#             return "high"
#         elif score > 0.3:
#             return "medium"
#         else:
#             return "low"
    
#     def get_stats(self) -> Dict:
#         """Get system statistics"""
#         table_chunks = sum(
#             1 for c in self.chunks 
#             if c['metadata'].get('contains_table', False)
#         )
        
#         return {
#             "total_chunks": len(self.chunks),
#             "table_chunks": table_chunks,
#             "embedding_model": config.TIER3_EMBEDDING_MODEL,
#             "reranker_model": config.TIER3_RERANKER_MODEL,
#             "collection_name": config.CHROMA_COLLECTION_NAME
#         }

# # ============================================================
# # MODULE TEST
# # ============================================================

# if __name__ == "__main__":
#     print("="*60)
#     print("TESTING TIER 3 MODULE")
#     print("="*60)
    
#     # Initialize
#     rag = Tier3RAGSystem()
#     rag.initialize()
    
#     # Test queries
#     test_queries = [
#         "What is the interest rate for personal loans?",
#         "What is the LTV ratio for home loans?",
#     ]
    
#     print("\n🧪 Testing queries...\n")
#     for query in test_queries:
#         result = rag.query(query, context_only=True)
        
#         print(f"✅ {query}")
#         print(f"   Confidence: {result['confidence']} ({result['confidence_score']:.3f})")
#         print(f"   Sources: {len(result['sources'])}")
#         for src in result['sources'][:2]:
#             print(f"      • {src['source']} (Page {src['page']})")
#         print()
    
#     # Print stats
#     print("📊 Stats:", rag.get_stats())


"""
Tier 3: RAG System
Advanced retrieval for complex queries (~10% coverage)
"""

import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from typing import List, Dict, Optional
import re
import os
import config

class Tier3RAGSystem:
    """
    Retrieval-Augmented Generation system
    
    Features:
    - Hybrid search (BM25 + semantic)
    - Reciprocal Rank Fusion
    - Cross-encoder reranking
    - Confidence scoring
    """
    
    def __init__(self):
        """Initialize the RAG system"""
        self.embedding_model = None
        self.reranker = None
        self.collection = None
        self.bm25 = None
        self.chunks = []
        
        print(f"🔧 Initializing Tier 3 (RAG System)...")
    
    def load_vector_db(self):
        """Load existing vector database"""
        print(f"   📂 Loading vector database...")
        
        # Connect to ChromaDB - FIXED: Use PersistentClient with correct path
        chroma_client = chromadb.PersistentClient(
            path=os.path.join(config.INDEX_PATH, 'chroma')  # Include 'chroma' subdirectory
        )
        
        # Get collection
        try:
            self.collection = chroma_client.get_collection(
                name=config.CHROMA_COLLECTION_NAME
            )
            print(f"   ✅ Found collection: {config.CHROMA_COLLECTION_NAME}")
            
            # Get all documents to build BM25 index
            results = self.collection.get()
            
            self.chunks = [
                {
                    'text': doc,
                    'metadata': meta,
                    'id': doc_id
                }
                for doc, meta, doc_id in zip(
                    results['documents'],
                    results['metadatas'],
                    results['ids']
                )
            ]
            
            print(f"   ✅ Loaded {len(self.chunks)} chunks")
            
        except Exception as e:
            print(f"   ❌ Failed to load vector DB: {e}")
            print(f"   ⚠️  Run vector_db_creator.py first!")
            raise
    
    def create_bm25_index(self):
        """Create BM25 index for keyword search"""
        print(f"   🔄 Creating BM25 index...")
        
        # Simple tokenization
        tokenized_chunks = [
            chunk['text'].lower().split() 
            for chunk in self.chunks
        ]
        
        self.bm25 = BM25Okapi(tokenized_chunks)
        
        print(f"   ✅ BM25 index created")
    
    def load_models(self):
        """Load embedding and reranking models"""
        print(f"   📊 Loading models...")
        
        # Embedding model
        self.embedding_model = SentenceTransformer(
            config.TIER3_EMBEDDING_MODEL
        )
        
        # Reranker
        self.reranker = CrossEncoder(
            config.TIER3_RERANKER_MODEL
        )
        
        print(f"   ✅ Models loaded")
    
    def initialize(self):
        """Initialize the RAG system"""
        self.load_vector_db()
        self.create_bm25_index()
        self.load_models()
        
        print(f"✅ Tier 3 ready")
    
    def dense_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Semantic search using embeddings"""
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        )[0]
        
        # Search in vector DB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': 1 - results['distances'][0][i],
                'source': 'dense'
            })
        
        return formatted_results
    
    def bm25_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Keyword search using BM25"""
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Format results
        formatted_results = []
        for idx in top_indices:
            formatted_results.append({
                'id': self.chunks[idx]['id'],
                'text': self.chunks[idx]['text'],
                'metadata': self.chunks[idx]['metadata'],
                'score': scores[idx],
                'source': 'bm25'
            })
        
        return formatted_results
    
    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict],
        bm25_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """
        Combine results using Reciprocal Rank Fusion
        
        RRF score = sum(1 / (k + rank))
        """
        rrf_scores = {}
        
        # Add dense results
        for rank, result in enumerate(dense_results, 1):
            doc_id = result['id']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank))
        
        # Add BM25 results
        for rank, result in enumerate(bm25_results, 1):
            doc_id = result['id']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank))
        
        # Combine all documents
        all_docs = {r['id']: r for r in dense_results + bm25_results}
        
        # Sort by RRF score
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Format results
        results = []
        for doc_id, score in sorted_docs:
            if doc_id in all_docs:
                result = all_docs[doc_id].copy()
                result['rrf_score'] = score
                results.append(result)
        
        return results
    
    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: int = None
    ) -> tuple[List[Dict], float]:
        """
        Rerank results using cross-encoder
        
        Returns:
            Reranked results and average confidence score
        """
        top_k = top_k or config.TIER3_TOP_K
        
        if len(results) == 0:
            return [], 0.0
        
        # Create query-document pairs
        pairs = [[query, r['text']] for r in results]
        
        # Get reranking scores
        scores = self.reranker.predict(pairs)
        
        # Add scores to results
        for i, result in enumerate(results):
            result['rerank_score'] = float(scores[i])
        
        # Sort by rerank score
        reranked = sorted(
            results,
            key=lambda x: x['rerank_score'],
            reverse=True
        )[:top_k]
        
        # Calculate average confidence
        avg_confidence = np.mean([r['rerank_score'] for r in reranked])
        
        return reranked, avg_confidence
    
    def hybrid_search(
        self,
        query: str,
        dense_k: int = 10,
        bm25_k: int = 10,
        fusion_top_k: int = 20,
        final_k: int = None
    ) -> tuple[List[Dict], float]:
        """
        Complete hybrid search pipeline
        
        Steps:
        1. Dense search
        2. BM25 search
        3. Reciprocal Rank Fusion
        4. Cross-encoder reranking
        
        Returns:
            Top results and confidence score
        """
        final_k = final_k or config.TIER3_TOP_K
        
        # Step 1: Dense search
        dense_results = self.dense_search(query, top_k=dense_k)
        
        # Step 2: BM25 search
        bm25_results = self.bm25_search(query, top_k=bm25_k)
        
        # Step 3: RRF fusion
        fused_results = self.reciprocal_rank_fusion(
            dense_results,
            bm25_results
        )[:fusion_top_k]
        
        # Step 4: Reranking
        final_results, confidence = self.rerank(
            query,
            fused_results,
            top_k=final_k
        )
        
        return final_results, confidence
    
    def query(
        self,
        user_query: str,
        context_only: bool = False
    ) -> Optional[Dict]:
        """
        Process a user query
        
        Args:
            user_query: User's question
            context_only: Return context without generation
            
        Returns:
            Response dict with answer and sources
        """
        # Search
        results, confidence = self.hybrid_search(user_query)
        
        # Check confidence threshold
        if confidence < config.TIER3_CONFIDENCE_THRESHOLD:
            return {
                "tier": "tier_3",
                "answer": "I could not find sufficient information in the official documents to answer this question accurately.",
                "confidence": "low",
                "confidence_score": confidence,
                "sources": []
            }
        
        # Format sources
        sources = [
            {
                'source': r['metadata']['source'],
                'page': r['metadata']['page'],
                'contains_table': r['metadata'].get('contains_table', False),
                'score': r['rerank_score']
            }
            for r in results
        ]
        
        # If context_only, return retrieved context
        if context_only:
            context_text = "\n\n".join([
                self._clean_text(r['text'])
                for r in results[:3]
            ])
            
            return {
                "tier": "tier_3",
                "answer": f"Retrieved context:\n\n{context_text}",
                "confidence": self._get_confidence_level(confidence),
                "confidence_score": confidence,
                "sources": sources,
                "context": context_text
            }
        
        # Otherwise, return with sources for generation
        # (Generation will be handled by Tier 2 SLM in orchestrator)
        return {
            "tier": "tier_3",
            "confidence": self._get_confidence_level(confidence),
            "confidence_score": confidence,
            "sources": sources,
            "results": results  # For generation
        }
    
    def _clean_text(self, text: str) -> str:
        """Remove markers and clean text"""
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'---.*?---', '', text)
        return text.strip()
    
    def _get_confidence_level(self, score: float) -> str:
        """Convert score to confidence level"""
        if score > 0.6:
            return "high"
        elif score > 0.3:
            return "medium"
        else:
            return "low"
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        if not hasattr(self, 'chunks') or not self.chunks:
            return {
                "total_chunks": 0,
                "table_chunks": 0,
                "embedding_model": config.TIER3_EMBEDDING_MODEL,
                "reranker_model": config.TIER3_RERANKER_MODEL,
                "collection_name": config.CHROMA_COLLECTION_NAME,
                "status": "not_initialized"
            }
        
        table_chunks = sum(
            1 for c in self.chunks 
            if c['metadata'].get('contains_table', False)
        )
        
        return {
            "total_chunks": len(self.chunks),
            "table_chunks": table_chunks,
            "embedding_model": config.TIER3_EMBEDDING_MODEL,
            "reranker_model": config.TIER3_RERANKER_MODEL,
            "collection_name": config.CHROMA_COLLECTION_NAME,
            "status": "ready"
        }

# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("TESTING TIER 3 MODULE")
    print("="*60)
    
    # Initialize
    rag = Tier3RAGSystem()
    rag.initialize()
    
    # Test queries
    test_queries = [
        "What is the interest rate for personal loans?",
        "What is the LTV ratio for home loans?",
    ]
    
    print("\n🧪 Testing queries...\n")
    for query in test_queries:
        result = rag.query(query, context_only=True)
        
        print(f"✅ {query}")
        print(f"   Confidence: {result['confidence']} ({result['confidence_score']:.3f})")
        print(f"   Sources: {len(result['sources'])}")
        for src in result['sources'][:2]:
            print(f"      • {src['source']} (Page {src['page']})")
        print()
    
    # Print stats
    print("📊 Stats:", rag.get_stats())
