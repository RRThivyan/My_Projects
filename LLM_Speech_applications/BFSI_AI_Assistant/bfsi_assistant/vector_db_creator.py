# """
# Vector Database Creator
# Creates and manages vector database for RAG system
# """

# import os
# import re
# import pdfplumber
# import fitz
# import chromadb
# from chromadb.config import Settings
# from sentence_transformers import SentenceTransformer
# from tqdm import tqdm
# from typing import List, Dict
# import config

# class VectorDBCreator:
#     """
#     Creates and manages vector database for document retrieval
    
#     Features:
#     - PDF extraction with table handling
#     - Smart chunking with overlap
#     - Semantic embeddings
#     - ChromaDB storage
#     """
    
#     def __init__(
#         self,
#         docs_path: str = None,
#         index_path: str = None,
#         embedding_model: str = None
#     ):
#         """
#         Initialize the vector DB creator
        
#         Args:
#             docs_path: Path to PDF documents
#             index_path: Path to store index
#             embedding_model: Embedding model name
#         """
#         self.docs_path = docs_path or config.DOCS_PATH
#         self.index_path = index_path or config.INDEX_PATH
#         self.embedding_model_name = embedding_model or config.TIER3_EMBEDDING_MODEL
        
#         self.documents = []
#         self.chunks = []
#         self.embeddings = None
#         self.embedding_model = None
#         self.collection = None
        
#         print(f"🔧 Initializing Vector DB Creator...")
#         print(f"   Documents: {self.docs_path}")
#         print(f"   Index: {self.index_path}")
    
#     def extract_pdf(self, filepath: str) -> Dict:
#         """Extract text from PDF with table handling"""
#         filename = os.path.basename(filepath)
#         full_text = ""
#         tables_found = 0
        
#         try:
#             # Try pdfplumber first (better for tables)
#             with pdfplumber.open(filepath) as pdf:
#                 for page_num, page in enumerate(pdf.pages, 1):
#                     full_text += f"\n--- Page {page_num} ---\n"
                    
#                     # Extract text
#                     page_text = page.extract_text()
#                     if page_text:
#                         full_text += page_text + "\n"
                    
#                     # Extract tables
#                     tables = page.extract_tables()
#                     if tables:
#                         for table in tables:
#                             tables_found += 1
#                             full_text += "\n[TABLE]\n"
                            
#                             # Clean table
#                             for row in table:
#                                 clean_row = [str(cell or "").strip() for cell in row]
#                                 full_text += " | ".join(clean_row) + "\n"
                
#                 return {
#                     'filename': filename,
#                     'text': full_text,
#                     'num_pages': len(pdf.pages),
#                     'tables_found': tables_found
#                 }
        
#         except Exception as e:
#             # Fallback to fitz
#             try:
#                 doc = fitz.open(filepath)
#                 for page_num, page in enumerate(doc, 1):
#                     full_text += f"\n--- Page {page_num} ---\n"
#                     full_text += page.get_text() + "\n"
                
#                 return {
#                     'filename': filename,
#                     'text': full_text,
#                     'num_pages': page_num,
#                     'tables_found': 0
#                 }
#             except Exception as e2:
#                 print(f"   ❌ Failed to extract {filename}: {e2}")
#                 return None
    
#     def load_documents(self) -> List[Dict]:
#         """Load all PDF documents"""
#         print(f"\n📚 Loading documents...")
        
#         pdf_files = [f for f in os.listdir(self.docs_path) if f.endswith('.pdf')]
#         print(f"   Found {len(pdf_files)} PDF files")
        
#         documents = []
#         total_tables = 0
        
#         for filename in pdf_files:
#             filepath = os.path.join(self.docs_path, filename)
#             doc = self.extract_pdf(filepath)
            
#             if doc:
#                 documents.append(doc)
#                 total_tables += doc['tables_found']
#                 print(f"   ✅ {filename}: {doc['num_pages']} pages, {doc['tables_found']} tables")
        
#         print(f"\n   📊 Total: {len(documents)} documents, {total_tables} tables")
        
#         self.documents = documents
#         return documents
    
#     def chunk_text(self, text: str) -> List[str]:
#         """
#         Smart text chunking with overlap
        
#         Preserves table boundaries
#         """
#         chunks = []
#         current_chunk = ""
#         in_table = False
        
#         lines = text.split('\n')
        
#         for line in lines:
#             # Detect table boundaries
#             if '[TABLE]' in line:
#                 if current_chunk.strip():
#                     chunks.append(current_chunk.strip())
#                     current_chunk = ""
#                 in_table = True
            
#             current_chunk += line + "\n"
            
#             # Check if we should create a chunk
#             if in_table:
#                 if '[TABLE]' in line and current_chunk.count('[TABLE]') >= 2:
#                     chunks.append(current_chunk.strip())
#                     current_chunk = ""
#                     in_table = False
#             else:
#                 if len(current_chunk) >= config.TIER3_CHUNK_SIZE:
#                     # Try to break at sentence
#                     sentences = current_chunk.split('. ')
#                     if len(sentences) > 1:
#                         chunks.append('. '.join(sentences[:-1]) + '.')
#                         current_chunk = sentences[-1]
#                     else:
#                         chunks.append(current_chunk[:config.TIER3_CHUNK_SIZE])
#                         current_chunk = current_chunk[config.TIER3_CHUNK_SIZE - config.TIER3_CHUNK_OVERLAP:]
        
#         if current_chunk.strip():
#             chunks.append(current_chunk.strip())
        
#         return chunks
    
#     def create_chunks(self) -> List[Dict]:
#         """Create chunks from all documents"""
#         print(f"\n📝 Creating chunks...")
#         print(f"   Chunk size: {config.TIER3_CHUNK_SIZE}")
#         print(f"   Overlap: {config.TIER3_CHUNK_OVERLAP}")
        
#         all_chunks = []
        
#         for doc in self.documents:
#             chunks = self.chunk_text(doc['text'])
            
#             for i, chunk in enumerate(chunks):
#                 # Extract page number
#                 page_match = re.search(r'--- Page (\d+) ---', chunk)
#                 page = page_match.group(1) if page_match else '1'
                
#                 all_chunks.append({
#                     'text': chunk,
#                     'contextualized_text': f"[{doc['filename']} p{page}] {chunk}",
#                     'metadata': {
#                         'source': doc['filename'],
#                         'page': page,
#                         'chunk_id': i,
#                         'contains_table': '[TABLE]' in chunk
#                     }
#                 })
        
#         table_chunks = sum(1 for c in all_chunks if c['metadata']['contains_table'])
#         print(f"   ✅ Created {len(all_chunks)} chunks ({table_chunks} with tables)")
        
#         self.chunks = all_chunks
#         return all_chunks
    
#     def create_embeddings(self):
#         """Create embeddings for all chunks"""
#         print(f"\n🔄 Creating embeddings...")
#         print(f"   Model: {self.embedding_model_name}")
        
#         self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
#         texts = [c['contextualized_text'] for c in self.chunks]
        
#         self.embeddings = self.embedding_model.encode(
#             texts,
#             batch_size=32,
#             normalize_embeddings=True,
#             show_progress_bar=True
#         )
        
#         print(f"   ✅ Created {len(self.embeddings)} embeddings")
    
#     def store_in_chromadb(self):
#         """Store chunks and embeddings in ChromaDB"""
#         print(f"\n💾 Storing in ChromaDB...")
        
#         # Initialize client
#         chroma_client = chromadb.Client(Settings(
#             persist_directory=self.index_path,
#             anonymized_telemetry=False
#         ))
        
#         # Delete existing collection
#         try:
#             chroma_client.delete_collection(config.CHROMA_COLLECTION_NAME)
#             print(f"   🗑️  Deleted existing collection")
#         except:
#             pass
        
#         # Create collection
#         self.collection = chroma_client.create_collection(
#             name=config.CHROMA_COLLECTION_NAME,
#             metadata={"description": "BFSI documents"}
#         )
        
#         # Add documents in batches
#         batch_size = 100
#         for i in tqdm(range(0, len(self.chunks), batch_size), desc="Uploading"):
#             batch_chunks = self.chunks[i:i + batch_size]
#             batch_embeddings = self.embeddings[i:i + batch_size]
            
#             self.collection.add(
#                 embeddings=batch_embeddings.tolist(),
#                 documents=[c['contextualized_text'] for c in batch_chunks],
#                 metadatas=[c['metadata'] for c in batch_chunks],
#                 ids=[f"chunk_{j}" for j in range(i, i + len(batch_chunks))]
#             )
        
#         print(f"   ✅ Stored {len(self.chunks)} chunks")
    
#     def create_database(self):
#         """Complete database creation pipeline"""
#         print("="*60)
#         print("CREATING VECTOR DATABASE")
#         print("="*60)
        
#         # Load documents
#         self.load_documents()
        
#         # Create chunks
#         self.create_chunks()
        
#         # Create embeddings
#         self.create_embeddings()
        
#         # Store in ChromaDB
#         self.store_in_chromadb()
        
#         print("\n" + "="*60)
#         print("✅ VECTOR DATABASE CREATED")
#         print("="*60)
        
#         return {
#             'num_documents': len(self.documents),
#             'num_chunks': len(self.chunks),
#             'num_tables': sum(d['tables_found'] for d in self.documents),
#             'embedding_dim': self.embeddings.shape[1]
#         }

# # ============================================================
# # MODULE TEST
# # ============================================================

# if __name__ == "__main__":
#     print("="*60)
#     print("TESTING VECTOR DB CREATOR")
#     print("="*60)
    
#     # Create database
#     creator = VectorDBCreator()
#     stats = creator.create_database()
    
#     print("\n📊 Database Stats:")
#     for key, value in stats.items():
#         print(f"   {key}: {value}")



"""
Vector Database Creator - FIXED VERSION
Explicitly persists ChromaDB to disk
"""

import os
import re
import pdfplumber
import fitz
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import List, Dict
import config
import shutil

class VectorDBCreator:
    """
    Creates and manages vector database for document retrieval
    
    Features:
    - PDF extraction with table handling
    - Smart chunking with overlap
    - Semantic embeddings
    - ChromaDB storage with explicit persistence
    """
    
    def __init__(
        self,
        docs_path: str = None,
        index_path: str = None,
        embedding_model: str = None
    ):
        """
        Initialize the vector DB creator
        
        Args:
            docs_path: Path to PDF documents
            index_path: Path to store index
            embedding_model: Embedding model name
        """
        self.docs_path = docs_path or config.DOCS_PATH
        self.index_path = index_path or config.INDEX_PATH
        self.embedding_model_name = embedding_model or config.TIER3_EMBEDDING_MODEL
        
        self.documents = []
        self.chunks = []
        self.embeddings = None
        self.embedding_model = None
        self.collection = None
        self.chroma_client = None
        
        print(f"🔧 Initializing Vector DB Creator...")
        print(f"   Documents: {self.docs_path}")
        print(f"   Index: {self.index_path}")
    
    def extract_pdf(self, filepath: str) -> Dict:
        """Extract text from PDF with table handling"""
        filename = os.path.basename(filepath)
        full_text = ""
        tables_found = 0
        
        try:
            # Try pdfplumber first (better for tables)
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    full_text += f"\n--- Page {page_num} ---\n"
                    
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
                    
                    # Extract tables
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            tables_found += 1
                            full_text += "\n[TABLE]\n"
                            
                            # Clean table
                            for row in table:
                                clean_row = [str(cell or "").strip() for cell in row]
                                full_text += " | ".join(clean_row) + "\n"
                
                return {
                    'filename': filename,
                    'text': full_text,
                    'num_pages': len(pdf.pages),
                    'tables_found': tables_found
                }
        
        except Exception as e:
            # Fallback to fitz
            try:
                doc = fitz.open(filepath)
                for page_num, page in enumerate(doc, 1):
                    full_text += f"\n--- Page {page_num} ---\n"
                    full_text += page.get_text() + "\n"
                
                return {
                    'filename': filename,
                    'text': full_text,
                    'num_pages': page_num,
                    'tables_found': 0
                }
            except Exception as e2:
                print(f"   ❌ Failed to extract {filename}: {e2}")
                return None
    
    def load_documents(self) -> List[Dict]:
        """Load all PDF documents"""
        print(f"\n📚 Loading documents...")
        
        pdf_files = [f for f in os.listdir(self.docs_path) if f.endswith('.pdf')]
        print(f"   Found {len(pdf_files)} PDF files")
        
        documents = []
        total_tables = 0
        
        for filename in pdf_files:
            filepath = os.path.join(self.docs_path, filename)
            doc = self.extract_pdf(filepath)
            
            if doc:
                documents.append(doc)
                total_tables += doc['tables_found']
                print(f"   ✅ {filename}: {doc['num_pages']} pages, {doc['tables_found']} tables")
        
        print(f"\n   📊 Total: {len(documents)} documents, {total_tables} tables")
        
        self.documents = documents
        return documents
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Smart text chunking with overlap
        
        Preserves table boundaries
        """
        chunks = []
        current_chunk = ""
        in_table = False
        
        lines = text.split('\n')
        
        for line in lines:
            # Detect table boundaries
            if '[TABLE]' in line:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                in_table = True
            
            current_chunk += line + "\n"
            
            # Check if we should create a chunk
            if in_table:
                if '[TABLE]' in line and current_chunk.count('[TABLE]') >= 2:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    in_table = False
            else:
                if len(current_chunk) >= config.TIER3_CHUNK_SIZE:
                    # Try to break at sentence
                    sentences = current_chunk.split('. ')
                    if len(sentences) > 1:
                        chunks.append('. '.join(sentences[:-1]) + '.')
                        current_chunk = sentences[-1]
                    else:
                        chunks.append(current_chunk[:config.TIER3_CHUNK_SIZE])
                        current_chunk = current_chunk[config.TIER3_CHUNK_SIZE - config.TIER3_CHUNK_OVERLAP:]
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def create_chunks(self) -> List[Dict]:
        """Create chunks from all documents"""
        print(f"\n📝 Creating chunks...")
        print(f"   Chunk size: {config.TIER3_CHUNK_SIZE}")
        print(f"   Overlap: {config.TIER3_CHUNK_OVERLAP}")
        
        all_chunks = []
        
        for doc in self.documents:
            chunks = self.chunk_text(doc['text'])
            
            for i, chunk in enumerate(chunks):
                # Extract page number
                page_match = re.search(r'--- Page (\d+) ---', chunk)
                page = page_match.group(1) if page_match else '1'
                
                all_chunks.append({
                    'text': chunk,
                    'contextualized_text': f"[{doc['filename']} p{page}] {chunk}",
                    'metadata': {
                        'source': doc['filename'],
                        'page': page,
                        'chunk_id': i,
                        'contains_table': '[TABLE]' in chunk
                    }
                })
        
        table_chunks = sum(1 for c in all_chunks if c['metadata']['contains_table'])
        print(f"   ✅ Created {len(all_chunks)} chunks ({table_chunks} with tables)")
        
        self.chunks = all_chunks
        return all_chunks
    
    def create_embeddings(self):
        """Create embeddings for all chunks"""
        print(f"\n🔄 Creating embeddings...")
        print(f"   Model: {self.embedding_model_name}")
        
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        texts = [c['contextualized_text'] for c in self.chunks]
        
        self.embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        print(f"   ✅ Created {len(self.embeddings)} embeddings")
    
    def store_in_chromadb(self):
        """Store chunks and embeddings in ChromaDB with explicit persistence"""
        print(f"\n💾 Storing in ChromaDB...")
        
        # Ensure directory exists
        os.makedirs(self.index_path, exist_ok=True)
        
        # Clear existing database if any
        chroma_dir = os.path.join(self.index_path, 'chroma')
        if os.path.exists(chroma_dir):
            print(f"   🗑️  Removing existing database...")
            shutil.rmtree(chroma_dir)
        
        # Initialize client with explicit persistence
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_dir
        )
        
        # Create collection
        self.collection = self.chroma_client.create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"description": "BFSI documents"}
        )
        
        # Add documents in batches
        batch_size = 100
        for i in tqdm(range(0, len(self.chunks), batch_size), desc="Uploading"):
            batch_chunks = self.chunks[i:i + batch_size]
            batch_embeddings = self.embeddings[i:i + batch_size]
            
            self.collection.add(
                embeddings=batch_embeddings.tolist(),
                documents=[c['contextualized_text'] for c in batch_chunks],
                metadatas=[c['metadata'] for c in batch_chunks],
                ids=[f"chunk_{j}" for j in range(i, i + len(batch_chunks))]
            )
        
        # Force persist
        self.chroma_client.persist()
        
        print(f"   ✅ Stored {len(self.chunks)} chunks")
        print(f"   📁 Database location: {chroma_dir}")
    
    def verify_database(self):
        """Verify the database was created correctly"""
        print(f"\n🔍 Verifying database...")
        
        # Check if directory exists
        chroma_dir = os.path.join(self.index_path, 'chroma')
        if os.path.exists(chroma_dir):
            files = os.listdir(chroma_dir)
            print(f"   📁 Database directory: {chroma_dir}")
            print(f"   📊 Files: {len(files)} items")
            if files:
                print(f"   ✅ Database files present")
                
                # Try to load and query
                try:
                    client = chromadb.PersistentClient(path=chroma_dir)
                    collection = client.get_collection(config.CHROMA_COLLECTION_NAME)
                    count = collection.count()
                    print(f"   ✅ Collection '{config.CHROMA_COLLECTION_NAME}' has {count} chunks")
                    return True
                except Exception as e:
                    print(f"   ❌ Failed to verify: {e}")
                    return False
        else:
            print(f"   ❌ Database directory not found")
            return False
    
    def create_database(self):
        """Complete database creation pipeline"""
        print("="*60)
        print("CREATING VECTOR DATABASE")
        print("="*60)
        
        # Load documents
        self.load_documents()
        
        # Create chunks
        self.create_chunks()
        
        # Create embeddings
        self.create_embeddings()
        
        # Store in ChromaDB
        self.store_in_chromadb()
        
        # Verify
        self.verify_database()
        
        print("\n" + "="*60)
        print("✅ VECTOR DATABASE CREATED")
        print("="*60)
        
        return {
            'num_documents': len(self.documents),
            'num_chunks': len(self.chunks),
            'num_tables': sum(d['tables_found'] for d in self.documents),
            'embedding_dim': self.embeddings.shape[1]
        }

# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("TESTING VECTOR DB CREATOR")
    print("="*60)
    
    # Create database
    creator = VectorDBCreator()
    stats = creator.create_database()
    
    print("\n📊 Database Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
