"""
Document Processing Module
Handles chunking, preprocessing, and metadata extraction

Supports multiple chunking strategies:
- fixed_size: Split at character boundaries with overlap
- sentence: Group sentences up to chunk_size
- paragraph: Split by paragraph boundaries
- semantic: Use embedding similarity to detect topic shifts
- contextual: Paragraph-aware with sentence boundaries
- token_based: Split by approximate token count
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import re


@dataclass
class Chunk:
    """Represents a document chunk with metadata"""
    text: str
    chunk_id: str
    context_id: str
    chunk_index: int
    total_chunks: int
    chunk_length: int
    chunk_words: int
    original_context: str
    start_char: int
    end_char: int


class DocumentProcessor:
    """
    Handles document processing: chunking, deduplication, metadata extraction
    
    Chunking Strategies:
    - fixed_size: Simple character-based splitting with overlap
    - sentence: Groups complete sentences up to chunk_size
    - paragraph: Respects paragraph boundaries (double newlines)
    - semantic: Detects topic shifts using embedding similarity
    - contextual: Paragraph-aware chunking that respects sentence boundaries
    - token_based: Splits by approximate token count (words / 0.75)
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        strategy: str = "fixed_size",
        min_chunk_length: int = 50,
        semantic_threshold: float = 0.5
    ):
        """
        Initialize Document Processor
        
        Args:
            chunk_size: Maximum chunk size in characters (or tokens for token_based)
            chunk_overlap: Overlap between chunks
            strategy: Chunking strategy
            min_chunk_length: Minimum chunk length to keep
            semantic_threshold: Similarity threshold for semantic chunking (0-1)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.min_chunk_length = min_chunk_length
        self.semantic_threshold = semantic_threshold
        self._embedder = None  # Lazy load for semantic chunking
        
    def chunk_documents(
        self,
        documents: List[str],
        document_ids: List[str] = None
    ) -> List[Chunk]:
        """
        Chunk all documents and return with metadata
        
        Args:
            documents: List of document texts
            document_ids: Optional list of document IDs
            
        Returns:
            List of Chunk objects with metadata
        """
        if document_ids is None:
            document_ids = [f"doc_{i}" for i in range(len(documents))]
            
        all_chunks = []
        
        for doc_idx, (document, doc_id) in enumerate(zip(documents, document_ids)):
            # Select chunking strategy
            if self.strategy == "fixed_size":
                chunks = self._chunk_fixed_size(document)
            elif self.strategy == "sentence" or self.strategy == "sentence_based":
                chunks = self._chunk_by_sentence(document)
            elif self.strategy == "paragraph":
                chunks = self._chunk_by_paragraph(document)
            elif self.strategy == "semantic":
                chunks = self._chunk_semantic(document)
            elif self.strategy == "contextual":
                chunks = self._chunk_contextual(document)
            elif self.strategy == "token_based":
                chunks = self._chunk_by_tokens(document)
            else:
                raise ValueError(f"Unknown chunking strategy: {self.strategy}. "
                               f"Available: fixed_size, sentence, paragraph, semantic, contextual, token_based")
            
            # Create Chunk objects with metadata
            for chunk_idx, (chunk_text, start, end) in enumerate(chunks):
                if len(chunk_text.strip()) < self.min_chunk_length:
                    continue
                    
                chunk_obj = Chunk(
                    text=chunk_text,
                    chunk_id=f"chunk_{len(all_chunks)}",
                    context_id=doc_id,
                    chunk_index=chunk_idx,
                    total_chunks=len(chunks),
                    chunk_length=len(chunk_text),
                    chunk_words=len(chunk_text.split()),
                    original_context=document,
                    start_char=start,
                    end_char=end
                )
                all_chunks.append(chunk_obj)
        
        return all_chunks
    
    # =========================================================================
    # CHUNKING STRATEGIES
    # =========================================================================
    
    def _chunk_fixed_size(self, text: str) -> List[tuple]:
        """
        Split text into fixed-size overlapping chunks
        Simple but may cut mid-sentence
        
        Returns:
            List of (chunk_text, start_pos, end_pos) tuples
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunks.append((chunk_text, start, end))
            
            # Move start position with overlap
            start += (self.chunk_size - self.chunk_overlap)
            
            if end >= text_length:
                break
        
        return chunks
    
    def _chunk_by_sentence(self, text: str) -> List[tuple]:
        """
        Chunk by sentences - keeps complete sentences together
        Better for maintaining semantic coherence
        
        Returns:
            List of (chunk_text, start_pos, end_pos) tuples
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        start_pos = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                end_pos = start_pos + len(chunk_text)
                chunks.append((chunk_text, start_pos, end_pos))
                
                # Start new chunk with overlap (keep last sentence)
                if self.chunk_overlap > 0 and len(current_chunk) > 1:
                    start_pos = end_pos - len(current_chunk[-1])
                    current_chunk = [current_chunk[-1], sentence]
                    current_length = len(current_chunk[-1]) + sentence_length
                else:
                    start_pos = end_pos
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            end_pos = start_pos + len(chunk_text)
            chunks.append((chunk_text, start_pos, end_pos))
        
        return chunks
    
    def _chunk_by_paragraph(self, text: str) -> List[tuple]:
        """
        Chunk by paragraph boundaries (double newlines)
        Best for well-structured documents with clear paragraphs
        
        Returns:
            List of (chunk_text, start_pos, end_pos) tuples
        """
        # Split by double newlines (paragraph boundaries)
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        start_pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_length = len(para)
            
            # If single paragraph exceeds chunk_size, use sentence chunking for it
            if para_length > self.chunk_size:
                # First, save current chunk if exists
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    end_pos = start_pos + len(chunk_text)
                    chunks.append((chunk_text, start_pos, end_pos))
                    start_pos = end_pos + 2  # Account for \n\n
                    current_chunk = []
                    current_length = 0
                
                # Chunk the long paragraph by sentences
                para_chunks = self._chunk_by_sentence(para)
                for chunk_text, _, _ in para_chunks:
                    chunks.append((chunk_text, start_pos, start_pos + len(chunk_text)))
                    start_pos += len(chunk_text) + 2
            
            elif current_length + para_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                end_pos = start_pos + len(chunk_text)
                chunks.append((chunk_text, start_pos, end_pos))
                
                # Start new chunk
                start_pos = end_pos + 2
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length + 2  # +2 for \n\n separator
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            end_pos = start_pos + len(chunk_text)
            chunks.append((chunk_text, start_pos, end_pos))
        
        return chunks
    
    def _chunk_semantic(self, text: str) -> List[tuple]:
        """
        Semantic chunking using embedding similarity
        Detects topic shifts by comparing sentence embeddings
        
        Best for: Documents with multiple topics, maintaining coherence
        
        Returns:
            List of (chunk_text, start_pos, end_pos) tuples
        """
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return [(text, 0, len(text))]
        
        # Get embeddings for sentences
        embeddings = self._get_sentence_embeddings(sentences)
        if embeddings is None:
            # Fallback to sentence chunking if embeddings unavailable
            return self._chunk_by_sentence(text)
        
        # Find breakpoints based on similarity
        breakpoints = [0]
        for i in range(1, len(sentences)):
            similarity = self._cosine_similarity(embeddings[i-1], embeddings[i])
            if similarity < self.semantic_threshold:
                breakpoints.append(i)
        breakpoints.append(len(sentences))
        
        # Create chunks from breakpoints
        chunks = []
        start_pos = 0
        
        for i in range(len(breakpoints) - 1):
            chunk_sentences = sentences[breakpoints[i]:breakpoints[i+1]]
            chunk_text = ' '.join(chunk_sentences)
            
            # If chunk is too large, split further
            if len(chunk_text) > self.chunk_size:
                sub_chunks = self._chunk_by_sentence(chunk_text)
                for sub_text, _, _ in sub_chunks:
                    chunks.append((sub_text, start_pos, start_pos + len(sub_text)))
                    start_pos += len(sub_text) + 1
            else:
                end_pos = start_pos + len(chunk_text)
                chunks.append((chunk_text, start_pos, end_pos))
                start_pos = end_pos + 1
        
        return chunks
    
    def _chunk_contextual(self, text: str) -> List[tuple]:
        """
        Contextual chunking: paragraph-aware with sentence boundaries
        Combines paragraph and sentence strategies for best of both
        
        Best for: General documents, maintains context and readability
        
        Returns:
            List of (chunk_text, start_pos, end_pos) tuples
        """
        # First, split by paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        start_pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Split paragraph into sentences
            sentences = self._split_into_sentences(para)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                sentence_length = len(sentence)
                
                if current_length + sentence_length > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    end_pos = start_pos + len(chunk_text)
                    chunks.append((chunk_text, start_pos, end_pos))
                    
                    # Start new chunk with context overlap
                    overlap_sentences = []
                    overlap_length = 0
                    for s in reversed(current_chunk):
                        if overlap_length + len(s) <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_length += len(s)
                        else:
                            break
                    
                    start_pos = end_pos - overlap_length
                    current_chunk = overlap_sentences + [sentence]
                    current_length = overlap_length + sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length + 1
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            end_pos = start_pos + len(chunk_text)
            chunks.append((chunk_text, start_pos, end_pos))
        
        return chunks
    
    def _chunk_by_tokens(self, text: str) -> List[tuple]:
        """
        Token-based chunking (approximate)
        Uses word count * 1.3 as token estimate (good for most LLMs)
        
        Best for: LLM context window management
        
        Returns:
            List of (chunk_text, start_pos, end_pos) tuples
        """
        # Approximate tokens: words * 1.3 (accounts for subword tokenization)
        words = text.split()
        tokens_per_word = 1.3
        max_words = int(self.chunk_size / tokens_per_word)
        overlap_words = int(self.chunk_overlap / tokens_per_word)
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + max_words, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            # Calculate character positions (approximate)
            start_char = len(' '.join(words[:start])) + (1 if start > 0 else 0)
            end_char = start_char + len(chunk_text)
            
            if chunk_text.strip():
                chunks.append((chunk_text, start_char, end_char))
            
            # Move with overlap
            start += max(1, max_words - overlap_words)
            
            if end >= len(words):
                break
        
        return chunks
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex"""
        # Handle common abbreviations and edge cases
        text = re.sub(r'(Mr|Mrs|Ms|Dr|Prof|Jr|Sr|Inc|Ltd|Corp|vs|etc|al|eg|ie)\.\s', r'\1<DOT> ', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Restore dots in abbreviations
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_sentence_embeddings(self, sentences: List[str]) -> Optional[np.ndarray]:
        """Get embeddings for sentences (lazy load embedder)"""
        try:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
            return self._embedder.encode(sentences, show_progress_bar=False)
        except Exception as e:
            print(f"[!] Semantic chunking unavailable: {e}")
            return None
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def extract_unique_contexts(
        self,
        dataset: List[Dict[str, Any]],
        context_key: str = 'context'
    ) -> tuple:
        """
        Extract unique contexts from dataset (for SQuAD-like data)
        
        Args:
            dataset: List of dataset examples
            context_key: Key name for context field
            
        Returns:
            Tuple of (unique_contexts, context_metadata)
        """
        contexts = []
        context_to_id = {}
        context_metadata = []
        
        for i, example in enumerate(dataset):
            context = example[context_key]
            
            if context not in context_to_id:
                context_id = f"doc_{len(contexts)}"
                context_to_id[context] = context_id
                contexts.append(context)
                
                context_metadata.append({
                    'context_id': context_id,
                    'context_length': len(context),
                    'context_words': len(context.split()),
                    'first_seen_idx': i
                })
        
        return contexts, context_metadata
    
    def get_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about chunked documents"""
        if not chunks:
            return {}
        
        chunk_lengths = [c.chunk_length for c in chunks]
        chunk_words = [c.chunk_words for c in chunks]
        
        return {
            'total_chunks': len(chunks),
            'unique_contexts': len(set(c.context_id for c in chunks)),
            'avg_chunk_length': np.mean(chunk_lengths),
            'min_chunk_length': np.min(chunk_lengths),
            'max_chunk_length': np.max(chunk_lengths),
            'avg_chunk_words': np.mean(chunk_words),
            'chunks_per_document': len(chunks) / len(set(c.context_id for c in chunks)),
            'strategy': self.strategy
        }
