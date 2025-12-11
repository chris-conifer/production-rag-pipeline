"""
Reranker Factory - Multiple Reranker Options
Supports: CrossEncoder, Cohere, Jina, BGE, RankGPT
"""

import os
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
import torch


class BaseReranker(ABC):
    """Abstract base class for all rerankers"""
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents for a query
        
        Args:
            query: Query text
            documents: List of document texts
            top_k: Number of top results to return
            
        Returns:
            List of dicts with 'text', 'original_index', 'rerank_score'
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get reranker information"""
        pass


class CrossEncoderReranker(BaseReranker):
    """
    Cross-Encoder Reranker (Local, Free)
    
    Models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast)
    - cross-encoder/ms-marco-MiniLM-L-12-v2 (better)
    - cross-encoder/stsb-roberta-base (alternative)
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "auto",
        batch_size: int = 32
    ):
        from sentence_transformers import CrossEncoder
        
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        self.model = CrossEncoder(model_name, device=self.device)
        print(f"  ✓ CrossEncoderReranker loaded: {model_name} on {self.device}")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []
        
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        
        results = [
            {
                'text': doc,
                'original_index': i,
                'rerank_score': float(score)
            }
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]
        
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        return results[:top_k]
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'cross_encoder',
            'model_name': self.model_name,
            'device': self.device,
            'cost': 'free'
        }


class CohereReranker(BaseReranker):
    """
    Cohere Rerank API - State-of-the-art commercial reranker
    
    Models:
    - rerank-english-v3.0 (best for English)
    - rerank-multilingual-v3.0 (100+ languages)
    - rerank-english-v2.0 (faster, cheaper)
    
    Pricing: ~$1 per 1K searches
    Docs: https://docs.cohere.com/docs/rerank
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "rerank-english-v3.0"
    ):
        import cohere
        
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY required. Set in .env or pass api_key parameter.")
        
        self.model = model
        self.client = cohere.Client(self.api_key)
        print(f"  ✓ CohereReranker initialized: {model}")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []
        
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_k,
            return_documents=True
        )
        
        results = []
        for result in response.results:
            results.append({
                'text': result.document.text,
                'original_index': result.index,
                'rerank_score': result.relevance_score  # 0-1 scale
            })
        
        return results
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'cohere',
            'model': self.model,
            'cost': '~$1/1K searches'
        }


class JinaReranker(BaseReranker):
    """
    Jina Reranker - Best open-source option
    
    Models:
    - jinaai/jina-reranker-v2-base-multilingual (best, 278M params)
    - jinaai/jina-reranker-v1-base-en (English only)
    - jinaai/jina-reranker-v1-tiny-en (fast, 33M params)
    
    Docs: https://jina.ai/reranker
    HuggingFace: https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual
    """
    
    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        device: str = "auto",
        batch_size: int = 32
    ):
        from sentence_transformers import CrossEncoder
        
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        # Jina models require trust_remote_code
        self.model = CrossEncoder(
            model_name,
            device=self.device,
            trust_remote_code=True
        )
        print(f"  ✓ JinaReranker loaded: {model_name} on {self.device}")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []
        
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        
        results = [
            {
                'text': doc,
                'original_index': i,
                'rerank_score': float(score)
            }
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]
        
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        return results[:top_k]
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'jina',
            'model_name': self.model_name,
            'device': self.device,
            'cost': 'free'
        }


class BGEReranker(BaseReranker):
    """
    BGE Reranker from BAAI - Top performer on HuggingFace leaderboard
    
    Models (ranked by quality):
    - BAAI/bge-reranker-v2-m3 (BEST multilingual, 568M params)
    - BAAI/bge-reranker-large (high quality English, 560M params)
    - BAAI/bge-reranker-base (balanced, 278M params)
    
    Benchmark: https://huggingface.co/spaces/mteb/leaderboard (Reranking tab)
    Docs: https://github.com/FlagOpen/FlagEmbedding
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "auto",
        batch_size: int = 32
    ):
        from sentence_transformers import CrossEncoder
        
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        self.model = CrossEncoder(model_name, device=self.device)
        print(f"  [OK] BGEReranker loaded: {model_name} on {self.device}")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []
        
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        
        results = [
            {
                'text': doc,
                'original_index': i,
                'rerank_score': float(score)
            }
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]
        
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        return results[:top_k]
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'bge',
            'model_name': self.model_name,
            'device': self.device,
            'cost': 'free'
        }


class BGELargeReranker(BaseReranker):
    """
    BGE Reranker Large - High quality English reranker
    
    Model: BAAI/bge-reranker-large (560M params)
    Slightly better for English-only tasks than v2-m3
    
    Benchmark: Top 5 on MTEB Reranking leaderboard
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str = "auto",
        batch_size: int = 32
    ):
        from sentence_transformers import CrossEncoder
        
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        self.model = CrossEncoder(model_name, device=self.device)
        print(f"  [OK] BGELargeReranker loaded: {model_name} on {self.device}")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []
        
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        
        results = [
            {
                'text': doc,
                'original_index': i,
                'rerank_score': float(score)
            }
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]
        
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        return results[:top_k]
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'bge_large',
            'model_name': self.model_name,
            'device': self.device,
            'cost': 'free'
        }


class LLMReranker(BaseReranker):
    """
    LLM-as-Reranker (RankGPT style)
    
    Uses LLM (GPT-4, Claude, local) to rerank documents
    Best for complex queries, multi-hop reasoning
    
    Paper: https://arxiv.org/abs/2304.09542
    """
    
    def __init__(
        self,
        provider: str = "openai",  # openai, anthropic, local
        model: str = "gpt-3.5-turbo",
        api_key: str = None
    ):
        self.provider = provider
        self.model = model
        
        if provider == "openai":
            from openai import OpenAI
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY required")
            self.client = OpenAI(api_key=self.api_key)
            
        elif provider == "anthropic":
            import anthropic
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY required")
            self.client = anthropic.Anthropic(api_key=self.api_key)
        
        print(f"  ✓ LLMReranker initialized: {provider}/{model}")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []
        
        # Truncate documents for prompt
        truncated_docs = [doc[:500] for doc in documents]
        
        # Create prompt
        docs_text = "\n".join([f"{i}. {doc}" for i, doc in enumerate(truncated_docs)])
        
        prompt = f"""Given the query and documents below, rank the documents by relevance to the query.
Output ONLY the document numbers in order of relevance (most relevant first), separated by commas.
Do not include any explanation.

Query: {query}

Documents:
{docs_text}

Ranking (comma-separated numbers, most relevant first):"""
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=100
                )
                ranking_str = response.choices[0].message.content.strip()
                
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}]
                )
                ranking_str = response.content[0].text.strip()
            
            # Parse ranking
            rankings = []
            for x in ranking_str.replace('\n', ',').split(','):
                x = x.strip()
                if x.isdigit():
                    idx = int(x)
                    if 0 <= idx < len(documents):
                        rankings.append(idx)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_rankings = []
            for idx in rankings:
                if idx not in seen:
                    seen.add(idx)
                    unique_rankings.append(idx)
            
            # Add any missing indices
            for i in range(len(documents)):
                if i not in seen:
                    unique_rankings.append(i)
            
            # Create results
            results = []
            for rank, idx in enumerate(unique_rankings[:top_k]):
                results.append({
                    'text': documents[idx],
                    'original_index': idx,
                    'rerank_score': 1.0 - (rank / len(unique_rankings))  # Normalized score
                })
            
            return results
            
        except Exception as e:
            print(f"  ⚠ LLMReranker error: {e}. Returning original order.")
            return [
                {'text': doc, 'original_index': i, 'rerank_score': 1.0 - (i / len(documents))}
                for i, doc in enumerate(documents[:top_k])
            ]
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'llm',
            'provider': self.provider,
            'model': self.model,
            'cost': 'variable'
        }


class RerankerFactory:
    """
    Factory for creating rerankers
    
    Supported types:
    - cross_encoder: Local cross-encoder (default, free)
    - cohere: Cohere Rerank API (best quality, paid)
    - jina: Jina Reranker (best open-source)
    - bge: BGE Reranker (fast & good)
    - llm: LLM-as-reranker (GPT-4, Claude)
    """
    
    # Model presets for easy selection
    PRESETS = {
        # CrossEncoder presets
        "cross_encoder_fast": {
            "type": "cross_encoder",
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"
        },
        "cross_encoder_quality": {
            "type": "cross_encoder",
            "model_name": "cross-encoder/ms-marco-MiniLM-L-12-v2"
        },
        
        # Cohere presets
        "cohere_v3": {
            "type": "cohere",
            "model": "rerank-english-v3.0"
        },
        "cohere_multilingual": {
            "type": "cohere",
            "model": "rerank-multilingual-v3.0"
        },
        
        # Jina presets
        "jina_v2": {
            "type": "jina",
            "model_name": "jinaai/jina-reranker-v2-base-multilingual"
        },
        "jina_tiny": {
            "type": "jina",
            "model_name": "jinaai/jina-reranker-v1-tiny-en"
        },
        
        # BGE presets
        "bge_v2": {
            "type": "bge",
            "model_name": "BAAI/bge-reranker-v2-m3"
        },
        "bge_large": {
            "type": "bge",
            "model_name": "BAAI/bge-reranker-large"
        },
        
        # LLM presets
        "gpt4_reranker": {
            "type": "llm",
            "provider": "openai",
            "model": "gpt-4"
        },
        "gpt35_reranker": {
            "type": "llm",
            "provider": "openai",
            "model": "gpt-3.5-turbo"
        }
    }
    
    @classmethod
    def create(
        cls,
        reranker_type: str = "cross_encoder",
        preset: str = None,
        **kwargs
    ) -> BaseReranker:
        """
        Create a reranker instance
        
        Args:
            reranker_type: Type of reranker (cross_encoder, cohere, jina, bge, llm)
            preset: Optional preset name (e.g., "jina_v2", "cohere_v3")
            **kwargs: Additional arguments for the reranker
            
        Returns:
            BaseReranker instance
        """
        # Use preset if provided
        if preset and preset in cls.PRESETS:
            preset_config = cls.PRESETS[preset].copy()
            reranker_type = preset_config.pop("type")
            kwargs = {**preset_config, **kwargs}
        
        # Create reranker
        if reranker_type == "cross_encoder":
            return CrossEncoderReranker(**kwargs)
        
        elif reranker_type == "cohere":
            return CohereReranker(**kwargs)
        
        elif reranker_type == "jina":
            return JinaReranker(**kwargs)
        
        elif reranker_type == "bge":
            return BGEReranker(**kwargs)
        
        elif reranker_type == "llm":
            return LLMReranker(**kwargs)
        
        else:
            raise ValueError(f"Unknown reranker type: {reranker_type}. "
                           f"Choose from: cross_encoder, cohere, jina, bge, llm")
    
    @classmethod
    def list_presets(cls) -> Dict[str, Dict]:
        """List all available presets"""
        return cls.PRESETS
    
    @classmethod
    def list_types(cls) -> List[str]:
        """List all supported reranker types"""
        return ["cross_encoder", "cohere", "jina", "bge", "llm"]


# Convenience function
def create_reranker(
    reranker_type: str = "cross_encoder",
    preset: str = None,
    **kwargs
) -> BaseReranker:
    """
    Convenience function to create a reranker
    
    Examples:
        # Default cross-encoder
        reranker = create_reranker()
        
        # Use preset
        reranker = create_reranker(preset="jina_v2")
        
        # Cohere with API key
        reranker = create_reranker("cohere", api_key="your-key")
        
        # Custom model
        reranker = create_reranker("bge", model_name="BAAI/bge-reranker-large")
    """
    return RerankerFactory.create(reranker_type, preset, **kwargs)

