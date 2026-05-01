"""
retriever.py — BM25 based document retrieval with domain-aware filtering.

Builds a BM25 index from the corpus and supports queries filtered by domain.
Returns top-K relevant document chunks with similarity scores.
"""

import re
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Lowercase and split text by non-alphanumeric characters."""
    return re.findall(r'\w+', text.lower())


class BM25Retriever:
    """BM25 based retriever with per-domain filtering."""
    
    def __init__(self, documents: list[dict]):
        """
        Build the BM25 index from corpus documents.
        
        Args:
            documents: List of document dicts from corpus_loader.
        """
        self.documents = documents
        
        # Pre-compute domain indices for fast filtering
        self._domain_indices = {}
        for i, doc in enumerate(documents):
            domain = doc['domain']
            if domain not in self._domain_indices:
                self._domain_indices[domain] = []
            self._domain_indices[domain].append(i)
            
        # Build per-domain BM25 indices
        self._domain_bm25 = {}
        for domain, indices in self._domain_indices.items():
            domain_texts = []
            for i in indices:
                doc = documents[i]
                combined = f"{doc['title']}\n\n{doc['content']}"
                domain_texts.append(tokenize(combined))
            self._domain_bm25[domain] = BM25Okapi(domain_texts)
            
        # Build global BM25 index for unknown domains
        global_texts = []
        for doc in documents:
            combined = f"{doc['title']}\n\n{doc['content']}"
            global_texts.append(tokenize(combined))
        self.global_bm25 = BM25Okapi(global_texts)
    
    def retrieve(
        self,
        query: str,
        domain: str | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[dict]:
        """
        Retrieve top-K documents matching the query.
        
        Args:
            query: Search query text.
            domain: Filter to this domain (hackerrank/claude/visa). None for all.
            top_k: Number of results to return.
            min_score: Minimum similarity score threshold.
            
        Returns:
            List of dicts with keys: document, score, rank
        """
        tokenized_query = tokenize(query)
        
        # Compute similarities
        if domain and domain in self._domain_bm25:
            bm25 = self._domain_bm25[domain]
            scores = bm25.get_scores(tokenized_query)
            indices = self._domain_indices[domain]
            scored = [(indices[i], scores[i]) for i in range(len(indices))]
        else:
            scores = self.global_bm25.get_scores(tokenized_query)
            scored = [(i, scores[i]) for i in range(len(self.documents))]
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Apply min_score filter and take top_k
        results = []
        for rank, (idx, score) in enumerate(scored[:top_k]):
            if score < min_score:
                continue
            results.append({
                'document': self.documents[idx],
                'score': float(score),
                'rank': rank + 1,
            })
        
        return results
    
    def retrieve_across_domains(self, query: str, top_k: int = 3) -> dict[str, list[dict]]:
        """
        Retrieve from all domains separately and return per-domain results.
        Useful for domain classification when company is None.
        """
        results = {}
        for domain in self._domain_indices:
            results[domain] = self.retrieve(query, domain=domain, top_k=top_k)
        return results
    
    def get_best_domain(self, query: str) -> tuple[str, float]:
        """
        Determine which domain best matches the query based on top retrieval scores.
        
        Returns:
            (domain_name, confidence_score)
        """
        domain_results = self.retrieve_across_domains(query, top_k=3)
        
        domain_scores = {}
        for domain, results in domain_results.items():
            if results:
                # Use average of top scores as domain confidence
                scores = [r['score'] for r in results]
                domain_scores[domain] = sum(scores) / len(scores)
            else:
                domain_scores[domain] = 0.0
        
        if not domain_scores:
            return 'unknown', 0.0
        
        best_domain = max(domain_scores, key=domain_scores.get)
        return best_domain, domain_scores[best_domain]


def format_context(results: list[dict], max_chars: int = 8000) -> str:
    """
    Format retrieval results into a context string for the LLM prompt.
    
    Args:
        results: Retrieval results from retrieve().
        max_chars: Maximum character budget for context.
    """
    if not results:
        return "No relevant documentation found."
    
    parts = []
    total = 0
    for r in results:
        doc = r['document']
        header = f"--- [{doc['domain'].upper()}] {doc['title']} (score: {r['score']:.3f}) ---"
        content = doc['content']
        
        chunk = f"{header}\n{content}\n"
        if total + len(chunk) > max_chars:
            # Truncate this chunk to fit
            remaining = max_chars - total - len(header) - 10
            if remaining > 200:
                chunk = f"{header}\n{content[:remaining]}...\n"
            else:
                break
        
        parts.append(chunk)
        total += len(chunk)
    
    return "\n".join(parts)
