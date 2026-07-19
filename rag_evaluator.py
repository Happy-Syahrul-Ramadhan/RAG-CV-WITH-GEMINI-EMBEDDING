"""
RAG Evaluator Module.

Evaluasi sistem RAG dengan berbagai metrik:
- Retrieval metrics (Precision, Recall, MRR, NDCG)
- Generation metrics (Semantic Similarity, BLEU, ROUGE)
- End-to-end metrics (Faithfulness, Answer Relevancy)
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

from embedding_service import GeminiEmbeddingService, GroqGenerationService
from chromadb_service import ChromaDBService


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    # Retrieval metrics
    precision: float
    recall: float
    mrr: float
    ndcg: float
    
    # Generation metrics
    semantic_similarity: float
    answer_length: int
    
    # End-to-end metrics
    faithfulness_score: float
    answer_relevancy: float
    
    # Performance metrics
    retrieval_time: float
    generation_time: float
    total_time: float
    
    # Retrieved documents info
    num_retrieved_docs: int
    retrieved_docs: List[str]


class RAGEvaluator:
    """Evaluator for RAG system."""
    
    def __init__(
        self,
        chroma_service: ChromaDBService,
        embedding_service: GeminiEmbeddingService,
        generation_service: GroqGenerationService
    ):
        self.chroma_service = chroma_service
        self.embedding_service = embedding_service
        self.generation_service = generation_service
    
    def evaluate_query(
        self,
        query: str,
        ground_truth_answer: str,
        relevant_doc_ids: List[str],
        top_k: int = 5
    ) -> EvaluationResult:
        """
        Evaluate a single query.
        
        Args:
            query: User question
            ground_truth_answer: Expected answer
            relevant_doc_ids: List of IDs of documents that should be retrieved
            top_k: Number of documents to retrieve
            
        Returns:
            EvaluationResult with all metrics
        """
        total_start = time.time()
        
        # 1. Retrieval phase
        retrieval_start = time.time()
        query_embedding = self.embedding_service.embed_text(query)
        
        results = self.chroma_service.query_embeddings(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        retrieved_ids = results['ids'][0]
        retrieved_docs = results['documents'][0]
        distances = results['distances'][0]
        
        retrieval_time = time.time() - retrieval_start
        
        # 2. Calculate retrieval metrics
        precision = self._calculate_precision(retrieved_ids, relevant_doc_ids)
        recall = self._calculate_recall(retrieved_ids, relevant_doc_ids)
        mrr = self._calculate_mrr(retrieved_ids, relevant_doc_ids)
        ndcg = self._calculate_ndcg(retrieved_ids, relevant_doc_ids, distances)
        
        # 3. Generation phase
        generation_start = time.time()
        generated_answer = self.generation_service.ask_with_context(
            question=query,
            context_chunks=retrieved_docs,
            source_name="document"
        )
        generation_time = time.time() - generation_start
        
        # 4. Calculate generation metrics
        semantic_sim = self._calculate_semantic_similarity(
            generated_answer,
            ground_truth_answer
        )
        
        # 5. Calculate end-to-end metrics
        faithfulness = self._calculate_faithfulness(
            generated_answer,
            retrieved_docs
        )
        
        answer_relevancy = self._calculate_answer_relevancy(
            query,
            generated_answer
        )
        
        total_time = time.time() - total_start
        
        return EvaluationResult(
            precision=precision,
            recall=recall,
            mrr=mrr,
            ndcg=ndcg,
            semantic_similarity=semantic_sim,
            answer_length=len(generated_answer.split()),
            faithfulness_score=faithfulness,
            answer_relevancy=answer_relevancy,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            num_retrieved_docs=len(retrieved_docs),
            retrieved_docs=retrieved_docs
        )
    
    def _calculate_precision(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """Calculate precision: relevant retrieved / total retrieved."""
        if not retrieved_ids:
            return 0.0
        
        relevant_retrieved = set(retrieved_ids) & set(relevant_ids)
        return len(relevant_retrieved) / len(retrieved_ids)
    
    def _calculate_recall(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """Calculate recall: relevant retrieved / total relevant."""
        if not relevant_ids:
            return 0.0
        
        relevant_retrieved = set(retrieved_ids) & set(relevant_ids)
        return len(relevant_retrieved) / len(relevant_ids)
    
    def _calculate_mrr(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / i
        return 0.0
    
    def _calculate_ndcg(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        distances: List[float]
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        # Create relevance scores (1 if relevant, 0 if not)
        relevance = [1 if doc_id in relevant_ids else 0 for doc_id in retrieved_ids]
        
        if sum(relevance) == 0:
            return 0.0
        
        # Calculate DCG
        dcg = relevance[0] + sum(
            rel / np.log2(i + 1) for i, rel in enumerate(relevance[1:], 2)
        )
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = ideal_relevance[0] + sum(
            rel / np.log2(i + 1) for i, rel in enumerate(ideal_relevance[1:], 2)
        )
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_semantic_similarity(
        self,
        generated_answer: str,
        ground_truth: str
    ) -> float:
        """Calculate semantic similarity using embeddings."""
        try:
            gen_embedding = self.embedding_service.embed_text(generated_answer)
            gt_embedding = self.embedding_service.embed_text(ground_truth)
            
            similarity = cosine_similarity(
                [gen_embedding],
                [gt_embedding]
            )[0][0]
            
            return float(similarity)
        except Exception:
            return 0.0
    
    def _calculate_faithfulness(
        self,
        generated_answer: str,
        retrieved_docs: List[str]
    ) -> float:
        """
        Calculate faithfulness: how much the answer is grounded in context.
        Simple version: check if key phrases from answer exist in context.
        """
        if not generated_answer or not retrieved_docs:
            return 0.0
        
        # Combine all retrieved documents
        context = " ".join(retrieved_docs).lower()
        
        # Split answer into sentences
        sentences = [s.strip() for s in generated_answer.split('.') if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Check how many sentences have support in context
        supported = 0
        for sentence in sentences:
            # Check if main words in sentence appear in context
            words = [w.lower() for w in sentence.split() if len(w) > 3]
            if not words:
                continue
            
            words_in_context = sum(1 for w in words if w in context)
            if words_in_context / len(words) > 0.5:  # At least 50% words found
                supported += 1
        
        return supported / len(sentences)
    
    def _calculate_answer_relevancy(
        self,
        query: str,
        generated_answer: str
    ) -> float:
        """
        Calculate answer relevancy: how well answer addresses the query.
        Uses semantic similarity between query and answer.
        """
        try:
            query_embedding = self.embedding_service.embed_text(query)
            answer_embedding = self.embedding_service.embed_text(generated_answer)
            
            similarity = cosine_similarity(
                [query_embedding],
                [answer_embedding]
            )[0][0]
            
            return float(similarity)
        except Exception:
            return 0.0


def aggregate_results(results: List[EvaluationResult]) -> Dict[str, Any]:
    """Aggregate multiple evaluation results."""
    if not results:
        return {}
    
    return {
        "num_queries": len(results),
        
        # Retrieval metrics (average)
        "avg_precision": np.mean([r.precision for r in results]),
        "avg_recall": np.mean([r.recall for r in results]),
        "avg_mrr": np.mean([r.mrr for r in results]),
        "avg_ndcg": np.mean([r.ndcg for r in results]),
        
        # Generation metrics
        "avg_semantic_similarity": np.mean([r.semantic_similarity for r in results]),
        "avg_answer_length": np.mean([r.answer_length for r in results]),
        
        # End-to-end metrics
        "avg_faithfulness": np.mean([r.faithfulness_score for r in results]),
        "avg_answer_relevancy": np.mean([r.answer_relevancy for r in results]),
        
        # Performance metrics
        "avg_retrieval_time": np.mean([r.retrieval_time for r in results]),
        "avg_generation_time": np.mean([r.generation_time for r in results]),
        "avg_total_time": np.mean([r.total_time for r in results]),
        
        # Standard deviations
        "std_precision": np.std([r.precision for r in results]),
        "std_recall": np.std([r.recall for r in results]),
        "std_semantic_similarity": np.std([r.semantic_similarity for r in results]),
    }
