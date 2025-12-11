#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 2: Final Evaluation on Golden Test Set

This script runs the BEST configuration from Phase 1 (grid search) 
on a held-out golden test set to get unbiased final metrics.

Usage:
    python scripts/run_final_evaluation.py --config best_config.yaml
    python scripts/run_final_evaluation.py --num-qa 100
    python scripts/run_final_evaluation.py --num-qa 500 --num-docs 5000
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

import torch
import yaml
import pandas as pd
from datasets import load_dataset

# Import core components
from src.core.document_processor import DocumentProcessor
from src.core.embedder import Embedder
from src.core.retriever import Retriever
from src.core.hybrid_retriever import HybridRetriever
from src.core.reranker_factory import RerankerFactory
from src.core.generator import Generator

# Import shared evaluation
try:
    from shared_evaluation.metrics.retrieval_metrics import RetrievalMetrics
    from shared_evaluation.metrics.generation_metrics import GenerationMetrics
    EVAL_AVAILABLE = True
except ImportError as e:
    print(f"[!] Evaluation framework not available: {e}")
    EVAL_AVAILABLE = False

# MLflow (optional)
MLFLOW_AVAILABLE = False
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    pass


def get_best_config_from_grid_search(output_dir: Path) -> dict:
    """Load best configuration from grid search results."""
    csv_path = output_dir / "grid_search_results.csv"
    
    if not csv_path.exists():
        print(f"[!] No grid search results found at {csv_path}")
        print("    Run Phase 1 (grid search) first!")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Sort by composite score (descending) and get best
    df_sorted = df.sort_values('composite_score', ascending=False)
    best_row = df_sorted.iloc[0]
    
    # Extract configuration
    config = {
        'chunk_strategy': best_row.get('chunk_strategy', 'fixed_size'),
        'chunk_size': int(best_row.get('chunk_size', 512)),
        'chunk_overlap': int(best_row.get('chunk_overlap', 100)),
        'embedding_model': best_row.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'),
        'index_type': best_row.get('index_type', 'IndexFlatL2'),
        'reranker_type': best_row.get('reranker_type', 'cross_encoder'),
        'top_k': int(best_row.get('top_k', 10)),
        'top_n': int(best_row.get('top_n', 3)),
        'llm_model': best_row.get('llm_model', 'google/flan-t5-base'),
    }
    
    print(f"\n[OK] Loaded BEST config from Phase 1 grid search:")
    print(f"     Name: {best_row['name']}")
    print(f"     Composite Score: {best_row['composite_score']:.4f}")
    
    return config


def load_golden_test_set(num_docs: int, num_qa: int):
    """
    Load held-out golden test set from SQuAD VALIDATION split.
    This data was NOT used in Phase 1 tuning.
    """
    print("\n[Phase 2] Loading GOLDEN TEST SET (SQuAD Validation)...")
    
    # Load VALIDATION split - held out from tuning
    dataset = load_dataset("squad", split="validation")
    
    # Get unique documents
    seen_contexts = set()
    documents = []
    document_ids = []
    
    for example in dataset:
        context = example['context']
        doc_id = example['id']
        
        if context not in seen_contexts and len(documents) < num_docs:
            seen_contexts.add(context)
            documents.append(context)
            document_ids.append(doc_id)
    
    print(f"   [OK] Loaded {len(documents)} unique documents from VALIDATION split")
    
    # Build context to ID mapping
    indexed_contexts = {doc: doc_id for doc, doc_id in zip(documents, document_ids)}
    
    # Get Q&A pairs that have answers in indexed documents
    qa_pairs = []
    for example in dataset:
        if len(qa_pairs) >= num_qa:
            break
        
        context = example['context']
        if context in indexed_contexts:
            qa_pairs.append({
                'question': example['question'],
                'answer': example['answers']['text'][0] if example['answers']['text'] else "",
                'context': context,
                'id': example['id']
            })
    
    print(f"   [OK] Created {len(qa_pairs)} Q&A pairs for FINAL evaluation")
    print(f"   [OK] All Q&A pairs are answerable (context in indexed docs)")
    
    return documents, document_ids, qa_pairs, indexed_contexts


def run_final_evaluation(config: dict, documents: list, document_ids: list, 
                         qa_pairs: list, indexed_contexts: dict, device: str):
    """Run final evaluation with locked-in best configuration."""
    
    results = {
        'config': config,
        'metrics': {},
        'latencies': {},
        'status': 'success'
    }
    
    try:
        # === DOCUMENT PROCESSING ===
        t0 = time.time()
        processor = DocumentProcessor(
            strategy=config['chunk_strategy'],
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap']
        )
        chunks = processor.chunk_documents(documents)
        chunk_texts = [c.text for c in chunks]
        results['latencies']['chunking_ms'] = (time.time() - t0) * 1000
        results['metrics']['num_chunks'] = len(chunk_texts)
        print(f"   Chunks: {len(chunk_texts)}")
        
        # === EMBEDDING ===
        t0 = time.time()
        embedder = Embedder(model_name=config['embedding_model'], device=device)
        embeddings = embedder.encode(chunk_texts)
        results['latencies']['embedding_ms'] = (time.time() - t0) * 1000
        
        # === INDEXING (Dense or Hybrid) ===
        t0 = time.time()
        retrieval_type = config.get('retrieval_type', 'dense')
        use_hybrid = retrieval_type == 'hybrid'
        
        if use_hybrid:
            retriever = HybridRetriever(
                embedding_dimension=embeddings.shape[1],
                fusion_strategy=config.get('fusion_strategy', 'rrf')
            )
            chunk_map = [{'content': text, 'metadata': {'index': i}} for i, text in enumerate(chunk_texts)]
            retriever.build_index(embeddings, chunk_map)
        else:
            retriever = Retriever(embedding_dim=embeddings.shape[1])
            retriever.build_index(embeddings, chunk_texts)
        results['latencies']['indexing_ms'] = (time.time() - t0) * 1000
        
        # === RERANKER ===
        reranker = RerankerFactory.create(
            reranker_type=config['reranker_type'],
            device=device
        )
        
        # === GENERATOR ===
        generator = Generator(
            model_name=config['llm_model'],
            device=device
        )
        
        # === EVALUATION LOOP ===
        correct = 0
        hallucination = 0
        appropriate_abstention = 0
        missed = 0
        
        all_retrieved_ids = []
        all_relevant_ids = []
        all_predictions = []
        all_references = []
        
        retrieval_times = []
        rerank_times = []
        generation_times = []
        
        print(f"\n   Evaluating {len(qa_pairs)} Q&A pairs...")
        
        for i, qa in enumerate(qa_pairs):
            question = qa['question']
            expected_answer = qa['answer']
            expected_context = qa['context']
            
            # Retrieval
            t0 = time.time()
            query_embedding = embedder.encode([question])
            if use_hybrid:
                retrieved = retriever.search(query_embedding, question, top_k=config['top_k'])
                retrieved_texts = [r['chunk_content'] for r in retrieved]
            else:
                retrieved = retriever.search(query_embedding, top_k=config['top_k'])
                retrieved_texts = [r['text'] for r in retrieved]
            retrieval_times.append((time.time() - t0) * 1000)
            
            # Reranking
            t0 = time.time()
            reranked = reranker.rerank(question, retrieved_texts, top_k=config['top_n'])
            rerank_times.append((time.time() - t0) * 1000)
            
            # Build context for generation
            context_for_gen = "\n\n".join([r['text'] for r in reranked])
            
            # Generation
            t0 = time.time()
            prompt = f"""Answer the question based on the context below. If the context doesn't contain enough information to answer, say "I cannot answer this based on the available context."

Context:
{context_for_gen}

Question: {question}

Answer:"""
            
            generated = generator.generate(prompt, max_length=150, temperature=0.7, num_beams=4)
            generation_times.append((time.time() - t0) * 1000)
            
            # Store for metrics
            all_predictions.append(generated)
            all_references.append(expected_answer)
            
            # Check if correct context was retrieved
            retrieved_context_ids = []
            for r in retrieved:
                for ctx, ctx_id in indexed_contexts.items():
                    if r['text'] in ctx:
                        retrieved_context_ids.append(ctx_id)
                        break
            
            # Use the document ID from indexed_contexts, NOT the SQuAD question ID
            expected_id = indexed_contexts.get(qa['context'])
            all_retrieved_ids.append(retrieved_context_ids)
            all_relevant_ids.append([expected_id] if expected_id else [])
            
            # Evaluate answer quality
            answer_lower = generated.lower().strip()
            expected_lower = expected_answer.lower().strip()
            
            is_abstention = "cannot answer" in answer_lower or "not enough" in answer_lower
            
            if expected_lower in answer_lower or answer_lower in expected_lower:
                correct += 1
            elif is_abstention:
                # Check if abstention was appropriate
                context_has_answer = expected_lower in context_for_gen.lower()
                if not context_has_answer:
                    appropriate_abstention += 1
                else:
                    missed += 1
            else:
                # Check for hallucination
                context_has_answer = expected_lower in context_for_gen.lower()
                if not context_has_answer:
                    hallucination += 1
                else:
                    missed += 1
            
            # Progress
            if (i + 1) % 50 == 0:
                print(f"      Processed {i + 1}/{len(qa_pairs)}")
        
        # === COMPUTE METRICS ===
        total_qa = len(qa_pairs)
        
        results['metrics']['total_qa'] = total_qa
        results['metrics']['correct'] = correct
        results['metrics']['appropriate_abstention'] = appropriate_abstention
        results['metrics']['hallucination'] = hallucination
        results['metrics']['missed'] = missed
        results['metrics']['accuracy'] = correct / total_qa if total_qa > 0 else 0
        results['metrics']['quality_score'] = (correct + appropriate_abstention) / total_qa if total_qa > 0 else 0
        results['metrics']['hallucination_rate'] = hallucination / total_qa if total_qa > 0 else 0
        
        # Retrieval metrics
        if EVAL_AVAILABLE:
            # Calculate retrieval metrics per query then average
            mrr_scores = []
            hit_rate_3 = []
            hit_rate_5 = []
            precision_1 = []
            precision_3 = []
            recall_3 = []
            
            for retrieved, relevant in zip(all_retrieved_ids, all_relevant_ids):
                relevant_set = set(relevant)
                mrr_scores.append(RetrievalMetrics.mrr(retrieved, relevant_set))
                hit_rate_3.append(RetrievalMetrics.hit_rate_at_k(retrieved, relevant_set, k=3))
                hit_rate_5.append(RetrievalMetrics.hit_rate_at_k(retrieved, relevant_set, k=5))
                precision_1.append(RetrievalMetrics.precision_at_k(retrieved, relevant_set, k=1))
                precision_3.append(RetrievalMetrics.precision_at_k(retrieved, relevant_set, k=3))
                recall_3.append(RetrievalMetrics.recall_at_k(retrieved, relevant_set, k=3))
            
            results['metrics']['ret_mrr'] = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
            results['metrics']['ret_hit_rate@3'] = sum(hit_rate_3) / len(hit_rate_3) if hit_rate_3 else 0
            results['metrics']['ret_hit_rate@5'] = sum(hit_rate_5) / len(hit_rate_5) if hit_rate_5 else 0
            results['metrics']['ret_precision@1'] = sum(precision_1) / len(precision_1) if precision_1 else 0
            results['metrics']['ret_precision@3'] = sum(precision_3) / len(precision_3) if precision_3 else 0
            results['metrics']['ret_recall@3'] = sum(recall_3) / len(recall_3) if recall_3 else 0
            
            # Generation metrics (compute average across all predictions)
            f1_scores = [GenerationMetrics.f1_score(p, r) for p, r in zip(all_predictions, all_references)]
            em_scores = [GenerationMetrics.exact_match(p, r) for p, r in zip(all_predictions, all_references)]
            rouge_results = [GenerationMetrics.rouge_l(p, r) for p, r in zip(all_predictions, all_references)]
            rouge_scores = [r['f1'] for r in rouge_results]  # Extract F1 from dict
            bleu_scores = [GenerationMetrics.bleu_score(p, r, n=1) for p, r in zip(all_predictions, all_references)]
            
            results['metrics']['gen_f1_score'] = sum(f1_scores) / len(f1_scores) if f1_scores else 0
            results['metrics']['gen_exact_match'] = sum(em_scores) / len(em_scores) if em_scores else 0
            results['metrics']['gen_rouge_l'] = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
            results['metrics']['gen_bleu_1'] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        
        # Latencies
        results['latencies']['avg_retrieval_ms'] = sum(retrieval_times) / len(retrieval_times)
        results['latencies']['avg_rerank_ms'] = sum(rerank_times) / len(rerank_times)
        results['latencies']['avg_generation_ms'] = sum(generation_times) / len(generation_times)
        results['latencies']['total_ms'] = (
            results['latencies']['chunking_ms'] +
            results['latencies']['embedding_ms'] +
            results['latencies']['indexing_ms'] +
            sum(retrieval_times) + sum(rerank_times) + sum(generation_times)
        )
        
        # Composite score
        accuracy_weight = 0.5
        latency_weight = 0.3
        cost_weight = 0.2
        
        accuracy_score = results['metrics']['quality_score']
        latency_score = 1.0 / (1.0 + results['latencies']['avg_generation_ms'] / 1000)
        cost_score = 1.0  # Open source = no cost
        
        results['metrics']['composite_score'] = (
            accuracy_weight * accuracy_score +
            latency_weight * latency_score +
            cost_weight * cost_score
        )
        
    except Exception as e:
        results['status'] = f'error: {str(e)}'
        import traceback
        traceback.print_exc()
    
    return results


def save_final_results(results: dict, output_dir: Path):
    """Save final evaluation results."""
    
    # Create Phase 2 output directory
    phase2_dir = output_dir / "phase2_final_evaluation"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_path = phase2_dir / f"final_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[OK] JSON saved: {json_path}")
    
    # Save detailed report
    report_path = phase2_dir / f"FINAL_EVALUATION_REPORT_{timestamp}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PHASE 2: FINAL EVALUATION REPORT\n")
        f.write("Golden Test Set (SQuAD Validation - Held Out)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("CONFIGURATION (Best from Phase 1 Grid Search):\n")
        f.write("-" * 40 + "\n")
        for k, v in results['config'].items():
            f.write(f"  {k}: {v}\n")
        
        f.write("\n\nFINAL METRICS:\n")
        f.write("-" * 40 + "\n")
        
        metrics = results['metrics']
        f.write(f"\n  ACCURACY METRICS:\n")
        f.write(f"    Total Q&A Pairs:        {metrics.get('total_qa', 'N/A')}\n")
        f.write(f"    Correct Answers:        {metrics.get('correct', 'N/A')}\n")
        f.write(f"    Appropriate Abstention: {metrics.get('appropriate_abstention', 'N/A')}\n")
        f.write(f"    Hallucinations:         {metrics.get('hallucination', 'N/A')}\n")
        f.write(f"    Missed:                 {metrics.get('missed', 'N/A')}\n")
        f.write(f"    \n")
        f.write(f"    Accuracy:               {metrics.get('accuracy', 0):.2%}\n")
        f.write(f"    Quality Score:          {metrics.get('quality_score', 0):.2%}\n")
        f.write(f"    Hallucination Rate:     {metrics.get('hallucination_rate', 0):.2%}\n")
        
        f.write(f"\n  RETRIEVAL METRICS:\n")
        mrr = metrics.get('ret_mrr', 0)
        f.write(f"    MRR:                    {mrr:.4f}\n")
        f.write(f"    Hit Rate@3:             {metrics.get('ret_hit_rate@3', 0):.4f}\n")
        f.write(f"    Hit Rate@5:             {metrics.get('ret_hit_rate@5', 0):.4f}\n")
        f.write(f"    Precision@1:            {metrics.get('ret_precision@1', 0):.4f}\n")
        f.write(f"    Precision@3:            {metrics.get('ret_precision@3', 0):.4f}\n")
        f.write(f"    Recall@3:               {metrics.get('ret_recall@3', 0):.4f}\n")
        
        f.write(f"\n  GENERATION METRICS:\n")
        f.write(f"    F1 Score:               {metrics.get('gen_f1_score', 0):.4f}\n")
        f.write(f"    Exact Match:            {metrics.get('gen_exact_match', 0):.4f}\n")
        f.write(f"    ROUGE-L:                {metrics.get('gen_rouge_l', 0):.4f}\n")
        f.write(f"    BLEU-1:                 {metrics.get('gen_bleu_1', 0):.4f}\n")
        
        f.write(f"\n  COMPOSITE SCORE:          {metrics.get('composite_score', 0):.4f}\n")
        
        f.write(f"\n\nLATENCY:\n")
        f.write("-" * 40 + "\n")
        latencies = results['latencies']
        f.write(f"    Chunking:               {latencies.get('chunking_ms', 0):.0f} ms\n")
        f.write(f"    Embedding:              {latencies.get('embedding_ms', 0):.0f} ms\n")
        f.write(f"    Indexing:               {latencies.get('indexing_ms', 0):.0f} ms\n")
        f.write(f"    Avg Retrieval:          {latencies.get('avg_retrieval_ms', 0):.1f} ms\n")
        f.write(f"    Avg Reranking:          {latencies.get('avg_rerank_ms', 0):.1f} ms\n")
        f.write(f"    Avg Generation:         {latencies.get('avg_generation_ms', 0):.1f} ms\n")
        f.write(f"    Total:                  {latencies.get('total_ms', 0):.0f} ms\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF FINAL EVALUATION REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"[OK] Report saved: {report_path}")
    
    # Save CSV for easy comparison
    csv_path = phase2_dir / f"final_metrics_{timestamp}.csv"
    
    flat_results = {'timestamp': timestamp, 'status': results['status']}
    flat_results.update(results['config'])
    flat_results.update(results['metrics'])
    flat_results.update({f"latency_{k}": v for k, v in results['latencies'].items()})
    
    df = pd.DataFrame([flat_results])
    df.to_csv(csv_path, index=False)
    print(f"[OK] CSV saved: {csv_path}")
    
    return phase2_dir


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Final Evaluation on Golden Test Set")
    parser.add_argument('--num-docs', type=int, default=1000,
                        help='Number of documents to load (default: 1000)')
    parser.add_argument('--num-qa', type=int, default=100,
                        help='Number of Q&A pairs for evaluation (default: 100)')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML (optional, defaults to best from grid search)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PHASE 2: FINAL EVALUATION ON GOLDEN TEST SET")
    print("=" * 80)
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[OK] GPU: {gpu_name}")
    else:
        print("[!] Running on CPU")
    
    output_dir = Path(args.output_dir)
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"\n[OK] Loaded config from: {args.config}")
    else:
        config = get_best_config_from_grid_search(output_dir)
        if config is None:
            print("\n[!] No config available. Using defaults.")
            config = {
                'chunk_strategy': 'fixed_size',
                'chunk_size': 512,
                'chunk_overlap': 100,
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'index_type': 'IndexFlatL2',
                'reranker_type': 'cross_encoder',
                'top_k': 10,
                'top_n': 3,
                'llm_model': 'google/flan-t5-base',
            }
    
    print("\n[Configuration]")
    for k, v in config.items():
        print(f"   {k}: {v}")
    
    # Load golden test set (VALIDATION split - held out)
    documents, document_ids, qa_pairs, indexed_contexts = load_golden_test_set(
        args.num_docs, args.num_qa
    )
    
    # Run final evaluation
    print("\n" + "=" * 80)
    print("RUNNING FINAL EVALUATION...")
    print("=" * 80)
    
    results = run_final_evaluation(
        config, documents, document_ids, qa_pairs, indexed_contexts, device
    )
    
    # Save results
    phase2_dir = save_final_results(results, output_dir)
    
    # Print summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    metrics = results['metrics']
    print(f"""
   ACCURACY:
      Correct:              {metrics.get('correct', 'N/A')}/{metrics.get('total_qa', 'N/A')}
      Quality Score:        {metrics.get('quality_score', 0):.2%}
      Hallucination Rate:   {metrics.get('hallucination_rate', 0):.2%}
   
   RETRIEVAL:
      MRR:                  {metrics.get('ret_mrr', 0):.4f}
      Hit Rate@3:           {metrics.get('ret_hit_rate@3', 0):.4f}
   
   GENERATION:
      F1 Score:             {metrics.get('gen_f1_score', 0):.4f}
      ROUGE-L:              {metrics.get('gen_rouge_l', 0):.4f}
   
   COMPOSITE SCORE:         {metrics.get('composite_score', 0):.4f}
    """)
    
    print("=" * 80)
    print("PHASE 2 COMPLETE")
    print(f"Results saved to: {phase2_dir}")
    print("=" * 80)
    
    # Log to MLflow
    if MLFLOW_AVAILABLE:
        mlflow_dir = output_dir / "mlflow_tracking"
        mlflow.set_tracking_uri(mlflow_dir.absolute().as_uri())
        mlflow.set_experiment("RAG_Final_Evaluation")
        
        with mlflow.start_run(run_name="FINAL_EVALUATION"):
            for k, v in config.items():
                mlflow.log_param(k, v)
            
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    safe_name = k.replace('@', '_at_')
                    mlflow.log_metric(safe_name, v)
            
            mlflow.set_tag("phase", "2_final_evaluation")
            mlflow.set_tag("dataset", "squad_validation")
        
        print(f"\n[MLflow] Final evaluation logged to experiment 'RAG_Final_Evaluation'")


if __name__ == "__main__":
    main()

