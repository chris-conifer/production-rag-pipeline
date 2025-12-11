#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production RAG Pipeline - Grid Search Optimization
Runs experiments with different configurations and logs to MLflow
"""

import argparse
import sys
import os
import time
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))  # For shared_evaluation

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

# Import shared evaluation framework
try:
    from shared_evaluation.metrics.retrieval_metrics import RetrievalMetrics
    from shared_evaluation.metrics.generation_metrics import GenerationMetrics
    EVAL_AVAILABLE = True
    print("[OK] Shared evaluation framework loaded")
except ImportError as e:
    print(f"[!] Evaluation framework not available: {e}")
    EVAL_AVAILABLE = False
    RetrievalMetrics = None
    GenerationMetrics = None

# Import MLflow tracker (optional - may have protobuf conflicts)
MLFLOW_AVAILABLE = False
MLflowTracker = None
try:
    import mlflow
    MLFLOW_AVAILABLE = True
    print("[OK] MLflow tracking available")
except ImportError as e:
    print(f"[!] MLflow not available: {e}")


# Confidence threshold for abstention
CONFIDENCE_THRESHOLD = 0.3  # If top rerank score < this, abstain
ABSTENTION_PHRASES = ["cannot answer", "don't have enough", "no information", "not enough context", "unable to answer"]


def check_is_abstention(answer):
    """Check if the answer is an abstention (says 'I don't know')"""
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in ABSTENTION_PHRASES)


def check_context_contains_answer(context_texts, expected_answer):
    """Check if the expected answer appears in any of the retrieved contexts"""
    combined_context = " ".join(context_texts).lower()
    return expected_answer.lower() in combined_context


def compute_retrieval_metrics(retrieved_chunks, relevant_context, k_values=[1, 3, 5, 10]):
    """
    Compute retrieval metrics using shared evaluation framework
    
    Args:
        retrieved_chunks: List of retrieved chunk texts
        relevant_context: The ground truth context that should have been retrieved
        k_values: List of k values for metrics
        
    Returns:
        Dictionary of retrieval metrics
    """
    if not EVAL_AVAILABLE or RetrievalMetrics is None:
        return {}
    
    metrics = {}
    
    # Check which retrieved chunks contain the relevant content
    # Create pseudo document IDs based on whether chunk contains answer
    retrieved_ids = []
    for i, chunk in enumerate(retrieved_chunks):
        # Check if this chunk contains substantial overlap with relevant context
        chunk_lower = chunk.lower()
        relevant_lower = relevant_context.lower()
        
        # Consider a hit if there's significant word overlap
        chunk_words = set(chunk_lower.split())
        relevant_words = set(relevant_lower.split())
        overlap = len(chunk_words & relevant_words)
        
        if overlap > 10:  # At least 10 common words
            retrieved_ids.append(f"relevant_{i}")
        else:
            retrieved_ids.append(f"irrelevant_{i}")
    
    # The relevant set is any chunk that matches
    relevant_set = {rid for rid in retrieved_ids if rid.startswith("relevant")}
    
    # Calculate metrics at different k values
    for k in k_values:
        if k <= len(retrieved_ids):
            metrics[f'precision@{k}'] = RetrievalMetrics.precision_at_k(retrieved_ids, relevant_set, k)
            metrics[f'recall@{k}'] = RetrievalMetrics.recall_at_k(retrieved_ids, relevant_set, k)
    
    # MRR and Hit Rate (using correct method names)
    metrics['mrr'] = RetrievalMetrics.mrr(retrieved_ids, relevant_set)
    metrics['hit_rate@3'] = RetrievalMetrics.hit_rate_at_k(retrieved_ids, relevant_set, k=3)
    metrics['hit_rate@5'] = RetrievalMetrics.hit_rate_at_k(retrieved_ids, relevant_set, k=5)
    
    return metrics


def compute_generation_metrics(generated_answer, ground_truth_answer):
    """
    Compute generation metrics using shared evaluation framework
    
    Args:
        generated_answer: The model's generated answer
        ground_truth_answer: The ground truth answer
        
    Returns:
        Dictionary of generation metrics
    """
    if not EVAL_AVAILABLE or GenerationMetrics is None:
        return {}
    
    metrics = {}
    
    try:
        # Exact match
        metrics['exact_match'] = GenerationMetrics.exact_match(generated_answer, ground_truth_answer)
        
        # F1 score (token-level)
        metrics['f1_score'] = GenerationMetrics.f1_score(generated_answer, ground_truth_answer)
        
        # BLEU scores
        metrics['bleu_1'] = GenerationMetrics.bleu_score(generated_answer, ground_truth_answer, n=1)
        metrics['bleu_2'] = GenerationMetrics.bleu_score(generated_answer, ground_truth_answer, n=2)
        
        # ROUGE scores
        rouge_1_result = GenerationMetrics.rouge_n(generated_answer, ground_truth_answer, n=1)
        rouge_2_result = GenerationMetrics.rouge_n(generated_answer, ground_truth_answer, n=2)
        rouge_l_result = GenerationMetrics.rouge_l(generated_answer, ground_truth_answer)
        
        metrics['rouge_1'] = rouge_1_result.get('f1', 0) if rouge_1_result else 0
        metrics['rouge_2'] = rouge_2_result.get('f1', 0) if rouge_2_result else 0
        metrics['rouge_l'] = rouge_l_result.get('f1', 0) if rouge_l_result else 0
    except Exception as e:
        # If any metric fails, return partial results
        pass
    
    return metrics


def run_single_experiment(config, documents, document_ids, qa_pairs, indexed_contexts, device="cuda"):
    """
    Run a single experiment with given configuration
    
    Tracks:
    - correct: Answer matches expected
    - appropriate_abstention: Said "I don't know" when answer wasn't in retrieved context
    - hallucination: Gave wrong answer when should have abstained
    - missed: Had context but failed to extract correct answer
    """
    results = {
        'config': config.copy(),
        'metrics': {},
        'latencies': {}
    }
    
    try:
        total_start = time.time()
        
        # 1. Chunking
        chunk_start = time.time()
        processor = DocumentProcessor(
            strategy=config.get('chunk_strategy', 'fixed_size'),
            chunk_size=config.get('chunk_size', 512),
            chunk_overlap=config.get('chunk_overlap', 100)
        )
        all_chunks = processor.chunk_documents(documents, document_ids)
        chunk_texts = [c.text for c in all_chunks]
        results['latencies']['chunking_ms'] = (time.time() - chunk_start) * 1000
        results['metrics']['num_chunks'] = len(chunk_texts)
        
        # 2. Embedding
        embed_start = time.time()
        embedder = Embedder(
            model_name=config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'),
            device=device
        )
        chunk_embeddings = embedder.encode(chunk_texts)
        results['latencies']['embedding_ms'] = (time.time() - embed_start) * 1000
        
        # 3. Indexing (Dense or Hybrid)
        index_start = time.time()
        retrieval_type = config.get('retrieval_type', 'dense')
        use_hybrid = retrieval_type == 'hybrid'
        
        if use_hybrid:
            # Hybrid retrieval (Dense + BM25)
            retriever = HybridRetriever(
                embedding_dimension=chunk_embeddings.shape[1],
                index_type=config.get('index_type', 'IndexFlatL2'),
                fusion_strategy=config.get('fusion_strategy', 'rrf'),
                dense_weight=config.get('dense_weight', 0.7),
                sparse_weight=config.get('sparse_weight', 0.3)
            )
            # Build chunk_map for hybrid retriever
            chunk_map = [{'content': text, 'metadata': {'index': i}} for i, text in enumerate(chunk_texts)]
            retriever.build_index(chunk_embeddings, chunk_map)
        else:
            # Dense-only retrieval (FAISS)
            retriever = Retriever(
                embedding_dim=chunk_embeddings.shape[1],
                index_type=config.get('index_type', 'IndexFlatL2')
            )
            retriever.build_index(chunk_embeddings, chunk_texts)
        
        results['latencies']['indexing_ms'] = (time.time() - index_start) * 1000
        
        # 4. Create reranker
        reranker = RerankerFactory.create(
            reranker_type=config.get('reranker_type', 'cross_encoder'),
            device=device
        )
        
        # 5. Create generator
        generator = Generator(
            model_name=config.get('llm_model', 'google/flan-t5-base'),
            device=device
        )
        
        # 6. Evaluate on Q&A pairs with detailed tracking
        correct = 0
        appropriate_abstention = 0
        hallucination = 0
        missed = 0
        answerable_count = 0
        unanswerable_count = 0
        
        total_retrieval_time = 0
        total_rerank_time = 0
        total_generation_time = 0
        
        # Accumulate metrics across all Q&A pairs
        all_retrieval_metrics = []
        all_generation_metrics = []
        
        for qa in qa_pairs:
            question = qa['question']
            expected_answer = qa['answer']
            qa_context = qa['context']
            
            # Check if this Q&A's context is in our indexed documents
            is_answerable = qa_context in indexed_contexts
            if is_answerable:
                answerable_count += 1
            else:
                unanswerable_count += 1
            
            # Retrieve
            ret_start = time.time()
            query_emb = embedder.encode([question])
            
            if use_hybrid:
                # Hybrid retrieval needs query text for BM25
                retrieved = retriever.search(query_emb, question, top_k=config.get('top_k', 10))
                retrieved_texts = [r['chunk_content'] for r in retrieved]
            else:
                # Dense-only retrieval
                retrieved = retriever.search(query_emb, top_k=config.get('top_k', 10))
                retrieved_texts = [r['text'] for r in retrieved]
            
            total_retrieval_time += time.time() - ret_start
            
            # Rerank
            rerank_start = time.time()
            reranked = reranker.rerank(question, retrieved_texts, top_k=config.get('top_n', 3))
            total_rerank_time += time.time() - rerank_start
            
            # Check confidence (top rerank score)
            top_score = reranked[0]['rerank_score'] if reranked else 0
            context_has_answer = check_context_contains_answer([r['text'] for r in reranked], expected_answer)
            
            # Generate with abstention-aware prompt
            gen_start = time.time()
            context_str = "\n\n".join([r['text'] for r in reranked])
            
            # Updated prompt that allows abstention
            prompt = f"""Answer the question based ONLY on the context provided below.
If the context does not contain enough information to answer the question, respond with: "I cannot answer this based on the available context."

Context:
{context_str}

Question: {question}

Answer:"""
            
            answer = generator.generate(prompt, max_length=100)
            total_generation_time += time.time() - gen_start
            
            # Classify the response
            is_abstention = check_is_abstention(answer)
            is_correct = expected_answer.lower() in answer.lower()
            
            if is_correct:
                # Correct answer
                correct += 1
            elif is_abstention:
                if not context_has_answer:
                    # Appropriately said "I don't know" when answer wasn't retrievable
                    appropriate_abstention += 1
                else:
                    # Had the answer but abstained (missed opportunity)
                    missed += 1
            else:
                # Gave a wrong answer
                if not context_has_answer:
                    # Hallucinated when should have abstained
                    hallucination += 1
                else:
                    # Had context but extracted wrong answer
                    missed += 1
            
            # Compute retrieval metrics for this Q&A
            ret_metrics = compute_retrieval_metrics(
                [r['text'] for r in reranked], 
                qa_context,
                k_values=[1, 3, 5]
            )
            if ret_metrics:
                all_retrieval_metrics.append(ret_metrics)
            
            # Compute generation metrics for this Q&A (only if not abstention)
            if not is_abstention:
                gen_metrics = compute_generation_metrics(answer, expected_answer)
                if gen_metrics:
                    all_generation_metrics.append(gen_metrics)
        
        # Calculate metrics
        num_qa = len(qa_pairs)
        
        # Core metrics
        results['metrics']['correct'] = correct
        results['metrics']['appropriate_abstention'] = appropriate_abstention
        results['metrics']['hallucination'] = hallucination
        results['metrics']['missed'] = missed
        results['metrics']['total_qa'] = num_qa
        results['metrics']['answerable_qa'] = answerable_count
        results['metrics']['unanswerable_qa'] = unanswerable_count
        
        # Rates
        results['metrics']['accuracy'] = correct / num_qa if num_qa > 0 else 0
        results['metrics']['abstention_rate'] = (appropriate_abstention + missed) / num_qa if num_qa > 0 else 0
        results['metrics']['hallucination_rate'] = hallucination / num_qa if num_qa > 0 else 0
        
        # Quality score (rewards correct answers AND appropriate abstentions, penalizes hallucinations)
        quality_score = (correct + appropriate_abstention) / num_qa if num_qa > 0 else 0
        results['metrics']['quality_score'] = quality_score
        
        # Latencies
        results['latencies']['avg_retrieval_ms'] = (total_retrieval_time / num_qa) * 1000 if num_qa > 0 else 0
        results['latencies']['avg_rerank_ms'] = (total_rerank_time / num_qa) * 1000 if num_qa > 0 else 0
        results['latencies']['avg_generation_ms'] = (total_generation_time / num_qa) * 1000 if num_qa > 0 else 0
        results['latencies']['total_ms'] = (time.time() - total_start) * 1000
        
        # =====================================================================
        # RETRIEVAL METRICS (averaged across all Q&A pairs)
        # =====================================================================
        if all_retrieval_metrics:
            # Average each metric across all Q&A pairs
            ret_metric_names = all_retrieval_metrics[0].keys()
            for metric_name in ret_metric_names:
                values = [m.get(metric_name, 0) for m in all_retrieval_metrics]
                results['metrics'][f'ret_{metric_name}'] = sum(values) / len(values)
        
        # =====================================================================
        # GENERATION METRICS (averaged across non-abstention Q&A pairs)
        # =====================================================================
        if all_generation_metrics:
            # Average each metric across all Q&A pairs
            gen_metric_names = all_generation_metrics[0].keys()
            for metric_name in gen_metric_names:
                values = [m.get(metric_name, 0) for m in all_generation_metrics]
                results['metrics'][f'gen_{metric_name}'] = sum(values) / len(values)
        
        # Composite score (quality weighted highest, penalize hallucinations)
        # Now also considers retrieval quality (MRR) if available
        ret_mrr = results['metrics'].get('ret_mrr', 0.5)
        gen_f1 = results['metrics'].get('gen_f1_score', 0)
        
        results['metrics']['composite_score'] = (
            0.25 * quality_score +
            0.20 * results['metrics']['accuracy'] +
            0.15 * (1.0 - results['metrics']['hallucination_rate']) +  # Reward low hallucination
            0.15 * ret_mrr +  # Retrieval quality
            0.15 * gen_f1 +   # Generation quality
            0.05 * (1.0 / (1.0 + results['latencies']['avg_retrieval_ms'] / 1000)) +
            0.05 * (1.0 / (1.0 + results['latencies']['avg_generation_ms'] / 1000))
        )
        
        results['status'] = 'success'
        
    except Exception as e:
        results['status'] = f'error: {str(e)[:100]}'
        results['metrics']['accuracy'] = 0
        results['metrics']['composite_score'] = 0
        results['metrics']['hallucination_rate'] = 0
        results['metrics']['quality_score'] = 0
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Production RAG Pipeline - Grid Search Optimization")
    
    parser.add_argument("--num-docs", type=int, default=200, help="Number of documents (default: 200)")
    parser.add_argument("--num-qa", type=int, default=20, help="Number of Q&A pairs for evaluation (default: 20)")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--max-experiments", type=int, default=10, help="Max experiments to run (default: 10)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PRODUCTION RAG PIPELINE - GRID SEARCH OPTIMIZATION")
    print("=" * 80)
    
    # Check GPU
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[!] No GPU - using CPU")
    
    print(f"\n[Config]")
    print(f"   Documents: {args.num_docs}")
    print(f"   Q&A Pairs: {args.num_qa}")
    print(f"   Max Experiments: {args.max_experiments}")
    print(f"   Output: {args.output_dir}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize MLflow tracking
    if MLFLOW_AVAILABLE:
        mlflow_dir = output_dir / "mlflow_tracking"
        mlflow_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(mlflow_dir.absolute().as_uri())
        mlflow.set_experiment("RAG_Grid_Search")
        print(f"[OK] MLflow tracking: {mlflow_dir}")
    
    # Load dataset
    print("Step 1: Loading dataset...")
    dataset = load_dataset("squad", split="train")
    
    # Get unique documents
    seen_contexts = set()
    documents = []
    document_ids = []
    qa_pairs = []
    
    for example in dataset:
        ctx = example['context']
        if ctx not in seen_contexts and len(documents) < args.num_docs:
            seen_contexts.add(ctx)
            documents.append(ctx)
            document_ids.append(example['id'])
        
        if len(qa_pairs) < args.num_qa:
            qa_pairs.append({
                'question': example['question'],
                'answer': example['answers']['text'][0] if example['answers']['text'] else "",
                'context': example['context']
            })
    
    print(f"   [OK] Loaded {len(documents)} unique documents")
    print(f"   [OK] Created {len(qa_pairs)} Q&A pairs for evaluation")
    
    # Track which Q&A pairs have their context in indexed documents
    indexed_contexts = set(documents)
    answerable = sum(1 for qa in qa_pairs if qa['context'] in indexed_contexts)
    print(f"   [OK] {answerable}/{len(qa_pairs)} Q&A pairs are answerable (context in indexed docs)\n")
    
    # =========================================================================
    # YAML-DRIVEN GRID SEARCH CONFIGURATION
    # =========================================================================
    # Load configuration from YAML file (dynamic, no hardcoding!)
    grid_config_path = project_root / "configs" / "grid_search_config.yaml"
    
    if grid_config_path.exists():
        print(f"[OK] Loading grid search config from: {grid_config_path.name}")
        with open(grid_config_path, 'r') as f:
            grid_config = yaml.safe_load(f)
        
        # Extract baseline from YAML (all 'baseline' values)
        params = grid_config.get('parameters', {})
        baseline = {}
        for param_name, param_config in params.items():
            if isinstance(param_config, dict) and 'baseline' in param_config:
                baseline[param_name] = param_config['baseline']
        
        # Generate experiments: baseline first, then one-at-a-time variations
        experiments = [('baseline', baseline.copy())]
        
        # For each parameter, test options that differ from baseline
        for param_name, param_config in params.items():
            if isinstance(param_config, dict):
                baseline_val = param_config.get('baseline')
                options = param_config.get('options', [])
                
                for option in options:
                    # Only create experiment if option differs from baseline
                    if option != baseline_val:
                        config = baseline.copy()
                        config[param_name] = option
                        experiments.append((f"{param_name}={option}", config))
        
        print(f"   Generated {len(experiments)} experiments from YAML")
    else:
        # Fallback to hardcoded defaults if YAML not found
        print(f"[!] YAML config not found, using hardcoded defaults")
        baseline = {
            'chunk_strategy': 'sentence',
            'chunk_size': 512,
            'chunk_overlap': 100,
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'index_type': 'IndexFlatL2',
            'retrieval_type': 'hybrid',
            'reranker_type': 'bge',
            'top_k': 10,
            'top_n': 3,
            'llm_model': 'google/flan-t5-base'
        }
        experiments = [('baseline', baseline.copy())]
    
    # Limit experiments (command-line override)
    experiments = experiments[:args.max_experiments]
    
    print(f"Step 2: Running {len(experiments)} experiments...")
    print("=" * 80)
    
    all_results = []
    
    for i, (name, config) in enumerate(experiments):
        print(f"\n[Experiment {i+1}/{len(experiments)}] {name}")
        print("-" * 40)
        
        result = run_single_experiment(config, documents, document_ids, qa_pairs, indexed_contexts, device)
        result['name'] = name
        all_results.append(result)
        
        # Log to MLflow
        if MLFLOW_AVAILABLE and result['status'] == 'success':
            with mlflow.start_run(run_name=name):
                # Log parameters (config)
                for key, value in config.items():
                    mlflow.log_param(key, value)
                
                # Log all metrics (sanitize names for MLflow - replace @ with _at_)
                for metric_name, metric_value in result['metrics'].items():
                    if isinstance(metric_value, (int, float)):
                        safe_name = metric_name.replace('@', '_at_')
                        mlflow.log_metric(safe_name, metric_value)
                
                # Log latencies
                for latency_name, latency_value in result['latencies'].items():
                    if isinstance(latency_value, (int, float)):
                        mlflow.log_metric(f"latency_{latency_name}", latency_value)
                
                # Log tags
                mlflow.set_tag("experiment_name", name)
                mlflow.set_tag("device", device)
        
        if result['status'] == 'success':
            print(f"   Correct: {result['metrics']['correct']}/{result['metrics']['total_qa']}")
            print(f"   Hallucinations: {result['metrics']['hallucination']}")
            # Retrieval metrics
            if 'ret_mrr' in result['metrics']:
                print(f"   Retrieval MRR: {result['metrics']['ret_mrr']:.3f}")
            if 'ret_hit_rate@3' in result['metrics']:
                print(f"   Hit Rate@3: {result['metrics']['ret_hit_rate@3']:.3f}")
            # Generation metrics
            if 'gen_f1_score' in result['metrics']:
                print(f"   Gen F1: {result['metrics']['gen_f1_score']:.3f}")
            if 'gen_rouge_l' in result['metrics']:
                print(f"   ROUGE-L: {result['metrics']['gen_rouge_l']:.3f}")
            print(f"   Composite: {result['metrics']['composite_score']:.4f}")
            if MLFLOW_AVAILABLE:
                print(f"   [MLflow] Run logged")
        else:
            print(f"   Status: {result['status']}")
    
    # Create results DataFrame
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    results_data = []
    for r in all_results:
        row = {
            'name': r['name'],
            'status': r['status'],
            'correct': r['metrics'].get('correct', 0),
            'appropriate_abstention': r['metrics'].get('appropriate_abstention', 0),
            'hallucination': r['metrics'].get('hallucination', 0),
            'missed': r['metrics'].get('missed', 0),
            'accuracy': r['metrics'].get('accuracy', 0),
            'quality_score': r['metrics'].get('quality_score', 0),
            'hallucination_rate': r['metrics'].get('hallucination_rate', 0),
            'composite_score': r['metrics'].get('composite_score', 0),
            # Retrieval metrics
            'ret_mrr': r['metrics'].get('ret_mrr', 0),
            'ret_hit_rate@3': r['metrics'].get('ret_hit_rate@3', 0),
            'ret_hit_rate@5': r['metrics'].get('ret_hit_rate@5', 0),
            'ret_precision@1': r['metrics'].get('ret_precision@1', 0),
            'ret_precision@3': r['metrics'].get('ret_precision@3', 0),
            'ret_recall@3': r['metrics'].get('ret_recall@3', 0),
            # Generation metrics
            'gen_exact_match': r['metrics'].get('gen_exact_match', 0),
            'gen_f1_score': r['metrics'].get('gen_f1_score', 0),
            'gen_bleu_1': r['metrics'].get('gen_bleu_1', 0),
            'gen_rouge_1': r['metrics'].get('gen_rouge_1', 0),
            'gen_rouge_l': r['metrics'].get('gen_rouge_l', 0),
            # Latencies
            'num_chunks': r['metrics'].get('num_chunks', 0),
            'total_ms': r['latencies'].get('total_ms', 0),
            'avg_retrieval_ms': r['latencies'].get('avg_retrieval_ms', 0),
            'avg_rerank_ms': r['latencies'].get('avg_rerank_ms', 0),
            'avg_generation_ms': r['latencies'].get('avg_generation_ms', 0),
        }
        row.update(r['config'])
        results_data.append(row)
    
    df = pd.DataFrame(results_data)
    
    # Sort by composite score
    df_sorted = df.sort_values('composite_score', ascending=False)
    
    print("\nTop 10 Configurations:")
    cols_to_show = ['name', 'correct', 'hallucination', 'ret_mrr', 'gen_f1_score', 'composite_score']
    cols_available = [c for c in cols_to_show if c in df_sorted.columns]
    print(df_sorted[cols_available].head(10).to_string())
    
    # Save results
    csv_path = output_dir / "grid_search_results.csv"
    df_sorted.to_csv(csv_path, index=False)
    print(f"\n[OK] Results saved to: {csv_path}")
    
    # Save summary
    summary_path = output_dir / "experiment_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("GRID SEARCH RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Experiments: {len(experiments)}\n")
        f.write(f"Documents: {len(documents)}\n")
        f.write(f"Q&A Pairs: {len(qa_pairs)}\n\n")
        f.write("TOP 10 CONFIGURATIONS:\n")
        f.write("-" * 60 + "\n")
        f.write(df_sorted.head(10).to_string())
        f.write("\n\nBEST CONFIGURATION:\n")
        f.write("-" * 60 + "\n")
        best = df_sorted.iloc[0]
        for col in df_sorted.columns:
            f.write(f"   {col}: {best[col]}\n")
    
    print(f"[OK] Summary saved to: {summary_path}")
    
    # Print best
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)
    best = df_sorted.iloc[0]
    print(f"\n   Name: {best['name']}")
    print(f"   Correct Answers: {best['correct']}")
    print(f"   Hallucinations: {best['hallucination']} (lower is better)")
    print(f"   Quality Score: {best['quality_score']:.2%}")
    print(f"   Hallucination Rate: {best['hallucination_rate']:.2%}")
    print(f"\n   --- RETRIEVAL METRICS ---")
    print(f"   MRR: {best.get('ret_mrr', 0):.3f}")
    print(f"   Hit Rate@3: {best.get('ret_hit_rate@3', 0):.3f}")
    print(f"   Precision@3: {best.get('ret_precision@3', 0):.3f}")
    print(f"\n   --- GENERATION METRICS ---")
    print(f"   F1 Score: {best.get('gen_f1_score', 0):.3f}")
    print(f"   ROUGE-L: {best.get('gen_rouge_l', 0):.3f}")
    print(f"   BLEU-1: {best.get('gen_bleu_1', 0):.3f}")
    print(f"\n   --- CONFIG ---")
    print(f"   Composite Score: {best['composite_score']:.4f}")
    print(f"   Total Latency: {best['total_ms']:.0f}ms")
    print(f"   Reranker: {best['reranker_type']}")
    print(f"   Chunk Size: {best['chunk_size']}")
    
    print("\n" + "=" * 80)
    print("[GRID SEARCH COMPLETE]")
    print("=" * 80)
    
    # Print MLflow UI instructions
    if MLFLOW_AVAILABLE:
        print(f"\n[MLflow] To view experiment dashboard, run:")
        print(f"   mlflow ui --backend-store-uri {(output_dir / 'mlflow_tracking').absolute()}")
        print(f"   Then open: http://localhost:5000")
    print("")


if __name__ == "__main__":
    main()



