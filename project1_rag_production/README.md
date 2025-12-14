# 🚀 Production RAG Pipeline

##  Overview

This is a **production-ready, scalable RAG (Retrieval-Augmented Generation) pipeline** designed to showcase expertise in GenAI architecture, optimization, and evaluation.

### ✨ Key Features

- ✅ **Full OOP Architecture**: Modular, reusable components
- ✅ **Comprehensive Evaluation**: Retrieval, Generation, RAGAS, DeepEval metrics
- ✅ **Grid Search Optimization**: Systematic hyperparameter tuning
- ✅ **MLflow Tracking**: Complete experiment logging
- ✅ **Cost & Latency Optimization**: Real-time performance tracking
- ✅ **Visualization**: 3D tradeoff plots, Pareto frontiers
- ✅ **Reranker Integration**: Cross-encoder for improved accuracy
- ✅ **Golden Dataset**: 100 stratified Q&A examples
- ✅ **Google Colab Ready**: Run on free GPU
- ✅ **Modular for 4 Projects**: Shared evaluation framework

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Pipeline Architecture                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Documents  →  Chunking  →  Embedding  →  FAISS Index      │
│                                                             │
│  Query  →  Embed  →  Retrieve (Top-K)  →  Rerank (Top-N)  │
│                                         ↓                    │
│                                    LLM Generate             │
│                                         ↓                    │
│                                      Answer                  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    Evaluation Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Golden Dataset (100 Q&A)  →  Run Pipeline  →  Evaluate    │
│                                                             │
│  Metrics:                                                   │
│    • Retrieval: P@K, R@K, MRR, NDCG, MAP                   │
│    • Generation: BLEU, ROUGE, F1, EM                        │
│    • RAGAS: Faithfulness, Relevancy                         │
│    • DeepEval: Hallucination Detection                      │
│    • Performance: Latency, Cost                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Project Structure

```
project1_rag_production/
├── src/
│   ├── core/                    # Core RAG components (OOP)
│   │   ├── document_processor.py   # Chunking with metadata
│   │   ├── embedder.py             # Sentence transformers
│   │   ├── retriever.py            # FAISS vector search
│   │   ├── reranker.py             # Cross-encoder reranking
│   │   └── generator.py            # LLM generation
│   │
│   ├── pipeline/                # Pipeline orchestration
│   │   ├── rag_pipeline.py         # Main RAG pipeline
│   │   └── pipeline_factory.py     # Config-based creation
│   │
│   ├── evaluation/              # Evaluation (imports from shared)
│   │   └── __init__.py             # Imports shared_evaluation
│   │
│   └── optimization/            # Grid search
│       └── grid_search_orchestrator.py
│
├── shared_evaluation/           # 🔄 SHARED ACROSS 4 PROJECTS
│   ├── metrics/
│   │   ├── retrieval_metrics.py
│   │   ├── generation_metrics.py
│   │   ├── ragas_evaluator.py
│   │   └── deepeval_evaluator.py
│   ├── composite_evaluator.py
│   ├── mlflow_tracker.py
│   ├── visualizer.py
│   ├── export_utils.py
│   └── golden_dataset.py
│
├── configs/
│   ├── base_config.yaml         # Default configuration
│   └── grid_search_config.yaml  # Parameter grid
│
├── scripts/
│   ├── run_grid_search.py       # CLI: Full optimization
│   └── demo_single_query.py     # CLI: Single query test
│
├── notebooks/
│   └── Google_Colab_Production_RAG.ipynb
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/MonoRepo.git
cd MonoRepo/project1_rag_production

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Single Query Demo

```bash
python scripts/demo_single_query.py \
    --config ./configs/base_config.yaml \
    --query "What is machine learning?" \
    --num-docs 50
```

### 3. Full Grid Search Optimization

```bash
python scripts/run_grid_search.py \
    --base-config ./configs/base_config.yaml \
    --grid-config ./configs/grid_search_config.yaml \
    --num-docs 100 \
    --output-dir ./outputs
```

### 4. Google Colab

Open `notebooks/Google_Colab_Production_RAG.ipynb` in Google Colab for GPU-accelerated execution.

---

## 🔍 Core Components

### 1. Document Processor
- **Chunking strategies**: Fixed size, sentence-based
- **Metadata extraction**: Length, word count, index
- **Overlap control**: Configurable overlap between chunks

### 2. Embedder
- **Models**: Sentence Transformers (all-MiniLM, MPNet, BGE)
- **Batch processing**: GPU-accelerated
- **Normalization**: Automatic embedding normalization

### 3. Retriever
- **FAISS indexes**: Flat, HNSW, IVF
- **Metrics**: L2, Inner Product
- **Scalability**: Handles millions of vectors

### 4. Reranker ⭐
- **Cross-encoder models**: MS MARCO, STSB
- **Two-stage retrieval**: Broad recall → Precise reranking
- **Significant accuracy boost**: +10-15% typical improvement

### 5. Generator
- **LLM support**: T5, LLaMA, Phi, Mistral
- **Quantization**: 4-bit, 8-bit for efficiency
- **Generation params**: Temperature, top-p, num_beams

---

## 📊 Evaluation Framework

### Metrics Tracked

#### Retrieval Metrics
- **Precision@K**: Relevance of top K results
- **Recall@K**: Coverage of relevant documents
- **MRR**: Mean Reciprocal Rank
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP**: Mean Average Precision
- **Hit Rate**: At least one relevant in top K

#### Generation Metrics
- **BLEU**: N-gram overlap with references
- **ROUGE**: Recall-oriented overlap
- **F1 Score**: Token-level precision & recall
- **Exact Match**: Exact string match

#### RAGAS Metrics
- **Faithfulness**: Answer grounded in context
- **Answer Relevancy**: Answer addresses question
- **Context Precision**: Relevant context retrieved
- **Context Recall**: All relevant context retrieved

#### DeepEval Metrics
- **Hallucination Detection**: Identifies fabricated information
- **Bias Detection**: Identifies unfair bias
- **Toxicity**: Identifies harmful content

#### Performance Metrics
- **Latency**: Total, retrieval, reranking, generation (ms)
- **Cost**: Embedding, LLM tokens ($)
- **Throughput**: Queries per second

### Composite Score

```python
composite_score = (
    0.5 * accuracy +  # Highest weight
    0.3 * (1 / latency) +  # Medium weight
    0.2 * (1 / cost)  # Lower weight
)
```

---

## 🧪 Grid Search

### Strategy: One Parameter at a Time

1. **Baseline**: Run with default config
2. **Chunking**: Vary chunk_size, chunk_overlap, strategy
3. **Embedding**: Vary model, batch_size
4. **Retrieval**: Vary index_type, top_k
5. **Reranking**: Vary model, top_n, enabled/disabled
6. **LLM**: Vary model, quantization
7. **Generation**: Vary max_length, temperature, num_beams

### Configuration

Edit `configs/grid_search_config.yaml` to define parameter spaces:

```yaml
chunking_grid:
  chunk_size:
    baseline: 512
    options: [256, 512, 768, 1024]
  
  chunk_overlap:
    baseline: 100
    options: [0, 50, 100, 150, 200]
```

---

## 📈 Visualization & Reporting

### Outputs Generated

1. **CSV Files**:
   - `all_experiments.csv`: All runs with parameters & metrics
   - `top_10_architectures.csv`: Best configurations

2. **TXT Reports**:
   - `experiment_summary.txt`: Human-readable summary

3. **JSON**:
   - `detailed_results.json`: Complete structured results

4. **Plots**:
   - `3d_tradeoff.png`: Accuracy vs Latency vs Cost
   - `pareto_frontier.png`: Optimal tradeoffs
   - `top_architectures_comparison.png`: Side-by-side comparison
   - `metric_distributions.png`: Distribution histograms

### MLflow UI

```bash
mlflow ui --backend-store-uri ./mlflow_tracking
```

Visit http://localhost:5000 to explore experiments interactively.

---

## 🔄 Modular Design: 4 Projects

The `shared_evaluation/` framework is **reusable across**:

1. **Project 1**: Basic RAG (this project)
2. **Project 2**: RAG with LLM Judge
3. **Project 3**: Agentic RAG
4. **Project 4**: Multi-modal Agent System

### Benefits

- ✅ **No code duplication**: Write once, use everywhere
- ✅ **Consistent metrics**: Same evaluation across projects
- ✅ **Easy maintenance**: Fix bugs in one place
- ✅ **Rapid development**: Focus on core logic, not boilerplate

---

## 💡 Best Practices

### 1. Golden Dataset
- Use stratified sampling (difficulty, length, type)
- 100-500 examples recommended
- Include edge cases

### 2. Grid Search
- Start with baseline
- Change one parameter at a time
- Log everything to MLflow

### 3. Reranking
- Always enable for production
- 2-stage retrieval significantly improves accuracy
- Slight latency increase (~50-100ms) is worth it

### 4. Cost Optimization
- Use smaller models for embedding (all-MiniLM)
- Apply quantization for LLMs (4-bit)
- Cache embeddings when possible

### 5. Latency Optimization
- Batch queries when possible
- Use HNSW index for large corpora
- Limit top_k to necessary size

---

## 🎓 Technical Highlights:

### Architecture Expertise
- ✅ Clean OOP design with SOLID principles
- ✅ Factory pattern for configuration-based instantiation
- ✅ Modular components with clear interfaces
- ✅ Separation of concerns (pipeline vs evaluation vs optimization)

### Optimization Expertise
- ✅ Systematic hyperparameter tuning (grid search)
- ✅ Multi-objective optimization (accuracy vs latency vs cost)
- ✅ Pareto frontier analysis
- ✅ Composite ranking scores

### Evaluation Expertise
- ✅ Comprehensive metric suite (retrieval + generation + frameworks)
- ✅ Multiple evaluation frameworks (RAGAS, DeepEval)
- ✅ Ranking & non-ranking metrics
- ✅ Golden dataset methodology

### Production Readiness
- ✅ MLflow experiment tracking
- ✅ Complete logging & monitoring
- ✅ Cost & latency tracking
- ✅ CSV/JSON exports for analysis
- ✅ CLI & notebook interfaces
- ✅ Google Colab compatible

### No Hallucination Focus
- ✅ Faithfulness metrics (RAGAS)
- ✅ Hallucination detection (DeepEval)
- ✅ Context grounding validation
- ✅ Reranker for precision

---

## 📚 References

- **FAISS**: [facebook/faiss](https://github.com/facebookresearch/faiss)
- **Sentence Transformers**: [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- **RAGAS**: [explodinggradients/ragas](https://github.com/explodinggradients/ragas)
- **DeepEval**: [confident-ai/deepeval](https://github.com/confident-ai/deepeval)
- **MLflow**: [mlflow.org](https://mlflow.org/)

---

## 📞 Contact

**Author**: Christian Dudziak  
**GitHub**: [GitHub](https://github.com/chris-conifer)  
**LinkedIn**: [LinkedIn](https://www.linkedin.com/in/christian-dudziak-b9193931/)  

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- HuggingFace for datasets and models
- Facebook Research for FAISS
- The open-source community for amazing tools

---

**⭐ If you find this project useful, please star it on GitHub!**



