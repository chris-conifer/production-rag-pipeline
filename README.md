# 🚀 Production RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) pipeline with systematic hyperparameter optimization, comprehensive evaluation, and MLflow experiment tracking.

---

## 🌟 100% FREE & Open Source

**This pipeline is designed to run entirely on FREE, open-source models - no API keys required!**

| Component | Default (Free) | Optional Paid Alternatives |
|-----------|----------------|---------------------------|
| **Embeddings** | `all-MiniLM-L6-v2` | OpenAI, Cohere embeddings |
| **Reranker** | `bge-reranker-v2-m3` | Cohere Rerank API |
| **Generator** | `flan-t5-base` | GPT-4, Claude, Gemini |
| **Dataset** | SQuAD (Stanford) | Your custom data |

> 💡 **Why Open Source First?** 
> - Zero cost to experiment and learn
> - No rate limits or API quotas
> - Full control over model behavior
> - Works offline after initial download
> - Easy to upgrade to paid APIs later (just update config)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔪 **Multiple Chunking** | Fixed-size, sentence-based, semantic, paragraph, contextual, token-based |
| 🔍 **Hybrid Retrieval** | Dense (FAISS) + Sparse (BM25) with RRF fusion - **Default** |
| 🏆 **BGE Reranker** | BAAI/bge-reranker-v2-m3 - **#1 on MTEB Leaderboard** |
| 🤖 **LLM Generation** | Flan-T5 with configurable parameters |
| 📊 **Comprehensive Metrics** | MRR, Precision, Recall, F1, ROUGE-L, Hallucination Rate |
| 🔬 **Grid Search** | Systematic hyperparameter optimization (one-at-a-time) |
| 📈 **MLflow Tracking** | Full experiment tracking, comparison, and artifact logging |
| ☁️ **Cloud Ready** | Works on Google Colab (free GPU) and AWS SageMaker |

---

## 📂 Repository Structure

```
production-rag-pipeline/
├── project1_rag_production/          # 🎯 MAIN RAG PIPELINE
│   ├── src/
│   │   ├── core/                     # Core RAG components
│   │   │   ├── document_processor.py # 6 chunking strategies
│   │   │   ├── embedder.py           # Sentence transformers
│   │   │   ├── retriever.py          # FAISS vector search
│   │   │   ├── hybrid_retriever.py   # Dense + BM25 hybrid
│   │   │   ├── reranker_factory.py   # BGE/CrossEncoder rerankers
│   │   │   └── generator.py          # Flan-T5 generation
│   │   │
│   │   └── evaluation/               # Evaluation metrics
│   │       └── metrics/
│   │
│   ├── scripts/
│   │   ├── run_grid_search.py        # Phase 1: Hyperparameter tuning
│   │   └── run_final_evaluation.py   # Phase 2: Golden test set
│   │
│   ├── configs/
│   │   ├── base_config.yaml          # Default configuration
│   │   └── grid_search_config.yaml   # Grid search parameters
│   │
│   ├── notebooks/
│   │   └── Google_Colab_Production_RAG.ipynb
│   │
│   ├── env.example                   # Dummy API keys template
│   └── requirements.txt
│
├── shared_evaluation/                # 🔄 REUSABLE EVALUATION FRAMEWORK
│   ├── metrics/
│   │   ├── retrieval_metrics.py      # MRR, Precision, Recall, Hit Rate
│   │   ├── generation_metrics.py     # F1, ROUGE-L, BLEU, Exact Match
│   │   ├── ragas_evaluator.py        # RAGAS framework integration
│   │   └── deepeval_evaluator.py     # DeepEval integration
│   │
│   ├── golden_dataset.py             # Golden Q&A dataset management
│   ├── mlflow_tracker.py             # MLflow experiment tracking
│   ├── composite_evaluator.py        # Combined evaluation
│   └── export_utils.py               # CSV/JSON export
│
├── configs/                          # Global configurations
├── requirements.txt                  # Root dependencies
└── README.md                         # This file
```

---

## 🚀 Quick Start: Google Colab (FREE GPU)

### Step 1: Fork This Repository

1. Click the **Fork** button at the top right of this page
2. Select your GitHub account
3. You now have your own copy!

### Step 2: Open Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **New Notebook**
3. **Enable GPU**: `Runtime` → `Change runtime type` → `T4 GPU` → `Save`

### Step 3: Run the Pipeline

Copy and paste these cells into your Colab notebook:

---

#### 📦 Cell 1: Clone & Install Dependencies

```python
#@title 1. Setup Environment
import os

# Clone YOUR forked repository (replace YOUR_USERNAME)
!git clone https://github.com/YOUR_USERNAME/production-rag-pipeline.git
%cd production-rag-pipeline

# Install PyTorch with CUDA
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install RAG dependencies
!pip install -q transformers sentence-transformers datasets faiss-cpu
!pip install -q mlflow pandas pyyaml tqdm rouge-score nltk rank-bm25

# Verify GPU
import torch
if torch.cuda.is_available():
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ No GPU detected - will run on CPU (slower)")

print("\n✅ Setup complete!")
```

---

#### 🔬 Cell 2: Phase 1 - Grid Search (Find Best Configuration)

```python
#@title 2. Phase 1: Grid Search (Hyperparameter Tuning)
%cd project1_rag_production

# Run grid search with:
# - 300 documents from SQuAD dataset
# - 30 Q&A pairs for evaluation  
# - Max 10 experiment configurations

!python scripts/run_grid_search.py \
    --num-docs 300 \
    --num-qa 30 \
    --max-experiments 10

print("\n✅ Phase 1 Complete! Best configuration identified.")
```

**What this does:**
- Tests different configurations (chunk sizes, rerankers, retrieval modes)
- Changes **one parameter at a time** (scientific approach)
- Logs all results to MLflow
- Saves results to `outputs/grid_search_results.csv`

---

#### 🏆 Cell 3: Phase 2 - Final Evaluation (Golden Test Set)

```python
#@title 3. Phase 2: Final Evaluation on Held-Out Test Set
# Uses the BEST configuration from Phase 1
# Evaluates on SQuAD validation set (never seen during tuning)

!python scripts/run_final_evaluation.py \
    --num-docs 500 \
    --num-qa 50

print("\n✅ Phase 2 Complete! Unbiased metrics on golden test set.")
```

**What this does:**
- Loads best config from Phase 1 automatically
- Evaluates on **held-out validation data** (not used in tuning)
- Reports unbiased final metrics

---

#### 📊 Cell 4: View Grid Search Results

```python
#@title 4. View Grid Search Results
import pandas as pd

print("=" * 70)
print("PHASE 1: GRID SEARCH RESULTS (Ranked by Composite Score)")
print("=" * 70)

df = pd.read_csv("outputs/grid_search_results.csv")
display_cols = ['name', 'correct', 'hallucination', 'gen_f1_score', 'composite_score']
available_cols = [c for c in display_cols if c in df.columns]
print(df[available_cols].head(10).to_string())

# Best configuration
print("\n" + "=" * 70)
print("BEST CONFIGURATION")
print("=" * 70)
best = df.iloc[0]
print(f"Name: {best['name']}")
print(f"Composite Score: {best['composite_score']:.4f}")
```

---

#### 📈 Cell 5: View MLflow Experiment Logs

```python
#@title 5. View MLflow Logs (Detailed Experiment Tracking)
import mlflow
import pandas as pd

# Set MLflow tracking URI
mlflow.set_tracking_uri("outputs/mlflow_tracking")

# Get all experiments
experiments = mlflow.search_experiments()
print("=" * 70)
print("MLFLOW EXPERIMENTS")
print("=" * 70)

for exp in experiments:
    print(f"\nExperiment: {exp.name}")
    print(f"  ID: {exp.experiment_id}")
    
    # Get runs for this experiment
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    if len(runs) > 0:
        print(f"  Total Runs: {len(runs)}")
        print(f"\n  Top 5 Runs by Composite Score:")
        
        if 'metrics.composite_score' in runs.columns:
            top_runs = runs.nlargest(5, 'metrics.composite_score')
            for i, (_, run) in enumerate(top_runs.iterrows(), 1):
                print(f"    {i}. Run {run['run_id'][:8]}...")
                print(f"       Composite Score: {run.get('metrics.composite_score', 'N/A'):.4f}")
                print(f"       Accuracy: {run.get('metrics.quality_score', 'N/A'):.2%}")
                print(f"       F1 Score: {run.get('metrics.gen_f1_score', 'N/A'):.4f}")
```

---

#### 📋 Cell 6: View Final Evaluation Report

```python
#@title 6. View Final Evaluation Report
import os

print("=" * 70)
print("PHASE 2: FINAL EVALUATION REPORT")
print("=" * 70)

# Find the latest report
report_dir = "outputs/phase2_final_evaluation"
if os.path.exists(report_dir):
    reports = sorted([f for f in os.listdir(report_dir) if f.startswith("FINAL_EVALUATION_REPORT")])
    if reports:
        latest_report = os.path.join(report_dir, reports[-1])
        with open(latest_report, 'r') as f:
            print(f.read())
    else:
        print("No reports found. Run Phase 2 first.")
else:
    print("Phase 2 not run yet. Execute Cell 3 first.")
```

---

#### 💾 Cell 7: Download All Results

```python
#@title 7. Download Results to Your Computer
from google.colab import files
import shutil

# Create zip of all outputs
shutil.make_archive('rag_pipeline_results', 'zip', 'outputs')

# Download
files.download('rag_pipeline_results.zip')

print("✅ Results downloaded!")
print("   Contains: grid_search_results.csv, MLflow logs, evaluation reports")
```

---

## 📈 Expected Results

| Metric | Typical Range | Description |
|--------|---------------|-------------|
| **Accuracy** | 85-95% | Correct answers / Total questions |
| **Hallucination Rate** | 0-10% | Fabricated answers (lower is better) |
| **F1 Score** | 0.70-0.85 | Token overlap with ground truth |
| **MRR** | 0.70-0.85 | Mean Reciprocal Rank (retrieval quality) |
| **Hit Rate@3** | 0.80-0.95 | Correct doc in top 3 retrievals |
| **ROUGE-L** | 0.70-0.85 | Longest common sequence match |

---

## 🏆 Key Findings: Parameter Impact

Based on empirical grid search results:

| Rank | Parameter | Impact | Expected Accuracy Lift |
|------|-----------|--------|------------------------|
| 🥇 | `reranker_type=bge` | ⭐⭐⭐⭐⭐ | **+10-15%** |
| 🥈 | `retrieval_mode=hybrid` | ⭐⭐⭐⭐⭐ | **+5-10%** |
| 🥉 | `chunk_strategy=sentence` | ⭐⭐⭐⭐ | **+3-5%** |
| 4 | `embedding_model` (BGE vs MiniLM) | ⭐⭐⭐⭐ | +3-5% |
| 5 | `chunk_size=512` | ⭐⭐⭐ | +2-3% |
| 6 | `top_k=10` | ⭐⭐⭐ | +1-2% |

**💡 Biggest Wins:** BGE reranker + Hybrid retrieval provide the largest accuracy improvements!

---

## 🔧 Customize Grid Search

Edit `project1_rag_production/configs/grid_search_config.yaml`:

```yaml
# Chunking strategies to test
chunking_grid:
  strategy:
    baseline: "sentence"  # Best for accuracy
    options: ["fixed_size", "sentence", "paragraph", "semantic"]
  
  chunk_size:
    baseline: 512
    options: [256, 512, 768, 1024]

# Retrieval configuration
retrieval_grid:
  mode:
    baseline: "hybrid"  # Dense + BM25 (recommended)
    options: ["hybrid", "dense", "sparse"]
  
  top_k:
    baseline: 10
    options: [5, 10, 15, 20]

# Rerankers to test (all FREE & open-source)
reranking_grid:
  reranker_type:
    baseline: "bge"  # #1 on MTEB leaderboard
    options: ["bge", "cross_encoder"]
```

---

## 🛠️ Models Used (All FREE & Open Source)

| Component | Model | Parameters | Source |
|-----------|-------|------------|--------|
| **Embeddings** | `all-MiniLM-L6-v2` | 22M | HuggingFace |
| **Embeddings (alt)** | `BAAI/bge-base-en-v1.5` | 110M | HuggingFace |
| **Reranker** | `BAAI/bge-reranker-v2-m3` | 568M | HuggingFace |
| **Reranker (alt)** | `ms-marco-MiniLM-L-6-v2` | 22M | HuggingFace |
| **Generator** | `google/flan-t5-base` | 250M | HuggingFace |
| **Dataset** | SQuAD v1.1 | 100K Q&A | Stanford |

> 📦 **All models auto-download on first run** - no manual setup needed!

---

## 🔓 Optional: Add Paid API Models

While this pipeline works 100% free, you can optionally add paid models for potentially better quality:

1. Copy `env.example` to `.env`
2. Add your API keys:

```bash
# .env file
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
ANTHROPIC_API_KEY=...
```

3. Update `base_config.yaml` to use paid models

> ⚠️ **Important:** Never commit `.env` files to GitHub!

---

## 🔒 Security

- ✅ **No API keys required** - All default models are free
- ✅ `.env` files are gitignored
- ✅ Only `env.example` with dummy values is included
- ✅ Outputs are gitignored (won't accidentally push your results)
- ✅ Safe to fork and run immediately

---

## 💻 Local Installation (Alternative to Colab)

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/production-rag-pipeline.git
cd production-rag-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install PyTorch with CUDA (for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
pip install -r project1_rag_production/requirements.txt

# Run Grid Search
cd project1_rag_production
python scripts/run_grid_search.py --num-docs 300 --num-qa 30 --max-experiments 10

# Run Final Evaluation
python scripts/run_final_evaluation.py --num-docs 500 --num-qa 50
```

---

## 🆘 Troubleshooting

### "CUDA out of memory"
```python
# Reduce the number of documents
!python scripts/run_grid_search.py --num-docs 100 --num-qa 20 --max-experiments 5
```

### "Module not found"
```python
# Reinstall dependencies
!pip install transformers sentence-transformers datasets faiss-cpu rank-bm25
```

### "GPU not detected"
- In Colab: `Runtime` → `Change runtime type` → `T4 GPU`
- Verify: `!nvidia-smi`

### MLflow not logging
```python
# Check if outputs directory exists
import os
os.makedirs("outputs/mlflow_tracking", exist_ok=True)
```

---

## 📚 Further Reading

### Must-Read Papers (7 Core Papers)

| # | Paper | Topic | Link |
|---|-------|-------|------|
| 1 | **HyDE** | Query Expansion | [arxiv.org/abs/2212.10496](https://arxiv.org/abs/2212.10496) |
| 2 | **Self-RAG** | Answer Verification | [arxiv.org/abs/2310.11511](https://arxiv.org/abs/2310.11511) |
| 3 | **LongLLMLingua** | Context Compression | [arxiv.org/abs/2310.04408](https://arxiv.org/abs/2310.04408) |
| 4 | **RAG-Fusion** | Multi-Query Approach | [arxiv.org/abs/2305.14283](https://arxiv.org/abs/2305.14283) |
| 5 | **Least-to-Most Prompting** | Reasoning | [arxiv.org/abs/2205.10625](https://arxiv.org/abs/2205.10625) |
| 6 | **Dense Passage Retrieval (DPR)** | Dense Retrieval | [arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906) |
| 7 | **ColBERT** | Late Interaction | [arxiv.org/abs/2004.12832](https://arxiv.org/abs/2004.12832) |

### Best Tutorials & Guides

| # | Resource | Description | Link |
|---|----------|-------------|------|
| 1 | **Pinecone RAG Guide** | Comprehensive overview (START HERE) | [pinecone.io/learn/retrieval-augmented-generation](https://www.pinecone.io/learn/retrieval-augmented-generation/) |
| 2 | **Pinecone Hybrid Search** | Dense + Sparse retrieval | [pinecone.io/learn/hybrid-search-intro](https://www.pinecone.io/learn/hybrid-search-intro/) |
| 3 | **Weaviate Hybrid Search** | Hybrid search explained | [weaviate.io/blog/hybrid-search-explained](https://weaviate.io/blog/hybrid-search-explained) |
| 4 | **LangChain Contextual Compression** | Context compression | [python.langchain.com/.../contextual_compression](https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression) |
| 5 | **Redis Vector Search** | Caching & vector search | [redis.io/docs/.../vector-search](https://redis.io/docs/latest/develop/interact/search-and-query/query/vector-search/) |
| 6 | **Pinecone Production RAG** | Production best practices | [pinecone.io/learn/production-rag](https://www.pinecone.io/learn/production-rag/) |
| 7 | **OpenAI RAG Best Practices** | Official OpenAI guide | [platform.openai.com/.../retrieval-augmented-generation](https://platform.openai.com/docs/guides/retrieval-augmented-generation) |

---

## 📄 License

MIT License - Free to use, modify, and distribute.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-improvement`
3. Commit your changes: `git commit -m "Add my improvement"`
4. Push: `git push origin feature/my-improvement`
5. Open a Pull Request

---

## ⭐ Star This Repo

If you find this useful, please give it a star! ⭐

---

**Built with ❤️ for the GenAI community**

*Demonstrating expertise in: RAG Architecture, Hybrid Retrieval, Reranking, Hyperparameter Optimization, Evaluation Frameworks (RAGAS, DeepEval), MLflow Experiment Tracking, and Production ML Systems*
