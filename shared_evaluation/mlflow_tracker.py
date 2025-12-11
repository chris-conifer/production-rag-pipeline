"""
MLflow Experiment Tracker
Logs all metrics, parameters, and artifacts for RAG experiments
SHARED across all 4 projects
"""

from typing import Dict, Any, Optional, List
import mlflow
import time
from pathlib import Path
import json
import pandas as pd


class MLflowExperimentTracker:
    """
    Manages MLflow experiment tracking for RAG pipelines
    
    Logs:
    - All hyperparameters (chunking, embedding, retrieval, reranking, generation)
    - All metrics (retrieval, generation, RAGAS, DeepEval, latency, cost)
    - Artifacts (configs, results, plots)
    - Architecture design info
    
    SHARED across all 4 projects for consistency
    """
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "./mlflow_tracking",
        artifact_location: Optional[str] = None
    ):
        """
        Initialize MLflow tracker
        
        Args:
            experiment_name: Name of the experiment (e.g., "RAG_Optimization", "Agentic_RAG")
            tracking_uri: MLflow tracking server URI or local path
            artifact_location: Optional artifact storage location
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location
                )
                self.experiment = mlflow.get_experiment(experiment_id)
            print(f"📊 MLflow Experiment: {experiment_name}")
            print(f"   ID: {self.experiment.experiment_id}")
            print(f"   Location: {tracking_uri}")
        except Exception as e:
            print(f"⚠️ Error setting up MLflow: {e}")
            self.experiment = None
        
        self.current_run = None
        self.run_start_time = None
    
    def start_run(
        self,
        run_name: str,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Start a new MLflow run
        
        Args:
            run_name: Name for this run (e.g., "baseline_config_001")
            tags: Optional tags for the run
        """
        if self.experiment is None:
            print("⚠️ MLflow experiment not initialized. Skipping tracking.")
            return
        
        self.run_start_time = time.time()
        
        self.current_run = mlflow.start_run(
            experiment_id=self.experiment.experiment_id,
            run_name=run_name,
            tags=tags or {}
        )
        
        print(f"\n🚀 Started MLflow Run: {run_name}")
        print(f"   Run ID: {self.current_run.info.run_id}")
    
    def log_architecture_params(self, config: Dict[str, Any]):
        """
        Log all architecture parameters
        
        Args:
            config: Full configuration dictionary
        """
        if not self.current_run:
            return
        
        # Flatten config for MLflow
        flat_params = self._flatten_config(config)
        
        try:
            mlflow.log_params(flat_params)
            print(f"   ✓ Logged {len(flat_params)} parameters")
        except Exception as e:
            print(f"   ⚠️ Error logging params: {e}")
    
    def log_chunking_metrics(self, metrics: Dict[str, Any]):
        """Log document chunking metrics"""
        if not self.current_run:
            return
        
        prefixed_metrics = {f"chunking_{k}": v for k, v in metrics.items()}
        mlflow.log_metrics(prefixed_metrics)
        print(f"   ✓ Logged chunking metrics")
    
    def log_embedding_metrics(self, metrics: Dict[str, Any]):
        """Log embedding generation metrics"""
        if not self.current_run:
            return
        
        prefixed_metrics = {f"embedding_{k}": v for k, v in metrics.items()}
        mlflow.log_metrics(prefixed_metrics)
        print(f"   ✓ Logged embedding metrics")
    
    def log_retrieval_metrics(self, metrics: Dict[str, Any]):
        """Log retrieval evaluation metrics"""
        if not self.current_run:
            return
        
        prefixed_metrics = {f"retrieval_{k}": v for k, v in metrics.items()}
        mlflow.log_metrics(prefixed_metrics)
        print(f"   ✓ Logged retrieval metrics: {list(metrics.keys())}")
    
    def log_reranking_metrics(self, metrics: Dict[str, Any]):
        """Log reranking metrics"""
        if not self.current_run:
            return
        
        prefixed_metrics = {f"reranking_{k}": v for k, v in metrics.items()}
        mlflow.log_metrics(prefixed_metrics)
        print(f"   ✓ Logged reranking metrics")
    
    def log_generation_metrics(self, metrics: Dict[str, Any]):
        """Log generation quality metrics"""
        if not self.current_run:
            return
        
        prefixed_metrics = {f"generation_{k}": v for k, v in metrics.items()}
        mlflow.log_metrics(prefixed_metrics)
        print(f"   ✓ Logged generation metrics: {list(metrics.keys())}")
    
    def log_ragas_metrics(self, metrics: Dict[str, Any]):
        """Log RAGAS evaluation metrics"""
        if not self.current_run:
            return
        
        # RAGAS metrics already have 'ragas_' prefix from evaluator
        mlflow.log_metrics(metrics)
        print(f"   ✓ Logged RAGAS metrics: {list(metrics.keys())}")
    
    def log_deepeval_metrics(self, metrics: Dict[str, Any]):
        """Log DeepEval metrics"""
        if not self.current_run:
            return
        
        # DeepEval metrics already have 'deepeval_' prefix
        mlflow.log_metrics(metrics)
        print(f"   ✓ Logged DeepEval metrics: {list(metrics.keys())}")
    
    def log_cost_metrics(self, metrics: Dict[str, Any]):
        """
        Log cost metrics
        
        Expected keys:
        - total_cost_usd
        - embedding_cost_usd
        - llm_cost_usd
        - cost_per_query
        """
        if not self.current_run:
            return
        
        prefixed_metrics = {f"cost_{k}": v for k, v in metrics.items()}
        mlflow.log_metrics(prefixed_metrics)
        print(f"   ✓ Logged cost metrics")
    
    def log_latency_metrics(self, metrics: Dict[str, Any]):
        """
        Log latency metrics
        
        Expected keys:
        - total_latency_ms
        - retrieval_latency_ms
        - reranking_latency_ms
        - generation_latency_ms
        - avg_query_latency_ms
        """
        if not self.current_run:
            return
        
        prefixed_metrics = {f"latency_{k}": v for k, v in metrics.items()}
        mlflow.log_metrics(prefixed_metrics)
        print(f"   ✓ Logged latency metrics")
    
    def log_composite_score(
        self,
        accuracy_score: float,
        latency_score: float,
        cost_score: float,
        weights: Dict[str, float] = None
    ):
        """
        Log composite ranking score
        
        Args:
            accuracy_score: Normalized accuracy (0-1)
            latency_score: Normalized latency (0-1, higher is better)
            cost_score: Normalized cost (0-1, higher is better)
            weights: Dict with 'accuracy', 'latency', 'cost' weights
        """
        if not self.current_run:
            return
        
        if weights is None:
            weights = {'accuracy': 0.5, 'latency': 0.3, 'cost': 0.2}
        
        composite_score = (
            accuracy_score * weights['accuracy'] +
            latency_score * weights['latency'] +
            cost_score * weights['cost']
        )
        
        mlflow.log_metrics({
            'composite_score': composite_score,
            'accuracy_component': accuracy_score,
            'latency_component': latency_score,
            'cost_component': cost_score
        })
        print(f"   ✓ Logged composite score: {composite_score:.4f}")
    
    def log_artifact_json(self, data: Dict, filename: str):
        """Log a JSON artifact"""
        if not self.current_run:
            return
        
        temp_path = Path(f"/tmp/{filename}")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        mlflow.log_artifact(str(temp_path))
        temp_path.unlink()  # Clean up
        print(f"   ✓ Logged artifact: {filename}")
    
    def log_artifact_csv(self, df: pd.DataFrame, filename: str):
        """Log a CSV artifact"""
        if not self.current_run:
            return
        
        temp_path = Path(f"/tmp/{filename}")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(temp_path, index=False)
        mlflow.log_artifact(str(temp_path))
        temp_path.unlink()  # Clean up
        print(f"   ✓ Logged artifact: {filename}")
    
    def log_artifact_file(self, file_path: str):
        """Log an existing file as artifact"""
        if not self.current_run:
            return
        
        mlflow.log_artifact(file_path)
        print(f"   ✓ Logged artifact: {Path(file_path).name}")
    
    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run
        
        Args:
            status: Run status ("FINISHED", "FAILED", "KILLED")
        """
        if not self.current_run:
            return
        
        # Log total runtime
        if self.run_start_time:
            runtime = time.time() - self.run_start_time
            mlflow.log_metric("total_runtime_seconds", runtime)
        
        mlflow.end_run(status=status)
        print(f"✓ Ended MLflow Run (Status: {status})")
        self.current_run = None
    
    def _flatten_config(self, config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested config for MLflow logging"""
        flat = {}
        
        for key, value in config.items():
            new_key = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                flat.update(self._flatten_config(value, f"{new_key}_"))
            elif isinstance(value, (list, tuple)):
                flat[new_key] = str(value)
            elif isinstance(value, (int, float, str, bool)):
                flat[new_key] = value
            else:
                flat[new_key] = str(value)
        
        return flat
    
    def get_run_info(self) -> Dict[str, Any]:
        """Get current run information"""
        if not self.current_run:
            return {}
        
        return {
            'run_id': self.current_run.info.run_id,
            'experiment_id': self.current_run.info.experiment_id,
            'run_name': self.current_run.data.tags.get('mlflow.runName', ''),
            'start_time': self.current_run.info.start_time,
        }
    
    def search_runs(
        self,
        filter_string: str = "",
        order_by: List[str] = None,
        max_results: int = 100
    ) -> pd.DataFrame:
        """
        Search runs in the experiment
        
        Args:
            filter_string: MLflow filter string (e.g., "metrics.accuracy > 0.8")
            order_by: List of columns to order by (e.g., ["metrics.composite_score DESC"])
            max_results: Maximum number of results
            
        Returns:
            DataFrame of runs
        """
        if self.experiment is None:
            return pd.DataFrame()
        
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            filter_string=filter_string,
            order_by=order_by or ["start_time DESC"],
            max_results=max_results
        )
        
        return runs
    
    def get_best_run(self, metric: str = "composite_score") -> Optional[Dict]:
        """
        Get the best run based on a metric
        
        Args:
            metric: Metric to optimize (default: composite_score)
            
        Returns:
            Dict with run info and metrics
        """
        runs = self.search_runs(
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )
        
        if runs.empty:
            return None
        
        best_run = runs.iloc[0]
        return best_run.to_dict()



