"""
Export Utilities
Exports experiment results to various formats (CSV, TXT, JSON)
SHARED across all 4 projects
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import json
from pathlib import Path
from datetime import datetime


class ResultsExporter:
    """
    Exports RAG experiment results to multiple formats
    
    Features:
    - CSV export with all metrics
    - TXT summary reports
    - JSON detailed results
    - Top-K architecture summaries
    
    SHARED across all 4 projects
    """
    
    def __init__(self, output_dir: str = "./outputs"):
        """
        Initialize exporter
        
        Args:
            output_dir: Directory to save exports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_all_runs_csv(
        self,
        df: pd.DataFrame,
        filename: str = None,
        include_params: bool = True,
        include_metrics: bool = True
    ) -> str:
        """
        Export all MLflow runs to CSV
        
        Args:
            df: DataFrame from mlflow.search_runs()
            filename: Output filename (auto-generated if None)
            include_params: Include parameter columns
            include_metrics: Include metric columns
            
        Returns:
            Path to saved CSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"all_experiments_{timestamp}.csv"
        
        output_path = self.output_dir / filename
        
        # Filter columns
        columns_to_keep = []
        
        # Always include run info
        info_cols = ['run_id', 'experiment_id', 'status', 'start_time', 'end_time']
        columns_to_keep.extend([col for col in info_cols if col in df.columns])
        
        if include_params:
            param_cols = [col for col in df.columns if col.startswith('params.')]
            columns_to_keep.extend(param_cols)
        
        if include_metrics:
            metric_cols = [col for col in df.columns if col.startswith('metrics.')]
            columns_to_keep.extend(metric_cols)
        
        # Export
        df_export = df[columns_to_keep].copy()
        df_export.to_csv(output_path, index=False)
        
        print(f"✓ Exported {len(df_export)} runs to: {output_path}")
        print(f"  Columns: {len(columns_to_keep)}")
        
        return str(output_path)
    
    def export_top_k_summary_csv(
        self,
        df: pd.DataFrame,
        top_k: int = 10,
        sort_by: str = "metrics.composite_score",
        filename: str = None
    ) -> str:
        """
        Export top K architectures summary to CSV
        
        Args:
            df: DataFrame from mlflow.search_runs()
            top_k: Number of top architectures to export
            sort_by: Column to sort by
            filename: Output filename
            
        Returns:
            Path to saved CSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"top_{top_k}_architectures_{timestamp}.csv"
        
        output_path = self.output_dir / filename
        
        # Sort and get top K
        if sort_by in df.columns:
            df_sorted = df.sort_values(sort_by, ascending=False)
        else:
            df_sorted = df.copy()
        
        df_top = df_sorted.head(top_k).copy()
        
        # Add rank column
        df_top.insert(0, 'rank', range(1, len(df_top) + 1))
        
        df_top.to_csv(output_path, index=False)
        
        print(f"✓ Exported top {top_k} architectures to: {output_path}")
        
        return str(output_path)
    
    def export_summary_txt(
        self,
        df: pd.DataFrame,
        top_k: int = 10,
        sort_by: str = "metrics.composite_score",
        filename: str = None
    ) -> str:
        """
        Export human-readable TXT summary
        
        Args:
            df: DataFrame from mlflow.search_runs()
            top_k: Number of top architectures to include
            sort_by: Column to sort by
            filename: Output filename
            
        Returns:
            Path to saved TXT file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_summary_{timestamp}.txt"
        
        output_path = self.output_dir / filename
        
        # Sort
        if sort_by in df.columns:
            df_sorted = df.sort_values(sort_by, ascending=False)
        else:
            df_sorted = df.copy()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RAG PIPELINE EXPERIMENT SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Experiments: {len(df)}\n")
            f.write(f"Top {top_k} Architectures Shown\n\n")
            
            # Overall statistics
            f.write("-" * 80 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 80 + "\n\n")
            
            metric_cols = [col for col in df.columns if col.startswith('metrics.')]
            for col in metric_cols[:10]:  # Limit to first 10 metrics
                if df[col].notna().any():
                    f.write(f"{col.replace('metrics.', '')}:\n")
                    f.write(f"  Mean:   {df[col].mean():.4f}\n")
                    f.write(f"  Median: {df[col].median():.4f}\n")
                    f.write(f"  Std:    {df[col].std():.4f}\n")
                    f.write(f"  Min:    {df[col].min():.4f}\n")
                    f.write(f"  Max:    {df[col].max():.4f}\n\n")
            
            # Top K architectures
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"TOP {top_k} ARCHITECTURES\n")
            f.write("=" * 80 + "\n\n")
            
            for i, (idx, row) in enumerate(df_sorted.head(top_k).iterrows(), 1):
                f.write(f"{'#' * 3} RANK {i} {'#' * 3}\n")
                f.write(f"Run ID: {row.get('run_id', 'N/A')}\n\n")
                
                # Key metrics
                f.write("Key Metrics:\n")
                key_metrics = [
                    'composite_score',
                    'retrieval_precision_at_5',
                    'generation_f1_score',
                    'ragas_faithfulness',
                    'latency_avg_query_latency_ms',
                    'cost_cost_per_query'
                ]
                for metric in key_metrics:
                    col_name = f'metrics.{metric}'
                    if col_name in row.index and pd.notna(row[col_name]):
                        f.write(f"  {metric}: {row[col_name]:.4f}\n")
                
                # Key parameters
                f.write("\nKey Parameters:\n")
                param_cols = [col for col in row.index if col.startswith('params.')]
                for col in param_cols[:10]:  # Limit to first 10 params
                    if pd.notna(row[col]):
                        f.write(f"  {col.replace('params.', '')}: {row[col]}\n")
                
                f.write("\n" + "-" * 80 + "\n\n")
        
        print(f"✓ Exported summary to: {output_path}")
        
        return str(output_path)
    
    def export_detailed_json(
        self,
        df: pd.DataFrame,
        top_k: Optional[int] = None,
        sort_by: str = "metrics.composite_score",
        filename: str = None
    ) -> str:
        """
        Export detailed JSON with all run information
        
        Args:
            df: DataFrame from mlflow.search_runs()
            top_k: Number of top architectures (None = all)
            sort_by: Column to sort by
            filename: Output filename
            
        Returns:
            Path to saved JSON file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = f"_top_{top_k}" if top_k else "_all"
            filename = f"detailed_results{suffix}_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Sort
        if sort_by in df.columns:
            df_sorted = df.sort_values(sort_by, ascending=False)
        else:
            df_sorted = df.copy()
        
        # Limit to top K if specified
        if top_k:
            df_export = df_sorted.head(top_k)
        else:
            df_export = df_sorted
        
        # Convert to JSON-serializable format
        results = {
            'metadata': {
                'generated': datetime.now().isoformat(),
                'total_experiments': len(df),
                'exported_experiments': len(df_export),
                'sorted_by': sort_by
            },
            'experiments': []
        }
        
        for i, (idx, row) in enumerate(df_export.iterrows(), 1):
            experiment = {
                'rank': i,
                'run_id': row.get('run_id', 'N/A'),
                'parameters': {},
                'metrics': {}
            }
            
            # Extract parameters
            for col in row.index:
                if col.startswith('params.') and pd.notna(row[col]):
                    param_name = col.replace('params.', '')
                    experiment['parameters'][param_name] = row[col]
            
            # Extract metrics
            for col in row.index:
                if col.startswith('metrics.') and pd.notna(row[col]):
                    metric_name = col.replace('metrics.', '')
                    experiment['metrics'][metric_name] = float(row[col])
            
            results['experiments'].append(experiment)
        
        # Save
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Exported detailed results to: {output_path}")
        print(f"  Experiments: {len(results['experiments'])}")
        
        return str(output_path)
    
    def export_comparison_table(
        self,
        df: pd.DataFrame,
        architectures: List[str],
        metrics: List[str],
        filename: str = None
    ) -> str:
        """
        Export comparison table for specific architectures and metrics
        
        Args:
            df: DataFrame from mlflow.search_runs()
            architectures: List of run IDs to compare
            metrics: List of metrics to include
            filename: Output filename
            
        Returns:
            Path to saved CSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"architecture_comparison_{timestamp}.csv"
        
        output_path = self.output_dir / filename
        
        # Filter to specified architectures
        df_filtered = df[df['run_id'].isin(architectures)].copy()
        
        # Select columns
        columns = ['run_id'] + [f'metrics.{m}' if not m.startswith('metrics.') else m for m in metrics]
        columns = [col for col in columns if col in df_filtered.columns]
        
        df_export = df_filtered[columns]
        df_export.to_csv(output_path, index=False)
        
        print(f"✓ Exported comparison table to: {output_path}")
        
        return str(output_path)



