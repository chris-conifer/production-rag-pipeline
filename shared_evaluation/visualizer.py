"""
Visualization and Tradeoff Analysis
Creates plots for comparing RAG architectures
SHARED across all 4 projects
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


class RAGVisualizer:
    """
    Creates visualizations for RAG pipeline experiments
    
    Features:
    - Accuracy vs Latency vs Cost tradeoff plots
    - Pareto frontier analysis
    - Top-K architecture comparison
    - Metric distributions
    - Heatmaps for parameter sensitivity
    
    SHARED across all 4 projects
    """
    
    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        sns.set_palette("husl")
        self.fig_size = (12, 8)
    
    def plot_3d_tradeoff(
        self,
        df: pd.DataFrame,
        accuracy_col: str = "composite_accuracy",
        latency_col: str = "latency_avg_query_latency_ms",
        cost_col: str = "cost_cost_per_query",
        top_k: int = 10,
        output_path: Optional[str] = None
    ):
        """
        Create 3D scatter plot showing accuracy vs latency vs cost tradeoff
        
        Args:
            df: DataFrame with experiment results
            accuracy_col: Column name for accuracy metric
            latency_col: Column name for latency metric
            cost_col: Column name for cost metric
            top_k: Highlight top K architectures
            output_path: Path to save figure
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Prepare data
        if accuracy_col not in df.columns or latency_col not in df.columns or cost_col not in df.columns:
            print(f"⚠️ Missing required columns for 3D tradeoff plot")
            return
        
        # Sort by composite score to identify top K
        if 'metrics.composite_score' in df.columns:
            df_sorted = df.sort_values('metrics.composite_score', ascending=False)
        else:
            df_sorted = df.copy()
        
        top_k_df = df_sorted.head(top_k)
        rest_df = df_sorted.iloc[top_k:]
        
        # Plot all points (gray)
        if not rest_df.empty:
            ax.scatter(
                rest_df[latency_col],
                rest_df[cost_col],
                rest_df[accuracy_col],
                c='lightgray',
                s=50,
                alpha=0.4,
                label='Other Architectures'
            )
        
        # Plot top K (colored)
        scatter = ax.scatter(
            top_k_df[latency_col],
            top_k_df[cost_col],
            top_k_df[accuracy_col],
            c=range(len(top_k_df)),
            s=200,
            cmap='viridis',
            edgecolors='black',
            linewidth=2,
            alpha=0.8,
            label=f'Top {top_k} Architectures'
        )
        
        # Labels
        ax.set_xlabel('Latency (ms)', fontsize=12, labelpad=10)
        ax.set_ylabel('Cost per Query ($)', fontsize=12, labelpad=10)
        ax.set_zlabel('Accuracy Score', fontsize=12, labelpad=10)
        ax.set_title('RAG Architecture Tradeoff Analysis\nAccuracy vs Latency vs Cost', fontsize=14, pad=20)
        
        # Add colorbar for top K ranking
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Ranking (Best to Worst)', fontsize=10)
        
        ax.legend(loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved 3D tradeoff plot: {output_path}")
        
        return fig
    
    def plot_pareto_frontier(
        self,
        df: pd.DataFrame,
        metric1: str = "composite_accuracy",
        metric2: str = "latency_avg_query_latency_ms",
        metric1_label: str = "Accuracy",
        metric2_label: str = "Latency (ms)",
        minimize_metric2: bool = True,
        output_path: Optional[str] = None
    ):
        """
        Plot Pareto frontier for two objectives
        
        Args:
            df: DataFrame with experiment results
            metric1: First metric (to maximize)
            metric2: Second metric (to minimize or maximize)
            metric1_label: Label for metric 1
            metric2_label: Label for metric 2
            minimize_metric2: Whether to minimize metric2 (True for latency/cost)
            output_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        if metric1 not in df.columns or metric2 not in df.columns:
            print(f"⚠️ Missing required columns for Pareto plot")
            return
        
        # Extract data
        x = df[metric2].values
        y = df[metric1].values
        
        # Find Pareto frontier
        pareto_indices = self._find_pareto_frontier(y, x, minimize_metric2)
        
        # Plot all points
        ax.scatter(x, y, c='lightblue', s=100, alpha=0.5, label='All Architectures')
        
        # Plot Pareto frontier
        pareto_x = x[pareto_indices]
        pareto_y = y[pareto_indices]
        
        # Sort for line plot
        sorted_indices = np.argsort(pareto_x)
        ax.plot(
            pareto_x[sorted_indices],
            pareto_y[sorted_indices],
            'r--',
            linewidth=2,
            alpha=0.7
        )
        ax.scatter(
            pareto_x,
            pareto_y,
            c='red',
            s=200,
            edgecolors='black',
            linewidth=2,
            label='Pareto Optimal',
            zorder=5
        )
        
        # Annotate Pareto points
        for i, idx in enumerate(pareto_indices):
            ax.annotate(
                f'{i+1}',
                (x[idx], y[idx]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_xlabel(metric2_label, fontsize=12)
        ax.set_ylabel(metric1_label, fontsize=12)
        ax.set_title(f'Pareto Frontier: {metric1_label} vs {metric2_label}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved Pareto frontier plot: {output_path}")
        
        return fig
    
    def plot_top_architectures_comparison(
        self,
        df: pd.DataFrame,
        top_k: int = 10,
        metrics: List[str] = None,
        output_path: Optional[str] = None
    ):
        """
        Create bar chart comparing top K architectures across multiple metrics
        
        Args:
            df: DataFrame with experiment results
            top_k: Number of top architectures to show
            metrics: List of metrics to compare (if None, auto-select)
            output_path: Path to save figure
        """
        if metrics is None:
            # Auto-select key metrics
            metrics = [
                'composite_score',
                'retrieval_precision_at_5',
                'generation_f1_score',
                'ragas_faithfulness',
                'latency_avg_query_latency_ms',
                'cost_cost_per_query'
            ]
            # Filter to existing columns
            metrics = [m for m in metrics if f'metrics.{m}' in df.columns]
        
        # Sort by composite score
        if 'metrics.composite_score' in df.columns:
            df_sorted = df.sort_values('metrics.composite_score', ascending=False)
        else:
            df_sorted = df.copy()
        
        top_k_df = df_sorted.head(top_k)
        
        # Prepare data for plotting
        plot_data = {}
        for metric in metrics:
            col_name = f'metrics.{metric}' if not metric.startswith('metrics.') else metric
            if col_name in top_k_df.columns:
                # Normalize to 0-1 for visualization
                values = top_k_df[col_name].values
                if 'latency' in metric.lower() or 'cost' in metric.lower():
                    # Lower is better - invert
                    values = 1 / (1 + values)
                plot_data[metric.replace('metrics.', '')] = values
        
        if not plot_data:
            print("⚠️ No valid metrics found for comparison")
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(top_k)
        width = 0.8 / len(plot_data)
        
        for i, (metric_name, values) in enumerate(plot_data.items()):
            offset = (i - len(plot_data) / 2) * width + width / 2
            ax.bar(x + offset, values, width, label=metric_name, alpha=0.8)
        
        ax.set_xlabel('Architecture Rank', fontsize=12)
        ax.set_ylabel('Normalized Score (0-1)', fontsize=12)
        ax.set_title(f'Top {top_k} Architecture Comparison Across Metrics', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f'#{i+1}' for i in range(top_k)])
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved top architectures comparison: {output_path}")
        
        return fig
    
    def plot_metric_distributions(
        self,
        df: pd.DataFrame,
        metrics: List[str] = None,
        output_path: Optional[str] = None
    ):
        """
        Create distribution plots for key metrics
        
        Args:
            df: DataFrame with experiment results
            metrics: List of metrics to plot
            output_path: Path to save figure
        """
        if metrics is None:
            metrics = [
                'composite_score',
                'retrieval_precision_at_5',
                'generation_f1_score',
                'latency_avg_query_latency_ms'
            ]
        
        # Filter to existing columns
        existing_metrics = []
        for m in metrics:
            col_name = f'metrics.{m}' if not m.startswith('metrics.') else m
            if col_name in df.columns:
                existing_metrics.append(col_name)
        
        if not existing_metrics:
            print("⚠️ No valid metrics found for distribution plot")
            return
        
        n_metrics = len(existing_metrics)
        fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(14, 8))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for i, metric in enumerate(existing_metrics):
            ax = axes[i]
            data = df[metric].dropna()
            
            ax.hist(data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.3f}')
            ax.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data.median():.3f}')
            
            ax.set_xlabel(metric.replace('metrics.', ''), fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title(f'Distribution: {metric.replace("metrics.", "")}', fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Metric Distributions Across All Experiments', fontsize=14, y=1.00)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved metric distributions: {output_path}")
        
        return fig
    
    def _find_pareto_frontier(
        self,
        objective1: np.ndarray,
        objective2: np.ndarray,
        minimize_obj2: bool = True
    ) -> np.ndarray:
        """
        Find Pareto frontier for two objectives
        
        Args:
            objective1: First objective (to maximize)
            objective2: Second objective
            minimize_obj2: Whether to minimize objective2
            
        Returns:
            Indices of Pareto optimal points
        """
        # Convert to maximization problem
        if minimize_obj2:
            objective2 = -objective2
        
        is_pareto = np.ones(len(objective1), dtype=bool)
        
        for i in range(len(objective1)):
            for j in range(len(objective1)):
                if i != j:
                    # Check if j dominates i
                    if (objective1[j] >= objective1[i] and objective2[j] >= objective2[i] and
                        (objective1[j] > objective1[i] or objective2[j] > objective2[i])):
                        is_pareto[i] = False
                        break
        
        return np.where(is_pareto)[0]



