"""Visualization module for LLM evaluation results

Provides functions to create charts and dashboards for comparing models
and analyzing evaluation metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np


class EvaluationVisualizer:
    """Create visualizations for LLM evaluation results"""

    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """Initialize visualizer with matplotlib style

        Args:
            style: Matplotlib style to use for static plots
        """
        self.style = style
        plt.style.use(style)
        sns.set_palette("husl")

    def plot_benchmark_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        output_path: Optional[Union[str, Path]] = None,
        interactive: bool = False,
    ) -> Optional[go.Figure]:
        """Create bar chart comparing benchmark scores across models

        Args:
            results: Dict mapping model names to benchmark scores
                    e.g. {"llama3.2": {"mmlu": 0.65, "truthful": 0.58}}
            output_path: Path to save the chart (optional)
            interactive: If True, create interactive plotly chart

        Returns:
            Plotly figure if interactive=True, else None
        """
        # Convert to DataFrame for easier plotting
        df_data = []
        for model, scores in results.items():
            for benchmark, score in scores.items():
                df_data.append({"Model": model, "Benchmark": benchmark, "Score": score})
        df = pd.DataFrame(df_data)

        if interactive:
            fig = px.bar(
                df,
                x="Benchmark",
                y="Score",
                color="Model",
                barmode="group",
                title="Benchmark Comparison Across Models",
                labels={"Score": "Score (0-1)"},
                range_y=[0, 1],
            )
            if output_path:
                fig.write_html(str(output_path))
            return fig
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            df_pivot = df.pivot(index="Benchmark", columns="Model", values="Score")
            df_pivot.plot(kind="bar", ax=ax, width=0.8)
            ax.set_title("Benchmark Comparison Across Models", fontsize=14, fontweight="bold")
            ax.set_ylabel("Score (0-1)", fontsize=12)
            ax.set_xlabel("Benchmark", fontsize=12)
            ax.set_ylim(0, 1)
            ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(axis="y", alpha=0.3)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            if output_path:
                plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
                plt.close()
            else:
                plt.show()
            return None

    def plot_radar_chart(
        self, results: Dict[str, Dict[str, float]], output_path: Optional[Union[str, Path]] = None
    ) -> go.Figure:
        """Create radar chart for multi-metric comparison

        Args:
            results: Dict mapping model names to metric scores
                    e.g. {"llama3.2": {"accuracy": 0.7, "coherence": 0.8}}
            output_path: Path to save the chart (optional)

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        for model_name, metrics in results.items():
            categories = list(metrics.keys())
            values = list(metrics.values())
            # Close the radar chart by repeating first value
            values_closed = values + [values[0]]
            categories_closed = categories + [categories[0]]

            fig.add_trace(
                go.Scatterpolar(
                    r=values_closed, theta=categories_closed, fill="toself", name=model_name
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Multi-Metric Model Comparison",
            font=dict(size=12),
        )

        if output_path:
            fig.write_html(str(output_path))

        return fig

    def plot_performance_trends(
        self,
        time_series: Dict[str, List[tuple]],
        metric_name: str = "Response Time",
        output_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Create line chart showing performance over time

        Args:
            time_series: Dict mapping model names to (timestamp, value) tuples
            metric_name: Name of the metric being plotted
            output_path: Path to save the chart (optional)
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for model_name, data_points in time_series.items():
            timestamps = [point[0] for point in data_points]
            values = [point[1] for point in data_points]
            ax.plot(timestamps, values, marker="o", label=model_name, linewidth=2)

        ax.set_title(f"{metric_name} Trends Over Time", fontsize=14, fontweight="bold")
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_xlabel("Request Number", fontsize=12)
        ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if output_path:
            plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_model_heatmap(
        self, results: Dict[str, Dict[str, float]], output_path: Optional[Union[str, Path]] = None
    ) -> None:
        """Create heatmap showing all metrics for all models

        Args:
            results: Dict mapping model names to metric scores
            output_path: Path to save the chart (optional)
        """
        # Convert to DataFrame
        df = pd.DataFrame(results).T

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            df,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Score (0-1)"},
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title("Model Performance Heatmap", fontsize=14, fontweight="bold")
        ax.set_xlabel("Metrics", fontsize=12)
        ax.set_ylabel("Models", fontsize=12)
        plt.tight_layout()

        if output_path:
            plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_score_distribution(
        self, scores: Dict[str, List[float]], output_path: Optional[Union[str, Path]] = None
    ) -> None:
        """Create box plot showing score distribution per model

        Args:
            scores: Dict mapping model names to lists of scores
            output_path: Path to save the chart (optional)
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Prepare data for box plot
        data = []
        labels = []
        for model_name, score_list in scores.items():
            data.append(score_list)
            labels.append(model_name)

        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        # Color the boxes
        colors = sns.color_palette("husl", len(data))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_title("Score Distribution Across Models", fontsize=14, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12)
        ax.set_xlabel("Model", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if output_path:
            plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def create_dashboard(
        self,
        results: Dict[str, Dict[str, float]],
        output_path: Union[str, Path],
        detailed_results: Optional[Dict] = None,
    ) -> None:
        """Create comprehensive HTML dashboard with multiple visualizations and detailed stats

        Args:
            results: Dict mapping model names to all metrics
            output_path: Path to save the HTML dashboard
            detailed_results: Optional dict with full EvaluationResults objects for extra info
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        # Extract benchmark and performance metrics
        benchmark_metrics = {}
        performance_metrics = {}

        for model, metrics in results.items():
            benchmark_metrics[model] = {
                k: v for k, v in metrics.items() if k in ["mmlu", "truthful_qa", "hellaswag"]
            }
            # Performance metrics: show actual values with proper scaling
            perf = {}
            if "token_efficiency" in metrics:
                # Normalize to a reasonable scale: 400 tok/s = 1.0
                perf["tokens/sec"] = min(metrics["token_efficiency"] / 400.0, 1.0)
            if "avg_response_time" in metrics:
                # Show inverse: faster is better, normalize to 3s = 0, <0.5s = 1.0
                perf["speed"] = max(0, min((3.0 - metrics["avg_response_time"]) / 3.0, 1.0))
            performance_metrics[model] = perf

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Benchmark Scores (Academic)",
                "Performance Metrics",
                "Overall Comparison",
                "Model Rankings",
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "scatterpolar"}, {"type": "bar"}]],
        )

        # Add benchmark comparison with detailed hover info
        for model, scores in benchmark_metrics.items():
            hover_text = []
            for metric, value in scores.items():
                # Add extra info if available
                extra_info = ""
                if detailed_results and model in detailed_results:
                    res = detailed_results[model]
                    if hasattr(res, "detailed_metrics") and hasattr(
                        res.detailed_metrics, "benchmarks"
                    ):
                        bench_data = res.detailed_metrics.benchmarks.get(metric, {})
                        if isinstance(bench_data, dict):
                            questions = bench_data.get(
                                "questions_tested", bench_data.get("scenarios_tested", "")
                            )
                            if questions:
                                extra_info = f"<br>Questions: {questions}"

                hover_text.append(f"{metric}: {value:.1%}{extra_info}")

            fig.add_trace(
                go.Bar(
                    name=model,
                    x=list(scores.keys()),
                    y=list(scores.values()),
                    hovertext=hover_text,
                    hoverinfo="text",
                ),
                row=1,
                col=1,
            )

        # Add performance metrics with detailed hover info
        for model, perf in performance_metrics.items():
            hover_text = []
            display_values = []
            display_labels = []

            if detailed_results and model in detailed_results:
                res = detailed_results[model]
                if "tokens/sec" in perf:
                    hover_text.append(f"Token Efficiency<br>{res.token_efficiency:.1f} tokens/sec")
                    display_values.append(perf["tokens/sec"])
                    display_labels.append("tokens/sec")
                if "speed" in perf:
                    hover_text.append(f"Response Speed<br>{res.avg_response_time:.2f} seconds")
                    display_values.append(perf["speed"])
                    display_labels.append("speed")

            fig.add_trace(
                go.Bar(
                    name=model,
                    x=display_labels,
                    y=display_values,
                    hovertext=hover_text if hover_text else None,
                    hoverinfo="text" if hover_text else "y",
                ),
                row=1,
                col=2,
            )

        # Add radar chart with ONLY real academic benchmarks (not dummy metrics)
        # These are the only truly validated metrics from academic datasets
        radar_metrics = ["mmlu", "truthful_qa", "hellaswag"]
        for model, metrics in results.items():
            # Filter only real benchmarks
            filtered_categories = [k for k in radar_metrics if k in metrics]
            filtered_values = [metrics[k] for k in filtered_categories]

            # Better labels for display
            display_labels = {
                "mmlu": "MMLU (Knowledge)",
                "truthful_qa": "TruthfulQA (Factuality)",
                "hellaswag": "HellaSwag (Reasoning)",
            }
            pretty_labels = [display_labels.get(k, k) for k in filtered_categories]

            fig.add_trace(
                go.Scatterpolar(r=filtered_values, theta=pretty_labels, name=model, fill="toself"),
                row=2,
                col=1,
            )

        # Add overall rankings with detailed stats
        overall_scores = {}
        hover_texts = []

        # Use ONLY real academic benchmarks for overall score (not dummy metrics)
        score_metrics = ["mmlu", "truthful_qa", "hellaswag"]

        for model, metrics in results.items():
            # Calculate overall score only from real benchmarks
            valid_scores = [metrics[k] for k in score_metrics if k in metrics]
            overall_scores[model] = np.mean(valid_scores) if valid_scores else 0.0

            # Build detailed hover text
            hover_info = f"<b>{model}</b><br>"
            hover_info += f"Avg Benchmark Score: {overall_scores[model]:.1%}<br>"
            hover_info += "<br><b>Academic Benchmarks:</b><br>"
            if "mmlu" in metrics:
                hover_info += f"MMLU (Knowledge): {metrics['mmlu']:.1%}<br>"
            if "truthful_qa" in metrics:
                hover_info += f"TruthfulQA (Factuality): {metrics['truthful_qa']:.1%}<br>"
            if "hellaswag" in metrics:
                hover_info += f"HellaSwag (Reasoning): {metrics['hellaswag']:.1%}<br>"

            if detailed_results and model in detailed_results:
                res = detailed_results[model]
                hover_info += "<br><b>Performance:</b><br>"
                hover_info += f"Token Efficiency: {res.token_efficiency:.1f} tok/s<br>"
                hover_info += f"Avg Response Time: {res.avg_response_time:.2f}s<br>"

                # Add system info if available
                if hasattr(res, "system_info") and res.system_info:
                    sys_info = res.system_info
                    if "gpu_info" in sys_info and sys_info["gpu_info"]:
                        hover_info += "<br><b>Hardware:</b><br>"
                        hover_info += f"GPU: {sys_info['gpu_info']}<br>"
                        if "gpu_vram_gb" in sys_info and sys_info["gpu_vram_gb"]:
                            hover_info += f"VRAM: {sys_info['gpu_vram_gb']:.1f} GB<br>"

            hover_texts.append(hover_info)

        sorted_models = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)

        fig.add_trace(
            go.Bar(
                x=[m[0] for m in sorted_models],
                y=[m[1] for m in sorted_models],
                marker_color="lightblue",
                hovertext=hover_texts,
                hoverinfo="text",
            ),
            row=2,
            col=2,
        )

        # Update layout with better styling and annotations
        fig.update_layout(
            height=1200,
            title_text="LLM Evaluation Dashboard - Detailed Comparison",
            showlegend=True,
            font=dict(size=11),
            hovermode="closest",
            annotations=[
                # Benchmark Scores explanation
                dict(
                    text="<b>Academic Benchmarks</b> (validated datasets)<br>"
                    "• MMLU: 14K questions across 57 subjects (knowledge)<br>"
                    "• TruthfulQA: 817 questions testing factual accuracy<br>"
                    "• HellaSwag: 10K scenarios testing common sense",
                    xref="paper",
                    yref="paper",
                    x=0.22,
                    y=0.97,
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",
                    font=dict(size=9, color="#666"),
                    align="center",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#ccc",
                    borderwidth=1,
                    borderpad=4,
                ),
                # Performance Metrics explanation
                dict(
                    text="<b>Performance Metrics</b> (normalized 0-1)<br>"
                    "• tokens/sec: Speed (400 tok/s = 100%)<br>"
                    "• speed: Responsiveness (3s = 0%, <0.5s = 100%)<br>"
                    "Hover for actual values",
                    xref="paper",
                    yref="paper",
                    x=0.78,
                    y=0.97,
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",
                    font=dict(size=9, color="#666"),
                    align="center",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#ccc",
                    borderwidth=1,
                    borderpad=4,
                ),
                # Overall Comparison explanation
                dict(
                    text="<b>Radar Chart</b>: 3 academic benchmarks<br>"
                    "Larger area = Better overall performance<br>"
                    "Models near center = Weak on all benchmarks",
                    xref="paper",
                    yref="paper",
                    x=0.22,
                    y=0.43,
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",
                    font=dict(size=9, color="#666"),
                    align="center",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#ccc",
                    borderwidth=1,
                    borderpad=4,
                ),
                # Model Rankings explanation
                dict(
                    text="<b>Overall Score</b>: Average of 3 benchmarks<br>"
                    "Does NOT include performance metrics<br>"
                    "Hover for detailed breakdown + hardware info",
                    xref="paper",
                    yref="paper",
                    x=0.78,
                    y=0.43,
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",
                    font=dict(size=9, color="#666"),
                    align="center",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#ccc",
                    borderwidth=1,
                    borderpad=4,
                ),
            ],
        )

        # Update axes
        fig.update_yaxes(title_text="Score (0-1)", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="Score (0-1)", range=[0, 1], row=1, col=2)
        fig.update_yaxes(title_text="Avg Benchmark Score", range=[0, 1], row=2, col=2)

        fig.write_html(str(output_path))
        print(f"Dashboard saved to: {output_path}")


def quick_comparison(
    results: Dict[str, Dict[str, float]],
    output_dir: Union[str, Path] = "outputs",
    detailed_results: Optional[Dict] = None,
) -> None:
    """Generate all standard visualizations for model comparison

    Args:
        results: Dict mapping model names to metric scores
        output_dir: Directory to save all charts
        detailed_results: Optional dict with full EvaluationResults objects for extra info
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    viz = EvaluationVisualizer()

    print("Generating benchmark comparison...")
    viz.plot_benchmark_comparison(
        {
            model: {
                k: v for k, v in metrics.items() if "mmlu" in k or "truthful" in k or "hella" in k
            }
            for model, metrics in results.items()
        },
        output_path=output_dir / "benchmarks.png",
    )

    print("Generating radar chart...")
    viz.plot_radar_chart(results, output_path=output_dir / "radar.html")

    print("Generating heatmap...")
    viz.plot_model_heatmap(results, output_path=output_dir / "heatmap.png")

    print("Generating dashboard...")
    viz.create_dashboard(
        results, output_path=output_dir / "dashboard.html", detailed_results=detailed_results
    )

    print(f"\n✅ All visualizations saved to: {output_dir}")
