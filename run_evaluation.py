"""
Script untuk menjalankan evaluasi RAG dan visualisasi hasil.

Menghasilkan:
- Metrik numerik (Precision, Recall, MRR, NDCG, dll)
- Grafik visualisasi (bar chart, radar chart, heatmap)
- Report dalam format text dan HTML
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from rag_evaluator import RAGEvaluator, EvaluationResult, aggregate_results
from embedding_service import GeminiEmbeddingService, GroqGenerationService
from chromadb_service import ChromaDBService
from config import Config


def load_test_dataset(json_path: str) -> List[Dict]:
    """Load test dataset from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_visualizations(results: List[EvaluationResult], aggregated: Dict, output_dir: str = "evaluation_results"):
    """Create visualization charts for evaluation results."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Bar chart for all metrics
    fig, ax = plt.subplots(figsize=(14, 8))
    
    metrics = {
        'Precision': aggregated['avg_precision'],
        'Recall': aggregated['avg_recall'],
        'MRR': aggregated['avg_mrr'],
        'NDCG': aggregated['avg_ndcg'],
        'Semantic\nSimilarity': aggregated['avg_semantic_similarity'],
        'Faithfulness': aggregated['avg_faithfulness'],
        'Answer\nRelevancy': aggregated['avg_answer_relevancy']
    }
    
    bars = ax.bar(metrics.keys(), metrics.values(), color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22'])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('RAG System Evaluation Metrics', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_bar_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Radar chart for key metrics
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    categories = ['Precision', 'Recall', 'NDCG', 'Semantic\nSimilarity', 'Faithfulness', 'Answer\nRelevancy']
    values = [
        aggregated['avg_precision'],
        aggregated['avg_recall'],
        aggregated['avg_ndcg'],
        aggregated['avg_semantic_similarity'],
        aggregated['avg_faithfulness'],
        aggregated['avg_answer_relevancy']
    ]
    
    # Close the plot
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, color='#3498db', label='RAG System')
    ax.fill(angles, values, alpha=0.25, color='#3498db')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('RAG System Performance Radar', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_radar_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Time performance chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time_metrics = {
        'Retrieval\nTime': aggregated['avg_retrieval_time'],
        'Generation\nTime': aggregated['avg_generation_time'],
        'Total\nTime': aggregated['avg_total_time']
    }
    
    bars = ax.bar(time_metrics.keys(), time_metrics.values(), color=['#3498db', '#e74c3c', '#2ecc71'])
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('RAG System Performance Time', fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Per-query heatmap
    if len(results) > 1:
        fig, ax = plt.subplots(figsize=(12, len(results) * 0.5 + 2))
        
        metric_names = ['Precision', 'Recall', 'MRR', 'NDCG', 'Sem.Sim', 'Faithful', 'Relevancy']
        data = []
        
        for r in results:
            data.append([
                r.precision,
                r.recall,
                r.mrr,
                r.ndcg,
                r.semantic_similarity,
                r.faithfulness_score,
                r.answer_relevancy
            ])
        
        sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', 
                    xticklabels=metric_names,
                    yticklabels=[f'Query {i+1}' for i in range(len(results))],
                    cbar_kws={'label': 'Score'},
                    vmin=0, vmax=1,
                    ax=ax)
        
        ax.set_title('Per-Query Metrics Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/per_query_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"[SUCCESS] Visualizations saved to {output_dir}/")


def generate_report(results: List[EvaluationResult], aggregated: Dict, output_dir: str = "evaluation_results"):
    """Generate text and HTML report."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Text report
    report = []
    report.append("="*70)
    report.append("RAG SYSTEM EVALUATION REPORT")
    report.append("="*70)
    report.append(f"\nTotal Queries Evaluated: {aggregated['num_queries']}\n")
    
    report.append("-"*70)
    report.append("RETRIEVAL METRICS")
    report.append("-"*70)
    report.append(f"Average Precision:    {aggregated['avg_precision']:.4f} ± {aggregated['std_precision']:.4f}")
    report.append(f"Average Recall:       {aggregated['avg_recall']:.4f} ± {aggregated['std_recall']:.4f}")
    report.append(f"Average MRR:          {aggregated['avg_mrr']:.4f}")
    report.append(f"Average NDCG:         {aggregated['avg_ndcg']:.4f}")
    
    report.append("\n" + "-"*70)
    report.append("GENERATION METRICS")
    report.append("-"*70)
    report.append(f"Avg Semantic Similarity: {aggregated['avg_semantic_similarity']:.4f} ± {aggregated['std_semantic_similarity']:.4f}")
    report.append(f"Average Answer Length:   {aggregated['avg_answer_length']:.1f} words")
    
    report.append("\n" + "-"*70)
    report.append("END-TO-END METRICS")
    report.append("-"*70)
    report.append(f"Average Faithfulness:    {aggregated['avg_faithfulness']:.4f}")
    report.append(f"Average Answer Relevancy: {aggregated['avg_answer_relevancy']:.4f}")
    
    report.append("\n" + "-"*70)
    report.append("PERFORMANCE METRICS")
    report.append("-"*70)
    report.append(f"Avg Retrieval Time:   {aggregated['avg_retrieval_time']:.4f}s")
    report.append(f"Avg Generation Time:  {aggregated['avg_generation_time']:.4f}s")
    report.append(f"Avg Total Time:       {aggregated['avg_total_time']:.4f}s")
    
    report.append("\n" + "="*70)
    
    report_text = "\n".join(report)
    
    # Save text report
    with open(f"{output_dir}/evaluation_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # Save JSON report
    with open(f"{output_dir}/evaluation_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, indent=2)
    
    print(report_text)
    print(f"\n[SUCCESS] Reports saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system with metrics and visualizations")
    parser.add_argument("test_dataset", type=str, help="Path to test dataset JSON file")
    parser.add_argument("--collection", type=str, default=None, help="ChromaDB collection name")
    parser.add_argument("--cloud", action="store_true", help="Use ChromaDB Cloud")
    parser.add_argument("--local", action="store_true", help="Use local ChromaDB")
    parser.add_argument("--output", type=str, default="evaluation_results", help="Output directory for results")
    parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--delay", type=int, default=3, help="Delay in seconds between queries (to avoid rate limits)")
    
    args = parser.parse_args()
    
    # Validate config
    try:
        Config.validate()
    except ValueError as e:
        print(f"[ERROR] Configuration error: {e}")
        return
    
    # Determine use_cloud setting
    use_cloud = None
    if args.cloud:
        use_cloud = True
    elif args.local:
        use_cloud = False
    
    print("[INFO] Initializing services...")
    
    # Initialize services
    chroma_service = ChromaDBService(
        collection_name=args.collection,
        use_cloud=use_cloud
    )
    chroma_service.get_or_create_collection()
    
    embedding_service = GeminiEmbeddingService()
    generation_service = GroqGenerationService()
    
    evaluator = RAGEvaluator(
        chroma_service=chroma_service,
        embedding_service=embedding_service,
        generation_service=generation_service
    )
    
    # Load test dataset
    print(f"[INFO] Loading test dataset from {args.test_dataset}...")
    test_data = load_test_dataset(args.test_dataset)
    print(f"[INFO] Loaded {len(test_data)} test queries")
    
    # Run evaluation
    print("\n[INFO] Running evaluation...")
    print("[INFO] Adding delay between queries to avoid rate limits...")
    results = []
    
    for i, test_case in enumerate(test_data, 1):
        print(f"\n[INFO] Evaluating query {i}/{len(test_data)}: {test_case['query'][:50]}...")
        
        try:
            result = evaluator.evaluate_query(
                query=test_case['query'],
                ground_truth_answer=test_case['ground_truth'],
                relevant_doc_ids=test_case['relevant_docs'],
                top_k=args.top_k
            )
            
            results.append(result)
            print(f"  Precision: {result.precision:.3f}, Recall: {result.recall:.3f}, "
                  f"Semantic Sim: {result.semantic_similarity:.3f}")
            
            # Add delay between queries to avoid rate limits (except for last query)
            if i < len(test_data):
                import time
                print(f"  Waiting {args.delay} seconds before next query...")
                time.sleep(args.delay)
                
        except Exception as e:
            print(f"  [ERROR] Failed to evaluate query: {e}")
            print(f"  Skipping this query and continuing...")
            continue
    
    # Aggregate results
    print("\n[INFO] Aggregating results...")
    aggregated = aggregate_results(results)
    
    # Generate visualizations
    print("\n[INFO] Creating visualizations...")
    create_visualizations(results, aggregated, args.output)
    
    # Generate report
    print("\n[INFO] Generating report...")
    generate_report(results, aggregated, args.output)
    
    print("\n[SUCCESS] Evaluation complete!")
    print(f"[INFO] Results saved to: {args.output}/")


if __name__ == "__main__":
    main()
