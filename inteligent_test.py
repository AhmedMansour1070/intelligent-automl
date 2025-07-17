#!/usr/bin/env python
"""
Clean Intelligent Pipeline Selector with Warning Suppression

This version provides a clean user experience by suppressing all unnecessary warnings
while maintaining full functionality of the intelligent preprocessing selection.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# Suppress all warnings for clean output
warnings.filterwarnings('ignore')

from automl_framework.intelligence.pipeline_selector import (
    IntelligentPipelineSelector, create_intelligent_pipeline
)


def create_clean_intelligent_pipeline(df: pd.DataFrame, target_column: Optional[str] = None) -> 'DataPipeline':
    """
    Create an intelligent pipeline with clean output (no warnings).
    
    Args:
        df: Input DataFrame
        target_column: Name of target column (optional)
        
    Returns:
        Optimized DataPipeline
    """
    # Suppress all warnings during pipeline creation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        selector = IntelligentPipelineSelector(target_column=target_column)
        
        print("🔍 Analyzing data characteristics...")
        characteristics = selector.analyze_data(df)
        
        print("🧠 Generating intelligent preprocessing recommendations...")
        recommendations = selector.generate_recommendations()
        
        print("🔧 Building intelligent pipeline...")
        pipeline = selector.build_intelligent_pipeline()
        
        # Show clean summary without detailed explanations
        print(f"✅ Intelligent pipeline created with {len(pipeline)} steps")
        print(f"📊 Dataset: {characteristics.n_rows:,} rows × {characteristics.n_features} features")
        print(f"🎯 Target type: {characteristics.target_type or 'auto-detected'}")
        
        # Show high-confidence recommendations only
        high_conf_recs = [r for r in recommendations if r.confidence >= 0.8]
        if high_conf_recs:
            print(f"🟢 High-confidence steps: {', '.join(r.step_name for r in high_conf_recs)}")
        
        return pipeline


def analyze_real_dataset(file_path: str, target_column: str):
    """
    Analyze a real dataset and show intelligent preprocessing recommendations.
    
    Args:
        file_path: Path to the dataset file
        target_column: Name of the target column
    """
    print("=" * 80)
    print("🚀 INTELLIGENT AUTOML ANALYSIS")
    print("=" * 80)
    
    try:
        # Load data
        print(f"📁 Loading dataset: {file_path}")
        df = pd.read_csv(file_path)
        print(f"✅ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Create intelligent pipeline
        pipeline = create_clean_intelligent_pipeline(df, target_column=target_column)
        
        # Prepare data
        features = df.drop(target_column, axis=1)
        target = df[target_column]
        
        print("\n⚙️ Processing data through intelligent pipeline...")
        
        # Process data
        start_time = pd.Timestamp.now()
        processed_features = pipeline.fit_transform(features)
        processing_time = (pd.Timestamp.now() - start_time).total_seconds()
        
        # Results
        print(f"✅ Processing complete in {processing_time:.2f} seconds!")
        print(f"📈 Features: {features.shape[1]} → {processed_features.shape[1]}")
        print(f"🎯 Data quality: {processed_features.isnull().sum().sum()} missing values")
        print(f"💾 Memory usage: {processed_features.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Data quality summary
        print("\n📊 DATA QUALITY SUMMARY:")
        print(f"  • Original missing values: {features.isnull().sum().sum():,}")
        print(f"  • Final missing values: {processed_features.isnull().sum().sum()}")
        print(f"  • Feature expansion: {(processed_features.shape[1] / features.shape[1]):.1f}x")
        print(f"  • Processing speed: {len(df) / processing_time:.0f} rows/second")
        
        # Save results
        output_file = f"processed_{file_path.split('/')[-1]}"
        processed_df = processed_features.copy()
        processed_df[target_column] = target
        processed_df.to_csv(output_file, index=False)
        print(f"\n💾 Processed data saved to: {output_file}")
        
        # Performance insights
        print(f"\n🧠 INTELLIGENT INSIGHTS:")
        print(f"  • Your dataset is {get_dataset_complexity(df)}")
        print(f"  • Recommended for: {get_ml_task_recommendation(target)}")
        print(f"  • Processing efficiency: {get_efficiency_rating(processing_time, len(df))}")
        
        return pipeline, processed_features
        
    except Exception as e:
        print(f"❌ Error processing dataset: {str(e)}")
        print("💡 Tip: Check file path and target column name")
        return None, None


def get_dataset_complexity(df: pd.DataFrame) -> str:
    """Determine dataset complexity level."""
    n_rows, n_cols = df.shape
    missing_pct = (df.isnull().sum().sum() / (n_rows * n_cols)) * 100
    
    if n_rows < 1000 and n_cols < 10:
        return "Small and Simple"
    elif n_rows < 10000 and n_cols < 50:
        return "Medium Complexity"
    elif missing_pct > 20:
        return "High Complexity (lots of missing data)"
    elif n_cols > 100:
        return "High Dimensional"
    else:
        return "Large Scale"


def get_ml_task_recommendation(target: pd.Series) -> str:
    """Recommend ML task type based on target."""
    if target.dtype in ['object', 'category']:
        unique_vals = target.nunique()
        if unique_vals == 2:
            return "Binary Classification"
        elif unique_vals <= 20:
            return "Multi-class Classification"
        else:
            return "Text Classification"
    else:
        if target.nunique() <= 20 and target.min() >= 0:
            return "Classification (discrete)"
        else:
            return "Regression"


def get_efficiency_rating(processing_time: float, n_rows: int) -> str:
    """Rate processing efficiency."""
    rate = n_rows / processing_time
    if rate > 50000:
        return "Excellent (>50k rows/sec)"
    elif rate > 10000:
        return "Very Good (>10k rows/sec)"
    elif rate > 1000:
        return "Good (>1k rows/sec)"
    else:
        return "Acceptable"


def quick_analysis(file_path: str, target_column: str):
    """
    Quick analysis function for immediate insights.
    
    Args:
        file_path: Path to CSV file
        target_column: Target column name
    """
    print("🚀 Quick Intelligent Analysis")
    print("-" * 40)
    
    try:
        # Load and analyze
        df = pd.read_csv(file_path)
        pipeline = create_clean_intelligent_pipeline(df, target_column)
        
        # Quick processing test on sample
        sample_size = min(1000, len(df))
        sample_df = df.head(sample_size)
        features = sample_df.drop(target_column, axis=1)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            processed = pipeline.fit_transform(features)
        
        print(f"✅ Ready for production!")
        print(f"📊 Sample processing: {features.shape} → {processed.shape}")
        return pipeline
        
    except Exception as e:
        print(f"❌ Quick analysis failed: {str(e)}")
        return None


# Example usage functions
def example_usage():
    """Show example usage patterns."""
    print("💡 EXAMPLE USAGE PATTERNS:")
    print("-" * 40)
    
    print("\n1. Quick Analysis:")
    print("   pipeline = quick_analysis('test.csv', 'trip_duration')")
    
    print("\n2. Full Analysis:")
    print("   analyze_real_dataset('data.csv', 'target')")
    
    print("\n3. Custom Pipeline:")
    print("   df = pd.read_csv('data.csv')")
    print("   pipeline = create_clean_intelligent_pipeline(df, 'target')")
    print("   result = pipeline.fit_transform(df.drop('target', axis=1))")


if __name__ == "__main__":
    # Show usage examples
    example_usage()
    
    print("\n" + "=" * 80)
    print("Ready to analyze your dataset!")
    print("Use: analyze_real_dataset('your_file.csv', 'your_target_column')")
    print("=" * 80)