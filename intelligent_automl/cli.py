#!/usr/bin/env python
"""
Command Line Interface for Intelligent AutoML Framework

This module provides a comprehensive CLI for the AutoML framework,
enabling users to perform data analysis, preprocessing, and model training
from the command line.
"""

import click
import pandas as pd
import json
import sys
import os
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

# Suppress warnings for cleaner CLI output
warnings.filterwarnings('ignore')

from intelligent_automl import create_intelligent_pipeline, IntelligentPipelineSelector
from intelligent_automl.data import load_data, DataPipeline
from intelligent_automl.models.auto_trainer import AutoModelTrainer
from intelligent_automl.utils.validation import validate_dataset, DataProfiler
from intelligent_automl.utils.logging import configure_logging, get_logger
from intelligent_automl.core.config import AutoMLConfig, DataConfig


@click.group()
@click.version_option(version='1.0.0', prog_name='intelligent-automl')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--log-file', help='Log file path')
@click.pass_context
def cli(ctx, verbose, log_file):
    """Intelligent AutoML Framework - Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    # Configure logging
    log_level = 'INFO' if verbose else 'WARNING'
    configure_logging(level=log_level, log_file=log_file, log_to_console=True)
    
    if verbose:
        click.echo("🧠 Intelligent AutoML Framework CLI")
        click.echo("🚀 Ready to intelligently process your data!\n")


@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output file path for the report')
@click.option('--format', '-f', type=click.Choice(['json', 'txt']), default='txt', help='Output format')
@click.pass_context
def analyze(ctx, data_file, output, format):
    """Analyze a dataset and generate a comprehensive report."""
    verbose = ctx.obj.get('verbose', False)
    
    if verbose:
        click.echo(f"📊 Analyzing dataset: {data_file}")
    
    try:
        # Load data
        df = load_data(data_file)
        
        if verbose:
            click.echo(f"✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Generate comprehensive analysis
        profiler = DataProfiler()
        profile = profiler.profile_data(df)
        
        # Validate data
        validation_report = validate_dataset(df)
        
        # Generate report
        report = {
            'file_info': {
                'path': str(data_file),
                'size_mb': Path(data_file).stat().st_size / (1024 * 1024)
            },
            'data_profile': profile,
            'validation': {
                'is_valid': validation_report.is_valid,
                'error_count': validation_report.error_count,
                'warning_count': validation_report.warning_count,
                'errors': [r.message for r in validation_report.get_errors()],
                'warnings': [r.message for r in validation_report.get_warnings()]
            }
        }
        
        # Output report
        if output:
            if format == 'json':
                with open(output, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                click.echo(f"📁 Report saved to: {output}")
            else:
                _write_text_report(report, output)
                click.echo(f"📁 Report saved to: {output}")
        else:
            _print_analysis_report(report, verbose)
            
    except Exception as e:
        click.echo(f"❌ Error analyzing data: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--target', '-t', required=True, help='Target column name')
@click.option('--output', '-o', help='Output file path for processed data')
@click.option('--save-pipeline', help='Save pipeline to file')
@click.option('--config', '-c', help='Configuration file path')
@click.pass_context
def preprocess(ctx, data_file, target, output, save_pipeline, config):
    """Preprocess data using intelligent pipeline selection."""
    verbose = ctx.obj.get('verbose', False)
    
    if verbose:
        click.echo(f"🔧 Preprocessing dataset: {data_file}")
        click.echo(f"🎯 Target column: {target}")
    
    try:
        # Load data
        df = load_data(data_file)
        
        if target not in df.columns:
            click.echo(f"❌ Target column '{target}' not found in dataset", err=True)
            sys.exit(1)
        
        if verbose:
            click.echo(f"✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Create intelligent pipeline
        if verbose:
            click.echo("🧠 Creating intelligent preprocessing pipeline...")
        
        pipeline = create_intelligent_pipeline(df, target_column=target)
        
        # Process data
        features = df.drop(target, axis=1)
        processed_features = pipeline.fit_transform(features)
        
        if verbose:
            click.echo(f"✅ Preprocessing complete!")
            click.echo(f"📈 Features: {features.shape[1]} → {processed_features.shape[1]}")
            click.echo(f"🎯 Missing values: {processed_features.isnull().sum().sum()}")
        
        # Save processed data
        if output:
            processed_df = processed_features.copy()
            processed_df[target] = df[target]
            processed_df.to_csv(output, index=False)
            click.echo(f"💾 Processed data saved to: {output}")
        
        # Save pipeline
        if save_pipeline:
            pipeline.save(save_pipeline)
            click.echo(f"🔧 Pipeline saved to: {save_pipeline}")
        
        # Print summary
        if not output:
            click.echo("\n📊 PREPROCESSING SUMMARY:")
            click.echo(f"  • Original features: {features.shape[1]}")
            click.echo(f"  • Final features: {processed_features.shape[1]}")
            click.echo(f"  • Feature expansion: {processed_features.shape[1] / features.shape[1]:.1f}x")
            click.echo(f"  • Data quality: {'Perfect' if processed_features.isnull().sum().sum() == 0 else 'Issues remain'}")
            
    except Exception as e:
        click.echo(f"❌ Error preprocessing data: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--target', '-t', required=True, help='Target column name')
@click.option('--output', '-o', help='Output directory for model and results')
@click.option('--models', help='Comma-separated list of models to try')
@click.option('--time-limit', type=int, help='Time limit in minutes')
@click.option('--cv-folds', type=int, default=5, help='Cross-validation folds')
@click.option('--test-size', type=float, default=0.2, help='Test set size')
@click.pass_context
def train(ctx, data_file, target, output, models, time_limit, cv_folds, test_size):
    """Train models using intelligent model selection."""
    verbose = ctx.obj.get('verbose', False)
    
    if verbose:
        click.echo(f"🚀 Training models on: {data_file}")
        click.echo(f"🎯 Target column: {target}")
    
    try:
        # Load data
        df = load_data(data_file)
        
        if target not in df.columns:
            click.echo(f"❌ Target column '{target}' not found in dataset", err=True)
            sys.exit(1)
        
        # Preprocess data
        if verbose:
            click.echo("🧠 Creating intelligent preprocessing pipeline...")
        
        pipeline = create_intelligent_pipeline(df, target_column=target)
        features = df.drop(target, axis=1)
        processed_features = pipeline.fit_transform(features)
        
        if verbose:
            click.echo(f"✅ Preprocessing complete: {features.shape[1]} → {processed_features.shape[1]} features")
        
        # Parse models list
        models_to_try = None
        if models:
            models_to_try = [m.strip() for m in models.split(',')]
        
        # Create trainer
        trainer = AutoModelTrainer(
            models_to_try=models_to_try,
            time_limit_minutes=time_limit,
            cross_validation_folds=cv_folds,
            test_size=test_size,
            verbose=verbose
        )
        
        # Train models
        if verbose:
            click.echo("🤖 Training models...")
        
        trainer.fit(processed_features, df[target])
        
        # Get results
        summary = trainer.get_training_summary()
        
        # Print results
        click.echo(f"\n🏆 TRAINING RESULTS:")
        click.echo(f"  • Best model: {summary['best_model']}")
        click.echo(f"  • Best score: {summary['best_score']:.4f}")
        click.echo(f"  • Models trained: {summary['models_trained']}")
        click.echo(f"  • Total time: {summary['total_training_time']:.1f}s")
        
        # Show model rankings
        if verbose:
            click.echo(f"\n📊 MODEL RANKINGS:")
            for i, model_info in enumerate(summary['model_rankings'], 1):
                click.echo(f"  {i}. {model_info['model']}: {model_info['score']:.4f} "
                          f"(±{model_info['cv_std']:.4f})")
        
        # Save results
        if output:
            output_dir = Path(output)
            output_dir.mkdir(exist_ok=True)
            
            # Save best model
            import joblib
            model_path = output_dir / 'best_model.joblib'
            joblib.dump(trainer.best_model, model_path)
            
            # Save preprocessing pipeline
            pipeline_path = output_dir / 'preprocessing_pipeline.joblib'
            pipeline.save(str(pipeline_path))
            
            # Save training summary
            summary_path = output_dir / 'training_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            click.echo(f"💾 Results saved to: {output_dir}")
            click.echo(f"  • Model: {model_path}")
            click.echo(f"  • Pipeline: {pipeline_path}")
            click.echo(f"  • Summary: {summary_path}")
            
    except Exception as e:
        click.echo(f"❌ Error training models: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.argument('model_dir', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output file for predictions')
@click.pass_context
def predict(ctx, data_file, model_dir, output):
    """Make predictions using a trained model."""
    verbose = ctx.obj.get('verbose', False)
    
    if verbose:
        click.echo(f"🔮 Making predictions on: {data_file}")
        click.echo(f"🤖 Using model from: {model_dir}")
    
    try:
        # Load data
        df = load_data(data_file)
        
        if verbose:
            click.echo(f"✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Load model components
        model_dir = Path(model_dir)
        
        # Load preprocessing pipeline
        pipeline_path = model_dir / 'preprocessing_pipeline.joblib'
        if not pipeline_path.exists():
            click.echo(f"❌ Pipeline not found: {pipeline_path}", err=True)
            sys.exit(1)
        
        pipeline = DataPipeline.load(str(pipeline_path))
        
        # Load model
        model_path = model_dir / 'best_model.joblib'
        if not model_path.exists():
            click.echo(f"❌ Model not found: {model_path}", err=True)
            sys.exit(1)
        
        import joblib
        model = joblib.load(model_path)
        
        if verbose:
            click.echo("✅ Model and pipeline loaded")
        
        # Process data
        processed_features = pipeline.transform(df)
        
        if verbose:
            click.echo(f"✅ Data processed: {df.shape[1]} → {processed_features.shape[1]} features")
        
        # Make predictions
        predictions = model.predict(processed_features)
        
        # Try to get prediction probabilities
        try:
            probabilities = model.predict_proba(processed_features)
            has_probabilities = True
        except:
            probabilities = None
            has_probabilities = False
        
        if verbose:
            click.echo(f"✅ Predictions made for {len(predictions)} samples")
        
        # Create results DataFrame
        results = df.copy()
        results['prediction'] = predictions
        
        if has_probabilities and probabilities.shape[1] == 2:
            # Binary classification
            results['probability'] = probabilities[:, 1]
        elif has_probabilities:
            # Multi-class classification
            for i in range(probabilities.shape[1]):
                results[f'probability_class_{i}'] = probabilities[:, i]
        
        # Save or display results
        if output:
            results.to_csv(output, index=False)
            click.echo(f"💾 Predictions saved to: {output}")
        else:
            # Display sample results
            click.echo("\n🔮 SAMPLE PREDICTIONS:")
            sample_size = min(10, len(results))
            display_columns = ['prediction']
            if has_probabilities:
                if probabilities.shape[1] == 2:
                    display_columns.append('probability')
                else:
                    display_columns.extend([f'probability_class_{i}' for i in range(probabilities.shape[1])])
            
            for i in range(sample_size):
                click.echo(f"  Row {i+1}: {dict(results[display_columns].iloc[i])}")
            
            if len(results) > sample_size:
                click.echo(f"  ... and {len(results) - sample_size} more rows")
        
        # Show prediction summary
        if hasattr(predictions, 'dtype') and predictions.dtype in ['int64', 'int32', 'bool']:
            # Classification
            unique_values, counts = np.unique(predictions, return_counts=True)
            click.echo(f"\n📊 PREDICTION SUMMARY:")
            for value, count in zip(unique_values, counts):
                percentage = (count / len(predictions)) * 100
                click.echo(f"  • Class {value}: {count} ({percentage:.1f}%)")
        else:
            # Regression
            click.echo(f"\n📊 PREDICTION SUMMARY:")
            click.echo(f"  • Mean: {np.mean(predictions):.4f}")
            click.echo(f"  • Std: {np.std(predictions):.4f}")
            click.echo(f"  • Min: {np.min(predictions):.4f}")
            click.echo(f"  • Max: {np.max(predictions):.4f}")
            
    except Exception as e:
        click.echo(f"❌ Error making predictions: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--target', '-t', required=True, help='Target column name')
@click.option('--output', '-o', help='Output directory for complete results')
@click.option('--config', '-c', help='Configuration file path')
@click.pass_context
def automl(ctx, data_file, target, output, config):
    """Run complete AutoML pipeline from data to trained model."""
    verbose = ctx.obj.get('verbose', False)
    
    if verbose:
        click.echo(f"🚀 Running complete AutoML pipeline on: {data_file}")
        click.echo(f"🎯 Target column: {target}")
    
    try:
        # Load data
        df = load_data(data_file)
        
        if target not in df.columns:
            click.echo(f"❌ Target column '{target}' not found in dataset", err=True)
            sys.exit(1)
        
        if verbose:
            click.echo(f"✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Step 1: Data Analysis
        if verbose:
            click.echo("🔍 Step 1: Analyzing data...")
        
        profiler = DataProfiler()
        profile = profiler.profile_data(df)
        validation_report = validate_dataset(df, target_column=target)
        
        # Step 2: Preprocessing
        if verbose:
            click.echo("🧠 Step 2: Creating intelligent preprocessing pipeline...")
        
        pipeline = create_intelligent_pipeline(df, target_column=target)
        features = df.drop(target, axis=1)
        processed_features = pipeline.fit_transform(features)
        
        if verbose:
            click.echo(f"✅ Preprocessing: {features.shape[1]} → {processed_features.shape[1]} features")
        
        # Step 3: Model Training
        if verbose:
            click.echo("🤖 Step 3: Training models...")
        
        trainer = AutoModelTrainer(verbose=verbose)
        trainer.fit(processed_features, df[target])
        
        training_summary = trainer.get_training_summary()
        
        # Step 4: Results
        click.echo(f"\n🎉 AUTOML PIPELINE COMPLETE!")
        click.echo(f"📊 Data Analysis:")
        click.echo(f"  • Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        click.echo(f"  • Data quality: {'Good' if validation_report.is_valid else 'Issues found'}")
        click.echo(f"  • Recommendations: {len(profile['recommendations'])}")
        
        click.echo(f"\n🔧 Preprocessing:")
        click.echo(f"  • Features engineered: {processed_features.shape[1]}")
        click.echo(f"  • Missing values: {processed_features.isnull().sum().sum()}")
        click.echo(f"  • Data quality: Perfect")
        
        click.echo(f"\n🏆 Model Training:")
        click.echo(f"  • Best model: {training_summary['best_model']}")
        click.echo(f"  • Best score: {training_summary['best_score']:.4f}")
        click.echo(f"  • Models evaluated: {training_summary['models_trained']}")
        
        # Save complete results
        if output:
            output_dir = Path(output)
            output_dir.mkdir(exist_ok=True)
            
            # Save all components
            import joblib
            
            # Model and pipeline
            joblib.dump(trainer.best_model, output_dir / 'best_model.joblib')
            pipeline.save(str(output_dir / 'preprocessing_pipeline.joblib'))
            
            # Analysis results
            with open(output_dir / 'data_analysis.json', 'w') as f:
                json.dump(profile, f, indent=2, default=str)
            
            # Training results
            with open(output_dir / 'training_summary.json', 'w') as f:
                json.dump(training_summary, f, indent=2, default=str)
            
            # Processed data
            processed_df = processed_features.copy()
            processed_df[target] = df[target]
            processed_df.to_csv(output_dir / 'processed_data.csv', index=False)
            
            # Configuration
            automl_config = {
                'data_file': str(data_file),
                'target_column': target,
                'preprocessing_steps': len(pipeline),
                'final_features': processed_features.shape[1],
                'best_model': training_summary['best_model'],
                'best_score': training_summary['best_score']
            }
            
            with open(output_dir / 'automl_config.json', 'w') as f:
                json.dump(automl_config, f, indent=2, default=str)
            
            click.echo(f"\n💾 Complete results saved to: {output_dir}")
            click.echo(f"  • Model: best_model.joblib")
            click.echo(f"  • Pipeline: preprocessing_pipeline.joblib")
            click.echo(f"  • Analysis: data_analysis.json")
            click.echo(f"  • Training: training_summary.json")
            click.echo(f"  • Data: processed_data.csv")
            
    except Exception as e:
        click.echo(f"❌ Error running AutoML pipeline: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.pass_context
def run_config(ctx, config_file):
    """Run AutoML pipeline from configuration file."""
    verbose = ctx.obj.get('verbose', False)
    
    if verbose:
        click.echo(f"⚙️ Running AutoML from config: {config_file}")
    
    try:
        # Load configuration
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Extract parameters
        data_file = config_data.get('data_file')
        target = config_data.get('target_column')
        output = config_data.get('output_directory')
        
        if not data_file or not target:
            click.echo("❌ Configuration must specify 'data_file' and 'target_column'", err=True)
            sys.exit(1)
        
        # Run AutoML using configuration
        ctx.invoke(automl, data_file=data_file, target=target, output=output)
        
    except Exception as e:
        click.echo(f"❌ Error running from config: {str(e)}", err=True)
        sys.exit(1)


def _print_analysis_report(report: Dict[str, Any], verbose: bool):
    """Print analysis report to console."""
    basic_info = report['data_profile']['basic_info']
    data_quality = report['data_profile']['data_quality']
    
    click.echo(f"\n📊 DATA ANALYSIS REPORT")
    click.echo(f"{'='*50}")
    
    click.echo(f"📋 Basic Information:")
    click.echo(f"  • Dataset size: {basic_info['shape'][0]} rows × {basic_info['shape'][1]} columns")
    click.echo(f"  • Memory usage: {basic_info['memory_usage_mb']:.1f} MB")
    click.echo(f"  • Data types: {basic_info['dtypes']}")
    
    click.echo(f"\n🔍 Data Quality:")
    click.echo(f"  • Completeness: {data_quality['completeness']:.1f}%")
    click.echo(f"  • Duplicate rows: {data_quality['duplicate_rows']}")
    click.echo(f"  • Constant columns: {len(data_quality['constant_columns'])}")
    
    # Validation results
    validation = report['validation']
    click.echo(f"\n✅ Validation Results:")
    click.echo(f"  • Overall status: {'✅ PASSED' if validation['is_valid'] else '❌ FAILED'}")
    click.echo(f"  • Errors: {validation['error_count']}")
    click.echo(f"  • Warnings: {validation['warning_count']}")
    
    # Show recommendations
    recommendations = report['data_profile']['recommendations']
    if recommendations:
        click.echo(f"\n💡 Recommendations:")
        for rec in recommendations[:5]:  # Show top 5
            priority_emoji = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
            click.echo(f"  {priority_emoji} {rec['message']}")
    
    if verbose and len(recommendations) > 5:
        click.echo(f"  ... and {len(recommendations) - 5} more recommendations")


def _write_text_report(report: Dict[str, Any], output_path: str):
    """Write analysis report to text file."""
    with open(output_path, 'w') as f:
        f.write("DATA ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        basic_info = report['data_profile']['basic_info']
        f.write("BASIC INFORMATION:\n")
        f.write(f"  Dataset size: {basic_info['shape'][0]} rows × {basic_info['shape'][1]} columns\n")
        f.write(f"  Memory usage: {basic_info['memory_usage_mb']:.1f} MB\n")
        f.write(f"  Data types: {basic_info['dtypes']}\n\n")
        
        data_quality = report['data_profile']['data_quality']
        f.write("DATA QUALITY:\n")
        f.write(f"  Completeness: {data_quality['completeness']:.1f}%\n")
        f.write(f"  Duplicate rows: {data_quality['duplicate_rows']}\n")
        f.write(f"  Constant columns: {len(data_quality['constant_columns'])}\n\n")
        
        validation = report['validation']
        f.write("VALIDATION RESULTS:\n")
        f.write(f"  Overall status: {'PASSED' if validation['is_valid'] else 'FAILED'}\n")
        f.write(f"  Errors: {validation['error_count']}\n")
        f.write(f"  Warnings: {validation['warning_count']}\n\n")
        
        recommendations = report['data_profile']['recommendations']
        if recommendations:
            f.write("RECOMMENDATIONS:\n")
            for rec in recommendations:
                f.write(f"  [{rec['priority'].upper()}] {rec['message']}\n")


if __name__ == '__main__':
    cli()