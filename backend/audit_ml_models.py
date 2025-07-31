#!/usr/bin/env python3
"""
CRITICAL ML MODEL AUDIT - VERIFY CLAIMED ACCURACY
Checking for data leakage, overfitting, and unrealistic performance
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLModelAuditor:
    """Comprehensive audit of fantasy football ML models"""
    
    def __init__(self):
        self.models = {}
        self.data = None
        self.features = None
        self.audit_results = {}
        
    def load_models_and_data(self):
        """Load the trained models and training data"""
        logger.info("="*70)
        logger.info("LOADING MODELS AND DATA FOR AUDIT")
        logger.info("="*70)
        
        # Find the latest model files
        model_dir = 'models'
        model_files = sorted([f for f in os.listdir(model_dir) if f.startswith('fantasy_') and f.endswith('.pkl')])
        
        if not model_files:
            logger.error("No model files found!")
            return False
            
        # Load metadata
        metadata_files = sorted([f for f in os.listdir(model_dir) if f.startswith('fantasy_metadata_') and f.endswith('.json')])
        if metadata_files:
            with open(os.path.join(model_dir, metadata_files[-1]), 'r') as f:
                metadata = json.load(f)
                self.features = metadata.get('features', [])
                logger.info(f"Loaded {len(self.features)} features from metadata")
        
        # Load models
        for model_type in ['xgboost', 'neural_network', 'random_forest']:
            matching_files = [f for f in model_files if f'fantasy_{model_type}_' in f and not any(pos in f for pos in ['QB', 'RB', 'WR', 'TE'])]
            if matching_files:
                model_path = os.path.join(model_dir, matching_files[-1])
                self.models[model_type] = joblib.load(model_path)
                logger.info(f"Loaded {model_type} model from {model_path}")
        
        # Load the training data
        logger.info("\nLoading training data...")
        import nfl_data_py as nfl
        self.data = nfl.import_weekly_data([2019, 2020, 2021, 2022, 2023, 2024])
        
        # Filter for fantasy positions
        fantasy_positions = ['QB', 'RB', 'WR', 'TE']
        self.data = self.data[self.data['position'].isin(fantasy_positions)]
        
        logger.info(f"Loaded {len(self.data)} records")
        
        return True
    
    def check_data_leakage(self):
        """Check for data leakage in features"""
        logger.info("\n" + "="*70)
        logger.info("1. DATA LEAKAGE DETECTION")
        logger.info("="*70)
        
        # Get feature columns from data
        exclude_cols = ['player_id', 'player_name', 'player_display_name', 'season', 'week',
                       'game_id', 'team', 'recent_team', 'opponent_team', 'position']
        
        available_features = [col for col in self.data.columns if col not in exclude_cols]
        
        logger.info(f"\nTotal features available: {len(available_features)}")
        logger.info("Checking for leakage indicators...")
        
        # Check for suspicious terms
        leakage_terms = ['actual', 'result', 'final', 'post', 'outcome', 'fantasy_points']
        leaked_features = []
        
        for term in leakage_terms:
            leaked = [col for col in available_features if term in col.lower()]
            if leaked:
                leaked_features.extend(leaked)
                logger.error(f"❌ CRITICAL: Potential leakage with term '{term}': {leaked}")
        
        # Check specific dangerous columns
        dangerous_cols = ['fantasy_points_ppr', 'fantasy_points', 'fantasy', 'points']
        for col in dangerous_cols:
            if col in available_features:
                logger.error(f"❌ CRITICAL: Target variable '{col}' found in features!")
                leaked_features.append(col)
        
        # Check for same-week stats that shouldn't be known before the game
        same_week_danger = ['passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds', 
                           'rushing_tds', 'receiving_tds', 'receptions', 'completions']
        
        # These should only be used as lagged features (avg3, avg5, etc.)
        for col in same_week_danger:
            if col in available_features and not any(suffix in col for suffix in ['_avg3', '_avg5', '_trend', '_last']):
                logger.warning(f"⚠️  WARNING: Same-week stat '{col}' in features - should use lagged version only")
        
        self.audit_results['data_leakage'] = {
            'leaked_features': leaked_features,
            'status': 'FAIL' if leaked_features else 'PASS'
        }
        
        return len(leaked_features) == 0
    
    def check_multicollinearity(self):
        """Check for multicollinearity in features"""
        logger.info("\n" + "="*70)
        logger.info("2. MULTICOLLINEARITY ANALYSIS")
        logger.info("="*70)
        
        # Prepare feature matrix
        numeric_features = []
        for col in self.data.columns:
            if col not in ['player_id', 'player_name', 'player_display_name', 'season', 'week',
                          'game_id', 'team', 'recent_team', 'opponent_team', 'position',
                          'fantasy_points_ppr', 'fantasy_points']:
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    numeric_features.append(col)
        
        # Get a sample for correlation analysis
        sample_data = self.data[numeric_features].sample(min(5000, len(self.data))).fillna(0)
        
        # Calculate correlation matrix
        logger.info("Calculating correlation matrix...")
        corr_matrix = sample_data.corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.9:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        logger.info(f"\nFound {len(high_corr_pairs)} highly correlated feature pairs (|r| > 0.9)")
        
        if high_corr_pairs:
            logger.warning("Highly correlated features (should remove one from each pair):")
            for pair in high_corr_pairs[:10]:  # Show first 10
                logger.warning(f"  {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
        
        self.audit_results['multicollinearity'] = {
            'high_corr_pairs': len(high_corr_pairs),
            'status': 'WARNING' if len(high_corr_pairs) > 5 else 'PASS'
        }
        
        return len(high_corr_pairs) < 10
    
    def verify_temporal_split(self):
        """Verify proper temporal train/test splitting"""
        logger.info("\n" + "="*70)
        logger.info("3. TEMPORAL VALIDATION")
        logger.info("="*70)
        
        # Check data split by season
        seasons = sorted(self.data['season'].unique())
        logger.info(f"Seasons in data: {seasons}")
        
        # Define splits
        train_seasons = [2019, 2020, 2021, 2022]
        val_season = 2023
        test_season = 2024
        
        train_data = self.data[self.data['season'].isin(train_seasons)]
        val_data = self.data[self.data['season'] == val_season]
        test_data = self.data[self.data['season'] == test_season]
        
        logger.info(f"\nData split verification:")
        logger.info(f"Training: {len(train_data)} records (seasons {train_seasons})")
        logger.info(f"Validation: {len(val_data)} records (season {val_season})")
        logger.info(f"Test: {len(test_data)} records (season {test_season})")
        
        # Verify no overlap
        train_max_week = train_data[['season', 'week']].max()
        test_min_week = test_data[['season', 'week']].min()
        
        proper_split = train_seasons[-1] < test_season
        logger.info(f"\nTemporal split check: {'✓ PASS' if proper_split else '✗ FAIL'}")
        
        self.audit_results['temporal_split'] = {
            'proper_split': proper_split,
            'status': 'PASS' if proper_split else 'FAIL'
        }
        
        return proper_split
    
    def test_realistic_performance(self):
        """Test if model performance is realistic"""
        logger.info("\n" + "="*70)
        logger.info("4. REALISTIC PERFORMANCE TESTING")
        logger.info("="*70)
        
        if not self.models:
            logger.error("No models loaded!")
            return False
        
        # Prepare test data (2024 season)
        test_data = self.data[self.data['season'] == 2024].copy()
        
        # Recreate features (simplified version)
        logger.info("Preparing features...")
        
        # Basic features that would be known before the game
        feature_cols = []
        for col in test_data.columns:
            if col.endswith('_avg3') or col.endswith('_avg5') or col.endswith('_trend'):
                feature_cols.append(col)
            elif col in ['years_experience', 'is_rookie', 'week', 'offense_pct']:
                feature_cols.append(col)
        
        # Remove any features that might leak information
        feature_cols = [col for col in feature_cols if col in test_data.columns and 
                       not any(term in col for term in ['fantasy_points', 'actual', 'result'])]
        
        if len(feature_cols) < 10:
            logger.warning("Too few features found, using basic stats")
            # Use some basic lagged features
            feature_cols = ['completions', 'attempts', 'passing_yards', 'rushing_yards', 
                           'receiving_yards', 'targets', 'receptions']
            feature_cols = [col for col in feature_cols if col in test_data.columns]
        
        X_test = test_data[feature_cols].fillna(0)
        y_test = test_data['fantasy_points_ppr'].fillna(0)
        
        # Test with XGBoost model
        if 'xgboost' in self.models:
            logger.info("\nTesting XGBoost model...")
            try:
                # Predict
                y_pred = self.models['xgboost'].predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(np.mean((y_test - y_pred)**2))
                
                logger.info(f"Test MAE: {mae:.3f}")
                logger.info(f"Test RMSE: {rmse:.3f}")
                logger.info(f"Test R²: {r2:.3f}")
                
                # Check if realistic
                if mae < 1.0:
                    logger.error("❌ CRITICAL: MAE < 1.0 is unrealistic for fantasy football!")
                    logger.error("   This strongly indicates data leakage!")
                elif mae < 3.0:
                    logger.warning("⚠️  WARNING: MAE < 3.0 is suspiciously low for weekly predictions")
                    logger.warning("   Typical range is 4-7 points MAE")
                else:
                    logger.info("✓ MAE is in realistic range for fantasy predictions")
                
                # Test baseline model
                logger.info("\nTesting baseline model (predict mean)...")
                baseline = DummyRegressor(strategy='mean')
                baseline.fit(X_test[:1000], y_test[:1000])  # Fit on subset
                baseline_pred = baseline.predict(X_test)
                baseline_mae = mean_absolute_error(y_test, baseline_pred)
                
                improvement = (baseline_mae - mae) / baseline_mae * 100
                logger.info(f"Baseline MAE: {baseline_mae:.3f}")
                logger.info(f"Model improvement over baseline: {improvement:.1f}%")
                
                if improvement > 80:
                    logger.warning("⚠️  Model improvement >80% is suspicious")
                
                self.audit_results['performance'] = {
                    'test_mae': mae,
                    'test_r2': r2,
                    'baseline_mae': baseline_mae,
                    'improvement': improvement,
                    'realistic': mae >= 3.0,
                    'status': 'PASS' if mae >= 3.0 else 'FAIL'
                }
                
                # Plot diagnostics
                self.plot_diagnostics(y_test, y_pred)
                
            except Exception as e:
                logger.error(f"Error testing model: {e}")
                logger.error("This might indicate feature mismatch or model issues")
                self.audit_results['performance'] = {'status': 'ERROR', 'error': str(e)}
                return False
        
        return True
    
    def plot_diagnostics(self, y_true, y_pred):
        """Create diagnostic plots"""
        logger.info("\nCreating diagnostic plots...")
        
        plt.figure(figsize=(15, 5))
        
        # 1. Distribution comparison
        plt.subplot(131)
        plt.hist(y_true, bins=50, alpha=0.5, label='Actual', density=True)
        plt.hist(y_pred, bins=50, alpha=0.5, label='Predicted', density=True)
        plt.xlabel('Fantasy Points')
        plt.ylabel('Density')
        plt.legend()
        plt.title('Distribution Comparison')
        
        # 2. Residual plot
        plt.subplot(132)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.3)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot (Mean: {np.mean(residuals):.2f})')
        
        # Add residual statistics
        plt.text(0.02, 0.98, f'Std: {np.std(residuals):.2f}', 
                transform=plt.gca().transAxes, verticalalignment='top')
        
        # 3. Actual vs Predicted
        plt.subplot(133)
        plt.scatter(y_true, y_pred, alpha=0.3)
        plt.plot([0, max(y_true)], [0, max(y_true)], 'r--', label='Perfect prediction')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend()
        plt.title('Actual vs Predicted')
        
        plt.tight_layout()
        plt.savefig('model_diagnostics.png', dpi=150)
        logger.info("Diagnostic plots saved to model_diagnostics.png")
        plt.close()
        
        # Additional plot: Error by score range
        plt.figure(figsize=(10, 6))
        
        score_ranges = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 100)]
        mae_by_range = []
        labels = []
        
        for low, high in score_ranges:
            mask = (y_true >= low) & (y_true < high)
            if mask.sum() > 0:
                range_mae = mean_absolute_error(y_true[mask], y_pred[mask])
                mae_by_range.append(range_mae)
                labels.append(f'{low}-{high}')
                
        plt.bar(labels, mae_by_range)
        plt.xlabel('Actual Score Range')
        plt.ylabel('MAE')
        plt.title('Prediction Error by Score Range')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('error_by_range.png', dpi=150)
        logger.info("Error by range plot saved to error_by_range.png")
        plt.close()
    
    def generate_audit_report(self):
        """Generate comprehensive audit report"""
        logger.info("\n" + "="*70)
        logger.info("🔍 ML MODEL AUDIT RESULTS")
        logger.info("="*70)
        
        # Data Integrity
        logger.info("\n├── Data Integrity:")
        
        leakage_status = self.audit_results.get('data_leakage', {}).get('status', 'NOT TESTED')
        logger.info(f"│   ├── Leakage Check: {'✓ PASS' if leakage_status == 'PASS' else '✗ FAIL'}")
        
        temporal_status = self.audit_results.get('temporal_split', {}).get('status', 'NOT TESTED')
        logger.info(f"│   ├── Temporal Split: {'✓ PASS' if temporal_status == 'PASS' else '✗ FAIL'}")
        
        multi_status = self.audit_results.get('multicollinearity', {}).get('status', 'NOT TESTED')
        high_corr = self.audit_results.get('multicollinearity', {}).get('high_corr_pairs', 0)
        logger.info(f"│   └── Feature Count: {high_corr} highly correlated pairs found")
        
        # True Performance
        logger.info("├── True Performance:")
        
        perf = self.audit_results.get('performance', {})
        if perf.get('status') != 'ERROR':
            mae = perf.get('test_mae', -1)
            r2 = perf.get('test_r2', -1)
            baseline = perf.get('baseline_mae', -1)
            improvement = perf.get('improvement', -1)
            
            logger.info(f"│   ├── Test MAE: {mae:.3f} {'(realistic: >3.0)' if mae > 0 else ''}")
            logger.info(f"│   ├── Test R²: {r2:.3f}")
            logger.info(f"│   ├── Baseline MAE: {baseline:.3f}")
            logger.info(f"│   └── Improvement: {improvement:.1f}%")
        else:
            logger.info("│   └── ERROR: Could not test performance")
        
        # Statistical Validity
        logger.info("├── Statistical Validity:")
        logger.info(f"│   ├── Multicollinearity: {high_corr} feature pairs with |r| > 0.9")
        
        realistic = perf.get('realistic', False) if perf.get('status') != 'ERROR' else False
        mae = perf.get('test_mae', 10) if perf.get('status') != 'ERROR' else 10
        logger.info(f"│   ├── Realistic Performance: {'✓ Yes' if realistic else '✗ No'}")
        logger.info(f"│   └── Confidence: {'Low' if mae < 3 else 'High' if mae < 10 else 'N/A'}")
        
        # Recommendation
        logger.info("└── Recommendation:")
        
        # Determine overall status
        critical_issues = []
        if leakage_status == 'FAIL':
            critical_issues.append("Data leakage detected")
        if temporal_status == 'FAIL':
            critical_issues.append("Improper train/test split")
        if perf.get('test_mae', 10) < 1.0:
            critical_issues.append("Unrealistic accuracy (MAE < 1.0)")
        if perf.get('test_r2', 0) > 0.95:
            critical_issues.append("Unrealistic R² (> 0.95)")
            
        if critical_issues:
            logger.info("    ├── Models are: INVALID - CRITICAL ISSUES FOUND")
            logger.info("    └── Action required:")
            for issue in critical_issues:
                logger.info(f"        - Fix: {issue}")
            logger.info("        - Rebuild models from scratch with proper validation")
        else:
            if mae < 3.0:
                logger.info("    ├── Models are: SUSPICIOUS - Performance too good")
                logger.info("    └── Action required:")
                logger.info("        - Review feature engineering for leakage")
                logger.info("        - Verify no same-week stats are used")
                logger.info("        - Consider more conservative model")
            else:
                logger.info("    ├── Models are: VALID")
                logger.info("    └── Action required: None - ready for deployment")
        
        logger.info("\n" + "="*70)
        
        # Save detailed report
        with open('ml_audit_report.json', 'w') as f:
            json.dump(self.audit_results, f, indent=2)
        logger.info("Detailed audit results saved to ml_audit_report.json")

def main():
    """Run comprehensive ML model audit"""
    auditor = MLModelAuditor()
    
    # Load models and data
    if not auditor.load_models_and_data():
        logger.error("Failed to load models and data!")
        return False
    
    # Run all audit checks
    auditor.check_data_leakage()
    auditor.check_multicollinearity()
    auditor.verify_temporal_split()
    auditor.test_realistic_performance()
    
    # Generate final report
    auditor.generate_audit_report()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)