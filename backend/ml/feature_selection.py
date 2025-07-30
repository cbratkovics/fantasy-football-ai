"""
Advanced Feature Selection Process for Optimal Feature Sets
Combines multiple selection methods for robust feature identification
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
import logging
from datetime import datetime
import json

# Feature selection methods
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, RFECV, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error
import shap

from backend.models.database import SessionLocal, PlayerStats

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportance:
    """Container for feature importance scores"""
    feature_name: str
    importance_score: float
    selection_method: str
    rank: int
    stability_score: float = 0.0


@dataclass
class FeatureSelectionResult:
    """Result of feature selection process"""
    position: str
    selected_features: List[str]
    feature_scores: Dict[str, float]
    selection_methods_used: List[str]
    performance_improvement: float
    stability_metrics: Dict[str, float]
    feature_correlations: Dict[str, float]
    created_at: datetime


class AdvancedFeatureSelector:
    """
    Advanced feature selection system using multiple methods:
    1. Statistical tests (F-test, mutual information)
    2. Model-based selection (LASSO, Random Forest importance)
    3. Recursive feature elimination
    4. Stability selection
    5. SHAP-based importance
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
        # Feature selection methods
        self.selection_methods = {
            'f_test': self._f_test_selection,
            'mutual_info': self._mutual_info_selection,
            'lasso': self._lasso_selection,
            'random_forest': self._random_forest_selection,
            'rfe': self._rfe_selection,
            'stability': self._stability_selection,
            'shap': self._shap_selection
        }
        
        # Feature categories for domain knowledge
        self.feature_categories = {
            'basic': ['age', 'years_exp', 'games_played_recent'],
            'performance': ['avg_points_recent', 'total_points_recent', 'points_std'],
            'efficiency': ['efficiency_ratio', 'opp_efficiency', 'matchup_efficiency'],
            'momentum': ['momentum_3w', 'momentum_5w', 'trend_direction'],
            'consistency': ['consistency_score', 'floor', 'ceiling', 'volatility'],
            'opportunity': ['touches_per_game', 'target_share', 'targets_per_game'],
            'context': ['is_home', 'opponent_rank_vs_position', 'implied_team_total']
        }
        
        # Position-specific feature importance weights
        self.position_weights = {
            'QB': {
                'performance': 0.3,
                'efficiency': 0.25,
                'momentum': 0.2,
                'opportunity': 0.15,
                'context': 0.1
            },
            'RB': {
                'opportunity': 0.3,
                'performance': 0.25,
                'efficiency': 0.2,
                'context': 0.15,
                'momentum': 0.1
            },
            'WR': {
                'opportunity': 0.35,
                'efficiency': 0.25,
                'performance': 0.2,
                'momentum': 0.15,
                'context': 0.05
            },
            'TE': {
                'opportunity': 0.3,
                'performance': 0.25,
                'efficiency': 0.2,
                'consistency': 0.15,
                'momentum': 0.1
            }
        }
    
    def select_optimal_features(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        position: str,
        max_features: int = 15,
        min_features: int = 5,
        cv_folds: int = 5,
        stability_iterations: int = 50
    ) -> FeatureSelectionResult:
        """
        Select optimal feature set using ensemble of methods
        
        Args:
            X: Feature matrix
            y: Target values
            position: Player position
            max_features: Maximum number of features to select
            min_features: Minimum number of features to select
            cv_folds: Cross-validation folds
            stability_iterations: Iterations for stability selection
        
        Returns:
            FeatureSelectionResult with optimal features
        """
        logger.info(f"Starting feature selection for {position} with {X.shape[1]} features")
        
        # Remove low-variance features
        X_filtered = self._remove_low_variance_features(X)
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_filtered),
            columns=X_filtered.columns,
            index=X_filtered.index
        )
        
        # Apply all selection methods
        method_results = {}
        for method_name, method_func in self.selection_methods.items():
            try:
                features = method_func(
                    X_scaled, y, position, max_features, cv_folds, stability_iterations
                )
                method_results[method_name] = features
                logger.info(f"{method_name}: selected {len(features)} features")
            except Exception as e:
                logger.error(f"Error in {method_name}: {str(e)}")
                method_results[method_name] = []
        
        # Combine results using ensemble voting
        feature_votes = self._ensemble_voting(method_results, X_scaled.columns.tolist())
        
        # Apply domain knowledge weights
        weighted_scores = self._apply_domain_knowledge(feature_votes, position)
        
        # Select final feature set
        final_features = self._select_final_features(
            weighted_scores, min_features, max_features
        )
        
        # Evaluate performance improvement
        performance_improvement = self._evaluate_feature_set(
            X_scaled, y, final_features, cv_folds
        )
        
        # Calculate stability metrics
        stability_metrics = self._calculate_stability_metrics(
            method_results, final_features
        )
        
        # Calculate feature correlations
        feature_correlations = self._calculate_feature_correlations(
            X_scaled[final_features]
        )
        
        # Create result
        result = FeatureSelectionResult(
            position=position,
            selected_features=final_features,
            feature_scores=weighted_scores,
            selection_methods_used=list(method_results.keys()),
            performance_improvement=performance_improvement,
            stability_metrics=stability_metrics,
            feature_correlations=feature_correlations,
            created_at=datetime.utcnow()
        )
        
        logger.info(f"Selected {len(final_features)} features for {position}")
        return result
    
    def _remove_low_variance_features(self, X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """Remove features with low variance"""
        selector = VarianceThreshold(threshold=threshold)
        X_filtered = selector.fit_transform(X)
        
        selected_features = X.columns[selector.get_support()].tolist()
        return pd.DataFrame(X_filtered, columns=selected_features, index=X.index)
    
    def _f_test_selection(
        self, X: pd.DataFrame, y: np.ndarray, position: str,
        max_features: int, cv_folds: int, stability_iterations: int
    ) -> List[str]:
        """F-test based feature selection"""
        selector = SelectKBest(f_regression, k=min(max_features, X.shape[1]))
        selector.fit(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        return selected_features
    
    def _mutual_info_selection(
        self, X: pd.DataFrame, y: np.ndarray, position: str,
        max_features: int, cv_folds: int, stability_iterations: int
    ) -> List[str]:
        """Mutual information based feature selection"""
        selector = SelectKBest(mutual_info_regression, k=min(max_features, X.shape[1]))
        selector.fit(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        return selected_features
    
    def _lasso_selection(
        self, X: pd.DataFrame, y: np.ndarray, position: str,
        max_features: int, cv_folds: int, stability_iterations: int
    ) -> List[str]:
        """LASSO-based feature selection"""
        lasso = LassoCV(cv=cv_folds, random_state=42, max_iter=1000)
        lasso.fit(X, y)
        
        # Get features with non-zero coefficients
        selected_mask = np.abs(lasso.coef_) > 1e-6
        selected_features = X.columns[selected_mask].tolist()
        
        # If too many features, select top ones by coefficient magnitude
        if len(selected_features) > max_features:
            feature_importance = pd.Series(
                np.abs(lasso.coef_[selected_mask]),
                index=selected_features
            ).sort_values(ascending=False)
            selected_features = feature_importance.head(max_features).index.tolist()
        
        return selected_features
    
    def _random_forest_selection(
        self, X: pd.DataFrame, y: np.ndarray, position: str,
        max_features: int, cv_folds: int, stability_iterations: int
    ) -> List[str]:
        """Random Forest importance-based selection"""
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importances
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        selected_features = importance_df.head(max_features)['feature'].tolist()
        return selected_features
    
    def _rfe_selection(
        self, X: pd.DataFrame, y: np.ndarray, position: str,
        max_features: int, cv_folds: int, stability_iterations: int
    ) -> List[str]:
        """Recursive Feature Elimination"""
        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        selector = RFECV(
            estimator, cv=cv_folds, scoring='neg_mean_absolute_error',
            min_features_to_select=5, n_jobs=-1
        )
        
        try:
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            # Limit to max_features
            if len(selected_features) > max_features:
                # Rank by RFE ranking
                ranking_df = pd.DataFrame({
                    'feature': X.columns,
                    'ranking': selector.ranking_
                }).sort_values('ranking')
                selected_features = ranking_df.head(max_features)['feature'].tolist()
            
            return selected_features
        except Exception as e:
            logger.error(f"RFE failed: {str(e)}")
            return []
    
    def _stability_selection(
        self, X: pd.DataFrame, y: np.ndarray, position: str,
        max_features: int, cv_folds: int, stability_iterations: int
    ) -> List[str]:
        """Stability selection using bootstrap sampling"""
        n_samples = X.shape[0]
        feature_selection_freq = {col: 0 for col in X.columns}
        
        for i in range(stability_iterations):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=int(0.8 * n_samples), replace=True)
            X_boot = X.iloc[indices]
            y_boot = y[indices]
            
            # Apply LASSO
            try:
                lasso = LassoCV(cv=3, random_state=i, max_iter=500)
                lasso.fit(X_boot, y_boot)
                
                # Count selected features
                selected_mask = np.abs(lasso.coef_) > 1e-6
                for j, selected in enumerate(selected_mask):
                    if selected:
                        feature_selection_freq[X.columns[j]] += 1
            except:
                continue
        
        # Select features with high selection frequency
        freq_df = pd.DataFrame(
            list(feature_selection_freq.items()),
            columns=['feature', 'frequency']
        ).sort_values('frequency', ascending=False)
        
        # Select features appearing in at least 30% of iterations
        threshold = stability_iterations * 0.3
        stable_features = freq_df[freq_df['frequency'] >= threshold]['feature'].tolist()
        
        return stable_features[:max_features]
    
    def _shap_selection(
        self, X: pd.DataFrame, y: np.ndarray, position: str,
        max_features: int, cv_folds: int, stability_iterations: int
    ) -> List[str]:
        """SHAP-based feature selection"""
        try:
            # Train model for SHAP
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X.sample(min(500, len(X))))
            
            # Calculate mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(axis=0)
            
            # Select top features
            shap_importance = pd.DataFrame({
                'feature': X.columns,
                'shap_importance': mean_shap
            }).sort_values('shap_importance', ascending=False)
            
            selected_features = shap_importance.head(max_features)['feature'].tolist()
            return selected_features
            
        except Exception as e:
            logger.error(f"SHAP selection failed: {str(e)}")
            return []
    
    def _ensemble_voting(
        self, method_results: Dict[str, List[str]], all_features: List[str]
    ) -> Dict[str, float]:
        """Combine results from multiple methods using voting"""
        feature_votes = {feature: 0 for feature in all_features}
        
        # Weight each method equally
        for method, features in method_results.items():
            if features:  # Only if method succeeded
                for feature in features:
                    if feature in feature_votes:
                        feature_votes[feature] += 1
        
        # Normalize votes
        max_votes = max(feature_votes.values()) if feature_votes else 1
        normalized_votes = {
            feature: votes / max_votes
            for feature, votes in feature_votes.items()
        }
        
        return normalized_votes
    
    def _apply_domain_knowledge(
        self, feature_votes: Dict[str, float], position: str
    ) -> Dict[str, float]:
        """Apply domain knowledge to feature scores"""
        weighted_scores = {}
        position_weights = self.position_weights.get(position, {})
        
        for feature, vote_score in feature_votes.items():
            # Find feature category
            category_weight = 1.0
            for category, features in self.feature_categories.items():
                if feature in features:
                    category_weight = position_weights.get(category, 1.0)
                    break
            
            # Apply category weight
            weighted_scores[feature] = vote_score * category_weight
        
        return weighted_scores
    
    def _select_final_features(
        self, weighted_scores: Dict[str, float], min_features: int, max_features: int
    ) -> List[str]:
        """Select final feature set based on weighted scores"""
        # Sort by score
        sorted_features = sorted(
            weighted_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select features with positive scores
        positive_features = [f for f, s in sorted_features if s > 0]
        
        # Ensure we have at least min_features
        if len(positive_features) < min_features:
            final_features = [f for f, s in sorted_features[:min_features]]
        else:
            final_features = positive_features[:max_features]
        
        return final_features
    
    def _evaluate_feature_set(
        self, X: pd.DataFrame, y: np.ndarray, features: List[str], cv_folds: int
    ) -> float:
        """Evaluate performance improvement of selected features"""
        if not features:
            return 0.0
        
        # Baseline performance (using all features)
        rf_baseline = RandomForestRegressor(n_estimators=50, random_state=42)
        baseline_scores = cross_val_score(
            rf_baseline, X, y, cv=cv_folds,
            scoring='neg_mean_absolute_error', n_jobs=-1
        )
        baseline_mae = -baseline_scores.mean()
        
        # Selected features performance
        rf_selected = RandomForestRegressor(n_estimators=50, random_state=42)
        selected_scores = cross_val_score(
            rf_selected, X[features], y, cv=cv_folds,
            scoring='neg_mean_absolute_error', n_jobs=-1
        )
        selected_mae = -selected_scores.mean()
        
        # Calculate improvement
        improvement = (baseline_mae - selected_mae) / baseline_mae
        return improvement
    
    def _calculate_stability_metrics(
        self, method_results: Dict[str, List[str]], final_features: List[str]
    ) -> Dict[str, float]:
        """Calculate stability metrics for feature selection"""
        if not method_results:
            return {}
        
        # Count how many methods selected each final feature
        feature_method_count = {}
        for feature in final_features:
            count = sum(1 for features in method_results.values() if feature in features)
            feature_method_count[feature] = count / len(method_results)
        
        # Overall stability metrics
        avg_stability = np.mean(list(feature_method_count.values()))
        min_stability = min(feature_method_count.values()) if feature_method_count else 0
        
        return {
            'average_stability': avg_stability,
            'minimum_stability': min_stability,
            'features_in_all_methods': sum(1 for s in feature_method_count.values() if s == 1.0),
            'feature_stability': feature_method_count
        }
    
    def _calculate_feature_correlations(self, X: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlation matrix for selected features"""
        if X.empty:
            return {}
        
        corr_matrix = X.corr().abs()
        
        # Find maximum correlation for each feature (excluding self-correlation)
        max_correlations = {}
        for feature in X.columns:
            other_corr = corr_matrix[feature].drop(feature)
            max_correlations[feature] = other_corr.max() if not other_corr.empty else 0.0
        
        return max_correlations
    
    def create_feature_report(self, result: FeatureSelectionResult) -> Dict[str, Any]:
        """Create comprehensive feature selection report"""
        report = {
            'summary': {
                'position': result.position,
                'total_features_selected': len(result.selected_features),
                'performance_improvement': result.performance_improvement,
                'selection_date': result.created_at.isoformat()
            },
            'selected_features': result.selected_features,
            'feature_scores': result.feature_scores,
            'stability_metrics': result.stability_metrics,
            'correlation_analysis': {
                'max_correlation': max(result.feature_correlations.values()) if result.feature_correlations else 0,
                'avg_correlation': np.mean(list(result.feature_correlations.values())) if result.feature_correlations else 0,
                'highly_correlated_features': [
                    f for f, corr in result.feature_correlations.items() if corr > 0.8
                ]
            },
            'recommendations': self._generate_recommendations(result)
        }
        
        return report
    
    def _generate_recommendations(self, result: FeatureSelectionResult) -> List[str]:
        """Generate recommendations based on feature selection results"""
        recommendations = []
        
        # Performance recommendation
        if result.performance_improvement > 0.05:
            recommendations.append("Feature selection shows significant improvement (>5%)")
        elif result.performance_improvement < 0:
            recommendations.append("Consider using all features - selection may be reducing performance")
        
        # Stability recommendation
        avg_stability = result.stability_metrics.get('average_stability', 0)
        if avg_stability < 0.5:
            recommendations.append("Low feature stability - consider more robust selection methods")
        elif avg_stability > 0.8:
            recommendations.append("High feature stability - good consensus across methods")
        
        # Correlation recommendation
        max_corr = max(result.feature_correlations.values()) if result.feature_correlations else 0
        if max_corr > 0.9:
            recommendations.append("High feature correlation detected - consider dimensionality reduction")
        
        # Feature count recommendation
        if len(result.selected_features) < 5:
            recommendations.append("Very few features selected - may be underfitting")
        elif len(result.selected_features) > 20:
            recommendations.append("Many features selected - consider more aggressive selection")
        
        return recommendations


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 30
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create target with some features being more important
    important_features = [0, 2, 5, 8, 12]
    y = (X.iloc[:, important_features].sum(axis=1) + 
         np.random.randn(n_samples) * 0.1)
    
    # Run feature selection
    selector = AdvancedFeatureSelector()
    result = selector.select_optimal_features(X, y, "QB", max_features=10)
    
    print("Selected features:", result.selected_features)
    print("Performance improvement:", result.performance_improvement)
    
    # Generate report
    report = selector.create_feature_report(result)
    print("\nFeature Selection Report:")
    print(json.dumps(report, indent=2, default=str))