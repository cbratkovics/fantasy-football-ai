"""
Fantasy Football AI - Gaussian Mixture Model Clustering
Creates probabilistic player tiers for draft recommendations using GMM clustering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import logging

logger = logging.getLogger(__name__)

@dataclass
class PlayerTier:
    """Container for player tier assignment"""
    player_id: str
    position: str
    tier: int
    tier_probability: float
    tier_name: str
    expected_value: float
    risk_level: str

class FantasyGMM:
    """
    Gaussian Mixture Model for Fantasy Football player tier classification.
    
    Creates probabilistic clusters of players based on performance features,
    then maps these clusters to draft tiers (1-16) that correspond to 
    fantasy football draft rounds.
    
    Key advantages over K-means:
    - Soft clustering: Players can belong to multiple tiers with probabilities
    - Handles overlapping player archetypes naturally
    - Provides uncertainty estimates for tier assignments
    """
    
    def __init__(self, n_components_range: Tuple[int, int] = (3, 8), 
                 use_pca: bool = True, pca_components: int = 4):
        """
        Initialize GMM clustering system.
        
        Args:
            n_components_range: Range of cluster numbers to test for optimal selection
            use_pca: Whether to use PCA for dimensionality reduction
            pca_components: Number of PCA components to retain
        """
        self.n_components_range = n_components_range
        self.use_pca = use_pca
        self.pca_components = pca_components
        
        # Model storage by position
        self.models: Dict[str, GaussianMixture] = {}
        self.pca_models: Dict[str, PCA] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.tier_mappings: Dict[str, Dict[int, Dict]] = {}
        
        # Tier names for interpretability
        self.tier_names = {
            1: "Elite (Round 1-2)", 2: "High-End RB1/WR1 (Round 2-3)",
            3: "Solid RB1/WR1 (Round 3-4)", 4: "RB2/WR2 Upside (Round 4-5)",
            5: "Reliable RB2/WR2 (Round 5-6)", 6: "Flex/Bench Depth (Round 6-8)",
            7: "Handcuff/Lottery (Round 8-10)", 8: "Deep Sleeper (Round 10+)"
        }
    
    def fit(self, features_df: pd.DataFrame, target_col: str = 'ppg') -> Dict[str, Dict]:
        """
        Fit GMM models for each position.
        
        Args:
            features_df: DataFrame with engineered features
            target_col: Target column for tier value calculation
            
        Returns:
            Dictionary with fitting results for each position
        """
        logger.info("Fitting GMM models for player tier classification")
        
        results = {}
        positions = ['QB', 'RB', 'WR', 'TE']
        
        for position in positions:
            logger.info(f"Fitting GMM for position: {position}")
            
            # Filter data for position
            pos_data = features_df[features_df['position'] == position].copy()
            
            if len(pos_data) < 20:  # Need minimum samples
                logger.warning(f"Insufficient data for {position}: {len(pos_data)} samples")
                continue
            
            # Prepare feature matrix
            feature_cols = ['ppg', 'consistency_score', 'efficiency_ratio', 
                           'momentum_score', 'boom_bust_ratio', 'recent_trend']
            X = pos_data[feature_cols].values
            y = pos_data[target_col].values
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[position] = scaler
            
            # Apply PCA if requested
            if self.use_pca:
                pca = PCA(n_components=self.pca_components)
                X_final = pca.fit_transform(X_scaled)
                self.pca_models[position] = pca
                logger.info(f"{position} PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
            else:
                X_final = X_scaled
            
            # Find optimal number of components using BIC
            best_model, best_bic, n_components = self._find_optimal_components(X_final, position)
            self.models[position] = best_model
            
            # Create tier mapping based on cluster centroids
            tier_mapping = self._create_tier_mapping(best_model, X_final, y, position)
            self.tier_mappings[position] = tier_mapping
            
            results[position] = {
                'n_components': n_components,
                'bic_score': best_bic,
                'n_samples': len(pos_data),
                'pca_variance': self.pca_models[position].explained_variance_ratio_.sum() if self.use_pca else 1.0,
                'tier_mapping': tier_mapping
            }
            
            logger.info(f"{position} - Clusters: {n_components}, BIC: {best_bic:.2f}, Samples: {len(pos_data)}")
        
        return results
    
    def _find_optimal_components(self, X: np.ndarray, position: str) -> Tuple[GaussianMixture, float, int]:
        """Find optimal number of GMM components using BIC."""
        best_bic = np.inf
        best_model = None
        best_n = self.n_components_range[0]
        
        for n_components in range(self.n_components_range[0], self.n_components_range[1] + 1):
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='full',
                    random_state=42,
                    max_iter=200
                )
                gmm.fit(X)
                
                bic = gmm.bic(X)
                
                if bic < best_bic:
                    best_bic = bic
                    best_model = gmm
                    best_n = n_components
                    
            except Exception as e:
                logger.warning(f"{position} - Failed fitting {n_components} components: {e}")
                continue
        
        return best_model, best_bic, best_n
    
    def _create_tier_mapping(self, model: GaussianMixture, X: np.ndarray, 
                           y: np.ndarray, position: str) -> Dict[int, Dict]:
        """Create mapping from GMM clusters to fantasy draft tiers."""
        cluster_labels = model.predict(X)
        tier_mapping = {}
        
        # Calculate cluster statistics
        for cluster_id in range(model.n_components):
            cluster_mask = cluster_labels == cluster_id
            cluster_y = y[cluster_mask]
            
            if len(cluster_y) == 0:
                continue
                
            # Calculate cluster characteristics
            mean_value = np.mean(cluster_y)
            std_value = np.std(cluster_y)
            size = len(cluster_y)
            
            # Determine risk level based on standard deviation
            if std_value < np.percentile([np.std(y[cluster_labels == i]) 
                                        for i in range(model.n_components) if np.sum(cluster_labels == i) > 0], 33):
                risk_level = "Low"
            elif std_value < np.percentile([np.std(y[cluster_labels == i]) 
                                          for i in range(model.n_components) if np.sum(cluster_labels == i) > 0], 67):
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            tier_mapping[cluster_id] = {
                'mean_value': float(mean_value),
                'std_value': float(std_value),
                'size': int(size),
                'risk_level': risk_level
            }
        
        # Sort clusters by mean value (descending) and assign draft tiers
        sorted_clusters = sorted(tier_mapping.keys(), 
                               key=lambda x: tier_mapping[x]['mean_value'], 
                               reverse=True)
        
        # Map to draft tiers (1-8 for each position, corresponding to draft rounds)
        for rank, cluster_id in enumerate(sorted_clusters):
            draft_tier = min(rank + 1, 8)  # Cap at tier 8
            tier_mapping[cluster_id]['draft_tier'] = draft_tier
            tier_mapping[cluster_id]['tier_name'] = self.tier_names.get(draft_tier, f"Tier {draft_tier}")
        
        return tier_mapping
    
    def predict_tiers(self, features_df: pd.DataFrame) -> List[PlayerTier]:
        """
        Predict player tiers for new data.
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            List of PlayerTier objects with tier assignments
        """
        logger.info(f"Predicting tiers for {len(features_df)} players")
        
        tier_predictions = []
        
        for _, row in features_df.iterrows():
            position = row['position']
            
            if position not in self.models:
                logger.warning(f"No model found for position {position}")
                continue
            
            try:
                # Prepare features
                feature_cols = ['ppg', 'consistency_score', 'efficiency_ratio', 
                               'momentum_score', 'boom_bust_ratio', 'recent_trend']
                X = np.array([row[col] for col in feature_cols]).reshape(1, -1)
                
                # Transform features
                X_scaled = self.scalers[position].transform(X)
                if self.use_pca:
                    X_final = self.pca_models[position].transform(X_scaled)
                else:
                    X_final = X_scaled
                
                # Get cluster prediction and probabilities
                cluster_id = self.models[position].predict(X_final)[0]
                cluster_probs = self.models[position].predict_proba(X_final)[0]
                
                # Get tier information
                tier_info = self.tier_mappings[position][cluster_id]
                
                tier_prediction = PlayerTier(
                    player_id=str(row['player_id']),
                    position=position,
                    tier=tier_info['draft_tier'],
                    tier_probability=float(cluster_probs[cluster_id]),
                    tier_name=tier_info['tier_name'],
                    expected_value=tier_info['mean_value'],
                    risk_level=tier_info['risk_level']
                )
                
                tier_predictions.append(tier_prediction)
                
            except Exception as e:
                logger.error(f"Error predicting tier for player {row['player_id']}: {e}")
                continue
        
        logger.info(f"Successfully predicted tiers for {len(tier_predictions)} players")
        return tier_predictions
    
    def get_tier_summary(self, tier_predictions: List[PlayerTier]) -> pd.DataFrame:
        """Create summary DataFrame of tier predictions."""
        data = []
        for tier in tier_predictions:
            data.append({
                'player_id': tier.player_id,
                'position': tier.position,
                'tier': tier.tier,
                'tier_name': tier.tier_name,
                'tier_probability': tier.tier_probability,
                'expected_value': tier.expected_value,
                'risk_level': tier.risk_level
            })
        return pd.DataFrame(data)
    
    def save_models(self, filepath: str):
        """Save trained models to disk."""
        model_data = {
            'models': self.models,
            'pca_models': self.pca_models,
            'scalers': self.scalers,
            'tier_mappings': self.tier_mappings,
            'config': {
                'n_components_range': self.n_components_range,
                'use_pca': self.use_pca,
                'pca_components': self.pca_components
            }
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk."""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.pca_models = model_data['pca_models']
        self.scalers = model_data['scalers']
        self.tier_mappings = model_data['tier_mappings']
        
        config = model_data['config']
        self.n_components_range = config['n_components_range']
        self.use_pca = config['use_pca']
        self.pca_components = config['pca_components']
        
        logger.info(f"Models loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    # Create sample feature data
    np.random.seed(42)
    
    sample_features = []
    positions = ['QB', 'RB', 'WR', 'TE']
    
    for position in positions:
        n_players = 50  # 50 players per position
        
        for i in range(n_players):
            # Create realistic feature distributions by position
            if position == 'QB':
                base_ppg = np.random.normal(18, 4)
                consistency = np.random.normal(3, 1)
            elif position == 'RB':
                base_ppg = np.random.normal(14, 5)
                consistency = np.random.normal(2.5, 0.8)
            elif position == 'WR':
                base_ppg = np.random.normal(12, 6)
                consistency = np.random.normal(2.2, 0.9)
            else:  # TE
                base_ppg = np.random.normal(10, 4)
                consistency = np.random.normal(2.8, 0.7)
            
            sample_features.append({
                'player_id': f"{position}_{i}",
                'position': position,
                'ppg': max(0, base_ppg),
                'consistency_score': max(0.1, consistency),
                'efficiency_ratio': np.random.normal(1.0, 0.2),
                'momentum_score': base_ppg * np.random.normal(1.0, 0.15),
                'boom_bust_ratio': np.random.normal(0, 0.3),
                'recent_trend': np.random.normal(0, 2)
            })
    
    features_df = pd.DataFrame(sample_features)
    
    # Test GMM fitting
    print("Testing GMM Clustering...")
    gmm_system = FantasyGMM(n_components_range=(3, 6), use_pca=True, pca_components=4)
    
    # Fit models
    results = gmm_system.fit(features_df)
    
    print("\nFitting Results:")
    for position, result in results.items():
        print(f"{position}: {result['n_components']} clusters, BIC: {result['bic_score']:.2f}")
    
    # Test predictions
    tier_predictions = gmm_system.predict_tiers(features_df.head(20))
    tier_df = gmm_system.get_tier_summary(tier_predictions)
    
    print(f"\nSample Tier Predictions ({len(tier_predictions)} players):")
    print(tier_df.head(10))
    
    print("\nTier Distribution:")
    print(tier_df.groupby(['position', 'tier']).size().unstack(fill_value=0))