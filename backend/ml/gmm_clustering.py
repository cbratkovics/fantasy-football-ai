"""
Gaussian Mixture Model (GMM) Clustering for Fantasy Football Draft Tiers
Creates 16 probabilistic tiers for optimal draft strategy
Includes PCA for dimensionality reduction and BIC for model selection
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import logging
import joblib
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DraftTier:
    """Represents a player's draft tier assignment"""
    player_id: str
    player_name: str
    position: str
    tier: int
    probability: float
    tier_label: str
    alternative_tiers: Dict[int, float]  # {tier: probability}
    cluster_center_distance: float
    expected_points: float


class GMMDraftOptimizer:
    """
    Sophisticated GMM-based draft tier system
    Creates 16 tiers aligned with typical draft rounds
    """
    
    # Tier labels for interpretability
    TIER_LABELS = {
        1: "Elite - Round 1",
        2: "Premium - Round 2",
        3: "Core Starters - Round 3",
        4: "Solid Starters - Round 4",
        5: "Reliable Options - Round 5",
        6: "Value Picks - Round 6",
        7: "Upside Plays - Round 7",
        8: "Depth Pieces - Round 8",
        9: "High-Risk/Reward - Round 9",
        10: "Bench Starters - Round 10",
        11: "Handcuffs/Specialists - Round 11",
        12: "Deep Sleepers - Round 12",
        13: "Late-Round Fliers - Round 13",
        14: "Lottery Tickets - Round 14",
        15: "Deep Dynasty Stashes - Round 15",
        16: "Waiver Wire Candidates - Round 16+"
    }
    
    def __init__(
        self,
        n_components: int = 16,
        n_pca_components: int = 10,
        random_state: int = 42
    ):
        """
        Initialize GMM Draft Optimizer
        
        Args:
            n_components: Number of clusters (default 16 for draft rounds)
            n_pca_components: PCA components for dimensionality reduction
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.n_pca_components = n_pca_components
        self.random_state = random_state
        
        # Initialize models
        self.scaler = StandardScaler()
        self.pca = None  # Will be initialized in fit() with proper dimensions
        self.gmm = None
        
        # Storage for fitted data
        self.feature_columns = None
        self.is_fitted = False
    
    def find_optimal_components(
        self,
        features: np.ndarray,
        min_components: int = 10,
        max_components: int = 20
    ) -> Tuple[int, List[float]]:
        """
        Use BIC to find optimal number of components
        
        Args:
            features: Feature matrix
            min_components: Minimum clusters to test
            max_components: Maximum clusters to test
            
        Returns:
            Optimal number of components and BIC scores
        """
        bic_scores = []
        n_components_range = range(min_components, max_components + 1)
        
        for n in n_components_range:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type='full',
                random_state=self.random_state,
                n_init=3
            )
            gmm.fit(features)
            bic_scores.append(gmm.bic(features))
            logger.info(f"BIC for {n} components: {gmm.bic(features):.2f}")
        
        # Find elbow point
        optimal_n = n_components_range[np.argmin(bic_scores)]
        
        return optimal_n, bic_scores
    
    def fit(
        self,
        features: np.ndarray,
        feature_names: List[str],
        optimize_components: bool = False
    ) -> 'GMMDraftOptimizer':
        """
        Fit GMM model to player features
        
        Args:
            features: Feature matrix (n_players, n_features)
            feature_names: Names of features for interpretability
            optimize_components: Whether to optimize n_components using BIC
            
        Returns:
            Self for chaining
        """
        logger.info(f"Fitting GMM with {features.shape[0]} players")
        
        # Store feature names
        self.feature_columns = feature_names
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Initialize PCA with appropriate number of components
        n_features = features_scaled.shape[1]
        actual_pca_components = min(self.n_pca_components, n_features)
        self.pca = PCA(n_components=actual_pca_components, random_state=self.random_state)
        
        # Apply PCA for dimensionality reduction
        features_pca = self.pca.fit_transform(features_scaled)
        logger.info(f"PCA using {actual_pca_components} components (original: {n_features} features)")
        logger.info(f"PCA explained variance: {self.pca.explained_variance_ratio_.sum():.2%}")
        
        # Optimize components if requested
        if optimize_components:
            optimal_n, _ = self.find_optimal_components(features_pca)
            self.n_components = optimal_n
            logger.info(f"Optimal components: {optimal_n}")
        
        # Fit GMM
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            random_state=self.random_state,
            n_init=5,  # Multiple initializations
            max_iter=200
        )
        
        self.gmm.fit(features_pca)
        self.is_fitted = True
        
        # Log convergence
        logger.info(f"GMM converged: {self.gmm.converged_}")
        logger.info(f"Number of iterations: {self.gmm.n_iter_}")
        
        return self
    
    def predict_tiers(
        self,
        features: np.ndarray,
        player_ids: List[str],
        player_names: List[str],
        positions: List[str],
        expected_points: List[float]
    ) -> List[DraftTier]:
        """
        Predict draft tiers for players
        
        Args:
            features: Feature matrix
            player_ids: Player IDs
            player_names: Player names
            positions: Player positions
            expected_points: Expected fantasy points
            
        Returns:
            List of DraftTier objects
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Transform features
        features_scaled = self.scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        
        # Get cluster assignments and probabilities
        cluster_labels = self.gmm.predict(features_pca)
        cluster_probs = self.gmm.predict_proba(features_pca)
        
        # Calculate distances to cluster centers
        distances = self._calculate_cluster_distances(features_pca)
        
        # Reorder clusters by average expected points (best to worst)
        cluster_values = self._calculate_cluster_values(
            cluster_labels, 
            expected_points
        )
        cluster_mapping = self._create_tier_mapping(cluster_values)
        
        # Create DraftTier objects
        draft_tiers = []
        for i in range(len(player_ids)):
            # Map to tier (1-16)
            original_cluster = cluster_labels[i]
            tier = cluster_mapping[original_cluster] + 1
            
            # Get alternative tier probabilities
            alt_tiers = {}
            for j, prob in enumerate(cluster_probs[i]):
                if prob > 0.05 and j != original_cluster:  # 5% threshold
                    alt_tier = cluster_mapping[j] + 1
                    alt_tiers[alt_tier] = float(prob)
            
            draft_tier = DraftTier(
                player_id=player_ids[i],
                player_name=player_names[i],
                position=positions[i],
                tier=tier,
                probability=float(cluster_probs[i][original_cluster]),
                tier_label=self.TIER_LABELS[tier],
                alternative_tiers=alt_tiers,
                cluster_center_distance=float(distances[i]),
                expected_points=expected_points[i]
            )
            
            draft_tiers.append(draft_tier)
        
        return draft_tiers
    
    def _calculate_cluster_distances(
        self, 
        features_pca: np.ndarray
    ) -> np.ndarray:
        """Calculate distance to assigned cluster center"""
        distances = np.zeros(features_pca.shape[0])
        labels = self.gmm.predict(features_pca)
        
        for i in range(features_pca.shape[0]):
            cluster = labels[i]
            center = self.gmm.means_[cluster]
            distances[i] = np.linalg.norm(features_pca[i] - center)
        
        return distances
    
    def _calculate_cluster_values(
        self,
        labels: np.ndarray,
        expected_points: List[float]
    ) -> Dict[int, float]:
        """Calculate average value for each cluster"""
        cluster_values = {}
        
        for cluster in range(self.n_components):
            mask = labels == cluster
            if np.any(mask):
                cluster_values[cluster] = np.mean(
                    [expected_points[i] for i in range(len(labels)) if mask[i]]
                )
            else:
                cluster_values[cluster] = 0.0
        
        return cluster_values
    
    def _create_tier_mapping(
        self, 
        cluster_values: Dict[int, float]
    ) -> Dict[int, int]:
        """Map clusters to tiers based on value"""
        # Sort clusters by value (descending)
        sorted_clusters = sorted(
            cluster_values.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Create mapping
        mapping = {}
        for tier, (cluster, _) in enumerate(sorted_clusters):
            mapping[cluster] = tier
        
        return mapping
    
    def visualize_tiers(
        self,
        draft_tiers: List[DraftTier],
        save_path: Optional[str] = None
    ):
        """
        Visualize draft tier distributions
        
        Args:
            draft_tiers: List of DraftTier objects
            save_path: Path to save visualization
        """
        # Create DataFrame for visualization
        df = pd.DataFrame([
            {
                'Player': dt.player_name,
                'Position': dt.position,
                'Tier': dt.tier,
                'Expected Points': dt.expected_points,
                'Confidence': dt.probability
            }
            for dt in draft_tiers
        ])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Tier distribution by position
        ax = axes[0, 0]
        tier_counts = df.groupby(['Position', 'Tier']).size().unstack(fill_value=0)
        tier_counts.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
        ax.set_title('Tier Distribution by Position')
        ax.set_xlabel('Position')
        ax.set_ylabel('Number of Players')
        ax.legend(title='Tier', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Expected points by tier
        ax = axes[0, 1]
        df.boxplot(column='Expected Points', by='Tier', ax=ax)
        ax.set_title('Expected Points Distribution by Tier')
        ax.set_xlabel('Tier')
        ax.set_ylabel('Expected Fantasy Points')
        
        # 3. Confidence distribution
        ax = axes[1, 0]
        df['Confidence'].hist(bins=30, ax=ax, edgecolor='black')
        ax.axvline(df['Confidence'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {df["Confidence"].mean():.2f}')
        ax.set_title('Cluster Assignment Confidence Distribution')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Number of Players')
        ax.legend()
        
        # 4. Top players by tier
        ax = axes[1, 1]
        top_by_tier = []
        for tier in range(1, min(9, self.n_components + 1)):  # Show top 8 tiers
            tier_players = df[df['Tier'] == tier].nlargest(3, 'Expected Points')
            for _, player in tier_players.iterrows():
                top_by_tier.append(f"T{tier}: {player['Player'][:15]}")
        
        ax.text(0.1, 0.9, '\n'.join(top_by_tier), transform=ax.transAxes,
                verticalalignment='top', fontsize=10, family='monospace')
        ax.set_title('Top 3 Players per Tier (Tiers 1-8)')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def get_tier_summary(
        self, 
        draft_tiers: List[DraftTier]
    ) -> pd.DataFrame:
        """
        Get summary statistics for each tier
        
        Args:
            draft_tiers: List of DraftTier objects
            
        Returns:
            DataFrame with tier summaries
        """
        df = pd.DataFrame([
            {
                'Tier': dt.tier,
                'Position': dt.position,
                'Expected Points': dt.expected_points,
                'Confidence': dt.probability
            }
            for dt in draft_tiers
        ])
        
        summary = df.groupby('Tier').agg({
            'Expected Points': ['mean', 'std', 'min', 'max'],
            'Confidence': 'mean',
            'Position': 'count'
        }).round(2)
        
        summary.columns = ['Avg Points', 'Std Dev', 'Min Points', 
                          'Max Points', 'Avg Confidence', 'Player Count']
        summary['Tier Label'] = [self.TIER_LABELS[i] for i in summary.index]
        
        return summary
    
    def save_model(self, path: str):
        """Save fitted model to disk"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        model_data = {
            'gmm': self.gmm,
            'scaler': self.scaler,
            'pca': self.pca,
            'n_components': self.n_components,
            'feature_columns': self.feature_columns,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load fitted model from disk"""
        model_data = joblib.load(path)
        
        self.gmm = model_data['gmm']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.n_components = model_data['n_components']
        self.feature_columns = model_data['feature_columns']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {path}")
        logger.info(f"Model timestamp: {model_data['timestamp']}")


# Example usage
def example_usage():
    """Demonstrate GMM draft tier creation"""
    np.random.seed(42)
    
    # Generate sample data (300 players)
    n_players = 300
    
    # Create synthetic features
    features = np.random.randn(n_players, 20)  # 20 features
    
    # Add some structure (better players have higher values)
    for i in range(n_players):
        skill_factor = (n_players - i) / n_players
        features[i] += skill_factor * np.random.randn(20) * 0.5
    
    # Generate metadata
    player_ids = [f"player_{i}" for i in range(n_players)]
    player_names = [f"Player {i}" for i in range(n_players)]
    
    # Mix of positions
    positions = []
    for i in range(n_players):
        if i < 30:
            positions.append('QB')
        elif i < 90:
            positions.append('RB')
        elif i < 180:
            positions.append('WR')
        else:
            positions.append('TE')
    
    # Expected points (decreasing)
    expected_points = [25 - (i * 0.08) + np.random.randn() * 2 for i in range(n_players)]
    expected_points = [max(0, p) for p in expected_points]
    
    # Feature names
    feature_names = [f"feature_{i}" for i in range(20)]
    
    # Initialize and fit GMM
    optimizer = GMMDraftOptimizer(n_components=16)
    optimizer.fit(features, feature_names)
    
    # Predict tiers
    draft_tiers = optimizer.predict_tiers(
        features,
        player_ids,
        player_names,
        positions,
        expected_points
    )
    
    # Show top players in first few tiers
    print("Top Players by Tier:\n")
    for tier in range(1, 5):
        tier_players = [dt for dt in draft_tiers if dt.tier == tier]
        tier_players.sort(key=lambda x: x.expected_points, reverse=True)
        
        print(f"Tier {tier} - {optimizer.TIER_LABELS[tier]}")
        for player in tier_players[:5]:
            print(f"  {player.player_name} ({player.position}) - "
                  f"{player.expected_points:.1f} pts (confidence: {player.probability:.2%})")
        print()
    
    # Get tier summary
    summary = optimizer.get_tier_summary(draft_tiers)
    print("\nTier Summary:")
    print(summary)


if __name__ == "__main__":
    example_usage()