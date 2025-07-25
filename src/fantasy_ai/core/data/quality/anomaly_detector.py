"""
AI-Powered Data Quality Validator with Anomaly Detection and Statistical Validation.
Location: src/fantasy_ai/core/data/quality/anomaly_detector.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.stats import zscore, iqr

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from ..storage.models import (
    WeeklyStats, Player, DataQualityMetric, DataQualityStatus
)
from ..storage.database import get_db_session

logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    STATISTICAL_OUTLIER = "statistical_outlier"
    ISOLATION_FOREST = "isolation_forest"
    DBSCAN_OUTLIER = "dbscan_outlier"
    BUSINESS_RULE = "business_rule"
    CONSISTENCY_CHECK = "consistency_check"
    COMPLETENESS_CHECK = "completeness_check"

class SeverityLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class AnomalyResult:
    """Result of anomaly detection analysis."""
    anomaly_type: AnomalyType
    severity: SeverityLevel
    confidence: float  # 0-1
    description: str
    affected_fields: List[str]
    suggested_action: str
    details: Dict[str, Any]

@dataclass
class QualityMetrics:
    """Data quality metrics for a player/season."""
    completeness_score: float
    consistency_score: float
    anomaly_score: float
    overall_score: float
    anomalies: List[AnomalyResult]
    validation_timestamp: datetime

class DataQualityValidator:
    """
    Advanced data quality validator using multiple ML techniques for
    anomaly detection, statistical validation, and business rule checking.
    """
    
    def __init__(self):
        # ML Models (will be fitted during validation)
        self.isolation_forest = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42,
            n_estimators=100
        )
        
        self.dbscan = DBSCAN(
            eps=0.5,
            min_samples=5
        )
        
        # Scalers for different feature types
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        
        # Statistical thresholds
        self.z_score_threshold = 3.0
        self.iqr_multiplier = 1.5
        
        # Business rules configuration
        self.business_rules = self._load_business_rules()
        
        # Position-specific statistics for normalization
        self.position_stats = {}

    def _load_business_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load business rules for fantasy football data validation."""
        return {
            'QB': {
                'max_passing_yards': 600,
                'max_passing_tds': 7,
                'max_rushing_yards': 200,
                'max_interceptions': 6,
                'required_fields': ['passing_attempts', 'passing_completions', 'passing_yards']
            },
            'RB': {
                'max_rushing_yards': 400,
                'max_rushing_attempts': 50,
                'max_receiving_yards': 200,
                'max_total_tds': 5,
                'required_fields': ['rushing_attempts', 'rushing_yards']
            },
            'WR': {
                'max_receiving_yards': 300,
                'max_receptions': 20,
                'max_targets': 25,
                'max_receiving_tds': 4,
                'required_fields': ['receiving_targets', 'receptions', 'receiving_yards']
            },
            'TE': {
                'max_receiving_yards': 200,
                'max_receptions': 15,
                'max_targets': 20,
                'max_receiving_tds': 3,
                'required_fields': ['receiving_targets', 'receptions', 'receiving_yards']
            }
        }

    async def validate_player_stats(self, player_id: int, season: int, 
                                  week: Optional[int] = None) -> QualityMetrics:
        """
        Comprehensive validation of player statistics using multiple AI techniques.
        """
        logger.info(f"Validating data quality for player {player_id}, season {season}")
        
        async with get_db_session() as session:
            # Get player and position info
            player = session.query(Player).get(player_id)
            if not player:
                raise ValueError(f"Player {player_id} not found")
            
            # Get stats to validate
            query = session.query(WeeklyStats).filter(
                WeeklyStats.player_id == player_id,
                WeeklyStats.season == season
            )
            
            if week:
                query = query.filter(WeeklyStats.week == week)
            
            stats = query.all()
            
            if not stats:
                logger.warning(f"No stats found for player {player_id}, season {season}")
                return QualityMetrics(
                    completeness_score=0.0,
                    consistency_score=0.0,
                    anomaly_score=1.0,
                    overall_score=0.0,
                    anomalies=[],
                    validation_timestamp=datetime.now(timezone.utc)
                )
            
            # Run comprehensive validation
            anomalies = []
            
            # 1. Statistical Analysis
            statistical_anomalies = await self._detect_statistical_anomalies(
                session, player, stats
            )
            anomalies.extend(statistical_anomalies)
            
            # 2. ML-based Anomaly Detection
            ml_anomalies = await self._detect_ml_anomalies(
                session, player, stats
            )
            anomalies.extend(ml_anomalies)
            
            # 3. Business Rule Validation
            business_anomalies = await self._validate_business_rules(
                player, stats
            )
            anomalies.extend(business_anomalies)
            
            # 4. Consistency Checks
            consistency_anomalies = await self._check_consistency(
                session, player, stats
            )
            anomalies.extend(consistency_anomalies)
            
            # Calculate quality scores
            quality_metrics = self._calculate_quality_scores(stats, anomalies)
            
            # Store results in database
            await self._store_quality_metrics(session, player_id, season, quality_metrics)
            
            return quality_metrics

    async def _detect_statistical_anomalies(self, session: Session, player: Player, 
                                          stats: List[WeeklyStats]) -> List[AnomalyResult]:
        """Detect anomalies using statistical methods (Z-score, IQR)."""
        
        anomalies = []
        
        # Convert to DataFrame for easier analysis
        df = self._stats_to_dataframe(stats)
        
        if len(df) < 3:  # Need minimum data for statistical analysis
            return anomalies
        
        # Define statistical features to analyze by position
        feature_sets = {
            'QB': ['passing_yards', 'passing_touchdowns', 'rushing_yards', 'interceptions'],
            'RB': ['rushing_yards', 'rushing_attempts', 'receiving_yards', 'rushing_touchdowns'],
            'WR': ['receiving_yards', 'receptions', 'receiving_targets', 'receiving_touchdowns'],
            'TE': ['receiving_yards', 'receptions', 'receiving_targets', 'receiving_touchdowns']
        }
        
        features = feature_sets.get(player.position, feature_sets['WR'])
        
        for feature in features:
            if feature not in df.columns or df[feature].isna().all():
                continue
            
            values = df[feature].dropna()
            if len(values) < 3:
                continue
            
            # Z-score analysis
            z_scores = np.abs(zscore(values, nan_policy='omit'))
            outlier_mask = z_scores > self.z_score_threshold
            
            if outlier_mask.any():
                outlier_weeks = df.loc[values.index[outlier_mask], 'week'].tolist()
                outlier_values = values[outlier_mask].tolist()
                
                anomalies.append(AnomalyResult(
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    severity=SeverityLevel.MEDIUM,
                    confidence=min(max(z_scores[outlier_mask]).item() / 10.0, 1.0),
                    description=f"Statistical outlier detected in {feature}",
                    affected_fields=[feature],
                    suggested_action=f"Review {feature} values for weeks {outlier_weeks}",
                    details={
                        'weeks': outlier_weeks,
                        'values': outlier_values,
                        'z_scores': z_scores[outlier_mask].tolist(),
                        'mean': values.mean(),
                        'std': values.std()
                    }
                ))
            
            # IQR analysis for additional outlier detection
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.iqr_multiplier * IQR
            upper_bound = Q3 + self.iqr_multiplier * IQR
            
            iqr_outliers = (values < lower_bound) | (values > upper_bound)
            
            if iqr_outliers.any() and not outlier_mask.any():  # Don't double-report
                outlier_weeks = df.loc[values.index[iqr_outliers], 'week'].tolist()
                outlier_values = values[iqr_outliers].tolist()
                
                anomalies.append(AnomalyResult(
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    severity=SeverityLevel.LOW,
                    confidence=0.7,
                    description=f"IQR outlier detected in {feature}",
                    affected_fields=[feature],
                    suggested_action=f"Review {feature} values for weeks {outlier_weeks}",
                    details={
                        'weeks': outlier_weeks,
                        'values': outlier_values,
                        'method': 'IQR',
                        'bounds': {'lower': lower_bound, 'upper': upper_bound}
                    }
                ))
        
        return anomalies

    async def _detect_ml_anomalies(self, session: Session, player: Player,
                                 stats: List[WeeklyStats]) -> List[AnomalyResult]:
        """Detect anomalies using machine learning models."""
        
        anomalies = []
        
        # Get comparison data from similar players
        comparison_data = await self._get_comparison_data(session, player)
        
        if len(comparison_data) < 50:  # Need sufficient data for ML
            logger.warning(f"Insufficient comparison data for ML anomaly detection "
                          f"(only {len(comparison_data)} samples)")
            return anomalies
        
        try:
            # Prepare feature matrix
            features = self._prepare_ml_features(stats, player.position)
            comparison_features = self._prepare_ml_features_from_df(comparison_data, player.position)
            
            if len(features) == 0 or len(comparison_features) == 0:
                return anomalies
            
            # Combine data for training
            all_features = np.vstack([comparison_features, features])
            
            # Scale features
            scaled_features = self.robust_scaler.fit_transform(all_features)
            
            # Isolation Forest detection
            outliers = self.isolation_forest.fit_predict(scaled_features)
            player_outliers = outliers[-len(features):]  # Last entries are player data
            
            # Identify anomalous weeks
            anomalous_weeks = []
            for i, is_outlier in enumerate(player_outliers):
                if is_outlier == -1:  # -1 indicates outlier in sklearn
                    week = stats[i].week
                    anomalous_weeks.append(week)
            
            if anomalous_weeks:
                # Calculate anomaly scores
                anomaly_scores = self.isolation_forest.decision_function(
                    scaled_features[-len(features):]
                )
                
                anomalies.append(AnomalyResult(
                    anomaly_type=AnomalyType.ISOLATION_FOREST,
                    severity=SeverityLevel.HIGH,
                    confidence=0.8,
                    description=f"ML anomaly detection identified unusual patterns",
                    affected_fields=['multiple_stats'],
                    suggested_action=f"Review comprehensive stats for weeks {anomalous_weeks}",
                    details={
                        'anomalous_weeks': anomalous_weeks,
                        'anomaly_scores': anomaly_scores.tolist(),
                        'model': 'IsolationForest',
                        'comparison_samples': len(comparison_features)
                    }
                ))
            
            # DBSCAN clustering for additional pattern detection
            if len(scaled_features) >= 10:  # Minimum for meaningful clustering
                cluster_labels = self.dbscan.fit_predict(scaled_features)
                player_labels = cluster_labels[-len(features):]
                
                # Identify outliers (label -1)
                dbscan_outliers = [i for i, label in enumerate(player_labels) if label == -1]
                
                if dbscan_outliers:
                    outlier_weeks = [stats[i].week for i in dbscan_outliers]
                    
                    anomalies.append(AnomalyResult(
                        anomaly_type=AnomalyType.DBSCAN_OUTLIER,
                        severity=SeverityLevel.MEDIUM,
                        confidence=0.75,
                        description=f"Density-based clustering identified anomalous patterns",
                        affected_fields=['performance_patterns'],
                        suggested_action=f"Analyze performance context for weeks {outlier_weeks}",
                        details={
                            'outlier_weeks': outlier_weeks,
                            'cluster_info': {
                                'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                                'n_outliers': len(dbscan_outliers)
                            }
                        }
                    ))
        
        except Exception as e:
            logger.error(f"Error in ML anomaly detection: {e}")
        
        return anomalies

    async def _validate_business_rules(self, player: Player, 
                                     stats: List[WeeklyStats]) -> List[AnomalyResult]:
        """Validate data against fantasy football business rules."""
        
        anomalies = []
        rules = self.business_rules.get(player.position, {})
        
        for stat in stats:
            week_anomalies = []
            
            # Check maximum value rules
            for rule_name, max_value in rules.items():
                if not rule_name.startswith('max_'):
                    continue
                
                field_name = rule_name[4:]  # Remove 'max_' prefix
                
                # Map rule names to actual database fields
                field_mapping = {
                    'passing_yards': stat.passing_yards,
                    'passing_tds': stat.passing_touchdowns,
                    'rushing_yards': stat.rushing_yards,
                    'rushing_attempts': stat.rushing_attempts,
                    'receiving_yards': stat.receiving_yards,
                    'receptions': stat.receptions,
                    'targets': stat.receiving_targets,
                    'receiving_tds': stat.receiving_touchdowns,
                    'interceptions': stat.interceptions,
                    'total_tds': (stat.passing_touchdowns or 0) + (stat.rushing_touchdowns or 0) + (stat.receiving_touchdowns or 0)
                }
                
                if field_name in field_mapping:
                    value = field_mapping[field_name]
                    if value and value > max_value:
                        week_anomalies.append(AnomalyResult(
                            anomaly_type=AnomalyType.BUSINESS_RULE,
                            severity=SeverityLevel.HIGH,
                            confidence=1.0,
                            description=f"{field_name} ({value}) exceeds maximum expected value ({max_value})",
                            affected_fields=[field_name],
                            suggested_action=f"Verify {field_name} value for week {stat.week}",
                            details={
                                'week': stat.week,
                                'value': value,
                                'max_expected': max_value,
                                'rule': rule_name
                            }
                        ))
            
            # Check required fields
            required_fields = rules.get('required_fields', [])
            missing_fields = []
            
            field_values = {
                'passing_attempts': stat.passing_attempts,
                'passing_completions': stat.passing_completions,
                'passing_yards': stat.passing_yards,
                'rushing_attempts': stat.rushing_attempts,
                'rushing_yards': stat.rushing_yards,
                'receiving_targets': stat.receiving_targets,
                'receptions': stat.receptions,
                'receiving_yards': stat.receiving_yards
            }
            
            for field in required_fields:
                if field in field_values and (field_values[field] is None or field_values[field] == 0):
                    # Only flag as missing if player actually played (has some stats)
                    has_any_stats = any(v and v > 0 for v in field_values.values())
                    if has_any_stats:
                        missing_fields.append(field)
            
            if missing_fields:
                week_anomalies.append(AnomalyResult(
                    anomaly_type=AnomalyType.COMPLETENESS_CHECK,
                    severity=SeverityLevel.MEDIUM,
                    confidence=0.9,
                    description=f"Missing expected statistics for {player.position}",
                    affected_fields=missing_fields,
                    suggested_action=f"Check data completeness for week {stat.week}",
                    details={
                        'week': stat.week,
                        'missing_fields': missing_fields,
                        'position': player.position
                    }
                ))
            
            anomalies.extend(week_anomalies)
        
        return anomalies

    async def _check_consistency(self, session: Session, player: Player,
                               stats: List[WeeklyStats]) -> List[AnomalyResult]:
        """Check data consistency and logical relationships."""
        
        anomalies = []
        
        for stat in stats:
            consistency_issues = []
            
            # Passing consistency checks
            if stat.passing_completions and stat.passing_attempts:
                if stat.passing_completions > stat.passing_attempts:
                    consistency_issues.append({
                        'issue': 'passing_completions > passing_attempts',
                        'details': f"Completions: {stat.passing_completions}, Attempts: {stat.passing_attempts}"
                    })
            
            # Receiving consistency checks
            if stat.receptions and stat.receiving_targets:
                if stat.receptions > stat.receiving_targets:
                    consistency_issues.append({
                        'issue': 'receptions > receiving_targets',
                        'details': f"Receptions: {stat.receptions}, Targets: {stat.receiving_targets}"
                    })
            
            # Logical impossibilities
            if stat.receiving_yards and stat.receptions == 0:
                if stat.receiving_yards > 0:
                    consistency_issues.append({
                        'issue': 'receiving_yards without receptions',
                        'details': f"Yards: {stat.receiving_yards}, Receptions: 0"
                    })
            
            if stat.rushing_yards and stat.rushing_attempts == 0:
                if stat.rushing_yards > 0:
                    consistency_issues.append({
                        'issue': 'rushing_yards without attempts',
                        'details': f"Yards: {stat.rushing_yards}, Attempts: 0"
                    })
            
            # Create anomaly results for consistency issues
            for issue in consistency_issues:
                anomalies.append(AnomalyResult(
                    anomaly_type=AnomalyType.CONSISTENCY_CHECK,
                    severity=SeverityLevel.HIGH,
                    confidence=1.0,
                    description=f"Data consistency violation: {issue['issue']}",
                    affected_fields=['consistency'],
                    suggested_action=f"Fix data consistency for week {stat.week}",
                    details={
                        'week': stat.week,
                        'issue': issue['issue'],
                        'details': issue['details']
                    }
                ))
        
        return anomalies

    async def _get_comparison_data(self, session: Session, player: Player) -> pd.DataFrame:
        """Get comparison data from similar players for ML analysis."""
        
        # Get players in same position from recent seasons
        similar_players = session.query(Player).filter(
            Player.position == player.position,
            Player.is_active == True,
            Player.id != player.id
        ).limit(20).all()
        
        # Get their stats
        comparison_stats = []
        for p in similar_players:
            stats = session.query(WeeklyStats).filter(
                WeeklyStats.player_id == p.id,
                WeeklyStats.season.in_([2021, 2022, 2023])
            ).all()
            comparison_stats.extend(stats)
        
        return self._stats_to_dataframe(comparison_stats)

    def _stats_to_dataframe(self, stats: List[WeeklyStats]) -> pd.DataFrame:
        """Convert WeeklyStats objects to DataFrame."""
        
        data = []
        for stat in stats:
            data.append({
                'player_id': stat.player_id,
                'season': stat.season,
                'week': stat.week,
                'passing_attempts': stat.passing_attempts or 0,
                'passing_completions': stat.passing_completions or 0,
                'passing_yards': stat.passing_yards or 0,
                'passing_touchdowns': stat.passing_touchdowns or 0,
                'interceptions': stat.interceptions or 0,
                'rushing_attempts': stat.rushing_attempts or 0,
                'rushing_yards': stat.rushing_yards or 0,
                'rushing_touchdowns': stat.rushing_touchdowns or 0,
                'receiving_targets': stat.receiving_targets or 0,
                'receptions': stat.receptions or 0,
                'receiving_yards': stat.receiving_yards or 0,
                'receiving_touchdowns': stat.receiving_touchdowns or 0,
                'fumbles': stat.fumbles or 0,
                'fumbles_lost': stat.fumbles_lost or 0,
                'fantasy_points_ppr': stat.fantasy_points_ppr or 0
            })
        
        return pd.DataFrame(data)

    def _prepare_ml_features(self, stats: List[WeeklyStats], position: str) -> np.ndarray:
        """Prepare feature matrix for ML models."""
        
        df = self._stats_to_dataframe(stats)
        return self._prepare_ml_features_from_df(df, position)

    def _prepare_ml_features_from_df(self, df: pd.DataFrame, position: str) -> np.ndarray:
        """Prepare feature matrix from DataFrame."""
        
        if df.empty:
            return np.array([])
        
        # Position-specific feature selection
        feature_sets = {
            'QB': ['passing_attempts', 'passing_completions', 'passing_yards', 
                   'passing_touchdowns', 'interceptions', 'rushing_yards'],
            'RB': ['rushing_attempts', 'rushing_yards', 'rushing_touchdowns',
                   'receiving_targets', 'receptions', 'receiving_yards'],
            'WR': ['receiving_targets', 'receptions', 'receiving_yards',
                   'receiving_touchdowns', 'rushing_attempts', 'rushing_yards'],
            'TE': ['receiving_targets', 'receptions', 'receiving_yards',
                   'receiving_touchdowns']
        }
        
        features = feature_sets.get(position, feature_sets['WR'])
        
        # Extract feature columns
        feature_data = df[features].fillna(0)
        
        return feature_data.values

    def _calculate_quality_scores(self, stats: List[WeeklyStats], 
                                anomalies: List[AnomalyResult]) -> QualityMetrics:
        """Calculate comprehensive quality scores."""
        
        # Completeness score
        total_fields = len(stats) * 8  # 8 key statistical fields per week
        missing_fields = sum(
            1 for stat in stats
            for field in [stat.passing_yards, stat.rushing_yards, stat.receiving_yards,
                         stat.passing_touchdowns, stat.rushing_touchdowns, stat.receiving_touchdowns,
                         stat.receptions, stat.rushing_attempts]
            if field is None
        )
        completeness_score = max(0.0, 1.0 - (missing_fields / total_fields))
        
        # Consistency score (based on consistency anomalies)
        consistency_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.CONSISTENCY_CHECK]
        consistency_score = max(0.0, 1.0 - (len(consistency_anomalies) / len(stats)))
        
        # Anomaly score (inverse of anomaly density)
        high_severity_anomalies = [a for a in anomalies if a.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]]
        anomaly_score = len(high_severity_anomalies) / max(len(stats), 1)
        anomaly_score = min(1.0, anomaly_score)  # Cap at 1.0
        
        # Overall score (weighted combination)
        overall_score = (
            0.3 * completeness_score +
            0.4 * consistency_score +
            0.3 * (1.0 - anomaly_score)
        )
        
        return QualityMetrics(
            completeness_score=round(completeness_score, 3),
            consistency_score=round(consistency_score, 3),
            anomaly_score=round(anomaly_score, 3),
            overall_score=round(overall_score, 3),
            anomalies=anomalies,
            validation_timestamp=datetime.now(timezone.utc)
        )

    async def _store_quality_metrics(self, session: Session, player_id: int,
                                   season: int, metrics: QualityMetrics) -> None:
        """Store quality metrics in database."""
        
        try:
            # Get or create quality metric record
            quality_metric = session.query(DataQualityMetric).filter(
                DataQualityMetric.player_id == player_id,
                DataQualityMetric.season == season
            ).first()
            
            if not quality_metric:
                quality_metric = DataQualityMetric(
                    player_id=player_id,
                    season=season
                )
                session.add(quality_metric)
            
            # Update metrics
            quality_metric.completeness_score = metrics.completeness_score
            quality_metric.consistency_score = metrics.consistency_score
            quality_metric.anomaly_score = metrics.anomaly_score
            quality_metric.overall_quality_score = metrics.overall_score
            quality_metric.last_validation = metrics.validation_timestamp
            
            # Determine quality status
            if metrics.overall_score >= 0.8:
                quality_metric.quality_status = DataQualityStatus.VALID.value
            elif metrics.overall_score >= 0.6:
                quality_metric.quality_status = DataQualityStatus.UNKNOWN.value
            elif len([a for a in metrics.anomalies if a.severity == SeverityLevel.CRITICAL]) > 0:
                quality_metric.quality_status = DataQualityStatus.INVALID.value
            else:
                quality_metric.quality_status = DataQualityStatus.ANOMALY.value
            
            # Store anomaly details
            anomaly_summary = {
                'total_anomalies': len(metrics.anomalies),
                'by_type': {},
                'by_severity': {},
                'details': []
            }
            
            for anomaly in metrics.anomalies:
                # Count by type
                anomaly_type = anomaly.anomaly_type.value
                anomaly_summary['by_type'][anomaly_type] = anomaly_summary['by_type'].get(anomaly_type, 0) + 1
                
                # Count by severity
                severity = anomaly.severity.name
                anomaly_summary['by_severity'][severity] = anomaly_summary['by_severity'].get(severity, 0) + 1
                
                # Store details for high-severity anomalies
                if anomaly.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
                    anomaly_summary['details'].append({
                        'type': anomaly_type,
                        'severity': severity,
                        'description': anomaly.description,
                        'affected_fields': anomaly.affected_fields,
                        'confidence': anomaly.confidence
                    })
            
            quality_metric.validation_details = anomaly_summary
            
            # Update anomaly-specific counts
            quality_metric.outlier_weeks = len([
                a for a in metrics.anomalies 
                if a.anomaly_type in [AnomalyType.STATISTICAL_OUTLIER, AnomalyType.ISOLATION_FOREST]
            ])
            
            session.commit()
            
            logger.info(f"Stored quality metrics for player {player_id}, season {season}: "
                       f"overall_score={metrics.overall_score:.3f}, "
                       f"anomalies={len(metrics.anomalies)}")
            
        except Exception as e:
            logger.error(f"Error storing quality metrics: {e}")
            session.rollback()

    async def generate_quality_report(self, player_id: int, season: int) -> Dict[str, Any]:
        """Generate comprehensive data quality report for a player."""
        
        async with get_db_session() as session:
            # Get quality metrics
            quality_metric = session.query(DataQualityMetric).filter(
                DataQualityMetric.player_id == player_id,
                DataQualityMetric.season == season
            ).first()
            
            if not quality_metric:
                return {'error': 'No quality metrics found'}
            
            # Get player info
            player = session.query(Player).get(player_id)
            
            # Get recent stats
            stats = session.query(WeeklyStats).filter(
                WeeklyStats.player_id == player_id,
                WeeklyStats.season == season
            ).order_by(WeeklyStats.week).all()
            
            return {
                'player': {
                    'id': player.id,
                    'name': player.name,
                    'position': player.position,
                    'team': player.team.name if player.team else None
                },
                'season': season,
                'quality_summary': {
                    'overall_score': quality_metric.overall_quality_score,
                    'completeness_score': quality_metric.completeness_score,
                    'consistency_score': quality_metric.consistency_score,
                    'anomaly_score': quality_metric.anomaly_score,
                    'status': quality_metric.quality_status,
                    'last_validation': quality_metric.last_validation.isoformat()
                },
                'data_coverage': {
                    'total_weeks': len(stats),
                    'weeks_with_data': len([s for s in stats if s.fantasy_points_ppr]),
                    'missing_weeks': quality_metric.missing_weeks_count,
                    'zero_stat_weeks': quality_metric.zero_stat_weeks
                },
                'anomaly_details': quality_metric.validation_details,
                'recommendations': self._generate_recommendations(quality_metric)
            }

    def _generate_recommendations(self, quality_metric: DataQualityMetric) -> List[str]:
        """Generate actionable recommendations based on quality metrics."""
        
        recommendations = []
        
        if quality_metric.overall_quality_score < 0.6:
            recommendations.append("Data quality is below acceptable threshold - consider re-collection")
        
        if quality_metric.completeness_score < 0.8:
            recommendations.append("Significant missing data detected - verify data source completeness")
        
        if quality_metric.consistency_score < 0.7:
            recommendations.append("Data consistency issues found - review logical relationships between fields")
        
        if quality_metric.anomaly_score > 0.3:
            recommendations.append("High anomaly rate detected - investigate unusual statistical patterns")
        
        if quality_metric.outlier_weeks > 3:
            recommendations.append("Multiple outlier weeks detected - verify exceptional performance periods")
        
        if not recommendations:
            recommendations.append("Data quality is acceptable for analysis")
        
        return recommendations

# Utility functions for external use
async def validate_player_data(player_id: int, season: int) -> QualityMetrics:
    """Convenience function for validating a single player's data."""
    validator = DataQualityValidator()
    return await validator.validate_player_stats(player_id, season)

async def batch_validate_players(player_ids: List[int], season: int) -> Dict[int, QualityMetrics]:
    """Validate multiple players' data in batch."""
    validator = DataQualityValidator()
    results = {}
    
    for player_id in player_ids:
        try:
            results[player_id] = await validator.validate_player_stats(player_id, season)
        except Exception as e:
            logger.error(f"Error validating player {player_id}: {e}")
            # Continue with other players
    
    return results