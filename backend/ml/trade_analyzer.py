"""
Advanced Trade Analyzer for Multi-Team Fantasy Football Deals
Evaluates complex trades using value-based analysis and projections
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from itertools import combinations
import json

logger = logging.getLogger(__name__)


@dataclass
class Player:
    """Player information for trade analysis"""
    player_id: str
    name: str
    position: str
    team: str
    current_projection: float
    rest_of_season_projection: float
    tier: int
    bye_week: int
    injury_risk: float = 0.2  # 0-1 scale
    age: int = 27
    trade_value: float = 0.0  # Calculated


@dataclass
class TradeProposal:
    """Multi-team trade proposal"""
    trade_id: str
    teams: List[str]  # Team IDs involved
    player_movements: Dict[str, str]  # {player_id: receiving_team}
    draft_picks: Dict[str, List[str]] = field(default_factory=dict)  # {team: [pick_descriptions]}
    proposed_by: str = ""
    proposal_date: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TradeAnalysis:
    """Comprehensive trade analysis result"""
    trade_id: str
    overall_grade: str  # A+, A, B+, B, C+, C, D+, D, F
    fairness_score: float  # 0-1, 1 = perfectly fair
    team_grades: Dict[str, str]  # Grade for each team
    team_value_changes: Dict[str, float]  # Net value change per team
    winner: Optional[str]  # Team that benefits most
    analysis_details: Dict[str, Any]
    recommendations: List[str]
    confidence: float
    risk_factors: List[str]


class AdvancedTradeAnalyzer:
    """
    Advanced trade analyzer supporting:
    1. Multi-team trades (3+ teams)
    2. Complex player + pick combinations
    3. Value-based drafting (VBD) calculations
    4. Positional scarcity analysis
    5. Schedule and matchup considerations
    6. Risk assessment and injury factors
    """
    
    def __init__(self):
        # Value based drafting baselines by position
        self.vbd_baselines = {
            'QB': 18.0,   # Points per game for replacement level
            'RB': 8.0,
            'WR': 7.0,
            'TE': 5.0,
            'K': 6.0,
            'DST': 4.0
        }
        
        # Position scarcity factors (higher = more scarce)
        self.scarcity_factors = {
            'QB': 0.8,   # QBs are abundant
            'RB': 1.4,   # RBs are scarce
            'WR': 1.0,   # WRs are baseline
            'TE': 1.3,   # TEs are somewhat scarce
            'K': 0.6,    # Kickers are very replaceable
            'DST': 0.7   # Defenses are replaceable
        }
        
        # Draft pick values (based on position in draft)
        self.draft_pick_values = self._initialize_pick_values()
        
        # League settings (would be configurable)
        self.league_settings = {
            'teams': 12,
            'roster_spots': {
                'QB': 2, 'RB': 4, 'WR': 4, 'TE': 2, 'K': 1, 'DST': 1, 'BENCH': 6
            },
            'starting_lineup': {
                'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'K': 1, 'DST': 1
            },
            'scoring': 'ppr'
        }
    
    def _initialize_pick_values(self) -> Dict[str, float]:
        """Initialize draft pick trade values"""
        # Simplified pick values - in reality would be more complex
        pick_values = {}
        
        # First round picks
        for pick in range(1, 13):
            pick_values[f'2024_1_{pick}'] = 100 - (pick * 3)
        
        # Second round picks
        for pick in range(1, 13):
            pick_values[f'2024_2_{pick}'] = 70 - (pick * 2)
        
        # Third round picks
        for pick in range(1, 13):
            pick_values[f'2024_3_{pick}'] = 40 - pick
        
        # Later rounds have diminishing value
        for round_num in range(4, 17):
            for pick in range(1, 13):
                pick_values[f'2024_{round_num}_{pick}'] = max(1, 25 - round_num - pick)
        
        return pick_values
    
    def analyze_trade(
        self,
        trade_proposal: TradeProposal,
        team_rosters: Dict[str, List[Player]],  # {team_id: [players]}
        include_projections: bool = True,
        include_schedule_analysis: bool = True
    ) -> TradeAnalysis:
        """
        Analyze a multi-team trade proposal
        
        Args:
            trade_proposal: The proposed trade
            team_rosters: Current rosters for all teams
            include_projections: Include rest-of-season projections
            include_schedule_analysis: Include schedule/matchup analysis
        
        Returns:
            Comprehensive trade analysis
        """
        logger.info(f"Analyzing trade {trade_proposal.trade_id} involving {len(trade_proposal.teams)} teams")
        
        # Calculate player values
        player_values = self._calculate_player_values(team_rosters, include_projections)
        
        # Calculate team value changes
        team_value_changes = self._calculate_team_value_changes(
            trade_proposal, player_values, team_rosters
        )
        
        # Analyze positional needs
        positional_analysis = self._analyze_positional_needs(
            trade_proposal, team_rosters
        )
        
        # Calculate fairness score
        fairness_score = self._calculate_fairness_score(team_value_changes)
        
        # Determine winner and grades
        winner = self._determine_trade_winner(team_value_changes)
        team_grades = self._assign_team_grades(team_value_changes, positional_analysis)
        overall_grade = self._calculate_overall_grade(fairness_score, team_grades)
        
        # Risk assessment
        risk_factors = self._assess_trade_risks(trade_proposal, team_rosters, player_values)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            trade_proposal, team_value_changes, positional_analysis, risk_factors
        )
        
        # Schedule analysis (if requested)
        schedule_impact = {}
        if include_schedule_analysis:
            schedule_impact = self._analyze_schedule_impact(trade_proposal, team_rosters)
        
        # Calculate confidence
        confidence = self._calculate_analysis_confidence(
            trade_proposal, len(team_rosters), fairness_score
        )
        
        # Compile detailed analysis
        analysis_details = {
            'player_values': {pid: pval for pid, pval in player_values.items() 
                            if pid in trade_proposal.player_movements},
            'positional_analysis': positional_analysis,
            'schedule_impact': schedule_impact,
            'value_breakdown': self._create_value_breakdown(trade_proposal, player_values),
            'trade_timeline_impact': self._analyze_timeline_impact(trade_proposal)
        }
        
        return TradeAnalysis(
            trade_id=trade_proposal.trade_id,
            overall_grade=overall_grade,
            fairness_score=fairness_score,
            team_grades=team_grades,
            team_value_changes=team_value_changes,
            winner=winner,
            analysis_details=analysis_details,
            recommendations=recommendations,
            confidence=confidence,
            risk_factors=risk_factors
        )
    
    def _calculate_player_values(
        self, team_rosters: Dict[str, List[Player]], include_projections: bool
    ) -> Dict[str, float]:
        """Calculate trade values for all players"""
        player_values = {}
        
        # Collect all players
        all_players = []
        for roster in team_rosters.values():
            all_players.extend(roster)
        
        # Calculate VBD (Value Based Drafting) values
        for player in all_players:
            baseline = self.vbd_baselines.get(player.position, 5.0)
            scarcity = self.scarcity_factors.get(player.position, 1.0)
            
            # Base value calculation
            if include_projections:
                points_above_replacement = player.rest_of_season_projection - baseline
            else:
                points_above_replacement = player.current_projection - baseline
            
            # Apply scarcity multiplier
            base_value = points_above_replacement * scarcity
            
            # Age adjustment (prime ages get bonus)
            age_factor = 1.0
            if 24 <= player.age <= 28:
                age_factor = 1.1
            elif player.age > 32:
                age_factor = 0.85
            
            # Injury risk adjustment
            injury_factor = 1.0 - (player.injury_risk * 0.3)
            
            # Tier adjustment (elite players get bonus)
            tier_factor = 1.0
            if player.tier <= 2:
                tier_factor = 1.2
            elif player.tier <= 5:
                tier_factor = 1.1
            elif player.tier >= 12:
                tier_factor = 0.9
            
            # Final value calculation
            final_value = base_value * age_factor * injury_factor * tier_factor
            player_values[player.player_id] = max(0, final_value)
        
        return player_values
    
    def _calculate_team_value_changes(
        self,
        trade_proposal: TradeProposal,
        player_values: Dict[str, float],
        team_rosters: Dict[str, List[Player]]
    ) -> Dict[str, float]:
        """Calculate net value change for each team"""
        team_changes = {team: 0.0 for team in trade_proposal.teams}
        
        # Player value changes
        for player_id, receiving_team in trade_proposal.player_movements.items():
            player_value = player_values.get(player_id, 0)
            
            # Find current team
            current_team = None
            for team, roster in team_rosters.items():
                if any(p.player_id == player_id for p in roster):
                    current_team = team
                    break
            
            if current_team and current_team in team_changes:
                team_changes[current_team] -= player_value  # Losing player
                team_changes[receiving_team] += player_value  # Gaining player
        
        # Draft pick value changes
        for team, picks in trade_proposal.draft_picks.items():
            for pick in picks:
                pick_value = self.draft_pick_values.get(pick, 5.0)
                team_changes[team] += pick_value
        
        return team_changes
    
    def _analyze_positional_needs(
        self, trade_proposal: TradeProposal, team_rosters: Dict[str, List[Player]]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze how trade addresses positional needs"""
        analysis = {}
        
        for team in trade_proposal.teams:
            roster = team_rosters.get(team, [])
            
            # Count current positions
            position_counts = {}
            position_quality = {}
            
            for player in roster:
                pos = player.position
                position_counts[pos] = position_counts.get(pos, 0) + 1
                
                if pos not in position_quality:
                    position_quality[pos] = []
                position_quality[pos].append(player.tier)
            
            # Analyze changes from trade
            players_gained = []
            players_lost = []
            
            for player_id, receiving_team in trade_proposal.player_movements.items():
                if receiving_team == team:
                    # Find player details
                    for roster_team, roster_players in team_rosters.items():
                        for p in roster_players:
                            if p.player_id == player_id:
                                players_gained.append(p)
                                break
                elif any(p.player_id == player_id for p in roster):
                    # Player is leaving this team
                    for p in roster:
                        if p.player_id == player_id:
                            players_lost.append(p)
                            break
            
            analysis[team] = {
                'current_needs': self._identify_positional_needs(position_counts, position_quality),
                'players_gained': [{'name': p.name, 'position': p.position, 'tier': p.tier} for p in players_gained],
                'players_lost': [{'name': p.name, 'position': p.position, 'tier': p.tier} for p in players_lost],
                'need_fulfillment': self._calculate_need_fulfillment(
                    position_counts, players_gained, players_lost
                )
            }
        
        return analysis
    
    def _identify_positional_needs(
        self, position_counts: Dict[str, int], position_quality: Dict[str, List[int]]
    ) -> List[str]:
        """Identify positional needs based on roster construction"""
        needs = []
        
        required_positions = self.league_settings['roster_spots']
        
        for position, required_count in required_positions.items():
            if position == 'BENCH':
                continue
                
            current_count = position_counts.get(position, 0)
            
            # Check quantity need
            if current_count < required_count:
                needs.append(f"Need {required_count - current_count} more {position}")
            
            # Check quality need
            elif current_count >= 1:
                avg_tier = np.mean(position_quality.get(position, [10]))
                if avg_tier > 8:  # Poor quality
                    needs.append(f"Need higher quality {position} (current avg tier: {avg_tier:.1f})")
        
        return needs
    
    def _calculate_need_fulfillment(
        self, position_counts: Dict[str, int], gained: List[Player], lost: List[Player]
    ) -> Dict[str, float]:
        """Calculate how well trade fulfills positional needs"""
        fulfillment = {}
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            current = position_counts.get(position, 0)
            gained_at_pos = len([p for p in gained if p.position == position])
            lost_at_pos = len([p for p in lost if p.position == position])
            
            net_change = gained_at_pos - lost_at_pos
            
            if current < 2 and net_change > 0:  # Addressing need
                fulfillment[position] = 1.0
            elif current >= 4 and net_change > 0:  # Adding depth
                fulfillment[position] = 0.5
            elif net_change < 0:  # Creating need
                fulfillment[position] = -0.5
            else:
                fulfillment[position] = 0.0
        
        return fulfillment
    
    def _calculate_fairness_score(self, team_value_changes: Dict[str, float]) -> float:
        """Calculate how fair the trade is (0-1, 1 = perfectly fair)"""
        if not team_value_changes:
            return 0.5
        
        values = list(team_value_changes.values())
        
        # Calculate standard deviation of value changes
        std_dev = np.std(values)
        
        # Convert to fairness score (lower std dev = more fair)
        fairness = max(0, min(1, 1 - (std_dev / 20)))  # 20 is arbitrary scaling factor
        
        return fairness
    
    def _determine_trade_winner(self, team_value_changes: Dict[str, float]) -> Optional[str]:
        """Determine which team benefits most from the trade"""
        if not team_value_changes:
            return None
        
        max_gain = max(team_value_changes.values())
        min_gain = min(team_value_changes.values())
        
        # If difference is significant, declare a winner
        if max_gain - min_gain > 10:  # Arbitrary threshold
            return max(team_value_changes, key=team_value_changes.get)
        
        return None  # Too close to call
    
    def _assign_team_grades(
        self, team_value_changes: Dict[str, float], positional_analysis: Dict[str, Any]
    ) -> Dict[str, str]:
        """Assign letter grades to each team"""
        grades = {}
        
        for team, value_change in team_value_changes.items():
            # Base grade on value change
            if value_change >= 15:
                base_grade = 'A'
            elif value_change >= 8:
                base_grade = 'B'
            elif value_change >= 0:
                base_grade = 'C'
            elif value_change >= -8:
                base_grade = 'D'
            else:
                base_grade = 'F'
            
            # Adjust based on need fulfillment
            pos_analysis = positional_analysis.get(team, {})
            need_fulfillment = pos_analysis.get('need_fulfillment', {})
            
            avg_fulfillment = np.mean(list(need_fulfillment.values())) if need_fulfillment else 0
            
            if avg_fulfillment > 0.5:
                # Upgrade grade
                if base_grade == 'B':
                    grades[team] = 'A-'
                elif base_grade == 'C':
                    grades[team] = 'B-'
                elif base_grade == 'D':
                    grades[team] = 'C-'
                else:
                    grades[team] = base_grade
            elif avg_fulfillment < -0.3:
                # Downgrade grade
                if base_grade == 'A':
                    grades[team] = 'B+'
                elif base_grade == 'B':
                    grades[team] = 'C+'
                elif base_grade == 'C':
                    grades[team] = 'D+'
                else:
                    grades[team] = base_grade
            else:
                grades[team] = base_grade
        
        return grades
    
    def _calculate_overall_grade(
        self, fairness_score: float, team_grades: Dict[str, str]
    ) -> str:
        """Calculate overall trade grade"""
        # Convert letter grades to numbers
        grade_values = {
            'A+': 97, 'A': 94, 'A-': 91,
            'B+': 87, 'B': 84, 'B-': 81,
            'C+': 77, 'C': 74, 'C-': 71,
            'D+': 67, 'D': 64, 'D-': 61,
            'F': 50
        }
        
        if not team_grades:
            return 'C'
        
        avg_grade_value = np.mean([grade_values.get(grade, 70) for grade in team_grades.values()])
        
        # Adjust based on fairness
        fairness_bonus = (fairness_score - 0.5) * 10  # -5 to +5 adjustment
        
        final_value = avg_grade_value + fairness_bonus
        
        # Convert back to letter grade
        if final_value >= 95:
            return 'A+'
        elif final_value >= 90:
            return 'A'
        elif final_value >= 87:
            return 'A-'
        elif final_value >= 84:
            return 'B+'
        elif final_value >= 80:
            return 'B'
        elif final_value >= 77:
            return 'B-'
        elif final_value >= 74:
            return 'C+'
        elif final_value >= 70:
            return 'C'
        elif final_value >= 67:
            return 'C-'
        elif final_value >= 64:
            return 'D+'
        elif final_value >= 60:
            return 'D'
        else:
            return 'F'
    
    def _assess_trade_risks(
        self,
        trade_proposal: TradeProposal,
        team_rosters: Dict[str, List[Player]],
        player_values: Dict[str, float]
    ) -> List[str]:
        """Assess potential risks in the trade"""
        risks = []
        
        # Check for injury-prone players
        for player_id in trade_proposal.player_movements:
            for roster in team_rosters.values():
                for player in roster:
                    if player.player_id == player_id and player.injury_risk > 0.4:
                        risks.append(f"{player.name} has high injury risk ({player.injury_risk:.1%})")
        
        # Check for age-related concerns
        for player_id in trade_proposal.player_movements:
            for roster in team_rosters.values():
                for player in roster:
                    if player.player_id == player_id and player.age > 32:
                        risks.append(f"{player.name} is aging (age {player.age}) - decline risk")
        
        # Check for over-concentration in one position
        for team in trade_proposal.teams:
            roster = team_rosters.get(team, [])
            gained_positions = []
            
            for player_id, receiving_team in trade_proposal.player_movements.items():
                if receiving_team == team:
                    for roster_team, roster_players in team_rosters.items():
                        for p in roster_players:
                            if p.player_id == player_id:
                                gained_positions.append(p.position)
                                break
            
            # Count current + gained by position
            from collections import Counter
            current_positions = Counter(p.position for p in roster)
            gained_counter = Counter(gained_positions)
            
            for pos, gained_count in gained_counter.items():
                total_after_trade = current_positions.get(pos, 0) + gained_count
                max_useful = self.league_settings['roster_spots'].get(pos, 2)
                
                if total_after_trade > max_useful + 1:
                    risks.append(f"Team {team} may have too many {pos}s after trade")
        
        # Check for unbalanced value
        team_values = list(trade_proposal.player_movements.values())
        value_spread = max(team_values) - min(team_values) if len(set(team_values)) > 1 else 0
        
        if value_spread > 25:
            risks.append("Significantly unbalanced trade - may be unfair to one party")
        
        return risks
    
    def _generate_recommendations(
        self,
        trade_proposal: TradeProposal,
        team_value_changes: Dict[str, float],
        positional_analysis: Dict[str, Any],
        risk_factors: List[str]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Overall recommendation
        fairness = self._calculate_fairness_score(team_value_changes)
        
        if fairness > 0.8:
            recommendations.append("Trade appears fair and beneficial for all parties")
        elif fairness > 0.6:
            recommendations.append("Reasonably balanced trade with minor disparities")
        else:
            recommendations.append("Trade appears unbalanced - consider additional compensation")
        
        # Team-specific recommendations
        for team, value_change in team_value_changes.items():
            if value_change > 10:
                recommendations.append(f"Team {team} should accept - significant value gain")
            elif value_change < -10:
                recommendations.append(f"Team {team} should reconsider - significant value loss")
        
        # Position-specific recommendations
        for team, analysis in positional_analysis.items():
            unfulfilled_needs = [need for need in analysis.get('current_needs', []) 
                               if 'Need' in need]
            if unfulfilled_needs:
                recommendations.append(f"Team {team} should address: {unfulfilled_needs[0]}")
        
        # Risk-based recommendations
        high_risk_count = len([r for r in risk_factors if 'high' in r.lower()])
        if high_risk_count > 0:
            recommendations.append("Consider injury insurance or handcuff players")
        
        return recommendations
    
    def _analyze_schedule_impact(
        self, trade_proposal: TradeProposal, team_rosters: Dict[str, List[Player]]
    ) -> Dict[str, Any]:
        """Analyze schedule impact of trade (simplified implementation)"""
        # Mock implementation - would integrate with actual schedule data
        return {
            'playoff_schedule': 'Trade may improve playoff matchups for involved teams',
            'bye_week_coverage': 'Addresses bye week concerns for teams involved',
            'strength_of_schedule': 'Neutral impact on remaining schedule difficulty'
        }
    
    def _calculate_analysis_confidence(
        self, trade_proposal: TradeProposal, num_teams: int, fairness_score: float
    ) -> float:
        """Calculate confidence in the analysis"""
        base_confidence = 0.75
        
        # More confidence with more data
        if num_teams >= 10:
            base_confidence += 0.1
        
        # More confidence with fairer trades
        base_confidence += (fairness_score - 0.5) * 0.2
        
        # Less confidence with complex multi-team trades
        if len(trade_proposal.teams) > 2:
            base_confidence -= 0.1
        
        return min(0.95, max(0.5, base_confidence))
    
    def _create_value_breakdown(
        self, trade_proposal: TradeProposal, player_values: Dict[str, float]
    ) -> Dict[str, Any]:
        """Create detailed value breakdown"""
        breakdown = {}
        
        for team in trade_proposal.teams:
            team_breakdown = {
                'players_gained': [],
                'players_lost': [],
                'picks_gained': trade_proposal.draft_picks.get(team, []),
                'net_player_value': 0,
                'net_pick_value': 0
            }
            
            for player_id, receiving_team in trade_proposal.player_movements.items():
                player_value = player_values.get(player_id, 0)
                
                if receiving_team == team:
                    team_breakdown['players_gained'].append({
                        'player_id': player_id,
                        'value': player_value
                    })
                    team_breakdown['net_player_value'] += player_value
                
                # Find if this team is losing the player
                # (simplified - would need actual roster data)
            
            # Calculate pick values
            for pick in team_breakdown['picks_gained']:
                pick_value = self.draft_pick_values.get(pick, 5.0)
                team_breakdown['net_pick_value'] += pick_value
            
            breakdown[team] = team_breakdown
        
        return breakdown
    
    def _analyze_timeline_impact(self, trade_proposal: TradeProposal) -> Dict[str, str]:
        """Analyze when trade impact will be felt"""
        return {
            'immediate_impact': 'Trade takes effect immediately for upcoming games',
            'playoff_impact': 'May significantly affect playoff positioning',
            'future_impact': 'Draft picks provide future value beyond current season'
        }
    
    def suggest_trade_improvements(
        self, analysis: TradeAnalysis, team_rosters: Dict[str, List[Player]]
    ) -> List[str]:
        """Suggest ways to improve the trade"""
        suggestions = []
        
        if analysis.fairness_score < 0.6:
            # Find the team that's losing the most value
            losing_team = min(analysis.team_value_changes, key=analysis.team_value_changes.get)
            losing_value = analysis.team_value_changes[losing_team]
            
            if losing_value < -10:
                suggestions.append(f"Team {losing_team} should receive additional compensation worth ~{abs(losing_value):.0f} points")
                suggestions.append("Consider adding a draft pick or lesser player to balance the trade")
        
        # Check for positional imbalances
        for team, details in analysis.analysis_details['positional_analysis'].items():
            if len(details['current_needs']) > 0:
                suggestions.append(f"Team {team} could address positional needs: {details['current_needs'][0]}")
        
        return suggestions


# Example usage
if __name__ == "__main__":
    # Create sample data
    analyzer = AdvancedTradeAnalyzer()
    
    # Sample players
    players = [
        Player("p1", "Christian McCaffrey", "RB", "SF", 18.5, 85.2, 1, 7),
        Player("p2", "Tyreek Hill", "WR", "MIA", 16.8, 78.3, 2, 11),
        Player("p3", "Josh Jacobs", "RB", "LV", 14.2, 65.8, 5, 6),
        Player("p4", "DK Metcalf", "WR", "SEA", 13.1, 60.7, 7, 5),
    ]
    
    # Sample rosters
    team_rosters = {
        "team1": [players[0], players[2]],
        "team2": [players[1], players[3]]
    }
    
    # Sample trade proposal
    trade = TradeProposal(
        trade_id="trade_001",
        teams=["team1", "team2"],
        player_movements={
            "p1": "team2",  # CMC to team2
            "p2": "team1",  # Tyreek to team1
            "p4": "team1"   # DK to team1
        }
    )
    
    # Analyze trade
    analysis = analyzer.analyze_trade(trade, team_rosters)
    
    print(f"Trade Analysis for {trade.trade_id}")
    print(f"Overall Grade: {analysis.overall_grade}")
    print(f"Fairness Score: {analysis.fairness_score:.2f}")
    print(f"Winner: {analysis.winner or 'Too close to call'}")
    print(f"\nTeam Grades:")
    for team, grade in analysis.team_grades.items():
        value_change = analysis.team_value_changes.get(team, 0)
        print(f"  {team}: {grade} ({value_change:+.1f} value)")
    
    print(f"\nRecommendations:")
    for rec in analysis.recommendations:
        print(f"  - {rec}")
    
    if analysis.risk_factors:
        print(f"\nRisk Factors:")
        for risk in analysis.risk_factors:
            print(f"  - {risk}")