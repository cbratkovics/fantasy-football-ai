#!/usr/bin/env python3
"""Test data quality and integrity for Fantasy Football AI MVP"""

import sys
import os
from pathlib import Path
import json
import requests
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from sqlalchemy import create_engine, text

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.core.config import settings


class DataQualityTester:
    """Run comprehensive data quality tests"""
    
    def __init__(self):
        self.engine = create_engine(settings.DATABASE_URL)
        self.api_base = "http://localhost:8000"
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "details": []
        }
    
    def log_result(self, test_name: str, passed: bool, message: str = "", warning: bool = False):
        """Log test result"""
        status = "WARNING" if warning else ("PASSED" if passed else "FAILED")
        
        if warning:
            self.test_results["warnings"] += 1
        elif passed:
            self.test_results["passed"] += 1
        else:
            self.test_results["failed"] += 1
        
        self.test_results["details"].append({
            "test": test_name,
            "status": status,
            "message": message
        })
        
        print(f"[{status}] {test_name}: {message}")
    
    def test_database_connectivity(self):
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                self.log_result("Database Connection", True, "Connected successfully")
        except Exception as e:
            self.log_result("Database Connection", False, str(e))
            return False
        return True
    
    def test_player_data_completeness(self):
        """Test player data completeness"""
        query = """
        SELECT 
            COUNT(*) as total_players,
            COUNT(DISTINCT position) as positions,
            COUNT(DISTINCT team) as teams,
            COUNT(CASE WHEN age = 0 OR age IS NULL THEN 1 END) as missing_age,
            COUNT(CASE WHEN team = 'FA' OR team IS NULL THEN 1 END) as free_agents
        FROM players
        WHERE status = 'Active'
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query)).fetchone()
            
            # Test total players
            if result.total_players > 500:
                self.log_result("Player Count", True, f"Found {result.total_players} active players")
            else:
                self.log_result("Player Count", False, f"Only {result.total_players} players (expected >500)")
            
            # Test positions
            if result.positions >= 6:
                self.log_result("Position Coverage", True, f"Found {result.positions} positions")
            else:
                self.log_result("Position Coverage", False, f"Only {result.positions} positions")
            
            # Test teams
            if result.teams >= 32:
                self.log_result("Team Coverage", True, f"Found {result.teams} teams")
            else:
                self.log_result("Team Coverage", False, f"Only {result.teams} teams")
            
            # Check data quality
            if result.missing_age < result.total_players * 0.1:
                self.log_result("Player Age Data", True, f"{result.missing_age} players missing age")
            else:
                self.log_result("Player Age Data", False, f"Too many missing ages: {result.missing_age}")
    
    def test_weekly_stats_coverage(self):
        """Test weekly stats data coverage"""
        query = """
        SELECT 
            COUNT(DISTINCT player_id) as players_with_stats,
            COUNT(DISTINCT week) as weeks_covered,
            MIN(week) as first_week,
            MAX(week) as last_week,
            COUNT(*) as total_entries,
            AVG(fantasy_points_ppr) as avg_ppr_points
        FROM weekly_stats
        WHERE season = 2024
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query)).fetchone()
            
            # Test coverage
            if result.weeks_covered >= 17:
                self.log_result("Week Coverage", True, f"Weeks {result.first_week}-{result.last_week} covered")
            else:
                self.log_result("Week Coverage", False, f"Only {result.weeks_covered} weeks covered")
            
            # Test player coverage
            if result.players_with_stats > 300:
                self.log_result("Player Stats Coverage", True, f"{result.players_with_stats} players have stats")
            else:
                self.log_result("Player Stats Coverage", False, f"Only {result.players_with_stats} players have stats")
            
            # Test data reasonableness
            if 5 < result.avg_ppr_points < 15:
                self.log_result("Average PPR Points", True, f"Avg: {result.avg_ppr_points:.1f} (reasonable)")
            else:
                self.log_result("Average PPR Points", False, f"Avg: {result.avg_ppr_points:.1f} (unexpected)")
    
    def test_scoring_calculations(self):
        """Test scoring calculations are reasonable"""
        query = """
        WITH player_averages AS (
            SELECT 
                p.position,
                AVG(ws.fantasy_points_standard) as avg_standard,
                AVG(ws.fantasy_points_ppr) as avg_ppr,
                AVG(ws.fantasy_points_half_ppr) as avg_half_ppr,
                COUNT(*) as sample_size
            FROM players p
            JOIN weekly_stats ws ON p.id = ws.player_id
            WHERE ws.season = 2024 AND p.status = 'Active'
            GROUP BY p.position
        )
        SELECT * FROM player_averages
        ORDER BY position
        """
        
        expected_ranges = {
            "QB": (15, 25),
            "RB": (8, 15),
            "WR": (8, 14),
            "TE": (5, 10),
            "K": (6, 10),
            "DEF": (6, 10)
        }
        
        with self.engine.connect() as conn:
            results = conn.execute(text(query)).fetchall()
            
            print("\nPosition Average Points:")
            for row in results:
                pos = row.position
                if pos in expected_ranges:
                    min_exp, max_exp = expected_ranges[pos]
                    
                    if min_exp <= row.avg_ppr <= max_exp:
                        self.log_result(
                            f"{pos} Scoring", 
                            True, 
                            f"PPR avg: {row.avg_ppr:.1f} (expected {min_exp}-{max_exp})"
                        )
                    else:
                        self.log_result(
                            f"{pos} Scoring", 
                            False, 
                            f"PPR avg: {row.avg_ppr:.1f} (expected {min_exp}-{max_exp})"
                        )
                    
                    # Verify PPR > Half-PPR > Standard
                    if row.avg_ppr > row.avg_half_ppr > row.avg_standard:
                        self.log_result(f"{pos} Scoring Order", True, "PPR > Half-PPR > Standard")
                    else:
                        self.log_result(f"{pos} Scoring Order", False, "Scoring order incorrect")
    
    def test_api_endpoints(self):
        """Test API endpoints return valid data"""
        
        # Test rankings endpoint
        try:
            response = requests.get(f"{self.api_base}/players/rankings?limit=10")
            if response.status_code == 200:
                data = response.json()
                if len(data) > 0 and "sleeper_id" in data[0]:
                    self.log_result("API Rankings", True, f"Returned {len(data)} players")
                else:
                    self.log_result("API Rankings", False, "Invalid response format")
            else:
                self.log_result("API Rankings", False, f"Status code: {response.status_code}")
        except Exception as e:
            self.log_result("API Rankings", False, str(e))
        
        # Test position filtering
        for position in ["QB", "RB", "WR", "TE"]:
            try:
                response = requests.get(f"{self.api_base}/players/rankings?position={position}&limit=5")
                if response.status_code == 200:
                    data = response.json()
                    if all(p["position"] == position for p in data):
                        self.log_result(f"API {position} Filter", True, "Position filter working")
                    else:
                        self.log_result(f"API {position} Filter", False, "Filter not working correctly")
            except Exception as e:
                self.log_result(f"API {position} Filter", False, str(e))
    
    def test_data_freshness(self):
        """Check if data appears to be from 2024 season"""
        query = """
        SELECT 
            MAX(week) as latest_week,
            COUNT(DISTINCT week) as total_weeks,
            MAX(updated_at) as last_update
        FROM weekly_stats
        WHERE season = 2024
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query)).fetchone()
            
            if result.total_weeks >= 17:
                self.log_result("Data Completeness", True, f"Full season data: {result.total_weeks} weeks")
            else:
                self.log_result("Data Completeness", False, f"Incomplete season: only {result.total_weeks} weeks")
            
            if result.last_update:
                days_old = (datetime.utcnow() - result.last_update).days
                if days_old < 7:
                    self.log_result("Data Freshness", True, f"Last updated {days_old} days ago")
                else:
                    self.log_result("Data Freshness", True, f"Data is {days_old} days old", warning=True)
    
    def test_top_players_sanity_check(self):
        """Verify top players are recognizable names"""
        query = """
        SELECT 
            p.full_name,
            p.position,
            p.team,
            ROUND(AVG(ws.fantasy_points_ppr), 1) as avg_ppr
        FROM players p
        JOIN weekly_stats ws ON p.id = ws.player_id
        WHERE ws.season = 2024 AND p.position IN ('QB', 'RB', 'WR')
        GROUP BY p.id, p.full_name, p.position, p.team
        HAVING COUNT(ws.id) >= 10
        ORDER BY avg_ppr DESC
        LIMIT 10
        """
        
        # Known elite players from 2024
        elite_players = [
            "Christian McCaffrey", "Tyreek Hill", "Josh Allen", 
            "Patrick Mahomes", "Justin Jefferson", "Travis Kelce",
            "Jalen Hurts", "Stefon Diggs", "Austin Ekeler"
        ]
        
        with self.engine.connect() as conn:
            results = conn.execute(text(query)).fetchall()
            
            print("\nTop 10 Players by PPR Average:")
            found_elite = 0
            for i, row in enumerate(results):
                print(f"{i+1}. {row.full_name} ({row.position}, {row.team}): {row.avg_ppr} pts")
                
                if any(elite in row.full_name for elite in elite_players):
                    found_elite += 1
            
            if found_elite >= 3:
                self.log_result("Elite Players Check", True, f"Found {found_elite} known elite players in top 10")
            else:
                self.log_result("Elite Players Check", False, f"Only {found_elite} elite players in top 10", warning=True)
    
    def generate_summary(self):
        """Generate test summary"""
        print("\n" + "="*50)
        print("DATA QUALITY TEST SUMMARY")
        print("="*50)
        print(f"Tests Passed: {self.test_results['passed']}")
        print(f"Tests Failed: {self.test_results['failed']}")
        print(f"Warnings: {self.test_results['warnings']}")
        
        total_tests = self.test_results['passed'] + self.test_results['failed']
        if total_tests > 0:
            success_rate = (self.test_results['passed'] / total_tests) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        
        if self.test_results['failed'] == 0:
            print("\nSTATUS: Data quality checks PASSED! Ready for production.")
        else:
            print("\nSTATUS: Some tests FAILED. Please review and fix issues.")
        
        # Save detailed results
        with open("logs/data_quality_test_results.json", "w") as f:
            json.dump(self.test_results, f, indent=2, default=str)
            print(f"\nDetailed results saved to: logs/data_quality_test_results.json")


def main():
    """Run all data quality tests"""
    print("Fantasy Football AI - Data Quality Testing")
    print("=" * 50)
    print(f"Started at: {datetime.now()}")
    
    tester = DataQualityTester()
    
    # Run all tests
    if tester.test_database_connectivity():
        tester.test_player_data_completeness()
        tester.test_weekly_stats_coverage()
        tester.test_scoring_calculations()
        tester.test_api_endpoints()
        tester.test_data_freshness()
        tester.test_top_players_sanity_check()
    
    # Generate summary
    tester.generate_summary()


if __name__ == "__main__":
    main()