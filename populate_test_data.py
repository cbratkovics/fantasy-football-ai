"""
Populate Test Data Script
Creates sample NFL data to test ML integration without needing API calls.
"""

import sys
import asyncio
import random
from pathlib import Path
from datetime import datetime

# Add your src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

async def populate_test_data():
    """Create sample NFL data for testing ML integration."""
    
    print("Fantasy Football AI - Populating Test Data")
    print("=" * 50)
    
    try:
        from fantasy_ai.core.data.storage.simple_database import get_simple_db_manager
        from fantasy_ai.core.data.storage.models import Team, Player, WeeklyStats
        
        db_manager = get_simple_db_manager()
        
        # Ensure tables exist
        await db_manager.create_tables(drop_existing=False)
        
        async with db_manager.get_session() as session:
            print("Creating sample teams...")
            
            # Create sample teams
            teams_data = [
                {"id": 1, "api_id": 1, "name": "Kansas City Chiefs", "code": "KC", "city": "Kansas City"},
                {"id": 2, "api_id": 2, "name": "Buffalo Bills", "code": "BUF", "city": "Buffalo"},
                {"id": 3, "api_id": 3, "name": "Philadelphia Eagles", "code": "PHI", "city": "Philadelphia"},
                {"id": 4, "api_id": 4, "name": "San Francisco 49ers", "code": "SF", "city": "San Francisco"},
                {"id": 5, "api_id": 5, "name": "Dallas Cowboys", "code": "DAL", "city": "Dallas"},
            ]
            
            for team_data in teams_data:
                team = Team(**team_data)
                session.add(team)
            
            print("Creating sample players...")
            
            # Create sample players
            players_data = [
                # QBs
                {"id": 1, "api_id": 101, "team_id": 1, "name": "Patrick Mahomes", "position": "QB"},
                {"id": 2, "api_id": 102, "team_id": 2, "name": "Josh Allen", "position": "QB"},
                {"id": 3, "api_id": 103, "team_id": 3, "name": "Jalen Hurts", "position": "QB"},
                {"id": 4, "api_id": 104, "team_id": 4, "name": "Brock Purdy", "position": "QB"},
                {"id": 5, "api_id": 105, "team_id": 5, "name": "Dak Prescott", "position": "QB"},
                
                # RBs
                {"id": 6, "api_id": 201, "team_id": 1, "name": "Isiah Pacheco", "position": "RB"},
                {"id": 7, "api_id": 202, "team_id": 2, "name": "James Cook", "position": "RB"},
                {"id": 8, "api_id": 203, "team_id": 3, "name": "D'Andre Swift", "position": "RB"},
                {"id": 9, "api_id": 204, "team_id": 4, "name": "Christian McCaffrey", "position": "RB"},
                {"id": 10, "api_id": 205, "team_id": 5, "name": "Tony Pollard", "position": "RB"},
                
                # WRs
                {"id": 11, "api_id": 301, "team_id": 1, "name": "Travis Kelce", "position": "WR"},
                {"id": 12, "api_id": 302, "team_id": 2, "name": "Stefon Diggs", "position": "WR"},
                {"id": 13, "api_id": 303, "team_id": 3, "name": "A.J. Brown", "position": "WR"},
                {"id": 14, "api_id": 304, "team_id": 4, "name": "Deebo Samuel", "position": "WR"},
                {"id": 15, "api_id": 305, "team_id": 5, "name": "CeeDee Lamb", "position": "WR"},
                
                # TEs
                {"id": 16, "api_id": 401, "team_id": 1, "name": "Travis Kelce", "position": "TE"},
                {"id": 17, "api_id": 402, "team_id": 2, "name": "Dalton Kincaid", "position": "TE"},
                {"id": 18, "api_id": 403, "team_id": 3, "name": "Dallas Goedert", "position": "TE"},
                {"id": 19, "api_id": 404, "team_id": 4, "name": "George Kittle", "position": "TE"},
                {"id": 20, "api_id": 405, "team_id": 5, "name": "Jake Ferguson", "position": "TE"},
            ]
            
            for player_data in players_data:
                player = Player(**player_data)
                session.add(player)
            
            print("Creating sample weekly stats...")
            
            # Create realistic weekly stats for 2023 season
            stats_created = 0
            
            for player_data in players_data:
                player_id = player_data["id"]
                position = player_data["position"]
                
                # Generate stats for 17 weeks
                for week in range(1, 18):
                    # Generate realistic stats based on position
                    if position == "QB":
                        # QB stats
                        passing_yards = random.randint(180, 350)
                        passing_tds = random.randint(0, 4)
                        interceptions = random.randint(0, 2)
                        rushing_yards = random.randint(0, 50)
                        rushing_tds = random.randint(0, 1)
                        
                        # Calculate fantasy points (standard scoring)
                        fantasy_points = (
                            passing_yards * 0.04 +  # 1 pt per 25 yards
                            passing_tds * 4 +       # 4 pts per TD
                            interceptions * -2 +    # -2 pts per INT
                            rushing_yards * 0.1 +   # 1 pt per 10 yards  
                            rushing_tds * 6         # 6 pts per TD
                        )
                        
                        stats = WeeklyStats(
                            player_id=player_id,
                            season=2023,
                            week=week,
                            passing_yards=passing_yards,
                            passing_touchdowns=passing_tds,
                            interceptions=interceptions,
                            rushing_yards=rushing_yards,
                            rushing_touchdowns=rushing_tds,
                            fantasy_points_standard=fantasy_points,
                            fantasy_points_ppr=fantasy_points,  # Same for QB
                            fantasy_points_half_ppr=fantasy_points
                        )
                        
                    elif position == "RB":
                        # RB stats
                        rushing_yards = random.randint(30, 150)
                        rushing_tds = random.randint(0, 2)
                        receiving_yards = random.randint(0, 80)
                        receptions = random.randint(0, 8)
                        receiving_tds = random.randint(0, 1)
                        
                        # Calculate fantasy points
                        standard_points = (
                            rushing_yards * 0.1 +
                            rushing_tds * 6 +
                            receiving_yards * 0.1 +
                            receiving_tds * 6
                        )
                        ppr_points = standard_points + receptions  # +1 per reception
                        half_ppr_points = standard_points + (receptions * 0.5)
                        
                        stats = WeeklyStats(
                            player_id=player_id,
                            season=2023,
                            week=week,
                            rushing_yards=rushing_yards,
                            rushing_touchdowns=rushing_tds,
                            receiving_yards=receiving_yards,
                            receptions=receptions,
                            receiving_touchdowns=receiving_tds,
                            fantasy_points_standard=standard_points,
                            fantasy_points_ppr=ppr_points,
                            fantasy_points_half_ppr=half_ppr_points
                        )
                        
                    elif position == "WR":
                        # WR stats
                        receiving_yards = random.randint(20, 150)
                        receptions = random.randint(2, 12)
                        receiving_tds = random.randint(0, 2)
                        rushing_yards = random.randint(0, 20)
                        rushing_tds = random.randint(0, 1)
                        
                        # Calculate fantasy points
                        standard_points = (
                            receiving_yards * 0.1 +
                            receiving_tds * 6 +
                            rushing_yards * 0.1 +
                            rushing_tds * 6
                        )
                        ppr_points = standard_points + receptions
                        half_ppr_points = standard_points + (receptions * 0.5)
                        
                        stats = WeeklyStats(
                            player_id=player_id,
                            season=2023,
                            week=week,
                            receiving_yards=receiving_yards,
                            receptions=receptions,
                            receiving_touchdowns=receiving_tds,
                            rushing_yards=rushing_yards,
                            rushing_touchdowns=rushing_tds,
                            fantasy_points_standard=standard_points,
                            fantasy_points_ppr=ppr_points,
                            fantasy_points_half_ppr=half_ppr_points
                        )
                        
                    else:  # TE
                        # TE stats
                        receiving_yards = random.randint(15, 100)
                        receptions = random.randint(1, 8)
                        receiving_tds = random.randint(0, 1)
                        
                        # Calculate fantasy points
                        standard_points = (
                            receiving_yards * 0.1 +
                            receiving_tds * 6
                        )
                        ppr_points = standard_points + receptions
                        half_ppr_points = standard_points + (receptions * 0.5)
                        
                        stats = WeeklyStats(
                            player_id=player_id,
                            season=2023,
                            week=week,
                            receiving_yards=receiving_yards,
                            receptions=receptions,
                            receiving_touchdowns=receiving_tds,
                            fantasy_points_standard=standard_points,
                            fantasy_points_ppr=ppr_points,
                            fantasy_points_half_ppr=half_ppr_points
                        )
                    
                    session.add(stats)
                    stats_created += 1
            
            # Commit all data
            await session.commit()
            
            print(f"Created {len(teams_data)} teams")
            print(f"Created {len(players_data)} players")
            print(f"Created {stats_created} weekly stats records")
            
            print("\nTest data population complete!")
            print("You can now test ML training with:")
            print("  python src/fantasy_ai/cli/main.py ml train --seasons 2023 --epochs 20")
            
    except Exception as e:
        print(f"Failed to populate test data: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(populate_test_data())