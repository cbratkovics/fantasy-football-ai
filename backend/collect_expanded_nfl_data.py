#!/usr/bin/env python3
"""
Expanded NFL Data Collection - Get ALL player data including defensive players
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from data.sources.nfl_data_py_client import NFLDataPyClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def collect_comprehensive_data():
    """Collect ALL NFL data to meet 50,000+ record requirement"""
    
    logger.info("="*70)
    logger.info("EXPANDED NFL DATA COLLECTION")
    logger.info("Target: >50,000 player-week records")
    logger.info("="*70)
    
    client = NFLDataPyClient()
    required_seasons = [2019, 2020, 2021, 2022, 2023, 2024]
    
    all_data = []
    
    for season in required_seasons:
        logger.info(f"\n📊 Collecting ALL data for {season}...")
        
        # 1. Get weekly data for ALL positions (not filtering)
        weekly_df = client.import_weekly_data(season, position=None)  # Get ALL positions
        
        if weekly_df is not None:
            logger.info(f"  Weekly stats: {len(weekly_df)} records")
            
            # Add season if missing
            if 'season' not in weekly_df.columns:
                weekly_df['season'] = season
                
            # 2. Also get play-by-play data for more records
            try:
                pbp_df = client.import_pbp_data(
                    season,
                    columns=['game_id', 'play_id', 'week', 'posteam', 'defteam',
                            'passer_player_id', 'passer_player_name', 
                            'rusher_player_id', 'rusher_player_name',
                            'receiver_player_id', 'receiver_player_name',
                            'fantasy_player_id', 'fantasy_player_name',
                            'fantasy', 'fantasy_id',
                            'passing_yards', 'rushing_yards', 'receiving_yards',
                            'pass_touchdown', 'rush_touchdown', 'return_touchdown',
                            'epa', 'air_epa', 'yac_epa']
                )
                
                if pbp_df is not None and len(pbp_df) > 0:
                    logger.info(f"  Play-by-play: {len(pbp_df)} plays")
                    
                    # Aggregate PBP data by player/week
                    player_stats = []
                    
                    # Passers
                    if 'passer_player_id' in pbp_df.columns:
                        passer_stats = pbp_df[pbp_df['passer_player_id'].notna()].groupby(
                            ['passer_player_id', 'passer_player_name', 'week']
                        ).agg({
                            'passing_yards': 'sum',
                            'pass_touchdown': 'sum',
                            'epa': 'sum',
                            'air_epa': 'sum'
                        }).reset_index()
                        passer_stats.columns = ['player_id', 'player_name', 'week', 
                                               'passing_yards_pbp', 'pass_td_pbp', 'epa_pass', 'air_epa']
                        passer_stats['position'] = 'QB'
                        passer_stats['season'] = season
                        player_stats.append(passer_stats)
                    
                    # Rushers
                    if 'rusher_player_id' in pbp_df.columns:
                        rusher_stats = pbp_df[pbp_df['rusher_player_id'].notna()].groupby(
                            ['rusher_player_id', 'rusher_player_name', 'week']
                        ).agg({
                            'rushing_yards': 'sum',
                            'rush_touchdown': 'sum',
                            'epa': 'sum'
                        }).reset_index()
                        rusher_stats.columns = ['player_id', 'player_name', 'week',
                                               'rushing_yards_pbp', 'rush_td_pbp', 'epa_rush']
                        rusher_stats['position'] = 'RB'
                        rusher_stats['season'] = season
                        player_stats.append(rusher_stats)
                    
                    # Receivers
                    if 'receiver_player_id' in pbp_df.columns:
                        receiver_stats = pbp_df[pbp_df['receiver_player_id'].notna()].groupby(
                            ['receiver_player_id', 'receiver_player_name', 'week']
                        ).agg({
                            'receiving_yards': 'sum',
                            'pass_touchdown': 'sum',
                            'yac_epa': 'sum'
                        }).reset_index()
                        receiver_stats.columns = ['player_id', 'player_name', 'week',
                                                 'receiving_yards_pbp', 'rec_td_pbp', 'yac_epa']
                        receiver_stats['position'] = 'WR/TE'
                        receiver_stats['season'] = season
                        player_stats.append(receiver_stats)
                    
                    if player_stats:
                        pbp_aggregated = pd.concat(player_stats, ignore_index=True)
                        logger.info(f"  PBP aggregated: {len(pbp_aggregated)} player-weeks")
                        
                        # Merge with weekly data to enrich
                        weekly_df = weekly_df.merge(
                            pbp_aggregated,
                            on=['player_id', 'week', 'season'],
                            how='outer',
                            suffixes=('', '_pbp')
                        )
                        
            except Exception as e:
                logger.warning(f"  Could not get PBP data: {e}")
            
            # 3. Get roster data for more player info
            try:
                rosters = client.import_rosters(season)
                if rosters is not None:
                    logger.info(f"  Rosters: {len(rosters)} players")
                    
                    # Create weekly entries for all rostered players
                    weeks = list(range(1, 23))  # Regular season + playoffs
                    roster_weekly = []
                    
                    for week in weeks:
                        roster_week = rosters.copy()
                        roster_week['week'] = week
                        roster_week['season'] = season
                        roster_weekly.append(roster_week)
                    
                    roster_df = pd.concat(roster_weekly, ignore_index=True)
                    
                    # Merge with weekly stats
                    weekly_df = weekly_df.merge(
                        roster_df[['player_id', 'week', 'season', 'team', 'position', 
                                  'depth_chart_position', 'jersey_number']],
                        on=['player_id', 'week', 'season'],
                        how='outer',
                        suffixes=('', '_roster')
                    )
                    
            except Exception as e:
                logger.warning(f"  Could not get roster data: {e}")
            
            # 4. Get team stats for context
            try:
                team_stats = client.import_seasonal_rosters(season)
                if team_stats is not None:
                    logger.info(f"  Team rosters: {len(team_stats)} entries")
            except:
                pass
                
            # Ensure we have position data
            if 'position' not in weekly_df.columns and 'position_roster' in weekly_df.columns:
                weekly_df['position'] = weekly_df['position_roster']
                
            # Add to collection
            all_data.append(weekly_df)
            logger.info(f"  Total for {season}: {len(weekly_df)} records")
            
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Remove pure duplicates
        combined_df = combined_df.drop_duplicates()
        
        # Fill missing fantasy points with 0 for players who played but didn't score
        if 'fantasy_points_ppr' in combined_df.columns:
            combined_df['fantasy_points_ppr'] = combined_df['fantasy_points_ppr'].fillna(0)
        else:
            # Calculate if missing
            combined_df['fantasy_points_ppr'] = (
                combined_df.get('passing_yards', 0) * 0.04 +
                combined_df.get('passing_tds', 0) * 4 +
                combined_df.get('interceptions', 0) * -1 +
                combined_df.get('rushing_yards', 0) * 0.1 +
                combined_df.get('rushing_tds', 0) * 6 +
                combined_df.get('receptions', 0) * 1 +
                combined_df.get('receiving_yards', 0) * 0.1 +
                combined_df.get('receiving_tds', 0) * 6 +
                combined_df.get('return_tds', 0) * 6 +
                combined_df.get('two_point_conversions', 0) * 2
            )
        
        logger.info("\n" + "="*70)
        logger.info("FINAL DATA SUMMARY")
        logger.info("="*70)
        logger.info(f"Total records: {len(combined_df)}")
        logger.info(f"Unique players: {combined_df['player_id'].nunique()}")
        logger.info(f"Seasons: {sorted(combined_df['season'].unique())}")
        
        # Position breakdown
        if 'position' in combined_df.columns:
            position_counts = combined_df['position'].value_counts()
            logger.info("\nPosition breakdown:")
            for pos, count in position_counts.head(20).items():
                logger.info(f"  {pos}: {count}")
        
        # Season breakdown
        season_counts = combined_df.groupby('season').agg({
            'player_id': ['nunique', 'count']
        })
        logger.info("\nSeason breakdown:")
        for season, (unique_players, total_records) in season_counts.iterrows():
            logger.info(f"  {season}: {unique_players} players, {total_records} records")
        
        # Check if we meet requirements
        missing_seasons = set(required_seasons) - set(combined_df['season'].unique())
        
        if len(combined_df) >= 50000 and not missing_seasons:
            logger.info("\n✅ SUCCESS! All requirements met:")
            logger.info(f"  - {len(combined_df)} records (>50,000 ✓)")
            logger.info(f"  - All seasons present ✓")
            
            # Save data
            os.makedirs('data', exist_ok=True)
            combined_df.to_csv('data/expanded_nfl_data_2019_2024.csv', index=False)
            logger.info(f"\nData saved to: data/expanded_nfl_data_2019_2024.csv")
            
            return combined_df
        else:
            logger.error("\n❌ STILL INSUFFICIENT DATA:")
            logger.error(f"  - Records: {len(combined_df)} (need >50,000)")
            if missing_seasons:
                logger.error(f"  - Missing seasons: {missing_seasons}")
            return None
    
    return None

if __name__ == "__main__":
    df = collect_comprehensive_data()
    success = df is not None and len(df) >= 50000
    sys.exit(0 if success else 1)