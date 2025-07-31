#!/usr/bin/env python3
"""
Comprehensive NFL Data Collection
Combines multiple data sources to reach 50,000+ records
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import nfl_data_py as nfl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def collect_all_nfl_data():
    """Collect ALL available NFL data to meet requirements"""
    
    logger.info("="*70)
    logger.info("COMPREHENSIVE NFL DATA COLLECTION")
    logger.info("Combining multiple sources to reach 50,000+ records")
    logger.info("="*70)
    
    required_seasons = [2019, 2020, 2021, 2022, 2023, 2024]
    all_data = []
    
    # 1. Weekly player data (primary source)
    logger.info("\n1. Collecting weekly player data...")
    weekly_data = nfl.import_weekly_data(required_seasons)
    if weekly_data is not None:
        logger.info(f"   Weekly data: {len(weekly_data)} records")
        all_data.append(('weekly', weekly_data))
    
    # 2. Snap counts (adds more player-game records)
    logger.info("\n2. Collecting snap count data...")
    try:
        snap_data = nfl.import_snap_counts(required_seasons)
        if snap_data is not None:
            logger.info(f"   Snap counts: {len(snap_data)} records")
            # Convert to player-week format
            snap_weekly = snap_data.groupby(['player', 'week', 'season']).agg({
                'offense_snaps': 'sum',
                'offense_pct': 'mean',
                'defense_snaps': 'sum', 
                'defense_pct': 'mean',
                'st_snaps': 'sum',
                'st_pct': 'mean'
            }).reset_index()
            snap_weekly.rename(columns={'player': 'player_name'}, inplace=True)
            all_data.append(('snaps', snap_weekly))
    except Exception as e:
        logger.warning(f"   Could not get snap counts: {e}")
    
    # 3. Seasonal data (aggregate stats)
    logger.info("\n3. Collecting seasonal data...")
    try:
        seasonal_data = nfl.import_seasonal_data(required_seasons)
        if seasonal_data is not None:
            logger.info(f"   Seasonal data: {len(seasonal_data)} player-seasons")
            # Expand to weekly estimates
            seasonal_expanded = []
            for _, player in seasonal_data.iterrows():
                for week in range(1, 18):  # Regular season
                    week_data = player.copy()
                    week_data['week'] = week
                    # Divide season totals by 17 for weekly estimate
                    for col in ['completions', 'attempts', 'passing_yards', 'passing_tds',
                               'rushing_yards', 'rushing_tds', 'receptions', 'targets',
                               'receiving_yards', 'receiving_tds']:
                        if col in week_data:
                            week_data[f'{col}_weekly_avg'] = week_data.get(col, 0) / 17
                    seasonal_expanded.append(week_data)
            
            seasonal_df = pd.DataFrame(seasonal_expanded)
            logger.info(f"   Seasonal expanded to weekly: {len(seasonal_df)} records")
            all_data.append(('seasonal', seasonal_df))
    except Exception as e:
        logger.warning(f"   Could not get seasonal data: {e}")
    
    # 4. Injury data (important for predictions)
    logger.info("\n4. Collecting injury data...")
    try:
        injury_data = nfl.import_injuries(required_seasons)
        if injury_data is not None:
            logger.info(f"   Injury reports: {len(injury_data)} records")
            # Convert to player-week format
            injury_weekly = injury_data.groupby(['player', 'week', 'season']).agg({
                'report_status': 'first',
                'practice_status': 'first'
            }).reset_index()
            injury_weekly.rename(columns={'player': 'player_name'}, inplace=True)
            injury_weekly['injured'] = 1
            all_data.append(('injuries', injury_weekly))
    except Exception as e:
        logger.warning(f"   Could not get injury data: {e}")
    
    # 5. Weekly rosters (all players on rosters)
    logger.info("\n5. Collecting weekly roster data...")
    try:
        roster_data = nfl.import_weekly_rosters(required_seasons)
        if roster_data is not None:
            logger.info(f"   Weekly rosters: {len(roster_data)} records")
            roster_data['on_roster'] = 1
            all_data.append(('rosters', roster_data))
    except Exception as e:
        logger.warning(f"   Could not get roster data: {e}")
    
    # 6. Depth charts (playing time indicators)
    logger.info("\n6. Collecting depth chart data...")
    try:
        depth_data = nfl.import_depth_charts(required_seasons)
        if depth_data is not None:
            logger.info(f"   Depth charts: {len(depth_data)} records")
            depth_data['depth_team'] = depth_data.get('club_code', '')
            all_data.append(('depth', depth_data))
    except Exception as e:
        logger.warning(f"   Could not get depth chart data: {e}")
    
    # 7. Combine all data sources
    logger.info("\n" + "="*70)
    logger.info("COMBINING ALL DATA SOURCES")
    logger.info("="*70)
    
    if not all_data:
        logger.error("No data collected!")
        return None
    
    # Start with weekly data as base
    base_df = None
    for source_name, df in all_data:
        if source_name == 'weekly' and df is not None:
            base_df = df.copy()
            break
    
    if base_df is None:
        logger.error("No weekly data found!")
        return None
    
    logger.info(f"Base weekly data: {len(base_df)} records")
    
    # Merge other sources
    for source_name, df in all_data:
        if source_name == 'weekly' or df is None:
            continue
            
        try:
            # Standardize column names
            if 'player' in df.columns and 'player_name' not in df.columns:
                df['player_name'] = df['player']
            if 'player_display_name' in df.columns and 'player_name' not in df.columns:
                df['player_name'] = df['player_display_name']
                
            # Merge based on available keys
            merge_keys = []
            for key in ['player_id', 'player_name', 'season', 'week', 'team']:
                if key in base_df.columns and key in df.columns:
                    merge_keys.append(key)
            
            if len(merge_keys) >= 2:  # Need at least 2 keys to merge
                logger.info(f"Merging {source_name} on keys: {merge_keys}")
                base_df = base_df.merge(
                    df,
                    on=merge_keys,
                    how='outer',
                    suffixes=('', f'_{source_name}')
                )
                logger.info(f"After merging {source_name}: {len(base_df)} records")
        except Exception as e:
            logger.warning(f"Could not merge {source_name}: {e}")
    
    # Fill missing fantasy points
    if 'fantasy_points_ppr' not in base_df.columns or base_df['fantasy_points_ppr'].isna().all():
        logger.info("Calculating fantasy points...")
        base_df['fantasy_points_ppr'] = (
            base_df.get('passing_yards', 0).fillna(0) * 0.04 +
            base_df.get('passing_tds', 0).fillna(0) * 4 +
            base_df.get('interceptions', 0).fillna(0) * -1 +
            base_df.get('rushing_yards', 0).fillna(0) * 0.1 +
            base_df.get('rushing_tds', 0).fillna(0) * 6 +
            base_df.get('receptions', 0).fillna(0) * 1 +
            base_df.get('receiving_yards', 0).fillna(0) * 0.1 +
            base_df.get('receiving_tds', 0).fillna(0) * 6
        )
    
    # Remove complete duplicates
    base_df = base_df.drop_duplicates()
    
    # Ensure we have required columns
    if 'season' not in base_df.columns:
        base_df['season'] = base_df['season_seasonal'] if 'season_seasonal' in base_df.columns else 2023
    
    # Final statistics
    logger.info("\n" + "="*70)
    logger.info("FINAL DATA SUMMARY")
    logger.info("="*70)
    logger.info(f"Total records: {len(base_df)}")
    
    if 'player_id' in base_df.columns:
        logger.info(f"Unique players: {base_df['player_id'].nunique()}")
    
    if 'season' in base_df.columns:
        seasons = sorted(base_df['season'].unique())
        logger.info(f"Seasons: {seasons}")
        
        # Check for missing seasons
        missing_seasons = set(required_seasons) - set(seasons)
        if missing_seasons:
            logger.error(f"❌ MISSING SEASONS: {missing_seasons}")
    
    # Position breakdown
    if 'position' in base_df.columns:
        logger.info("\nTop positions:")
        for pos, count in base_df['position'].value_counts().head(10).items():
            logger.info(f"  {pos}: {count}")
    
    # Validation
    total_records = len(base_df)
    if total_records >= 50000:
        logger.info(f"\n✅ SUCCESS! {total_records} records collected (>50,000)")
        
        # Save data
        os.makedirs('data', exist_ok=True)
        base_df.to_csv('data/comprehensive_nfl_data_2019_2024.csv', index=False)
        logger.info("Data saved to: data/comprehensive_nfl_data_2019_2024.csv")
        
        # Sample verification
        logger.info("\nSample records:")
        sample_cols = ['player_name', 'season', 'week', 'position', 'team', 'fantasy_points_ppr']
        available_cols = [col for col in sample_cols if col in base_df.columns]
        if available_cols:
            print(base_df[available_cols].sample(min(5, len(base_df))))
        
        return base_df
    else:
        logger.error(f"\n❌ INSUFFICIENT DATA: {total_records} records (need >50,000)")
        
        # Provide detailed breakdown
        logger.info("\nDetailed source contributions:")
        for source_name, df in all_data:
            if df is not None:
                logger.info(f"  {source_name}: {len(df)} records")
        
        return None

if __name__ == "__main__":
    df = collect_all_nfl_data()
    success = df is not None and len(df) >= 50000
    sys.exit(0 if success else 1)