#!/usr/bin/env python3
"""
STRICT ML TRAINING - FULL HISTORICAL DATA COLLECTION
Collects ALL NFL data from 2019-2024 seasons
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import asyncio

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from data.sources.nfl_data_py_client import NFLDataPyClient
from data.sources.weather_client import WeatherClient
from data.sleeper_client import SleeperAPIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FullHistoricalDataCollector:
    """Collects complete NFL data for 2019-2024"""
    
    def __init__(self):
        self.nfl_client = NFLDataPyClient()
        self.weather_client = None
        self.sleeper_client = SleeperAPIClient()
        self.required_seasons = [2019, 2020, 2021, 2022, 2023, 2024]
        self.collected_data = {}
        
    async def initialize(self):
        """Initialize async clients"""
        self.weather_client = WeatherClient()
        await self.weather_client.__aenter__()
        
    async def close(self):
        """Close async clients"""
        if self.weather_client:
            await self.weather_client.__aexit__(None, None, None)
    
    def collect_nfl_data(self):
        """Collect ALL NFL data for required seasons"""
        logger.info("="*70)
        logger.info("COLLECTING NFL DATA (2019-2024)")
        logger.info("="*70)
        
        all_weekly_data = []
        season_summary = {}
        
        for season in self.required_seasons:
            logger.info(f"\n📊 Processing {season} season...")
            
            try:
                # Get weekly stats
                weekly_df = self.nfl_client.import_weekly_data(season)
                
                if weekly_df is not None and len(weekly_df) > 0:
                    # Add season column if missing
                    if 'season' not in weekly_df.columns:
                        weekly_df['season'] = season
                    
                    all_weekly_data.append(weekly_df)
                    
                    # Track summary
                    season_summary[season] = {
                        'records': len(weekly_df),
                        'players': weekly_df['player_id'].nunique(),
                        'weeks': weekly_df['week'].nunique() if 'week' in weekly_df.columns else 0
                    }
                    
                    logger.info(f"  ✓ Collected {len(weekly_df)} records")
                    logger.info(f"  ✓ Players: {season_summary[season]['players']}")
                    logger.info(f"  ✓ Weeks: {season_summary[season]['weeks']}")
                    
                    # Try to add Next Gen Stats
                    try:
                        # Passing
                        ngs_pass = self.nfl_client.import_ngs_data('passing', season)
                        if ngs_pass is not None:
                            logger.info(f"  ✓ Added {len(ngs_pass)} passing NGS records")
                        
                        # Rushing  
                        ngs_rush = self.nfl_client.import_ngs_data('rushing', season)
                        if ngs_rush is not None:
                            logger.info(f"  ✓ Added {len(ngs_rush)} rushing NGS records")
                            
                        # Receiving
                        ngs_rec = self.nfl_client.import_ngs_data('receiving', season)
                        if ngs_rec is not None:
                            logger.info(f"  ✓ Added {len(ngs_rec)} receiving NGS records")
                            
                    except Exception as e:
                        logger.warning(f"  ⚠️  Could not get NGS data: {e}")
                        
                else:
                    logger.error(f"  ✗ NO DATA for {season}!")
                    season_summary[season] = {'records': 0, 'players': 0, 'weeks': 0}
                    
            except Exception as e:
                logger.error(f"  ✗ ERROR collecting {season}: {e}")
                season_summary[season] = {'error': str(e)}
        
        # Combine all data
        if all_weekly_data:
            combined_df = pd.concat(all_weekly_data, ignore_index=True)
            logger.info(f"\n📊 TOTAL NFL DATA: {len(combined_df)} records")
            
            # Verify all seasons present
            seasons_present = sorted(combined_df['season'].unique())
            logger.info(f"Seasons present: {seasons_present}")
            
            missing_seasons = set(self.required_seasons) - set(seasons_present)
            if missing_seasons:
                logger.error(f"❌ MISSING SEASONS: {missing_seasons}")
            else:
                logger.info("✅ All required seasons present!")
                
            self.collected_data['nfl'] = combined_df
            
            # Display verification queries
            self.verify_nfl_data(combined_df)
            
            return combined_df
        else:
            logger.error("❌ NO DATA COLLECTED!")
            return None
            
    def verify_nfl_data(self, df):
        """Run verification queries on collected data"""
        logger.info("\n" + "="*70)
        logger.info("DATA VERIFICATION")
        logger.info("="*70)
        
        # 1. Season coverage
        logger.info("\n1. SEASON COVERAGE:")
        season_stats = df.groupby('season').agg({
            'player_id': 'nunique',
            'player_name': 'count'
        }).rename(columns={'player_id': 'unique_players', 'player_name': 'total_records'})
        
        for season, row in season_stats.iterrows():
            status = "✓" if season in self.required_seasons else "✗"
            logger.info(f"  {status} {season}: {row['unique_players']} players, {row['total_records']} records")
        
        # 2. Position breakdown
        logger.info("\n2. POSITION BREAKDOWN:")
        if 'position' in df.columns:
            position_stats = df.groupby(['position', 'season']).size().unstack(fill_value=0)
            logger.info(position_stats.to_string())
        
        # 3. Data quality
        logger.info("\n3. DATA QUALITY:")
        if 'fantasy_points_ppr' in df.columns:
            quality_stats = df.groupby(['position', 'season'])['fantasy_points_ppr'].agg(['count', 'mean', 'std'])
            logger.info(quality_stats.head(20).to_string())
        
        # 4. Missing data check
        logger.info("\n4. MISSING DATA:")
        missing_seasons = set(self.required_seasons) - set(df['season'].unique())
        if missing_seasons:
            logger.error(f"❌ MISSING SEASONS: {missing_seasons}")
            logger.error("CANNOT PROCEED WITHOUT ALL SEASONS!")
        else:
            logger.info("✅ All required seasons present")
            
        # 5. Sample records
        logger.info("\n5. SAMPLE RECORDS:")
        sample_cols = ['player_name', 'season', 'week', 'position', 'fantasy_points_ppr']
        available_cols = [col for col in sample_cols if col in df.columns]
        if available_cols:
            sample = df[available_cols].sample(min(10, len(df)))
            logger.info(sample.to_string())
            
    async def add_weather_data(self, df):
        """Add weather data for outdoor games"""
        logger.info("\n" + "="*70)
        logger.info("ADDING WEATHER DATA")
        logger.info("="*70)
        
        # Get unique games
        if 'recent_team' not in df.columns:
            logger.warning("No team data available for weather matching")
            return df
            
        unique_games = df[['season', 'week', 'recent_team']].drop_duplicates()
        logger.info(f"Processing weather for {len(unique_games)} unique game-teams...")
        
        weather_success = 0
        weather_data = []
        
        # Process in batches
        batch_size = 100
        for i in range(0, len(unique_games), batch_size):
            batch = unique_games.iloc[i:i+batch_size]
            
            for _, game in batch.iterrows():
                try:
                    # Get schedule to find home team
                    schedules = self.nfl_client.import_schedules(game['season'])
                    if schedules is None:
                        continue
                        
                    game_info = schedules[
                        ((schedules['home_team'] == game['recent_team']) |
                         (schedules['away_team'] == game['recent_team'])) &
                        (schedules['week'] == game['week'])
                    ]
                    
                    if not game_info.empty:
                        home_team = game_info['home_team'].iloc[0]
                        game_date = pd.to_datetime(game_info['gameday'].iloc[0])
                        
                        weather = await self.weather_client.get_game_weather(
                            home_team, game_date, 13
                        )
                        
                        weather_data.append({
                            'season': game['season'],
                            'week': game['week'],
                            'team': game['recent_team'],
                            **weather
                        })
                        weather_success += 1
                        
                except Exception as e:
                    logger.debug(f"Weather error: {e}")
                    
            # Progress update
            if (i + batch_size) % 500 == 0:
                logger.info(f"  Processed {min(i + batch_size, len(unique_games))}/{len(unique_games)} games...")
                
        # Merge weather data
        if weather_data:
            weather_df = pd.DataFrame(weather_data)
            df = df.merge(
                weather_df,
                left_on=['season', 'week', 'recent_team'],
                right_on=['season', 'week', 'team'],
                how='left'
            )
            
            join_rate = (weather_success / len(unique_games)) * 100
            logger.info(f"\n✅ Weather data added: {weather_success}/{len(unique_games)} ({join_rate:.1f}%)")
            
            if join_rate < 95:
                logger.warning(f"⚠️  Weather join rate below 95%: {join_rate:.1f}%")
        else:
            logger.warning("❌ No weather data collected")
            
        return df
        
    async def add_college_stats(self, df):
        """Add college stats for rookies"""
        logger.info("\n" + "="*70)
        logger.info("ADDING COLLEGE STATS FOR ROOKIES")
        logger.info("="*70)
        
        # Identify rookies (first year players)
        if 'season' in df.columns and 'player_id' in df.columns:
            player_first_year = df.groupby('player_id')['season'].min().reset_index()
            player_first_year.columns = ['player_id', 'rookie_year']
            
            df = df.merge(player_first_year, on='player_id', how='left')
            df['is_rookie'] = df['season'] == df['rookie_year']
            
            rookies = df[df['is_rookie']]['player_id'].unique()
            logger.info(f"Identified {len(rookies)} unique rookies")
            
            # Note: College stats would need additional data source
            # For now, we'll flag rookies for special handling
            df['has_college_stats'] = False  # Placeholder
            
            logger.info("⚠️  College stats API not currently integrated")
            logger.info("   Rookies flagged for future enhancement")
            
        return df
        
    def create_progress_report(self):
        """Create detailed progress report"""
        report = """
📊 DATA COLLECTION PROGRESS
├── nfl_data_py: 
│   ├── Seasons: {} {}
│   ├── Records: {} (target: >50,000)
│   └── Players: {}
├── Weather data:
│   ├── Games covered: {}/{}
│   └── Join rate: {}%
├── Sleeper data:
│   ├── Players matched: {}/{}
│   └── Rookies with college stats: {}/{}
└── OVERALL: {} {}
"""
        
        # Fill in actual values
        if 'nfl' in self.collected_data:
            df = self.collected_data['nfl']
            seasons = sorted(df['season'].unique())
            all_seasons_present = set(self.required_seasons).issubset(set(seasons))
            
            return report.format(
                seasons,
                "✓" if all_seasons_present else "✗",
                len(df),
                df['player_id'].nunique() if 'player_id' in df.columns else 0,
                "TBD", "TBD", "TBD",
                "TBD", "TBD",
                "TBD", "TBD",
                "✓ READY" if all_seasons_present and len(df) > 50000 else "✗ INCOMPLETE",
                f"({len(df)} records)" if len(df) < 50000 else ""
            )
        else:
            return "❌ NO DATA COLLECTED YET"
            
    async def run_full_collection(self):
        """Run complete data collection pipeline"""
        logger.info("="*70)
        logger.info("STRICT ML TRAINING - FULL HISTORICAL DATA COLLECTION")
        logger.info("Required: 2019-2024 seasons, >50,000 records")
        logger.info("="*70)
        
        # 1. Collect NFL data
        nfl_data = self.collect_nfl_data()
        
        if nfl_data is None or len(nfl_data) == 0:
            logger.error("\n❌ CRITICAL ERROR: No NFL data collected!")
            logger.error("CANNOT PROCEED WITHOUT DATA")
            return False
            
        # Check if we have all required seasons
        seasons_present = set(nfl_data['season'].unique())
        missing_seasons = set(self.required_seasons) - seasons_present
        
        if missing_seasons:
            logger.error(f"\n❌ INCOMPLETE DATA: Only have seasons {sorted(seasons_present)}")
            logger.error(f"Missing seasons: {sorted(missing_seasons)}")
            logger.error("CANNOT PROCEED WITHOUT ALL SEASONS")
            
            # Check for 2024 specifically
            if 2024 in missing_seasons:
                logger.warning("\n⚠️  2024 season may not be complete in data source")
                logger.warning("   This is expected if season is ongoing")
                
            return False
            
        # 2. Add weather data
        nfl_data = await self.add_weather_data(nfl_data)
        
        # 3. Add college stats
        nfl_data = await self.add_college_stats(nfl_data)
        
        # 4. Final verification
        logger.info("\n" + "="*70)
        logger.info("FINAL VALIDATION")
        logger.info("="*70)
        
        total_records = len(nfl_data)
        total_players = nfl_data['player_id'].nunique() if 'player_id' in nfl_data.columns else 0
        
        logger.info(f"Total records: {total_records}")
        logger.info(f"Total players: {total_players}")
        logger.info(f"Seasons: {sorted(nfl_data['season'].unique())}")
        
        # Progress report
        logger.info(self.create_progress_report())
        
        # Save collected data
        self.collected_data['final'] = nfl_data
        
        # Check if we meet requirements
        if total_records >= 50000 and not missing_seasons:
            logger.info("\n✅ ALL REQUIREMENTS MET - READY FOR ML TRAINING")
            
            # Save to file for ML training
            nfl_data.to_csv('data/full_nfl_data_2019_2024.csv', index=False)
            logger.info(f"Data saved to: data/full_nfl_data_2019_2024.csv")
            
            return True
        else:
            logger.error(f"\n❌ REQUIREMENTS NOT MET:")
            if total_records < 50000:
                logger.error(f"   - Need >50,000 records, have {total_records}")
            if missing_seasons:
                logger.error(f"   - Missing seasons: {missing_seasons}")
            return False

async def main():
    """Main execution"""
    collector = FullHistoricalDataCollector()
    
    try:
        await collector.initialize()
        success = await collector.run_full_collection()
        
        if not success:
            logger.error("\n❌ DATA COLLECTION FAILED - CANNOT PROCEED WITH ML TRAINING")
            logger.error("Fix data issues before attempting model training")
            return False
            
        return True
        
    finally:
        await collector.close()

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    sys.exit(0 if success else 1)