import pandas as pd
from espn_api.football import League
import os
from dotenv import load_dotenv

## load anaconda environment variables
load_dotenv()

## access private credentials in .env file
ESPN_S2 = os.getenv("ESPN_S2")
SWID = os.getenv("SWID")

## print first few characters privately to verify access safely
print(f"ESPN_S2: {ESPN_S2[:6]}...")  
print(f"SWID: {SWID[:6]}...")

import pandas as pd
from espn_api.football import League

## ESPN API public credentials
LEAGUE_ID = 1099505687
SEASONS = [2022, 2023, 2024]  ## past 3 seasons
MATCHUP_PERIODS = range(1, 18)

## function to fetch weekly fantasy football stats from ESPN API for multiple seasons & combine into single dataset
def fetch_fantasy_data():
    all_seasons_data = []  ## list to store all seasons data

    try:
        for season in SEASONS:
            print(f"Fetching data for SEASON {season}...")

            try:
                ## initialize ESPN API Client for specified season
                league = League(league_id=LEAGUE_ID, year=season, espn_s2=ESPN_S2, swid=SWID)
                weekly_stats = []

                ## loop through each week
                for week in MATCHUP_PERIODS:
                    print(f"Fetching data for Week {week} in {season}...")
                    try:
                        matchups = league.box_scores(week)
                    except Exception as e:
                        print(f"Error fetching data for Week {week} in {season}: {e}")
                        continue  ## skip week if error occurs

                    ## structure player stats from each matchup
                    for matchup in matchups:
                        for player in matchup.away_lineup + matchup.home_lineup:
                            weekly_stats.append({
                                'season': season,
                                'week': week,
                                'player_id': player.playerId,
                                'player_name': player.name,
                                'team': getattr(player, 'proTeam', 'Unknown'),
                                'position': getattr(player, 'position', 'Unknown'),
                                'fantasy_points': getattr(player, 'points', 0),
                                'projected_points': getattr(player, 'projected_points', 0)
                            })

                ## convert to dataframe
                df = pd.DataFrame(weekly_stats)

                ## remove kickers (K) and defenses (D/ST)
                df = df[~df["position"].isin(["K", "D/ST"])]

                ## show unique positions
                print(f"Unique Positions After Removing K and DEF ({season}):", df["position"].unique())

                ## validate data pull
                if df.empty:
                    print(f"No data collected for {season}. Check your credentials and league settings.")
                    continue  ## Skip adding empty data

                ## display a sample of the data
                print(f"Data Sample for {season}:\n", df.head())

                ## save CSV for future use
                filename = f"./data/fantasy_weekly_stats_{season}.csv"
                df.to_csv(filename, index=False)
                print(f"Data successfully saved to {filename}")

                ## append season data to combined list
                all_seasons_data.append(df)

            except Exception as e:
                print(f"Error initializing ESPN API client for {season}: {e}")

        ## combine all season data into one dataframe
        if all_seasons_data:
            combined_df = pd.concat(all_seasons_data, ignore_index=True)
            print(f"Total records collected across all seasons: {combined_df.shape[0]}")

            ## save combined CSV for future use
            combined_df.to_csv("./data/fantasy_weekly_stats_combined.csv", index=False)
            print("Data successfully saved to fantasy_weekly_stats_combined.csv")

        else:
            print("No data collected for any season.")

    except Exception as e:
        print(f"Unexpected error: {e}")

## run function
fetch_fantasy_data()
