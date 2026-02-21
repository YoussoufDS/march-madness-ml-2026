"""Data Loader Agent for March Madness."""

import pandas as pd
from google.adk.agents import LlmAgent
from src.utils.config import DATA_DIR, GOOGLE_API_KEY

# Global state for sharing data
DATA = {}

def load_competition_data() -> dict:
    """Load all March Madness competition CSV files and return a summary."""
    global DATA
    
    try:
        # Set the correct path - adjust if needed
        data_path = DATA_DIR
        
        DATA['m_teams'] = pd.read_csv(f'{data_path}/MTeams.csv')
        DATA['w_teams'] = pd.read_csv(f'{data_path}/WTeams.csv')
        DATA['m_regular'] = pd.read_csv(f'{data_path}/MRegularSeasonCompactResults.csv')
        DATA['w_regular'] = pd.read_csv(f'{data_path}/WRegularSeasonCompactResults.csv')
        DATA['m_tourney'] = pd.read_csv(f'{data_path}/MNCAATourneyCompactResults.csv')
        DATA['w_tourney'] = pd.read_csv(f'{data_path}/WNCAATourneyCompactResults.csv')
        DATA['m_seeds'] = pd.read_csv(f'{data_path}/MNCAATourneySeeds.csv')
        DATA['w_seeds'] = pd.read_csv(f'{data_path}/WNCAATourneySeeds.csv')
        DATA['sample_sub'] = pd.read_csv(f'{data_path}/SampleSubmissionStage1.csv')
        
        # Try to load detailed data if available
        try:
            DATA['m_detailed'] = pd.read_csv(f'{data_path}/MRegularSeasonDetailedResults.csv')
            DATA['w_detailed'] = pd.read_csv(f'{data_path}/WRegularSeasonDetailedResults.csv')
        except:
            print("Detailed results not available, continuing with compact results")
        
        return {
            'status': 'success',
            'seasons': f"{DATA['m_regular']['Season'].min()}-{DATA['m_regular']['Season'].max()}",
            'mens_teams': len(DATA['m_teams']),
            'womens_teams': len(DATA['w_teams']),
            'regular_season_games': len(DATA['m_regular']) + len(DATA['w_regular']),
            'tourney_games': len(DATA['m_tourney']) + len(DATA['w_tourney']),
            'submission_rows': len(DATA['sample_sub']),
            'message': '✅ All data loaded successfully. Ready for feature engineering.'
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error loading data: {str(e)}',
            'error': str(e)
        }

def create_data_loader_agent():
    """Create and return a Data Loader Agent."""
    return LlmAgent(
        name="DataLoaderAgent",
        model="gemini-2.5-flash-lite",
        instruction="""You are a data loading specialist for the March Madness prediction pipeline.

Your job:
1. Call the `load_competition_data` tool to load all competition CSV files.
2. Report a brief summary of what was loaded (number of teams, games, seasons).
3. If there's an error, explain what went wrong.

Be concise — just the key numbers and a confirmation.""",
        description="Loads and summarizes the competition dataset.",
        tools=[load_competition_data],
        output_key="data_summary"
    )