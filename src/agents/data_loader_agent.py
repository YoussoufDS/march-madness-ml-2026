"""Data Loader Agent for March Madness."""

import pandas as pd
import os
from google.adk.agents import LlmAgent
from src.utils.config import DATA_DIR, GOOGLE_API_KEY

# Global state for sharing data
DATA = {}

def load_competition_data() -> dict:
    """Load all March Madness competition CSV files and return a summary."""
    global DATA
    
    try:
        # Correction du chemin : on vérifie si les données sont dans 'raw'
        base_path = DATA_DIR
        if os.path.exists(os.path.join(base_path, 'raw')):
            data_path = os.path.join(base_path, 'raw')
        else:
            data_path = base_path
            
        print(f"📂 Chargement des données depuis : {data_path}")

        # Utilisation de encoding='utf-8-sig' pour éviter les caractères invisibles en début de fichier
        def smart_read(filename):
            full_path = os.path.join(data_path, filename)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Fichier manquant : {filename}")
            return pd.read_csv(full_path, encoding='utf-8-sig')

        DATA['m_teams'] = smart_read('MTeams.csv')
        DATA['w_teams'] = smart_read('WTeams.csv')
        DATA['m_regular'] = smart_read('MRegularSeasonCompactResults.csv')
        DATA['w_regular'] = smart_read('WRegularSeasonCompactResults.csv')
        DATA['m_tourney'] = smart_read('MNCAATourneyCompactResults.csv')
        DATA['w_tourney'] = smart_read('WNCAATourneyCompactResults.csv')
        DATA['m_seeds'] = smart_read('MNCAATourneySeeds.csv')
        DATA['w_seeds'] = smart_read('WNCAATourneySeeds.csv')
        DATA['sample_sub'] = smart_read('SampleSubmissionStage1.csv')
        
        # Tentative de chargement des données détaillées
        try:
            DATA['m_detailed'] = smart_read('MRegularSeasonDetailedResults.csv')
            DATA['w_detailed'] = smart_read('WRegularSeasonDetailedResults.csv')
        except Exception as e:
            print(f"ℹ️ Résultats détaillés non chargés : {e}")
        
        # Vérification de la présence de la colonne Season pour le résumé
        if 'Season' not in DATA['m_regular'].columns:
            # Debug : affiche les vrais noms de colonnes si Season n'est pas trouvé
            cols = DATA['m_regular'].columns.tolist()
            raise KeyError(f"La colonne 'Season' est introuvable. Colonnes détectées : {cols}")

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
        print(f"❌ Erreur critique : {str(e)}")
        return {
            'status': 'error',
            'message': f'Error loading data: {str(e)}',
            'error': str(e)
        }

def create_data_loader_agent():
    """Create and return a Data Loader Agent."""
    return LlmAgent(
        name="DataLoaderAgent",
        model="gemini-1.5-flash", # Note: Gemini 2.5 n'existe pas encore, correction en 2.0
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