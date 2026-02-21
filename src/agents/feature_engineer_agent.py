"""Feature Engineer Agent for March Madness."""

import pandas as pd
import numpy as np
from google.adk.agents import LlmAgent
from src.utils.config import ELO_K, ELO_INIT, ELO_HCA
from src.agents.data_loader_agent import DATA

# Global state for Elo ratings
ELO = {}

def compute_elo_ratings() -> dict:
    """Compute Elo ratings for all men's and women's teams across all seasons."""
    global ELO
    
    def _run_elo(regular_df, tourney_df):
        elo = {}
        season_elos = {}
        all_games = pd.concat([regular_df, tourney_df]).sort_values(['Season', 'DayNum'])
        prev_season = None

        for _, row in all_games.iterrows():
            season = row['Season']
            if season != prev_season and prev_season is not None:
                for tid, r in elo.items():
                    season_elos[(prev_season, tid)] = r
                # Regression toward mean for new season
                elo = {tid: 0.75 * r + 0.25 * ELO_INIT for tid, r in elo.items()}
            prev_season = season

            w_id, l_id = row['WTeamID'], row['LTeamID']
            w_elo = elo.get(w_id, ELO_INIT)
            l_elo = elo.get(l_id, ELO_INIT)

            # Home court adjustment
            w_loc = row.get('WLoc', 'N')
            w_adj = w_elo + (ELO_HCA if w_loc == 'H' else (-ELO_HCA if w_loc == 'A' else 0))

            # Expected win probability & update
            exp_w = 1.0 / (1.0 + 10 ** ((l_elo - w_adj) / 400.0))
            elo[w_id] = w_elo + ELO_K * (1.0 - exp_w)
            elo[l_id] = l_elo + ELO_K * (0.0 - (1.0 - exp_w))

        if prev_season:
            for tid, r in elo.items():
                season_elos[(prev_season, tid)] = r
        return season_elos

    try:
        m_elos = _run_elo(DATA['m_regular'], DATA['m_tourney'])
        w_elos = _run_elo(DATA['w_regular'], DATA['w_tourney'])
        ELO.update(m_elos)
        ELO.update(w_elos)

        # Top teams for display
        m_names = dict(zip(DATA['m_teams']['TeamID'], DATA['m_teams']['TeamName']))
        w_names = dict(zip(DATA['w_teams']['TeamID'], DATA['w_teams']['TeamName']))
        latest_m = max(s for s, _ in m_elos.keys()) if m_elos else 0
        latest_w = max(s for s, _ in w_elos.keys()) if w_elos else 0
        
        top_m = sorted([(tid, r) for (s, tid), r in m_elos.items() if s == latest_m], key=lambda x: -x[1])[:5]
        top_w = sorted([(tid, r) for (s, tid), r in w_elos.items() if s == latest_w], key=lambda x: -x[1])[:5]

        return {
            'status': 'success',
            'total_ratings': len(ELO),
            'top_mens': [f"{m_names.get(t, t)}: {r:.0f}" for t, r in top_m],
            'top_womens': [f"{w_names.get(t, t)}: {r:.0f}" for t, r in top_w],
            'message': f'✅ Elo computed through {latest_m} (men) and {latest_w} (women).'
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error computing Elo: {str(e)}',
            'error': str(e)
        }

def create_feature_engineer_agent():
    """Create and return a Feature Engineer Agent."""
    return LlmAgent(
        name="FeatureEngineerAgent",
        model="gemini-2.5-flash",
        instruction="""You are a feature engineering specialist for March Madness.

Previous stage summary: {data_summary}

Your job:
1. Call `compute_elo_ratings` to calculate Elo ratings for all teams.
2. Report the top-rated men's and women's teams.
3. If there's an error, explain what went wrong.

Be concise.""",
        description="Computes Elo ratings as features for prediction.",
        tools=[compute_elo_ratings],
        output_key="feature_summary"
    )