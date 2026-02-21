"""Submission Generator Agent for March Madness."""

import pandas as pd
import numpy as np
import joblib
from google.adk.agents import LlmAgent
from src.agents.data_loader_agent import DATA
from src.agents.feature_engineer_agent import ELO
from src.agents.model_trainer_agent import MODEL
from src.utils.config import ELO_INIT, MODEL_DIR

def _parse_seed(seed_str):
    """Extract numeric seed from string like 'W01', 'X16a' → 1, 16."""
    try:
        return int(seed_str[1:3])
    except:
        return 8

def generate_submission() -> dict:
    """Generate predictions for every possible matchup and save submission.csv."""
    try:
        sub = DATA['sample_sub'].copy()

        # Seed lookup
        seed_map = {}
        for df in [DATA['m_seeds'], DATA['w_seeds']]:
            for _, row in df.iterrows():
                seed_map[(row['Season'], row['TeamID'])] = _parse_seed(row['Seed'])

        # Load model if not in memory
        if MODEL is None:
            model = joblib.load(f"{MODEL_DIR}/baseline_model.pkl")
        else:
            model = MODEL

        preds = []
        for _, row in sub.iterrows():
            parts = row['ID'].split('_')
            season = int(parts[0])
            t1, t2 = int(parts[1]), int(parts[2])  # t1 < t2 by construction

            # Use prior season's Elo
            latest_season = season - 1

            e1 = ELO.get((latest_season, t1), ELO_INIT)
            e2 = ELO.get((latest_season, t2), ELO_INIT)
            s1 = seed_map.get((season, t1), 8)
            s2 = seed_map.get((season, t2), 8)

            features = np.array([[e1 - e2, s2 - s1]])
            prob = model.predict_proba(features)[0][1]
            # Clip to avoid extreme probabilities
            prob = np.clip(prob, 0.01, 0.99)
            preds.append(prob)

        sub['Pred'] = preds
        output_path = '/kaggle/working/submission.csv'  # Pour Kaggle
        # Alternative pour local:
        # output_path = 'submission.csv'
        sub.to_csv(output_path, index=False)

        return {
            'status': 'success',
            'num_predictions': len(preds),
            'mean_pred': f"{np.mean(preds):.4f}",
            'std_pred': f"{np.std(preds):.4f}",
            'output_path': output_path,
            'sample_predictions': preds[:5],
            'message': f'✅ Submission saved to {output_path} with {len(preds)} predictions.'
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error generating submission: {str(e)}',
            'error': str(e)
        }

def create_submission_agent():
    """Create and return a Submission Generator Agent."""
    return LlmAgent(
        name="SubmissionAgent",
        model="gemini-2.5-flash-lite",
        instruction="""You are a submission generation specialist for March Madness.

Previous stage summary: {model_summary}

Your job:
1. Call `generate_submission` to create predictions for all matchups.
2. Report where the submission file was saved and basic prediction stats.
3. Suggest 2-3 concrete ideas for improving the model.
4. If there's an error, explain what went wrong.

Be concise.""",
        description="Generates the final submission CSV.",
        tools=[generate_submission],
        output_key="submission_summary"
    )