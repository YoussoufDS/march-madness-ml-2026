"""Model Trainer Agent for March Madness."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import joblib
from google.adk.agents import LlmAgent
from src.agents.data_loader_agent import DATA
from src.agents.feature_engineer_agent import ELO
from src.utils.config import ELO_INIT, MODEL_DIR

MODEL = None

def _parse_seed(seed_str):
    """Extract numeric seed from string like 'W01', 'X16a' → 1, 16."""
    try:
        return int(seed_str[1:3])
    except:
        return 8  # Default seed

def train_prediction_model() -> dict:
    """Train a logistic regression on Elo difference and seed difference."""
    global MODEL

    try:
        # Seed lookup
        seed_map = {}
        for df in [DATA['m_seeds'], DATA['w_seeds']]:
            for _, row in df.iterrows():
                seed_map[(row['Season'], row['TeamID'])] = _parse_seed(row['Seed'])

        # Build training set from tournament games (2003+)
        X, y = [], []
        for t_df in [DATA['m_tourney'], DATA['w_tourney']]:
            for _, row in t_df.iterrows():
                season = row['Season']
                if season < 2003:
                    continue

                w_id, l_id = row['WTeamID'], row['LTeamID']
                # Use PREVIOUS season Elo as pre-tournament rating
                w_elo = ELO.get((season - 1, w_id), ELO_INIT)
                l_elo = ELO.get((season - 1, l_id), ELO_INIT)
                w_seed = seed_map.get((season, w_id), 8)
                l_seed = seed_map.get((season, l_id), 8)

                # Convention: team1 = lower ID
                if w_id < l_id:
                    X.append([w_elo - l_elo, l_seed - w_seed])
                    y.append(1)
                else:
                    X.append([l_elo - w_elo, w_seed - l_seed])
                    y.append(0)

        X, y = np.array(X), np.array(y)

        # Train
        MODEL = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        MODEL.fit(X, y)

        # Cross-val Brier score
        cv_probs = cross_val_score(
            LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000), 
            X, y,
            scoring='neg_brier_score', 
            cv=5
        )
        brier = -cv_probs.mean()

        # Save model
        model_path = f"{MODEL_DIR}/baseline_model.pkl"
        joblib.dump(MODEL, model_path)

        return {
            'status': 'success',
            'training_games': len(y),
            'win_rate_label1': f"{y.mean():.3f}",
            'cv_brier_score': f"{brier:.4f}",
            'coefficients': {
                'elo_diff': f"{MODEL.coef_[0][0]:.6f}",
                'seed_diff': f"{MODEL.coef_[0][1]:.6f}",
                'intercept': f"{MODEL.intercept_[0]:.6f}"
            },
            'model_path': model_path,
            'message': f'✅ Model trained on {len(y)} games. CV Brier: {brier:.4f}'
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error training model: {str(e)}',
            'error': str(e)
        }

def create_model_trainer_agent():
    """Create and return a Model Trainer Agent."""
    return LlmAgent(
        name="ModelTrainerAgent",
        model="gemini-2.5-flash",
        instruction="""You are a model training specialist for March Madness.

Previous stage summary: {feature_summary}

Your job:
1. Call `train_prediction_model` to train a logistic regression model.
2. Report the cross-validation Brier score and model coefficients.
3. Briefly interpret which feature (Elo or seed) matters more.
4. If there's an error, explain what went wrong.

Be concise.""",
        description="Trains and evaluates the prediction model.",
        tools=[train_prediction_model],
        output_key="model_summary"
    )