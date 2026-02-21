"""Data drift detection for March Madness predictions."""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from typing import Dict, Any
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift, TestNumberOfDriftedColumns
from src.agents.data_loader_agent import DATA
from src.utils.config import DRIFT_REPORTS_DIR, DRIFT_THRESHOLD, RETRAIN_THRESHOLD

class DataDriftDetector:
    """Detect data drift between training and new data."""
    
    def __init__(self):
        self.report_dir = DRIFT_REPORTS_DIR
        os.makedirs(self.report_dir, exist_ok=True)
    
    def prepare_features(self, games_df: pd.DataFrame, with_outcome: bool = False) -> pd.DataFrame:
        """Extract features for drift detection."""
        if len(games_df) == 0:
            return pd.DataFrame()
            
        features = pd.DataFrame()
        
        # Need to join with Elo ratings? For now, use basic features
        features['score_diff'] = games_df['WScore'] - games_df['LScore']
        features['total_score'] = games_df['WScore'] + games_df['LScore']
        features['num_ot'] = games_df['NumOT']
        features['is_home_win'] = (games_df['WLoc'] == 'H').astype(int)
        
        return features
    
    def detect_drift(self, current_data: pd.DataFrame, reference_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect drift between reference and current data.
        """
        if len(reference_data) == 0 or len(current_data) == 0:
            return {
                'drift_detected': False,
                'message': 'Insufficient data for drift detection',
                'needs_retraining': False
            }
        
        # Prepare features
        ref_features = self.prepare_features(reference_data)
        current_features = self.prepare_features(current_data)
        
        if len(ref_features) == 0 or len(current_features) == 0:
            return {
                'drift_detected': False,
                'message': 'No features could be extracted',
                'needs_retraining': False
            }
        
        # Column mapping
        column_mapping = ColumnMapping(
            numerical_features=ref_features.columns.tolist()
        )
        
        # Create test suite
        test_suite = TestSuite(tests=[
            TestColumnDrift(column_name=col) for col in ref_features.columns[:3]  # Test first 3 columns
        ] + [TestNumberOfDriftedColumns()])
        
        test_suite.run(
            reference_data=ref_features,
            current_data=current_features,
            column_mapping=column_mapping
        )
        
        # Save reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_suite.save_html(f"{self.report_dir}/test_suite_{timestamp}.html")
        
        # Extract metrics
        test_results = test_suite.as_dict()
        
        result = {
            'timestamp': timestamp,
            'number_of_drifted_columns': 0,
            'drift_detected': False,
            'column_drifts': {},
            'recommendations': []
        }
        
        # Parse results
        for test in test_results.get('tests', []):
            if test.get('name') == 'Number of Drifted Columns':
                result['number_of_drifted_columns'] = test.get('parameters', {}).get('value', 0)
                result['drift_detected'] = test.get('status') == 'FAIL'
            elif 'Column Drift' in test.get('name', ''):
                col = test.get('parameters', {}).get('column_name', {}).get('value', 'unknown')
                result['column_drifts'][col] = {
                    'drift_score': test.get('parameters', {}).get('drift_score', 0),
                    'drift_detected': test.get('status') == 'FAIL'
                }
                
                if test.get('status') == 'FAIL':
                    result['recommendations'].append(
                        f"Significant drift detected in {col}. Consider retraining."
                    )
        
        # Save metrics
        with open(f"{self.report_dir}/drift_metrics_{timestamp}.json", 'w') as f:
            json.dump(result, f, indent=2)
        
        # Determine if retraining needed
        n_cols = len(result['column_drifts'])
        n_drifted = result['number_of_drifted_columns']
        
        needs_retrain = False
        if n_cols > 0 and n_drifted / n_cols > DRIFT_THRESHOLD:
            needs_retrain = True
        
        result['needs_retraining'] = needs_retrain
        
        return result


def detect_data_drift() -> dict:
    """Check for data drift between training data and new season data."""
    
    detector = DataDriftDetector()
    
    try:
        # Get historical data (pre-2026) as reference
        historical_games = pd.concat([
            DATA.get('m_regular', pd.DataFrame()),
            DATA.get('w_regular', pd.DataFrame())
        ])
        historical_games = historical_games[historical_games['Season'] < 2026]
        
        # Get current season data
        current_games = pd.concat([
            DATA.get('m_regular', pd.DataFrame()),
            DATA.get('w_regular', pd.DataFrame())
        ])
        current_games = current_games[current_games['Season'] == 2026]
        
        if len(current_games) == 0:
            return {
                'status': 'warning',
                'message': 'No current season data available for drift detection',
                'drift_detected': False,
                'needs_retraining': False
            }
        
        # Detect drift
        drift_result = detector.detect_drift(current_games, historical_games)
        
        return {
            'status': 'success',
            'drift_detected': drift_result['drift_detected'],
            'needs_retraining': drift_result['needs_retraining'],
            'drifted_columns': drift_result['number_of_drifted_columns'],
            'column_drifts': drift_result['column_drifts'],
            'recommendations': drift_result['recommendations'],
            'reports_dir': detector.report_dir,
            'timestamp': drift_result['timestamp'],
            'message': f"Drift analysis complete. {'⚠️ Retraining recommended.' if drift_result['needs_retraining'] else '✅ Model is stable.'}"
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error in drift detection: {str(e)}',
            'drift_detected': False,
            'needs_retraining': False,
            'error': str(e)
        }