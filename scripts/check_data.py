#!/usr/bin/env python
"""Analyse et visualisation des résultats du pipeline March Madness."""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# Ajouter le chemin src
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import DATA_DIR, MODEL_DIR

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_data():
    """Charge les données depuis les fichiers CSV."""
    print("📂 Chargement des données...")
    data = {}
    
    try:
        data['m_teams'] = pd.read_csv(f'{DATA_DIR}/MTeams.csv')
        data['w_teams'] = pd.read_csv(f'{DATA_DIR}/WTeams.csv')
        data['m_regular'] = pd.read_csv(f'{DATA_DIR}/MRegularSeasonCompactResults.csv')
        data['w_regular'] = pd.read_csv(f'{DATA_DIR}/WRegularSeasonCompactResults.csv')
        data['m_tourney'] = pd.read_csv(f'{DATA_DIR}/MNCAATourneyCompactResults.csv')
        data['w_tourney'] = pd.read_csv(f'{DATA_DIR}/WNCAATourneyCompactResults.csv')
        data['m_seeds'] = pd.read_csv(f'{DATA_DIR}/MNCAATourneySeeds.csv')
        data['w_seeds'] = pd.read_csv(f'{DATA_DIR}/WNCAATourneySeeds.csv')
        
        print(f"✅ Données chargées: {len(data['m_teams'])} équipes hommes, {len(data['w_teams'])} équipes femmes")
        return data
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None

def create_reports_dir():
    """Crée le dossier pour les rapports."""
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    return reports_dir

def plot_team_distribution(data):
    """Graphique de distribution des équipes hommes/femmes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Équipes hommes
    m_teams = data.get('m_teams', pd.DataFrame())
    if not m_teams.empty:
        m_teams['first_letter'] = m_teams['TeamName'].str[0]
        letter_counts = m_teams['first_letter'].value_counts().sort_index()
        
        axes[0].bar(letter_counts.index, letter_counts.values, color='steelblue', alpha=0.7)
        axes[0].set_title(f'Distribution équipes hommes (n={len(m_teams)})')
        axes[0].set_xlabel('Première lettre')
        axes[0].set_ylabel('Nombre')
    
    # Équipes femmes
    w_teams = data.get('w_teams', pd.DataFrame())
    if not w_teams.empty:
        w_teams['first_letter'] = w_teams['TeamName'].str[0]
        w_letter_counts = w_teams['first_letter'].value_counts().sort_index()
        
        axes[1].bar(w_letter_counts.index, w_letter_counts.values, color='coral', alpha=0.7)
        axes[1].set_title(f'Distribution équipes femmes (n={len(w_teams)})')
        axes[1].set_xlabel('Première lettre')
        axes[1].set_ylabel('Nombre')
    
    plt.tight_layout()
    reports_dir = create_reports_dir()
    plt.savefig(reports_dir / 'team_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_games_by_season(data):
    """Graphique du nombre de matchs par saison."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    datasets = [
        ('m_regular', 'Réguliers hommes', axes[0, 0], 'blue'),
        ('w_regular', 'Réguliers femmes', axes[0, 1], 'red'),
        ('m_tourney', 'Tournoi hommes', axes[1, 0], 'darkblue'),
        ('w_tourney', 'Tournoi femmes', axes[1, 1], 'darkred')
    ]
    
    for key, title, ax, color in datasets:
        df = data.get(key, pd.DataFrame())
        if not df.empty:
            by_season = df['Season'].value_counts().sort_index()
            ax.plot(by_season.index, by_season.values, 'o-', color=color, alpha=0.7)
            ax.fill_between(by_season.index, by_season.values, alpha=0.2, color=color)
            ax.set_title(title)
            ax.set_xlabel('Saison')
            ax.set_ylabel('Nombre')
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    reports_dir = create_reports_dir()
    plt.savefig(reports_dir / 'games_by_season.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_model():
    """Analyse du modèle entraîné."""
    model_path = Path(MODEL_DIR) / 'baseline_model.pkl'
    if not model_path.exists():
        print("⚠️ Modèle non trouvé")
        return
    
    model = joblib.load(model_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Coefficients du modèle
    features = ['Elo diff', 'Seed diff']
    coefs = model.coef_[0]
    
    bars = axes[0].bar(features, coefs, color=['blue', 'orange'])
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].set_title('Importance des features')
    axes[0].set_ylabel('Coefficient')
    
    # Ajouter les valeurs
    for bar, v in zip(bars, coefs):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{v:.6f}', ha='center', va='bottom' if v > 0 else 'top')
    
    # Simulation
    np.random.seed(42)
    elo_diffs = np.random.uniform(-300, 300, 1000)
    seed_diffs = np.random.uniform(-15, 15, 1000)
    X_test = np.column_stack([elo_diffs, seed_diffs])
    probs = model.predict_proba(X_test)[:, 1]
    
    scatter = axes[1].scatter(elo_diffs, seed_diffs, c=probs, cmap='RdBu', alpha=0.6, s=20)
    axes[1].set_xlabel('Différence Elo')
    axes[1].set_ylabel('Différence Seed')
    axes[1].set_title('Probabilité de victoire')
    plt.colorbar(scatter, ax=axes[1], label='P(Team1 gagne)')
    
    plt.tight_layout()
    reports_dir = create_reports_dir()
    plt.savefig(reports_dir / 'model_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n📊 RÉSUMÉ DU MODÈLE")
    print("="*40)
    print(f"Coefficient Elo diff: {coefs[0]:.6f}")
    print(f"Coefficient Seed diff: {coefs[1]:.6f}")
    print(f"Intercept: {model.intercept_[0]:.6f}")

def plot_submission_analysis():
    """Analyse du fichier de submission."""
    submission_path = Path('submission.csv')
    if not submission_path.exists():
        print("⚠️ Fichier submission.csv non trouvé")
        return
    
    df_sub = pd.read_csv(submission_path)
    df_sub['Season'] = df_sub['ID'].str[:4].astype(int)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribution des prédictions
    axes[0, 0].hist(df_sub['Pred'], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[0, 0].axvline(0.5, color='red', linestyle='--', label='50/50')
    axes[0, 0].set_title('Distribution des prédictions')
    axes[0, 0].set_xlabel('P(Team1 gagne)')
    axes[0, 0].set_ylabel('Fréquence')
    axes[0, 0].legend()
    
    # Stats
    stats_text = f"Moyenne: {df_sub['Pred'].mean():.4f}\nÉcart-type: {df_sub['Pred'].std():.4f}"
    axes[0, 0].text(0.02, 0.98, stats_text, transform=axes[0, 0].transAxes,
                   fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Par saison
    season_stats = df_sub.groupby('Season')['Pred'].agg(['mean', 'std'])
    axes[0, 1].errorbar(season_stats.index, season_stats['mean'], 
                        yerr=season_stats['std'], fmt='o-', capsize=5, color='purple')
    axes[0, 1].set_title('Prédictions moyennes par saison')
    axes[0, 1].set_xlabel('Saison')
    axes[0, 1].set_ylabel('P(Team1 gagne) moyenne')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Prédictions extrêmes
    extreme_preds = df_sub[(df_sub['Pred'] < 0.1) | (df_sub['Pred'] > 0.9)]
    if len(extreme_preds) > 0:
        axes[1, 0].hist(extreme_preds['Pred'], bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 0].set_title(f'Prédictions extrêmes (n={len(extreme_preds)})')
        axes[1, 0].set_xlabel('P(Team1 gagne)')
        axes[1, 0].set_ylabel('Fréquence')
    
    # Histogramme cumulé
    axes[1, 1].hist(df_sub['Pred'], bins=50, cumulative=True, density=True,
                    histtype='step', linewidth=2, color='teal')
    axes[1, 1].set_title('Distribution cumulative')
    axes[1, 1].set_xlabel('P(Team1 gagne)')
    axes[1, 1].set_ylabel('Probabilité cumulée')
    
    plt.tight_layout()
    reports_dir = create_reports_dir()
    plt.savefig(reports_dir / 'submission_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n📈 STATISTIQUES DES PRÉDICTIONS")
    print("="*40)
    print(f"Total: {len(df_sub):,} matchs")
    print(f"Moyenne: {df_sub['Pred'].mean():.4f}")
    print(f"Écart-type: {df_sub['Pred'].std():.4f}")
    print(f"Min: {df_sub['Pred'].min():.4f}")
    print(f"Max: {df_sub['Pred'].max():.4f}")

def generate_full_report():
    """Génère un rapport complet avec tous les graphiques."""
    print("\n" + "="*60)
    print("📊 GÉNÉRATION DU RAPPORT COMPLET")
    print("="*60)
    
    # Charger les données
    data = load_data()
    if data is None:
        return
    
    reports_dir = create_reports_dir()
    
    # 1. Distribution des équipes
    print("\n1️⃣ Distribution des équipes...")
    plot_team_distribution(data)
    
    # 2. Matchs par saison
    print("2️⃣ Matchs par saison...")
    plot_games_by_season(data)
    
    # 3. Analyse du modèle
    print("3️⃣ Analyse du modèle...")
    analyze_model()
    
    # 4. Analyse des prédictions
    print("4️⃣ Analyse des prédictions...")
    plot_submission_analysis()
    
    # Rapport HTML
    print("\n5️⃣ Création du rapport HTML...")
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rapport March Madness 2026</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            h1 {{ color: #1a237e; text-align: center; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .image-container {{ margin: 30px 0; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
            img {{ max-width: 100%; border: 1px solid #ddd; }}
            .footer {{ text-align: center; margin-top: 50px; color: #666; }}
            .timestamp {{ text-align: right; color: #666; font-style: italic; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏀 Rapport d'analyse March Madness 2026</h1>
            <div class="timestamp">Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}</div>
            
            <div class="image-container">
                <h2>Distribution des équipes</h2>
                <img src="team_distribution.png" alt="Distribution des équipes">
            </div>
            
            <div class="image-container">
                <h2>Matchs par saison</h2>
                <img src="games_by_season.png" alt="Matchs par saison">
            </div>
            
            <div class="image-container">
                <h2>Analyse du modèle</h2>
                <img src="model_analysis.png" alt="Analyse modèle">
            </div>
            
            <div class="image-container">
                <h2>Analyse des prédictions</h2>
                <img src="submission_analysis.png" alt="Analyse prédictions">
            </div>
            
            <div class="footer">
                Rapport généré automatiquement | Pipeline March Madness 2026
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(reports_dir / 'rapport_complet.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ Rapport généré dans '{reports_dir}/'")
    print("\n📁 Fichiers créés:")
    for f in sorted(reports_dir.glob('*')):
        print(f"   - {f.name}")
    
    print(f"\n🌐 Ouvre le rapport: {reports_dir.absolute()}/rapport_complet.html")

if __name__ == '__main__':
    generate_full_report()