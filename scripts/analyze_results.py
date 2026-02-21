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

from src.agents.data_loader_agent import DATA
from src.agents.feature_engineer_agent import ELO
from src.utils.config import MODEL_DIR

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def create_reports_dir():
    """Crée le dossier pour les rapports."""
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    return reports_dir

def plot_team_distribution():
    """Graphique de distribution des équipes hommes/femmes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Équipes hommes
    m_teams = DATA.get('m_teams', pd.DataFrame())
    if not m_teams.empty:
        # Distribution par première lettre
        m_teams['first_letter'] = m_teams['TeamName'].str[0]
        letter_counts = m_teams['first_letter'].value_counts().sort_index()
        
        axes[0].bar(letter_counts.index, letter_counts.values, color='steelblue', alpha=0.7)
        axes[0].set_title('Distribution des équipes hommes (par première lettre)')
        axes[0].set_xlabel('Première lettre')
        axes[0].set_ylabel('Nombre d\'équipes')
        axes[0].text(0.02, 0.98, f"Total: {len(m_teams)} équipes", 
                    transform=axes[0].transAxes, fontsize=10, verticalalignment='top')
    
    # Équipes femmes
    w_teams = DATA.get('w_teams', pd.DataFrame())
    if not w_teams.empty:
        w_teams['first_letter'] = w_teams['TeamName'].str[0]
        w_letter_counts = w_teams['first_letter'].value_counts().sort_index()
        
        axes[1].bar(w_letter_counts.index, w_letter_counts.values, color='coral', alpha=0.7)
        axes[1].set_title('Distribution des équipes femmes (par première lettre)')
        axes[1].set_xlabel('Première lettre')
        axes[1].set_ylabel('Nombre d\'équipes')
        axes[1].text(0.02, 0.98, f"Total: {len(w_teams)} équipes", 
                    transform=axes[1].transAxes, fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    reports_dir = create_reports_dir()
    plt.savefig(reports_dir / 'team_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_games_by_season():
    """Graphique du nombre de matchs par saison."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Matchs hommes réguliers
    m_reg = DATA.get('m_regular', pd.DataFrame())
    if not m_reg.empty:
        m_reg_by_season = m_reg['Season'].value_counts().sort_index()
        axes[0, 0].plot(m_reg_by_season.index, m_reg_by_season.values, 'o-', color='blue', alpha=0.7)
        axes[0, 0].fill_between(m_reg_by_season.index, m_reg_by_season.values, alpha=0.3)
        axes[0, 0].set_title('Matchs réguliers hommes par saison')
        axes[0, 0].set_xlabel('Saison')
        axes[0, 0].set_ylabel('Nombre de matchs')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Matchs femmes réguliers
    w_reg = DATA.get('w_regular', pd.DataFrame())
    if not w_reg.empty:
        w_reg_by_season = w_reg['Season'].value_counts().sort_index()
        axes[0, 1].plot(w_reg_by_season.index, w_reg_by_season.values, 'o-', color='red', alpha=0.7)
        axes[0, 1].fill_between(w_reg_by_season.index, w_reg_by_season.values, alpha=0.3)
        axes[0, 1].set_title('Matchs réguliers femmes par saison')
        axes[0, 1].set_xlabel('Saison')
        axes[0, 1].set_ylabel('Nombre de matchs')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Matchs tournoi hommes
    m_tourney = DATA.get('m_tourney', pd.DataFrame())
    if not m_tourney.empty:
        m_tourney_by_season = m_tourney['Season'].value_counts().sort_index()
        axes[1, 0].plot(m_tourney_by_season.index, m_tourney_by_season.values, 'o-', color='darkblue', alpha=0.7)
        axes[1, 0].fill_between(m_tourney_by_season.index, m_tourney_by_season.values, alpha=0.3)
        axes[1, 0].set_title('Matchs tournoi hommes par saison')
        axes[1, 0].set_xlabel('Saison')
        axes[1, 0].set_ylabel('Nombre de matchs')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Matchs tournoi femmes
    w_tourney = DATA.get('w_tourney', pd.DataFrame())
    if not w_tourney.empty:
        w_tourney_by_season = w_tourney['Season'].value_counts().sort_index()
        axes[1, 1].plot(w_tourney_by_season.index, w_tourney_by_season.values, 'o-', color='darkred', alpha=0.7)
        axes[1, 1].fill_between(w_tourney_by_season.index, w_tourney_by_season.values, alpha=0.3)
        axes[1, 1].set_title('Matchs tournoi femmes par saison')
        axes[1, 1].set_xlabel('Saison')
        axes[1, 1].set_ylabel('Nombre de matchs')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    reports_dir = create_reports_dir()
    plt.savefig(reports_dir / 'games_by_season.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_elo_ratings():
    """Visualisation des ratings Elo."""
    if not ELO:
        print("Pas de ratings Elo disponibles")
        return
    
    # Convertir ELO en DataFrame
    elo_data = []
    for (season, team_id), rating in ELO.items():
        elo_data.append({'Season': season, 'TeamID': team_id, 'Rating': rating})
    
    df_elo = pd.DataFrame(elo_data)
    
    # Top équipes par saison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Distribution des ratings
    axes[0].hist(df_elo['Rating'], bins=50, edgecolor='black', alpha=0.7, color='purple')
    axes[0].axvline(1500, color='red', linestyle='--', label='Rating initial (1500)')
    axes[0].set_title('Distribution des ratings Elo')
    axes[0].set_xlabel('Rating Elo')
    axes[0].set_ylabel('Fréquence')
    axes[0].legend()
    
    # Évolution par saison (moyenne)
    avg_rating = df_elo.groupby('Season')['Rating'].agg(['mean', 'std', 'count']).reset_index()
    axes[1].errorbar(avg_rating['Season'], avg_rating['mean'], 
                     yerr=avg_rating['std'], fmt='o-', capsize=5, alpha=0.7)
    axes[1].set_title('Évolution moyenne des ratings Elo')
    axes[1].set_xlabel('Saison')
    axes[1].set_ylabel('Rating Elo moyen')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    reports_dir = create_reports_dir()
    plt.savefig(reports_dir / 'elo_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Top 10 équipes hommes de tous les temps
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Hommes - Top 10
    m_names = dict(zip(DATA['m_teams']['TeamID'], DATA['m_teams']['TeamName']))
    m_elo_teams = df_elo[df_elo['TeamID'] < 2000]  # IDs hommes < 2000
    top_m = m_elo_teams.groupby('TeamID')['Rating'].max().sort_values(ascending=False).head(10)
    top_m_names = [m_names.get(tid, f"Team {tid}") for tid in top_m.index]
    
    axes[0].barh(range(len(top_m)), top_m.values, color='steelblue')
    axes[0].set_yticks(range(len(top_m)))
    axes[0].set_yticklabels(top_m_names)
    axes[0].set_title('Top 10 équipes hommes (meilleur rating historique)')
    axes[0].set_xlabel('Rating Elo')
    axes[0].invert_yaxis()
    
    # Femmes - Top 10
    w_names = dict(zip(DATA['w_teams']['TeamID'], DATA['w_teams']['TeamName']))
    w_elo_teams = df_elo[df_elo['TeamID'] >= 3000]  # IDs femmes >= 3000
    top_w = w_elo_teams.groupby('TeamID')['Rating'].max().sort_values(ascending=False).head(10)
    top_w_names = [w_names.get(tid, f"Team {tid}") for tid in top_w.index]
    
    axes[1].barh(range(len(top_w)), top_w.values, color='coral')
    axes[1].set_yticks(range(len(top_w)))
    axes[1].set_yticklabels(top_w_names)
    axes[1].set_title('Top 10 équipes femmes (meilleur rating historique)')
    axes[1].set_xlabel('Rating Elo')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(reports_dir / 'top_teams_elo.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_model():
    """Analyse du modèle entraîné."""
    model_path = MODEL_DIR / 'baseline_model.pkl'
    if not model_path.exists():
        print("Modèle non trouvé")
        return
    
    model = joblib.load(model_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Coefficients du modèle
    features = ['Elo diff', 'Seed diff']
    coefs = model.coef_[0]
    
    axes[0].bar(features, coefs, color=['blue', 'orange'])
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].set_title('Importance des features dans le modèle')
    axes[0].set_ylabel('Coefficient')
    
    # Ajouter les valeurs sur les barres
    for i, v in enumerate(coefs):
        axes[0].text(i, v + (0.001 if v > 0 else -0.01), f'{v:.6f}', 
                    ha='center', va='bottom' if v > 0 else 'top')
    
    # Simulation de prédictions
    np.random.seed(42)
    elo_diffs = np.random.uniform(-300, 300, 1000)
    seed_diffs = np.random.uniform(-15, 15, 1000)
    
    # Calculer les probabilités pour différentes combinaisons
    X_test = np.column_stack([elo_diffs, seed_diffs])
    probs = model.predict_proba(X_test)[:, 1]
    
    # Scatter plot
    scatter = axes[1].scatter(elo_diffs, seed_diffs, c=probs, cmap='RdBu', 
                              alpha=0.6, s=30, vmin=0, vmax=1)
    axes[1].set_xlabel('Différence Elo')
    axes[1].set_ylabel('Différence Seed')
    axes[1].set_title('Probabilité de victoire selon les features')
    plt.colorbar(scatter, ax=axes[1], label='P(Team1 gagne)')
    
    plt.tight_layout()
    reports_dir = create_reports_dir()
    plt.savefig(reports_dir / 'model_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Résumé du modèle
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DU MODÈLE")
    print("="*60)
    print(f"Type de modèle: {type(model).__name__}")
    print(f"Features: {features}")
    print(f"Coefficients: {dict(zip(features, coefs))}")
    print(f"Intercept: {model.intercept_[0]:.6f}")
    print(f"\nÉquation: log-odds = {model.intercept_[0]:.6f} + {coefs[0]:.6f}*Elo_diff + {coefs[1]:.6f}*Seed_diff")

def plot_submission_analysis():
    """Analyse du fichier de submission."""
    submission_path = Path('submission.csv')
    if not submission_path.exists():
        print("Fichier submission.csv non trouvé")
        return
    
    df_sub = pd.read_csv(submission_path)
    
    # Extraire les saisons et équipes
    df_sub['Season'] = df_sub['ID'].str[:4].astype(int)
    df_sub['Team1'] = df_sub['ID'].str[5:9].astype(int)
    df_sub['Team2'] = df_sub['ID'].str[10:14].astype(int)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribution des prédictions
    axes[0, 0].hist(df_sub['Pred'], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[0, 0].axvline(0.5, color='red', linestyle='--', label='50/50')
    axes[0, 0].set_title('Distribution des prédictions')
    axes[0, 0].set_xlabel('P(Team1 gagne)')
    axes[0, 0].set_ylabel('Nombre de matchs')
    axes[0, 0].legend()
    axes[0, 0].text(0.02, 0.98, f"Mean: {df_sub['Pred'].mean():.4f}\nStd: {df_sub['Pred'].std():.4f}", 
                   transform=axes[0, 0].transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Prédictions par saison
    season_stats = df_sub.groupby('Season')['Pred'].agg(['mean', 'std', 'count'])
    axes[0, 1].errorbar(season_stats.index, season_stats['mean'], 
                        yerr=season_stats['std'], fmt='o-', capsize=5, color='purple')
    axes[0, 1].set_title('Prédictions moyennes par saison')
    axes[0, 1].set_xlabel('Saison')
    axes[0, 1].set_ylabel('P(Team1 gagne) moyenne')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Heatmap des probabilités extrêmes
    extreme_preds = df_sub[(df_sub['Pred'] < 0.1) | (df_sub['Pred'] > 0.9)]
    if len(extreme_preds) > 0:
        axes[1, 0].hist(extreme_preds['Pred'], bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 0].set_title('Prédictions extrêmes (<0.1 ou >0.9)')
        axes[1, 0].set_xlabel('P(Team1 gagne)')
        axes[1, 0].set_ylabel('Nombre de matchs')
        axes[1, 0].text(0.02, 0.98, f"Total: {len(extreme_preds)}", 
                       transform=axes[1, 0].transAxes, fontsize=10, verticalalignment='top')
    
    # Distribution des équipes
    all_teams = pd.concat([df_sub['Team1'], df_sub['Team2']]).value_counts().head(20)
    axes[1, 1].barh(range(len(all_teams)), all_teams.values, color='teal')
    axes[1, 1].set_yticks(range(len(all_teams)))
    axes[1, 1].set_yticklabels([f"Team {t}" for t in all_teams.index])
    axes[1, 1].set_title('Top 20 équipes les plus fréquentes')
    axes[1, 1].set_xlabel('Nombre d\'apparitions')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    reports_dir = create_reports_dir()
    plt.savefig(reports_dir / 'submission_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Statistiques textuelles
    print("\n" + "="*60)
    print("📈 STATISTIQUES DES PRÉDICTIONS")
    print("="*60)
    print(f"Nombre total de prédictions: {len(df_sub):,}")
    print(f"Moyenne des prédictions: {df_sub['Pred'].mean():.4f}")
    print(f"Médiane: {df_sub['Pred'].median():.4f}")
    print(f"Écart-type: {df_sub['Pred'].std():.4f}")
    print(f"Min: {df_sub['Pred'].min():.4f}")
    print(f"Max: {df_sub['Pred'].max():.4f}")
    print(f"Prédictions > 0.9: {(df_sub['Pred'] > 0.9).sum():,} ({(df_sub['Pred'] > 0.9).mean()*100:.1f}%)")
    print(f"Prédictions < 0.1: {(df_sub['Pred'] < 0.1).sum():,} ({(df_sub['Pred'] < 0.1).mean()*100:.1f}%)")

def generate_full_report():
    """Génère un rapport complet avec tous les graphiques."""
    print("\n" + "="*60)
    print("📊 GÉNÉRATION DU RAPPORT COMPLET")
    print("="*60)
    
    reports_dir = create_reports_dir()
    
    # 1. Distribution des équipes
    print("\n1️⃣ Génération: Distribution des équipes...")
    plot_team_distribution()
    
    # 2. Matchs par saison
    print("2️⃣ Génération: Matchs par saison...")
    plot_games_by_season()
    
    # 3. Ratings Elo
    print("3️⃣ Génération: Analyse des ratings Elo...")
    plot_elo_ratings()
    
    # 4. Analyse du modèle
    print("4️⃣ Génération: Analyse du modèle...")
    analyze_model()
    
    # 5. Analyse des prédictions
    print("5️⃣ Génération: Analyse des prédictions...")
    plot_submission_analysis()
    
    # Rapport HTML
    print("\n📝 Création du rapport HTML...")
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rapport March Madness 2026</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            h1 {{ color: #1a237e; text-align: center; }}
            h2 {{ color: #0d47a1; border-bottom: 2px solid #bbdefb; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .image-container {{ margin: 30px 0; text-align: center; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            .footer {{ text-align: center; margin-top: 50px; color: #666; font-size: 0.9em; }}
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
                <h2>Distribution des ratings Elo</h2>
                <img src="elo_distribution.png" alt="Distribution Elo">
            </div>
            
            <div class="image-container">
                <h2>Top équipes par rating Elo</h2>
                <img src="top_teams_elo.png" alt="Top équipes Elo">
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
                Rapport généré automatiquement par le pipeline March Madness 2026
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(reports_dir / 'rapport_complet.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ Rapport complet généré dans le dossier '{reports_dir}'")
    print(f"📁 Chemin: {reports_dir.absolute()}")
    print(f"📊 Fichiers générés:")
    for f in reports_dir.glob('*'):
        print(f"   - {f.name}")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🔍 ANALYSE DES RÉSULTATS DU PIPELINE")
    print("="*60)
    
    # Vérifier que les données sont chargées
    if not DATA:
        print("⚠️  Les données ne sont pas chargées. Exécute d'abord le pipeline.")
        print("   python scripts/run_pipeline.py --all")
        sys.exit(1)
    
    generate_full_report()
    
    print("\n" + "="*60)
    print("✅ Analyse terminée !")
    print("📁 Ouvre le dossier 'reports' pour voir tous les graphiques")
    print("🌐 Ouvre 'reports/rapport_complet.html' dans ton navigateur")
    print("="*60)