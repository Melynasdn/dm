# ----------------- IMPORTS -----------------

# === Interface ===
from nicegui import ui

# === Outils de base ===
import io
import os
import copy
import time
import math
import base64
import random
import pickle
import asyncio
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import skew, iqr
    
# === Manipulation de données ===
import pandas as pd
import numpy as np

# === Visualisation ===
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === Statistiques & Analyse ===
from scipy import stats
from scipy.stats import (
    iqr, skew, boxcox, yeojohnson,
    chi2_contingency, f_oneway, pearsonr
)
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# === Préprocessing ===
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    LabelEncoder, OrdinalEncoder, OneHotEncoder, PowerTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectKBest, f_classif, chi2

# === Modélisation & Évaluation ===
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)

# === Algorithmes de Machine Learning ===
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids

# === Clustering hiérarchique ===
from scipy.cluster.hierarchy import linkage, dendrogram


# ----------------- IMPORTS -----------------

# Interface
from nicegui import ui

# Manipulation de données
import pandas as pd
import numpy as np

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import linkage, dendrogram

# Prétraitement
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from scipy.stats import zscore, skew, kurtosis

# Réduction de dimension
from sklearn.decomposition import PCA

# Clustering (non supervisé)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids

# Métriques de qualité de clustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from itertools import combinations

# Utilitaires
import io
import base64
import random

# ----------------- ÉTAT GLOBAL -----------------
state = {
    "raw_df": None,
    "numeric_columns": None,
    "X": None,
    "X_pca": None,
    "results": {},
    "optimal_k": {}  
}


# ----------------- GLOBAL DATA STORAGE -----------------
class GlobalData:
    df = None
    df_train = None
    df_val = None
    df_test = None

global_data = GlobalData()
# --------------------------------------------------------


def ensure_original_saved():
    """
    Sauvegarde df_original si pas déjà fait.
    À appeler au début de chaque fonction de transformation.
    """
    if "df_original" not in state:
        if "raw_df" in state:
            state["df_original"] = state["raw_df"].copy()
            print(" df_original sauvegardé automatiquement")
        else:
            print(" Impossible de sauvegarder df_original : raw_df absent")

# ----------------- FONCTIONS UTILITAIRES -----------------
# ... (plot_to_base64, preprocess_dataframe, scatter_plot_2d restent inchangées)

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

def preprocess_dataframe(df, handle_missing=True, normalization=None, encode_onehot=False, drop_duplicates=False):
    df_copy = df.replace('?', np.nan).copy()
    for col in df_copy.columns:
        try:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        except: pass

    if drop_duplicates: 
        df_copy = df_copy.drop_duplicates()

    if handle_missing:
        for col in df_copy.columns:
            if df_copy[col].dtype in [np.float64, np.int64]:
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
            else:
                df_copy[col].fillna(df_copy[col].mode()[0] if len(df_copy[col].mode())>0 else 'Unknown', inplace=True)

    if encode_onehot:
        cat_cols = df_copy.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            df_copy = pd.get_dummies(df_copy, columns=cat_cols)

    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    if normalization == 'zscore':
        df_copy[numeric_cols] = StandardScaler().fit_transform(df_copy[numeric_cols])
    elif normalization == 'minmax':
        df_copy[numeric_cols] = MinMaxScaler().fit_transform(df_copy[numeric_cols])

    X_numeric = df_copy.select_dtypes(include=[np.number])
    return df_copy, X_numeric.values.astype(np.float64)

def scatter_plot_2d(X_pca, labels, title):
    unique_labels = np.unique(labels)
    colors = ListedColormap(plt.cm.get_cmap('tab10').colors)
    fig, ax = plt.subplots(figsize=(5,5))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(X_pca[mask,0], X_pca[mask,1], label=f'Cluster {label}', color=colors(i%10))
    ax.set_title(title)
    ax.legend()
    return plot_to_base64(fig)

# NOTE: La fonction 'diagnose_dbscan' n'est pas définie dans app.py, elle doit être importée ou définie.
# Pour la démo, je vais la définir avec un retour simple pour éviter une erreur d'exécution.
def diagnose_dbscan(X):
    # Simuler le diagnostic, si vous n'avez pas le fichier custom_cluster_optimized.
    # Vous devriez utiliser une méthode comme K-distance plot pour une estimation réelle.
    return {'suggested_eps': 0.5} 

# ----------------- FONCTION MODIFIÉE: PLOT ELBOW (utilise scikit-learn) -----------------
def plot_elbow_curve(X, max_k=10, algo='kmeans'):
    """Génère le graphique Elbow Method et trouve le k optimal"""
    inertias = []
    
    for k in range(1, max_k + 1):
        if algo == 'kmeans':
            model = KMeans(n_clusters=k, random_state=0, n_init='auto') # Utilisation de scikit-learn KMeans
        else:  # kmedoids
            # Utilisation de scikit-learn-extra KMedoids
            model = KMedoids(n_clusters=k, random_state=0, method='pam')
        
        model.fit(X)
        inertias.append(model.inertia_) # KMeans et KMedoids de sklearn-extra ont l'attribut inertia_
    
    # Trouver le coude en utilisant la méthode de la dérivée seconde
    # Reste inchangé, mais peut être moins précis que des méthodes plus sophistiquées.
    differences = np.diff(inertias)
    second_diff = np.diff(differences)
    optimal_k = np.argmax(second_diff) + 2  # +2 car on a perdu 2 indices avec les diff
    
    # Si le k optimal est trop proche des bords, ajuster
    if optimal_k < 2:
        optimal_k = 2
    if optimal_k > max_k - 1:
        optimal_k = max_k - 1
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, max_k + 1), inertias, 'bo-', linewidth=2, markersize=8)
    ax.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, 
               label=f'K optimal suggéré = {optimal_k}')
    ax.scatter([optimal_k], [inertias[optimal_k-1]], color='red', s=200, zorder=5)
    ax.set_xlabel('Nombre de clusters (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Inertie (SSE)', fontsize=12, fontweight='bold')
    ax.set_title(f'Méthode du Coude - {algo.upper()}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    return plot_to_base64(fig), optimal_k



# ----------------- PAGE INDEX -----------------
@ui.page('/')
def home_page():

    # ---------------- HEADER ----------------
    with ui.header().classes("w-full justify-between items-center px-8 py-4 shadow-sm").style(
        "background-color:white !important;"
    ):
        ui.label("DataVision AI").style(
            "font-size:22px !important; font-weight:700 !important; color:#01335A !important;"
        )

        with ui.row().classes("gap-6 items-center"):
            ui.link("Documentation", "https://example.com").style(
                "color:#01335A !important; text-decoration:none !important;"
            ).classes("hover:underline")

            ui.link("Contact", "mailto:support@datavision.ai").style(
                "color:#01335A !important; text-decoration:none !important;"
            ).classes("hover:underline")

    # ---------------- PAGE CONTENT ----------------
    with ui.column().classes("w-full min-h-screen items-center justify-center gap-12 p-10").style(
        """
        background: url('https://raw.githubusercontent.com/creotiv/misc-assets/main/ai-lines-light.svg')
        no-repeat center/cover !important;
        background-color: #f5f6fa !important;
        font-family: 'Inter', sans-serif !important;
        """
    ):

        # ----------- TITRES -----------
        ui.label("Plateforme d'Analyse de Données").style(
            "font-weight:700 !important; font-size:40px !important; color:#01335A !important; "
            "text-align:center !important;"
        )

        ui.label("Choisissez un mode d'apprentissage pour commencer").style(
            "color:#09538C !important; font-size:18px !important; text-align:center !important;"
        )

        # ----------- CARTES -----------
        with ui.row().classes("gap-14 flex-wrap justify-center mt-4"):

            # ░░░░░░░░░░░░ NON SUPERVISÉ ░░░░░░░░░░░░
            with ui.card().classes(
                "p-8 w-120 shadow-md rounded-xl transition transform hover:scale-105 hover:shadow-xl "
                "flex flex-col items-center justify-center"
            ).style(
                "background-color:rgba(255,255,255,0.9) !important;"
                "backdrop-filter:blur(6px) !important;"
                "transition:all 0.3s ease !important;"
            ):
                ui.label("🔍").style("font-size:50px !important; color:#09538C !important;")

                ui.label("Non Supervisé").style(
                    "font-size:22px !important; font-weight:600 !important; color:#01335A !important; "
                    "margin-top:8px !important;"
                )

                ui.label("Analyse sans labels (clustering, anomalies)").style(
                    "font-size:15px !important; color:#636e72 !important; margin-top:10px !important;"
                    "text-align:center !important;"
                )

                ui.label("Exemples : segmentation clients, détection d'anomalies").style(
                    "font-size:15px !important; color:#636e72 !important; text-align:center !important;"
                )

                with ui.row().classes("gap-4 mt-6 w-full justify-center"):

                    ui.button(
                        "Commencer",
                        on_click=lambda: ui.run_javascript(
                            "window.location.href='/unsupervised/upload'"
                        )
                    ).style(
                        "background: linear-gradient(135deg, #01335A, #09538C) !important;"
                        "color:white !important; font-weight:600 !important;"
                        "height:48px !important; width:100% !important; border-radius:8px !important;"
                    )

                    ui.button(
                        "Voir Exemple",
                        on_click=lambda: ui.notify(
                            "Exemple : Analyse clustering avec Iris dataset.",
                            color="info",
                        )
                    ).style(
                        "background-color:#dfe6e9 !important; color:#2d3436 !important;"
                        "font-weight:500 !important; height:48px !important; width:100% !important;"
                        "border-radius:8px !important;"
                    )

            # ░░░░░░░░░░░░ SUPERVISÉ ░░░░░░░░░░░░
            with ui.card().classes(
                "p-8 w-120 shadow-md rounded-xl transition transform hover:scale-105 hover:shadow-xl "
                "flex flex-col items-center justify-center"
            ).style(
                "background-color:rgba(255,255,255,0.9) !important;"
                "backdrop-filter:blur(6px) !important;"
                "transition:all 0.3s ease !important;"
            ):
                ui.label("🧠").style("font-size:50px !important; color:#09538C !important;")

                ui.label("Supervisé").style(
                    "font-size:22px !important; font-weight:600 !important; color:#01335A !important;"
                    "margin-top:8px !important;"
                )

                ui.label("Prédiction supervisée avec labels").style(
                    "font-size:15px !important; color:#636e72 !important; margin-top:10px !important;"
                    "text-align:center !important;"
                )

                ui.label("Exemples : email spam, prédiction de prix").style(
                    "font-size:15px !important; color:#636e72 !important; text-align:center !important;"
                )

                with ui.row().classes("gap-4 mt-6 w-full justify-center"):

                    ui.button(
                        "Commencer",
                        on_click=lambda: ui.run_javascript(
                            "window.location.href='/supervised/upload'"
                        )
                    ).style(
                        "background: linear-gradient(135deg, #01335A, #09538C) !important;"
                        "color:white !important; font-weight:600 !important;"
                        "height:48px !important; width:100% !important; border-radius:8px !important;"
                    )

                    ui.button(
                        "Voir Exemple",
                        on_click=lambda: ui.notify(
                            "Exemple : Prédiction de prix avec Boston Housing.",
                            color="info",
                        )
                    ).style(
                        "background-color:#dfe6e9 !important; color:#2d3436 !important;"
                        "font-weight:500 !important; height:48px !important; width:100% !important;"
                        "border-radius:8px !important;"
                    )




# ----------------- PAGE UPLOAD ---------------



@ui.page('/supervised/upload')
def supervised_upload_page():


    # Conteneur principal
    with ui.column() as main_col:
        main_col.style(
            """
            width: 100% !important;
            min-height: 100vh !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: flex-start !important;
            background-color: #f5f6fa !important;
            padding-top: 60px !important;
            font-family: 'Inter', sans-serif !important;
            """
        )

        # Titre
        ui.label("Phase 2 : Chargement et Validation des Données").style(
            """
            font-weight: 700 !important;
            font-size: 32px !important;
            color: #01335A !important;
            margin-bottom: 24px !important;
            text-align: center !important;
            """
        )

        # Carte centrale
        with ui.card() as card:
            card.style(
                """
                padding: 32px !important;
                width: 800px !important;
                border-radius: 12px !important;
                box-shadow: 0 4px 20px rgba(0,0,0,0.12) !important;
                display: flex !important;
                flex-direction: column !important;
                align-items: center !important;
                background-color: white !important;
                """
            )

            ui.label("📂 Importer un Dataset CSV").style(
                """
                font-weight: 700 !important;
                font-size: 24px !important;
                color: #01335A !important;
                text-align: center !important;
                margin-bottom: 10px !important;
                """
            )

            ui.label("Glissez-déposez un fichier CSV ou cliquez pour parcourir").style(
                """
                color: #09538C !important;
                font-size: 15px !important;
                text-align: center !important;
                margin-bottom: 24px !important;
                """
            )

            status_label = ui.label("Aucun fichier chargé").style(
                """
                color: #e74c3c !important;
                font-size: 14px !important;
                margin-bottom: 18px !important;
                font-weight: 600 !important;
                """
            )

            table_placeholder = ui.column().style(
                """
                width: 100% !important;
                margin-top: 20px !important;
                border-top: 1px solid #ecf0f1 !important;
                padding-top: 20px !important;
                """
            )

            async def on_upload(e):
                try:
                    content = await e.file.read()
                    df = pd.read_csv(io.BytesIO(content))
                    state["raw_df"] = df

                    # Mise à jour du statut
                    status_label.text = f"Fichier chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes"
                    status_label.style(
                        """
                        color: #01335A !important;
                        font-size: 14px !important;
                        margin-bottom: 18px !important;
                        font-weight: 600 !important;
                        """
                    )

                    # Affichage aperçu
                    table_placeholder.clear()
                    with table_placeholder:
                        ui.label("Aperçu des 10 premières lignes :").style(
                            """
                            font-weight: 600 !important;
                            color: #01335A !important;
                            font-size: 16px !important;
                            margin-bottom: 8px !important;
                            """
                        )
                        ui.table(rows=df.head(10).to_dict(orient="records")).style(
                            """
                            width: 100% !important;
                            font-size: 14px !important;
                            border: 1px solid #dfe6e9 !important;
                            border-radius: 6px !important;
                            background-color: #fafafa !important;
                            """
                        )

                    btn_next.enable()
                    ui.notify("Dataset chargé avec succès !", color='positive')

                except Exception as err:
                    ui.notify(f"Erreur lors de l'import : {err}", color='negative')

            # Zone de drag & drop
            ui.upload(
                on_upload=on_upload,
                label="Glissez-déposez un fichier CSV ou cliquez ici"
            ).style(
                """
                width: 100% !important;
                padding: 24px !important;
                margin-bottom: 24px !important;
                border: 2px dashed #09538C !important;
                border-radius: 10px !important;
                text-align: center !important;
                font-size: 15px !important;
                color: #01335A !important;
                cursor: pointer !important;
                background-color: #fdfdfd !important;
                transition: all 0.2s ease !important;
                """
            ).props('accept=".csv"')

            # Boutons navigation
            with ui.row() as buttons_row:
                buttons_row.style(
                    """
                    width: 100% !important;
                    display: flex !important;
                    gap: 16px !important;
                    margin-top: 12px !important;
                    """
                )

                btn_next = ui.button("Continuer ")
                btn_next.disable()
                btn_next.style(
                    """
                    width: 100% !important;
                    height: 48px !important;
                    margin-top: 14px !important;
                    border-radius: 8px !important;
                    background: linear-gradient(0deg,rgba(1, 51, 90, 1) 0%, rgba(15, 50, 102, 1) 100%) !important;
                    color: white !important;
                    font-weight: 600 !important;
                    font-size: 15px !important;
                    border: none !important;
                    cursor: pointer !important;
                    """
                )

                ui.button(
                    " Retour",
                    on_click=lambda: ui.run_javascript("window.location.href='/'")
                ).style(
                    """
                    width: 100% !important;
                    height: 48px !important;
                    border-radius: 8px !important;
                    background: #dfe6e9 !important;
                    color: #2c3e50 !important;
                    font-weight: 600 !important;
                    border: none !important;
                    cursor: pointer !important;
                    """
                )

                btn_next.on_click(lambda: ui.run_javascript("window.location.href='/supervised/preprocessing'"))




# ----------------- PAGE PREPROCESS (SUPERVISE) -----------------

@ui.page('/supervised/preprocessing')
def supervised_preprocessing_page():
    df = state.get("raw_df", None)

    if df is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Aucun dataset chargé. Veuillez importer un fichier avant de continuer.").style(
                "font-size:18px !important; color:#c0392b !important; font-weight:600 !important;"
            )
            ui.button(" Retour à l'Upload", on_click=lambda: ui.run_javascript("window.location.href='/supervised/upload'")).style(
                "margin-top:20px !important; background:#01335A !important; color:white !important; font-weight:600 !important;"
            )
        return

    # ----------- STYLES GÉNÉRAUX -----------
    with ui.column().classes("w-full h-auto items-center p-10").style(
        "background-color:#f5f6fa !important; font-family:'Inter', sans-serif !important;"
    ):

        ui.label("Regardons notre DATA de plus PRES").style(
            "font-weight:700 !important; font-size:32px !important; color:#01335A !important; margin-bottom:32px !important; text-align:center !important;"
        )

        # ---------- SECTION A : VUE D'ENSEMBLE ----------
        with ui.card().classes("w-full max-w-5xl p-6 mb-8").style(
            "background-color:white !important; border-radius:12px !important; box-shadow:0 4px 15px rgba(0,0,0,0.08) !important;"
        ):
            ui.label(" Vue d’Ensemble du Dataset").style(
                "font-weight:700 !important; font-size:22px !important; color:#01335A !important; margin-bottom:12px !important;"
            )

            n_rows, n_cols = df.shape
            mem_mb = round(df.memory_usage(deep=True).sum() / 1e6, 2)

            with ui.row().classes("justify-around w-full mt-2"):
                def metric(label, value, color):
                    with ui.column().classes("items-center"):
                        ui.label(label).style("font-size:15px !important; color:#636e72 !important;")
                        ui.label(value).style(f"font-weight:700 !important; font-size:20px !important; color:{color} !important;")
                metric("Nombre de lignes", f"{n_rows:,}", "#01335A")
                metric("Nombre de colonnes", f"{n_cols:,}", "#01335A")
                metric("Taille mémoire", f"{mem_mb} MB", "#09538C")

        # ---------- SECTION B : TABLEAU DES COLONNES ----------
        with ui.card().classes("w-full max-w-6xl p-6").style(
            "background-color:white !important; border-radius:12px !important; box-shadow:0 4px 15px rgba(0,0,0,0.08) !important;"
        ):
            ui.label("🧾 Schéma du Dataset (Colonnes)").style(
                "font-weight:700 !important; font-size:22px !important; color:#01335A !important; margin-bottom:16px !important;"
            )

            # --------- CALCUL DES INFORMATIONS DE COLONNES ---------
            columns_info = []

            for col in df.columns:
                series = df[col]
                series_clean = series.replace('', np.nan).replace('?', np.nan)
                conv = pd.to_numeric(series_clean.dropna(), errors='coerce')
                detected_type = "Texte"  # default

                if conv.notna().sum() / max(1, len(series_clean.dropna())) >= 0.9:
                    n_unique = conv.nunique(dropna=True)
                    ratio_unique = n_unique / max(1, len(series_clean))
                    integer_like = np.all(np.abs(conv - conv.round()) < 1e-6)

                    if integer_like:
                        detected_type = "Numérique Discrète"
                    else:
                        if n_unique > 20 or ratio_unique > 0.05:
                            detected_type = "Numérique Continue"
                        else:
                            detected_type = "Numérique Discrète"

                missing_pct = round(series_clean.isna().mean() * 100, 2)
                cardinality = int(series_clean.nunique(dropna=True))
                unique_vals = series_clean.dropna().astype(str).unique()
                examples = ", ".join(unique_vals[:3]) if len(unique_vals) > 0 else ""

                columns_info.append({
                    "Colonne": col,
                    "Type Détecté": detected_type,
                    "Type Réel": detected_type,
                    "Rôle": "Feature",
                    "% Missing": f"{missing_pct}%",
                    "Cardinalité": cardinality,
                    "Exemples": examples
                })

            # Sauvegarde pour la page suivante
            state["columns_info"] = columns_info

            # --------- TABLEAU ---------
            ui.table(
                columns=[
                    {"name": "Colonne", "label": "Colonne", "field": "Colonne", "sortable": True},
                    {"name": "Type Détecté", "label": "Type Détecté", "field": "Type Détecté"},
                    {"name": "Type Réel", "label": "Type Réel", "field": "Type Réel"},
                    {"name": "Rôle", "label": "Rôle", "field": "Rôle"},
                    {"name": "% Missing", "label": "% Missing", "field": "% Missing"},
                    {"name": "Cardinalité", "label": "Cardinalité", "field": "Cardinalité"},
                    {"name": "Exemples", "label": "Exemples", "field": "Exemples"},
                ],
                rows=columns_info,
                row_key="Colonne",
            ).style(
                "width:100% !important; font-size:14px !important; background:#fafafa !important; border-radius:8px !important;"
            )

        # ---------- BOUTONS NAVIGATION ----------
        with ui.row().classes("justify-between w-full max-w-6xl mt-8"):
            ui.button(" Retour", on_click=lambda: ui.run_javascript("window.location.href='/supervised/upload'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )

            ui.button(" Étape suivante", on_click=lambda: ui.run_javascript("window.location.href='/supervised/user_decisions'")).style(
                "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )


# ----------------- PAGE /supervised/user_decisions -----------------

def map_detected_type(detected_type):
    # Remap pour dropdown
    if detected_type in ["Texte", "Catégorielle / Texte"]:
        return "Texte"
    elif detected_type == "Numérique Discrète":
        return "Numérique Discrète"
    elif detected_type == "Numérique Continue":
        return "Numérique Continue"
    else:
        return "Texte"


@ui.page('/supervised/user_decisions')
def user_decisions_page():
    import pandas as pd
    import numpy as np

    df = state.get("raw_df", None)
    columns_info = state.get("columns_info", None)

    if df is None or columns_info is None:
        with ui.column().classes("w-full h-screen items-center justify-center"):
            ui.label("❌ Aucun dataset chargé ou informations de colonnes manquantes.").style(
                "font-size:18px !important; color:#c0392b !important; font-weight:600 !important;"
            )
            ui.button(
                "Retour à l'Upload",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/upload'")
            ).style(
                "margin-top:20px !important; background:#01335A !important; color:white !important; "
                "font-weight:600 !important; padding:12px 32px !important; border-radius:8px !important;"
            )
        return

    # ---------------------- CORRECTIONS NUMÉRIQUES ----------------------
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            non_null = df[col].dropna()
            if len(non_null) > 0 and (non_null % 1 == 0).all():
                df[col] = df[col].astype('Int64')

    state["raw_df"] = df

    default_target = columns_info[-1]["Colonne"] if columns_info else None

    # ---------------------- FONCTIONS INTERNES ----------------------
    def detect_actual_type(col_info, col_name):
        col_data = df[col_name]
        n_unique = col_data.nunique()
        n_total = len(col_data.dropna())

        if n_unique == 1:
            return "Identifiant"
        if col_name.lower().startswith('id') or col_name.lower().endswith('_id'):
            return "Identifiant"
        if n_total > 0 and n_unique / n_total > 0.95:
            return "Identifiant"

        if pd.api.types.is_numeric_dtype(col_data):
            non_null = col_data.dropna()
            if len(non_null) > 0:
                is_integer = (non_null % 1 == 0).all()
                if is_integer:
                    if n_unique <= 20:
                        return "Numérique Discrète"
                    else:
                        unique_sorted = sorted(non_null.unique())
                        if len(unique_sorted) > 1:
                            gaps = np.diff(unique_sorted)
                            avg_gap = np.mean(gaps)
                            if avg_gap <= 5 and np.std(gaps) < avg_gap:
                                return "Numérique Continue"
                            else:
                                return "Numérique Discrète"
                        return "Numérique Discrète"
                else:
                    return "Numérique Continue"

        if pd.api.types.is_datetime64_any_dtype(col_data):
            return "Date/Datetime"

        if col_data.dtype == 'object' or pd.api.types.is_categorical_dtype(col_data):
            values_lower = [str(v).lower() for v in col_data.unique()[:20]]
            ordinal_patterns = [
                ['low', 'medium', 'high'],
                ['bad', 'good', 'excellent'],
                ['small', 'medium', 'large'],
                ['never', 'sometimes', 'often', 'always'],
                ['poor', 'fair', 'good', 'excellent'],
                ['xs', 's', 'm', 'l', 'xl', 'xxl']
            ]
            for pattern in ordinal_patterns:
                if any(p in values_lower for p in pattern):
                    return "Catégorielle Ordinale"
            return "Catégorielle Nominale"

        return "Texte"

    def on_target_change(target_col):
        if target_col is None:
            target_warning.text = ""
            imbalance_label.text = "Sélectionnez la target pour voir la distribution"
            smote_cb.disable()
            return

        n_unique = df[target_col].nunique(dropna=True)
        if n_unique > 20:
            target_warning.text = "⚠️ Plus de 20 valeurs uniques → Cela semble être une régression"
            imbalance_label.text = ""
            smote_cb.disable()
        else:
            target_warning.text = ""
            counts = df[target_col].value_counts()
            total = counts.sum()

            distribution_text = "📊 Distribution des classes : "
            for k, v in counts.items():
                pct = (v / total * 100)
                distribution_text += f"{k}: {v} ({pct:.1f}%) | "
            distribution_text = distribution_text.rstrip(" | ")
            imbalance_label.text = distribution_text

            min_class = counts.min()
            max_class = counts.max()
            if max_class / min_class > 1.5:
                smote_cb.enable()
                imbalance_label.text += "\n⚠️ Déséquilibre détecté - SMOTE recommandé"
                imbalance_label.style("color:#e74c3c !important; font-weight:600 !important;")
            else:
                smote_cb.enable()
                imbalance_label.text += "\n✅ Classes équilibrées"
                imbalance_label.style("color:#27ae60 !important; font-weight:600 !important;")

    def on_confirm():
        target_col = target_dropdown.value
        if target_col is None:
            ui.notify("⚠️ Veuillez sélectionner une colonne target", color="warning")
            return

        state["target_column"] = target_col

        for col_name, widget in column_type_widgets.items():
            state.setdefault("columns_types", {})[col_name] = widget.value

        columns_to_exclude = {}
        for col_name, cb in column_exclude_widgets.items():
            columns_to_exclude[col_name] = cb.value
        
        state["columns_exclude"] = columns_to_exclude

        cols_to_drop = [col for col, exclude in columns_to_exclude.items() if exclude and col in df.columns and col != target_col]
        
        if cols_to_drop:
            print(f"\n🗑️ EXCLUSION DE {len(cols_to_drop)} COLONNES:")
            print(f"    Colonnes: {cols_to_drop}")
            
            current_df = state["raw_df"].copy()
            current_df = current_df.drop(columns=cols_to_drop, errors='ignore')
            state["raw_df"] = current_df
            print(f"✅ raw_df: {current_df.shape}")
            
            split_data = state.get("split", {})
            if split_data:
                for split_key in ["X_train", "X_val", "X_test"]:
                    if split_key in split_data:
                        split_df = split_data[split_key]
                        cols_in_split = [c for c in cols_to_drop if c in split_df.columns]
                        if cols_in_split:
                            split_data[split_key] = split_df.drop(columns=cols_in_split, errors='ignore')
                            print(f"✅ {split_key}: {split_data[split_key].shape}")
                
                state["split"] = split_data
            
            ui.notify(f"🗑️ {len(cols_to_drop)} colonne(s) exclue(s)", color="info", timeout=2000)

        state["apply_smote"] = smote_cb.value

        ui.notify("✅ Décisions enregistrées avec succès !", color="positive")
         
        ui.run_javascript("setTimeout(() => window.location.href='/supervised/split', 1000);")

    def go_to_split():
        on_confirm()

    # ---------------------- UI ----------------------
    with ui.column().classes("w-full items-center").style(
        "min-height:100vh !important; background:#f0f2f5 !important; padding:48px 24px !important; "
        "font-family:'Inter', sans-serif !important;"
    ):
        ui.label("Configuration de l'Analyse").style(
            "font-weight:700 !important; font-size:36px !important; color:#01335A !important; "
            "margin-bottom:8px !important; text-align:center !important; letter-spacing:-0.5px !important;"
        )
        ui.label("Sélection de la target et configuration des types de colonnes").style(
            "font-size:16px !important; color:#636e72 !important; margin-bottom:48px !important; "
            "text-align:center !important; font-weight:400 !important;"
        )

        # ---------- SÉLECTION DE LA TARGET ----------
        with ui.card().classes("w-full max-w-4xl mb-6").style(
            "background:white !important; border-radius:16px !important; padding:32px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            ui.label("🎯 Sélection de la colonne Target").style(
                "font-weight:700 !important; font-size:22px !important; color:#01335A !important; "
                "margin-bottom:16px !important;"
            )
            ui.label("La colonne target est la variable que vous souhaitez prédire").style(
                "font-size:14px !important; color:#636e72 !important; margin-bottom:20px !important;"
            )

            target_dropdown = ui.select(
                options=[col["Colonne"] for col in columns_info],
                label="Colonne cible",
                value=default_target,
                on_change=lambda e: on_target_change(e.value)
            ).props("outlined").classes("w-full mb-4")

            target_warning = ui.label("").style(
                "color:#e74c3c !important; font-weight:600 !important; font-size:14px !important;"
            )

            imbalance_label = ui.label("").style(
                "font-size:14px !important; color:#2c3e50 !important; margin-top:12px !important;"
            )

            with ui.row().classes("w-full items-center gap-3 mt-6"):
                smote_cb = ui.checkbox("Appliquer un rééquilibrage (SMOTE)")
                smote_cb.disable()
                ui.label("Recommandé si classes déséquilibrées").style(
                    "font-size:13px !important; color:#7f8c8d !important; font-style:italic !important;"
                )

        # Info SMOTE
        with ui.card().classes("w-full max-w-4xl mb-6").style(
            "background:#e3f2fd !important; border-radius:16px !important; padding:24px !important; "
            "border-left:4px solid #2196f3 !important; box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            ui.label("💡 Qu'est-ce que SMOTE ?").style(
                "font-size:18px !important; font-weight:700 !important; color:#01335A !important; "
                "margin-bottom:12px !important;"
            )

            ui.label(
                "SMOTE (Synthetic Minority Oversampling Technique) génère de nouveaux exemples synthétiques "
                "pour la classe minoritaire en créant des points intermédiaires entre observations existantes."
            ).style(
                "font-size:14px !important; color:#2c3e50 !important; line-height:1.6 !important; "
                "margin-bottom:12px !important;"
            )

            ui.label(
                "✔ Utile pour classes déséquilibrées\n"
                "✔ Améliore les performances\n"
                "✔ Évite le simple oversampling"
            ).style(
                "font-size:13px !important; color:#01335A !important; line-height:1.8 !important; "
                "margin-bottom:8px !important;"
            )

            ui.label(
                "⚠ Attention : SMOTE appliqué uniquement sur l'ensemble d'entraînement"
            ).style(
                "font-size:13px !important; color:#e74c3c !important; font-weight:600 !important;"
            )

        # ---------- APERÇU DES DONNÉES ----------
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white !important; border-radius:16px !important; padding:32px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            ui.label("👁️ Aperçu des données").style(
                "font-weight:700 !important; font-size:22px !important; color:#01335A !important; "
                "margin-bottom:16px !important;"
            )
            
            ui.label(
                f"Visualisation des 10 premières lignes du dataset ({len(df)} lignes × {len(df.columns)} colonnes)"
            ).style(
                "font-size:14px !important; color:#636e72 !important; margin-bottom:20px !important;"
            )

            df_preview = df.head(10).copy()
            
            columns_for_table = []
            rows_for_table = []
            
            for col in df_preview.columns:
                columns_for_table.append({
                    "name": col,
                    "label": col,
                    "field": col,
                    "align": "left",
                    "sortable": True
                })
            
            for idx, row in df_preview.iterrows():
                row_dict = {"_index": str(idx)}
                for col in df_preview.columns:
                    val = row[col]
                    if pd.isna(val):
                        row_dict[col] = "NaN"
                    elif isinstance(val, (int, np.integer)):
                        row_dict[col] = str(val)
                    elif isinstance(val, (float, np.floating)):
                        row_dict[col] = f"{val:.2f}"
                    else:
                        row_dict[col] = str(val)[:50]
                rows_for_table.append(row_dict)
            
            columns_for_table.insert(0, {
                "name": "_index",
                "label": "Index",
                "field": "_index",
                "align": "center",
                "sortable": False
            })
            
            ui.table(
                columns=columns_for_table,
                rows=rows_for_table,
                row_key="_index"
            ).props("flat bordered dense").style(
                "width:100% !important; font-size:12px !important; max-height:400px !important; "
                "overflow-y:auto !important;"
            )
            
            # Statistiques compactes avec gradient
            with ui.row().classes("w-full items-center justify-around mt-6").style(
                "background:linear-gradient(135deg, #01335A 0%, #09538C 100%) !important; "
                "padding:20px !important; border-radius:12px !important;"
            ):
                with ui.row().classes("items-center gap-3"):
                    ui.icon("dataset", size="md").classes("text-white")
                    with ui.column().classes("gap-0"):
                        ui.label(f"{len(df):,}").style(
                            "font-weight:700 !important; font-size:24px !important; color:white !important; "
                            "line-height:1 !important;"
                        )
                        ui.label("lignes").style(
                            "font-size:12px !important; color:rgba(255,255,255,0.8) !important; margin-top:4px !important;"
                        )
                
                with ui.row().classes("items-center gap-3"):
                    ui.icon("table_chart", size="md").classes("text-white")
                    with ui.column().classes("gap-0"):
                        ui.label(f"{len(df.columns)}").style(
                            "font-weight:700 !important; font-size:24px !important; color:white !important; "
                            "line-height:1 !important;"
                        )
                        ui.label("colonnes").style(
                            "font-size:12px !important; color:rgba(255,255,255,0.8) !important; margin-top:4px !important;"
                        )
                
                with ui.row().classes("items-center gap-3"):
                    ui.icon("warning", size="md").classes("text-white")
                    with ui.column().classes("gap-0"):
                        total_missing = df.isna().sum().sum()
                        missing_pct = (total_missing / (len(df) * len(df.columns)) * 100)
                        ui.label(f"{total_missing:,}").style(
                            "font-weight:700 !important; font-size:24px !important; color:white !important; "
                            "line-height:1 !important;"
                        )
                        ui.label(f"missing ({missing_pct:.1f}%)").style(
                            "font-size:12px !important; color:rgba(255,255,255,0.8) !important; margin-top:4px !important;"
                        )
                
                with ui.row().classes("items-center gap-3"):
                    ui.icon("analytics", size="md").classes("text-white")
                    with ui.column().classes("gap-0"):
                        n_numeric = len(df.select_dtypes(include=[np.number]).columns)
                        ui.label(f"{n_numeric}").style(
                            "font-weight:700 !important; font-size:24px !important; color:white !important; "
                            "line-height:1 !important;"
                        )
                        ui.label("numériques").style(
                            "font-size:12px !important; color:rgba(255,255,255,0.8) !important; margin-top:4px !important;"
                        )
                
                with ui.row().classes("items-center gap-3"):
                    ui.icon("category", size="md").classes("text-white")
                    with ui.column().classes("gap-0"):
                        n_categorical = len(df.select_dtypes(include=['object', 'category']).columns)
                        ui.label(f"{n_categorical}").style(
                            "font-weight:700 !important; font-size:24px !important; color:white !important; "
                            "line-height:1 !important;"
                        )
                        ui.label("catégorielles").style(
                            "font-size:12px !important; color:rgba(255,255,255,0.8) !important; margin-top:4px !important;"
                        )

        # ---------- CONFIGURATION DES TYPES (2 PAR LIGNE) ----------
        column_type_widgets = {}
        column_exclude_widgets = {}

        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white !important; border-radius:16px !important; padding:32px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            ui.label("🛠 Configuration des types de colonnes").style(
                "font-weight:700 !important; font-size:22px !important; color:#01335A !important; "
                "margin-bottom:16px !important;"
            )
            ui.label("Vérifiez et corrigez les types détectés automatiquement").style(
                "font-size:14px !important; color:#636e72 !important; margin-bottom:20px !important;"
            )

            # ✅ DISPOSITION EN GRILLE : 2 COLONNES PAR LIGNE
            for i in range(0, len(columns_info), 2):
                with ui.row().classes("w-full gap-4 mb-4"):
                    # Colonne 1
                    col1 = columns_info[i]
                    col1_name = col1["Colonne"]
                    actual_type1 = detect_actual_type(col1, col1_name)
                    
                    with ui.card().classes("flex-1").style(
                        "background:#f8f9fa !important; padding:16px !important; "
                        "border-radius:12px !important; border:1px solid #e1e8ed !important;"
                    ):
                        ui.label(col1_name).style(
                            "font-weight:700 !important; font-size:15px !important; "
                            "color:#01335A !important; margin-bottom:12px !important;"
                        )
                        
                        col_type1 = ui.select(
                            options=[
                                "Numérique Continue", "Numérique Discrète",
                                "Catégorielle Nominale", "Catégorielle Ordinale",
                                "Date/Datetime", "Texte", "Identifiant"
                            ],
                            value=actual_type1,
                            label="Type"
                        ).props("outlined dense").classes("w-full")
                        
                        column_type_widgets[col1_name] = col_type1
                        
                        auto_exclude1 = False
                        if col1["Cardinalité"] == 1:
                            auto_exclude1 = True
                        if "%" in col1["% Missing"]:
                            try:
                                missing_pct = float(col1["% Missing"].replace("%", "").strip())
                                if missing_pct >= 100:
                                    auto_exclude1 = True
                            except:
                                pass
                        if actual_type1 == "Identifiant":
                            auto_exclude1 = True
                        
                        exclude_cb1 = ui.checkbox("Exclure cette colonne", value=auto_exclude1).classes("mt-2")
                        column_exclude_widgets[col1_name] = exclude_cb1
                    
                    # Colonne 2 (si elle existe)
                    if i + 1 < len(columns_info):
                        col2 = columns_info[i + 1]
                        col2_name = col2["Colonne"]
                        actual_type2 = detect_actual_type(col2, col2_name)
                        
                        with ui.card().classes("flex-1").style(
                            "background:#f8f9fa !important; padding:16px !important; "
                            "border-radius:12px !important; border:1px solid #e1e8ed !important;"
                        ):
                            ui.label(col2_name).style(
                                "font-weight:700 !important; font-size:15px !important; "
                                "color:#01335A !important; margin-bottom:12px !important;"
                            )
                            
                            col_type2 = ui.select(
                                options=[
                                    "Numérique Continue", "Numérique Discrète",
                                    "Catégorielle Nominale", "Catégorielle Ordinale",
                                    "Date/Datetime", "Texte", "Identifiant"
                                ],
                                value=actual_type2,
                                label="Type"
                            ).props("outlined dense").classes("w-full")
                            
                            column_type_widgets[col2_name] = col_type2
                            
                            auto_exclude2 = False
                            if col2["Cardinalité"] == 1:
                                auto_exclude2 = True
                            if "%" in col2["% Missing"]:
                                try:
                                    missing_pct = float(col2["% Missing"].replace("%", "").strip())
                                    if missing_pct >= 100:
                                        auto_exclude2 = True
                                except:
                                    pass
                            if actual_type2 == "Identifiant":
                                auto_exclude2 = True
                            
                            exclude_cb2 = ui.checkbox("Exclure cette colonne", value=auto_exclude2).classes("mt-2")
                            column_exclude_widgets[col2_name] = exclude_cb2

            # Info exclusions
            with ui.card().classes("w-full mt-6").style(
                "background:#134b78 !important; padding:20px !important; border-radius:12px !important; "
                "border-left:4px solid #01335A !important; box-shadow:none !important;"
            ):
                ui.label("💡 Exclusions automatiques détectées :").style(
                    "font-weight:700 !important; margin-bottom:12px !important; color:white !important;"
                )
                
                exclusions = [
                    "• Colonnes avec cardinalité = 1 (valeur unique)",
                    "• Colonnes avec 100% de valeurs manquantes",
                    "• Colonnes identifiants (détection automatique)"
                ]
                
                for excl in exclusions:
                    ui.label(excl).style(
                        "font-size:13px !important; color:white !important; margin-bottom:4px !important;"
                    )

        # ---------- BOUTONS ----------
        with ui.row().classes("w-full max-w-6xl justify-between gap-4 mt-8"):
            ui.button(
                "← Retour",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/upload'")
            ).style(
                "background:white !important; color:#01335A !important; font-weight:500 !important; "
                "border:1px solid #e1e8ed !important; border-radius:8px !important; height:50px !important; "
                "min-width:200px !important; font-size:14px !important; text-transform:none !important; "
                "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
            )
            
            ui.button(
                "Suivant →",
                on_click=go_to_split
            ).style(
                "background:#01335A !important; color:white !important; font-weight:600 !important; "
                "border-radius:8px !important; height:50px !important; min-width:200px !important; "
                "font-size:14px !important; text-transform:none !important;"
            )

    # Déclencher l'analyse initiale
    if default_target:
        on_target_change(default_target)

# Fonction helper (si pas déjà définie ailleurs)
def map_detected_type(detected_type):
    """Mappe le type détecté vers les options du dropdown"""
    mapping = {
        "int64": "Numérique Discrète",
        "float64": "Numérique Continue",
        "object": "Catégorielle Nominale",
        "datetime64": "Date/Datetime",
        "bool": "Catégorielle Nominale",
        "category": "Catégorielle Nominale"
    }
    return mapping.get(detected_type, "Catégorielle Nominale")




@ui.page('/supervised/split')
def split_page():
    from sklearn.model_selection import train_test_split
    
    df = state.get("raw_df")
    target_col = state.get("target_column")
    columns_exclude = state.get("columns_exclude", {})

    if df is None or target_col is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Dataset ou target manquants.").style(
                "font-size:18px; color:#c0392b; font-weight:600;"
            )
            ui.button(" Retour", on_click=lambda: ui.run_javascript(
                "window.location.href='/supervised/user_decisions'"
            )).style("margin-top:20px; background:#01335A; color:white; font-weight:600;")
        return

    # Colonnes utilisées pour l'analyse
    active_cols = [c for c in df.columns if not columns_exclude.get(c, False) and c != target_col]
    df_active = df[active_cols + [target_col]].copy()

    # ----------- STYLES GÉNÉRAUX -----------
    with ui.column().classes("w-full h-auto items-center p-10").style(
        "background-color:#f5f6fa; font-family:'Inter', sans-serif;"
    ):
        ui.label("Phase 3.2 : Split Train/Validation/Test").style(
            "font-weight:700; font-size:32px; color:#01335A; margin-bottom:32px; text-align:center;"
        )

        # ---------- 1️⃣ Vérification du déséquilibre ----------
        with ui.card().classes("w-full max-w-4xl p-6 mb-6").style(
            "background:white; border-radius:12px; box-shadow:0 4px 15px rgba(0,0,0,0.08);"
        ):
            ui.label(f"📊 Distribution de la target '{target_col}'").style(
                "font-weight:700; font-size:20px; color:#01335A; margin-bottom:12px;"
            )

            counts = df_active[target_col].value_counts()
            total = len(df_active)
            imbalance_detected = False

            for cls, cnt in counts.items():
                pct = round(cnt / total * 100, 1)
                with ui.row().classes("items-center gap-2 mb-1"):
                    ui.label(f"{cls}: {cnt} ({pct}%)").style("width:120px; font-family:monospace;")
                    ui.linear_progress(value=pct/100, color="#01335A").classes("w-full h-3 rounded-lg")

            if len(counts) > 1 and counts.min() / total * 100 < 30:
                imbalance_detected = True
                ui.label("⚠️ Déséquilibre détecté !").style(
                    "color:#01335A; font-weight:600; margin-top:6px;"
                )
                ui.label("Recommandation : Stratified split + métriques adaptées (F1-score, recall)").style(
                    "font-size:14px; margin-top:4px;"
                )

            smote_cb = ui.checkbox("Appliquer un rééquilibrage (SMOTE/undersampling)").disable()
            if imbalance_detected:
                smote_cb.enable()

        # ---------- 2️⃣ Configuration du Split ----------
        with ui.card().classes("w-full max-w-4xl p-6 mb-6").style(
            "background:white; border-radius:12px; box-shadow:0 4px 15px rgba(0,0,0,0.08);"
        ):
            ui.label("📂 Configuration du Split").style(
                "font-weight:700; font-size:20px; color:#01335A; margin-bottom:12px;"
            )

            # Default ratios
            n_samples = len(df_active)
            if n_samples < 1000:
                train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
            elif n_samples > 10000:
                train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
            else:
                train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

            train_slider = ui.slider(value=int(train_ratio*100), min=50, max=90, step=1).props("show-value")
            val_slider = ui.slider(value=int(val_ratio*100), min=5, max=30, step=1).props("show-value")
            test_slider = ui.slider(value=int(test_ratio*100), min=5, max=30, step=1).props("show-value")

            # Affichage dynamique des ratios
            train_label = ui.label(f"Train: {train_slider.value}%").style("width:100px;")
            val_label = ui.label(f"Validation: {val_slider.value}%").style("width:120px;")
            test_label = ui.label(f"Test: {test_slider.value}%").style("width:100px;")

            def update_ratios():
                total = train_slider.value + val_slider.value + test_slider.value
                if total != 100:
                    # Ajuste proportionnellement les autres
                    rem = 100 - train_slider.value
                    val_slider.value = int(val_slider.value / (val_slider.value + test_slider.value) * rem)
                    test_slider.value = rem - val_slider.value
                train_label.text = f"Train: {train_slider.value}%"
                val_label.text = f"Validation: {val_slider.value}%"
                test_label.text = f"Test: {test_slider.value}%"

            train_slider.on("update:model-value", lambda e: update_ratios())
            val_slider.on("update:model-value", lambda e: update_ratios())
            test_slider.on("update:model-value", lambda e: update_ratios())

            # Stratified / Random / Time-based
            ui.label("Stratégie de split").style(
                "font-weight:600; font-size:16px; color:#2c3e50; margin-top:20px; margin-bottom:8px;"
            )
            
            strategy_radio = ui.radio(
                options=["Stratified (recommandé)", "Random", "Time-based (si date)"],
                value="Stratified (recommandé)"
            ).style("margin-bottom:16px; color:#01335A !important;")

            # ✅ AMÉLIORATION : Explications avec meilleur design
            explanation_container = ui.column().classes("w-full")
            
            def update_explanation():
                explanation_container.clear()
                with explanation_container:
                    strategy = strategy_radio.value
                    
                    if "Stratified" in strategy:
                        with ui.card().classes("w-full p-6").style(
                            "background:white !important; border-radius:16px !important; "
                            "box-shadow:0 4px 12px rgba(1,51,90,0.08) !important; "
                            "border-left:5px solid #2196f3 !important;"
                        ):
                            # Header
                            with ui.row().classes("items-center gap-3 mb-4"):
                                with ui.card().style(
                                    "background:linear-gradient(135deg, #2196f3, #1976d2) !important; "
                                    "padding:12px !important; border-radius:12px !important; "
                                    "box-shadow:0 4px 8px rgba(33,150,243,0.3) !important; "
                                    "min-width:60px !important; min-height:60px !important; "
                                    "display:flex !important; align-items:center !important; justify-content:center !important;"
                                ):
                                    ui.label("🎯").style("font-size:32px !important;")
                                
                                with ui.column().classes("gap-1"):
                                    ui.label("Stratified Split").style(
                                        "font-weight:700 !important; font-size:20px !important; color:#01335A !important;"
                                    )
                                    ui.label("Méthode recommandée").style(
                                        "font-size:13px !important; color:#2196f3 !important; font-weight:600 !important;"
                                    )
                            
                            ui.separator().classes("my-3")
                            
                            # Principe
                            with ui.column().classes("gap-2 mb-4"):
                                ui.label("📌 Principe").style(
                                    "font-weight:700 !important; font-size:15px !important; color:#01335A !important;"
                                )
                                ui.label(
                                    "Préserve la distribution des classes dans chaque ensemble (train/val/test). "
                                    "Garantit que chaque sous-ensemble a la même proportion de classes que le dataset original."
                                ).style(
                                    "font-size:14px !important; color:#2c3e50 !important; line-height:1.7 !important;"
                                )
                            
                            # Exemple visuel
                            with ui.card().classes("w-full").style(
                                "background:#f8f9fa !important; padding:16px !important; "
                                "border-radius:12px !important; box-shadow:none !important; "
                                "border:1px solid #e1e8ed !important;"
                            ):
                                ui.label("💡 Exemple").style(
                                    "font-weight:700 !important; font-size:14px !important; "
                                    "color:#01335A !important; margin-bottom:8px !important;"
                                )
                                
                                with ui.row().classes("items-center gap-3 mb-2"):
                                    ui.label("Dataset original :").style("font-weight:600 !important; font-size:13px !important; width:140px !important;")
                                    ui.label("70% classe A, 30% classe B").style(
                                        "font-size:13px !important; color:#2c3e50 !important; "
                                        "background:white !important; padding:4px 12px !important; border-radius:6px !important;"
                                    )
                                
                                for dataset in ["Train", "Validation", "Test"]:
                                    with ui.row().classes("items-center gap-3 mb-2"):
                                        ui.label(f"{dataset} :").style("font-weight:600 !important; font-size:13px !important; width:140px !important;")
                                        ui.label("70% classe A, 30% classe B").style(
                                            "font-size:13px !important; color:#27ae60 !important; font-weight:600 !important; "
                                            "background:white !important; padding:4px 12px !important; border-radius:6px !important;"
                                        )
                            
                            ui.separator().classes("my-3")
                            
                            # Avantages
                            with ui.column().classes("gap-2 mb-3"):
                                ui.label("✅ Avantages").style(
                                    "font-weight:700 !important; font-size:15px !important; color:#27ae60 !important;"
                                )
                                for advantage in [
                                    "Garantit une distribution identique dans tous les ensembles",
                                    "Essentiel pour les données déséquilibrées",
                                    "Évite les biais lors de l'évaluation du modèle",
                                    "Résultats plus fiables et reproductibles"
                                ]:
                                    with ui.row().classes("items-start gap-2"):
                                        ui.label("•").style("color:#27ae60 !important; font-weight:700 !important; font-size:16px !important;")
                                        ui.label(advantage).style("font-size:13px !important; color:#2c3e50 !important; line-height:1.6 !important;")
                            
                            # Cas d'usage
                            with ui.column().classes("gap-2"):
                                ui.label("📊 Cas d'usage").style(
                                    "font-weight:700 !important; font-size:15px !important; color:#01335A !important;"
                                )
                                ui.label(
                                    "Classification avec déséquilibre • Détection de fraude • Diagnostic médical • "
                                    "Détection de spam • Prédiction de churn"
                                ).style(
                                    "font-size:13px !important; color:#636e72 !important; line-height:1.6 !important;"
                                )
                    
                    elif "Random" in strategy:
                        with ui.card().classes("w-full p-6").style(
                            "background:white !important; border-radius:16px !important; "
                            "box-shadow:0 4px 12px rgba(243,156,18,0.08) !important; "
                            "border-left:5px solid #f39c12 !important;"
                        ):
                            # Header
                            with ui.row().classes("items-center gap-3 mb-4"):
                                with ui.card().style(
                                    "background:linear-gradient(135deg, #f39c12, #e67e22) !important; "
                                    "padding:12px !important; border-radius:12px !important; "
                                    "box-shadow:0 4px 8px rgba(243,156,18,0.3) !important; "
                                    "min-width:60px !important; min-height:60px !important; "
                                    "display:flex !important; align-items:center !important; justify-content:center !important;"
                                ):
                                    ui.label("🎲").style("font-size:32px !important;")
                                
                                with ui.column().classes("gap-1"):
                                    ui.label("Random Split").style(
                                        "font-weight:700 !important; font-size:20px !important; color:#01335A !important;"
                                    )
                                    ui.label("Division aléatoire simple").style(
                                        "font-size:13px !important; color:#f39c12 !important; font-weight:600 !important;"
                                    )
                            
                            ui.separator().classes("my-3")
                            
                            # Principe
                            with ui.column().classes("gap-2 mb-4"):
                                ui.label("📌 Principe").style(
                                    "font-weight:700 !important; font-size:15px !important; color:#01335A !important;"
                                )
                                ui.label(
                                    "Division aléatoire du dataset sans tenir compte de la distribution des classes. "
                                    "Chaque échantillon a une probabilité égale d'être assigné à train/val/test."
                                ).style(
                                    "font-size:14px !important; color:#2c3e50 !important; line-height:1.7 !important;"
                                )
                            
                            # Exemple visuel
                            with ui.card().classes("w-full").style(
                                "background:#fff9f0 !important; padding:16px !important; "
                                "border-radius:12px !important; box-shadow:none !important; "
                                "border:1px solid #ffe8cc !important;"
                            ):
                                ui.label("⚠️ Attention au déséquilibre").style(
                                    "font-weight:700 !important; font-size:14px !important; "
                                    "color:#f39c12 !important; margin-bottom:8px !important;"
                                )
                                
                                with ui.row().classes("items-center gap-3 mb-2"):
                                    ui.label("Dataset original :").style("font-weight:600 !important; font-size:13px !important; width:140px !important;")
                                    ui.label("70% classe A, 30% classe B").style(
                                        "font-size:13px !important; color:#2c3e50 !important; "
                                        "background:white !important; padding:4px 12px !important; border-radius:6px !important;"
                                    )
                                
                                for dataset, distrib in [("Train", "68% A, 32% B"), ("Validation", "73% A, 27% B"), ("Test", "71% A, 29% B")]:
                                    with ui.row().classes("items-center gap-3 mb-2"):
                                        ui.label(f"{dataset} :").style("font-weight:600 !important; font-size:13px !important; width:140px !important;")
                                        ui.label(distrib).style(
                                            "font-size:13px !important; color:#f39c12 !important; font-weight:600 !important; "
                                            "background:white !important; padding:4px 12px !important; border-radius:6px !important;"
                                        )
                                
                                ui.label(" Les distributions peuvent varier légèrement").style(
                                    "font-size:12px !important; color:#856404 !important; margin-top:4px !important; font-style:italic !important;"
                                )
                            
                            ui.separator().classes("my-3")
                            
                            # Avantages
                            with ui.column().classes("gap-2 mb-3"):
                                ui.label("✅ Avantages").style(
                                    "font-weight:700 !important; font-size:15px !important; color:#27ae60 !important;"
                                )
                                for advantage in [
                                    "Simple et rapide à implémenter",
                                    "Bon pour les données parfaitement équilibrées",
                                    "Pas de contrainte sur la distribution"
                                ]:
                                    with ui.row().classes("items-start gap-2"):
                                        ui.label("•").style("color:#27ae60 !important; font-weight:700 !important; font-size:16px !important;")
                                        ui.label(advantage).style("font-size:13px !important; color:#2c3e50 !important; line-height:1.6 !important;")
                            
                            # Inconvénients
                            with ui.column().classes("gap-2 mb-3"):
                                ui.label("❌ Inconvénients").style(
                                    "font-weight:700 !important; font-size:15px !important; color:#e74c3c !important;"
                                )
                                for disadvantage in [
                                    "Peut créer des ensembles déséquilibrés par hasard",
                                    "Risque de sous-représenter certaines classes minoritaires",
                                    "Résultats moins reproductibles avec petits datasets"
                                ]:
                                    with ui.row().classes("items-start gap-2"):
                                        ui.label("•").style("color:#e74c3c !important; font-weight:700 !important; font-size:16px !important;")
                                        ui.label(disadvantage).style("font-size:13px !important; color:#2c3e50 !important; line-height:1.6 !important;")
                            
                            # Cas d'usage
                            with ui.column().classes("gap-2"):
                                ui.label("📊 Cas d'usage").style(
                                    "font-weight:700 !important; font-size:15px !important; color:#01335A !important;"
                                )
                                ui.label(
                                    "Régression • Données parfaitement équilibrées • Datasets très larges (>100k lignes) • "
                                    "Exploration rapide"
                                ).style(
                                    "font-size:13px !important; color:#636e72 !important; line-height:1.6 !important;"
                                )
                    
                    elif "Time-based" in strategy:
                        with ui.card().classes("w-full p-6").style(
                            "background:white !important; border-radius:16px !important; "
                            "box-shadow:0 4px 12px rgba(76,175,80,0.08) !important; "
                            "border-left:5px solid #4caf50 !important;"
                        ):
                            # Header
                            with ui.row().classes("items-center gap-3 mb-4"):
                                with ui.card().style(
                                    "background:linear-gradient(135deg, #4caf50, #388e3c) !important; "
                                    "padding:12px !important; border-radius:12px !important; "
                                    "box-shadow:0 4px 8px rgba(76,175,80,0.3) !important; "
                                    "min-width:60px !important; min-height:60px !important; "
                                    "display:flex !important; align-items:center !important; justify-content:center !important;"
                                ):
                                    ui.label("📅").style("font-size:32px !important;")
                                
                                with ui.column().classes("gap-1"):
                                    ui.label("Time-based Split").style(
                                        "font-weight:700 !important; font-size:20px !important; color:#01335A !important;"
                                    )
                                    ui.label("Division chronologique").style(
                                        "font-size:13px !important; color:#4caf50 !important; font-weight:600 !important;"
                                    )
                            
                            ui.separator().classes("my-3")
                            
                            # Principe
                            with ui.column().classes("gap-2 mb-4"):
                                ui.label("📌 Principe").style(
                                    "font-weight:700 !important; font-size:15px !important; color:#01335A !important;"
                                )
                                ui.label(
                                    "Division chronologique des données : les données passées pour l'entraînement, "
                                    "les plus récentes pour validation et test. Respecte l'ordre temporel naturel."
                                ).style(
                                    "font-size:14px !important; color:#2c3e50 !important; line-height:1.7 !important;"
                                )
                            
                            # Exemple visuel
                            with ui.card().classes("w-full").style(
                                "background:#f1f8f4 !important; padding:16px !important; "
                                "border-radius:12px !important; box-shadow:none !important; "
                                "border:1px solid #c8e6c9 !important;"
                            ):
                                ui.label("📆 Exemple temporel").style(
                                    "font-weight:700 !important; font-size:14px !important; "
                                    "color:#4caf50 !important; margin-bottom:8px !important;"
                                )
                                
                                timeline_data = [
                                    ("Train", "2020-2022", "70%", "#667eea"),
                                    ("Validation", "2023", "15%", "#f093fb"),
                                    ("Test", "2024", "15%", "#4facfe")
                                ]
                                
                                for dataset, period, pct, color in timeline_data:
                                    with ui.row().classes("items-center gap-3 mb-3"):
                                        with ui.card().style(
                                            f"background:{color} !important; min-width:100px !important; "
                                            "padding:8px 12px !important; border-radius:8px !important; "
                                            "box-shadow:0 2px 8px rgba(0,0,0,0.1) !important;"
                                        ):
                                            ui.label(dataset).style(
                                                "color:white !important; font-weight:700 !important; "
                                                "font-size:13px !important; text-align:center !important;"
                                            )
                                        
                                        ui.label("").style("font-size:20px !important; color:#4caf50 !important;")
                                        
                                        ui.label(period).style(
                                            "font-weight:600 !important; font-size:14px !important; "
                                            "color:#2c3e50 !important; background:white !important; "
                                            "padding:6px 12px !important; border-radius:8px !important;"
                                        )
                                        
                                        ui.label(f"({pct})").style(
                                            "font-size:13px !important; color:#636e72 !important; font-family:monospace !important;"
                                        )
                            
                            ui.separator().classes("my-3")
                            
                            # Avantages
                            with ui.column().classes("gap-2 mb-3"):
                                ui.label("✅ Avantages").style(
                                    "font-weight:700 !important; font-size:15px !important; color:#27ae60 !important;"
                                )
                                for advantage in [
                                    "Essentiel pour les séries temporelles",
                                    "Évite le data leakage (fuite d'information du futur vers le passé)",
                                    "Simule un scénario réel de prédiction",
                                    "Respecte la dépendance temporelle des données"
                                ]:
                                    with ui.row().classes("items-start gap-2"):
                                        ui.label("•").style("color:#27ae60 !important; font-weight:700 !important; font-size:16px !important;")
                                        ui.label(advantage).style("font-size:13px !important; color:#2c3e50 !important; line-height:1.6 !important;")
                            
                            # Attention
                            with ui.card().classes("w-full").style(
                                "background:#fff3e0 !important; padding:12px !important; "
                                "border-radius:8px !important; box-shadow:none !important; "
                                "border-left:3px solid #ff9800 !important;"
                            ):
                                ui.label("⚠️ Attention").style(
                                    "font-weight:700 !important; font-size:14px !important; color:#ff9800 !important; margin-bottom:4px !important;"
                                )
                                ui.label("Nécessite une colonne date/timestamp • Pas de mélange aléatoire possible").style(
                                    "font-size:13px !important; color:#e65100 !important; line-height:1.6 !important;"
                                )
                            
                            ui.separator().classes("my-3")
                            
                            # Cas d'usage
                            with ui.column().classes("gap-2"):
                                ui.label("📊 Cas d'usage").style(
                                    "font-weight:700 !important; font-size:15px !important; color:#01335A !important;"
                                )
                                ui.label(
                                    "Prévisions de ventes • Prédictions boursières • Analyse de séries temporelles • "
                                    "Prévisions météo • Détection d'anomalies temporelles"
                                ).style(
                                    "font-size:13px !important; color:#636e72 !important; line-height:1.6 !important;"
                                )
            
            strategy_radio.on_value_change(lambda: update_explanation())
            update_explanation()

            seed_input = ui.input(label="Random Seed", value=42).style("margin-top:16px; width:150px;")

        # ---------- 3️⃣ Bouton Split ----------
        # Container résultat
        result_container = ui.column().classes("w-full max-w-6xl mt-4")

        def do_split():
            tr = train_slider.value / 100
            vr = val_slider.value / 100
            te = test_slider.value / 100

            stratify_col = df_active[target_col] if "Stratified" in strategy_radio.value else None

            # Sauvegarder l'état original AVANT split
            if "df_original" not in state:
                state["df_original"] = df_active.copy()

            X = df_active.drop(columns=[target_col])
            y = df_active[target_col]

            # Premier split : Train vs (Val + Test)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, train_size=tr, random_state=int(seed_input.value), stratify=stratify_col
            )

            # Deuxième split : Val vs Test
            strat_temp = y_temp if stratify_col is not None else None
            val_size = vr / (vr + te)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, train_size=val_size, random_state=int(seed_input.value), stratify=strat_temp
            )

            state["split"] = {
                "X_train": X_train, "y_train": y_train,
                "X_val": X_val, "y_val": y_val,
                "X_test": X_test, "y_test": y_test
            }

            result_container.clear()
            
            with result_container: 
                # Titre du résultat avec stats globales
                with ui.column().classes("w-full items-center gap-2 mb-6"):
                    with ui.row().classes("items-center gap-3"):
                        ui.label("✅").style("font-size:32px;")
                        ui.label("Split effectué avec succès !").style(
                            "font-weight:700; font-size:26px; color:#01335A;"
                        )
                    ui.label(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}").style(
                        "font-weight:600; font-size:16px; color:#666; font-family:monospace;"
                    )
                
                # Cards des datasets
                with ui.row().classes("w-full gap-4 justify-center flex-wrap mb-8"):
                    for name, y_set, gradient, icon in [
                        ("Train", y_train, "linear-gradient(135deg, #667eea 0%, #764ba2 100%)", "🎓"),
                        ("Validation", y_val, "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)", "🔍"),
                        ("Test", y_test, "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)", "✨")
                    ]:
                        with ui.card().classes("p-6").style(
                            f"background:white; border-radius:16px; min-width:280px; max-width:320px;"
                            f"box-shadow:0 8px 24px rgba(0,0,0,0.12); border:2px solid #f0f0f0;"
                            f"transition:all 0.3s ease; cursor:default;"
                        ):
                            # Header avec icône et gradient
                            with ui.row().classes("items-center gap-3 mb-4 w-full"):
                                ui.label(icon).style("font-size:36px;")
                                with ui.column().classes("gap-1 flex-grow"):
                                    ui.label(f"{name} Set").style(
                                        f"font-weight:700; font-size:22px; color:#01335A;"
                                    )
                                    counts = y_set.value_counts()
                                    total = len(y_set)
                                    ui.label(f"{total:,} échantillons").style(
                                        "font-weight:500; font-size:14px; color:#666; letter-spacing:0.5px;"
                                    )
                            
                            # Séparateur avec gradient
                            ui.html(f'<div style="height:3px; width:100%; background:{gradient}; border-radius:2px; margin:12px 0;"></div>')
                            
                            # Distribution des classes
                            ui.label("Distribution des classes").style(
                                "font-weight:600; font-size:15px; color:#01335A; margin-bottom:12px;"
                            )
                            
                            for cls, cnt in counts.items():
                                pct = round(cnt / total * 100, 1)
                                
                                with ui.column().classes("w-full gap-2 mb-3"):
                                    # Label et pourcentage
                                    with ui.row().classes("items-center justify-between w-full"):
                                        ui.label(f"Classe: {cls}").style(
                                            "font-weight:600; font-size:14px; color:#333;"
                                        )
                                        ui.label(f"{cnt:,} ({pct}%)").style(
                                            "font-family:monospace; font-size:13px; color:#666; background:#f5f5f5; padding:4px 10px; border-radius:6px;"
                                        )
                                    
                                    # Barre de progression avec gradient
                                    with ui.row().classes("w-full").style("position:relative; height:10px; background:#e8e8e8; border-radius:8px; overflow:hidden;"):
                                        ui.html(f'<div style="position:absolute; left:0; top:0; height:100%; width:{pct}%; background:{gradient}; border-radius:8px; transition:width 0.5s ease;"></div>')
                
                # ✅ AJOUT : Vérification de la qualité du split
                with ui.card().classes("w-full p-6 mb-6").style(
                    "background:white; border-radius:16px; box-shadow:0 8px 24px rgba(0,0,0,0.12);"
                ):
                    ui.label("📊 Vérification de la Distribution").style(
                        "font-weight:700; font-size:22px; color:#01335A; margin-bottom:20px;"
                    )
                    
                    # Calculer les distributions
                    original_dist = (counts / total * 100).round(1)
                    train_dist = (y_train.value_counts() / len(y_train) * 100).round(1)
                    val_dist = (y_val.value_counts() / len(y_val) * 100).round(1)
                    test_dist = (y_test.value_counts() / len(y_test) * 100).round(1)
                    
                    # Tableau comparatif
                    comparison_rows = []
                    max_diff = 0
                    
                    for cls in counts.index:
                        orig_pct = original_dist.get(cls, 0)
                        train_pct = train_dist.get(cls, 0)
                        val_pct = val_dist.get(cls, 0)
                        test_pct = test_dist.get(cls, 0)
                        
                        # Calculer la différence max
                        diffs = [abs(train_pct - orig_pct), abs(val_pct - orig_pct), abs(test_pct - orig_pct)]
                        max_diff = max(max_diff, max(diffs))
                        
                        comparison_rows.append({
                            "Classe": str(cls),
                            "Original": f"{orig_pct}%",
                            "Train": f"{train_pct}%",
                            "Validation": f"{val_pct}%",
                            "Test": f"{test_pct}%",
                            "Max Diff": f"±{max(diffs):.1f}%"
                        })
                    
                    ui.table(
                        columns=[
                            {"name": "Classe", "label": "Classe", "field": "Classe", "align": "left"},
                            {"name": "Original", "label": "Original", "field": "Original", "align": "center"},
                            {"name": "Train", "label": "Train", "field": "Train", "align": "center"},
                            {"name": "Validation", "label": "Validation", "field": "Validation", "align": "center"},
                            {"name": "Test", "label": "Test", "field": "Test", "align": "center"},
                            {"name": "Max Diff", "label": "Diff Max", "field": "Max Diff", "align": "center"}
                        ],
                        rows=comparison_rows,
                        row_key="Classe"
                    ).props("flat bordered").style(
                        "width:100% !important; font-size:13px !important;"
                    )
                    
                    # Évaluation de la qualité du split
                    ui.separator().classes("my-4")
                    
                    if max_diff < 2:
                        quality_color = "#27ae60"
                        quality_icon = "✅"
                        quality_text = "Excellent"
                        quality_msg = "Le split est parfaitement équilibré ! Les distributions sont quasi-identiques."
                    elif max_diff < 5:
                        quality_color = "#2196f3"
                        quality_icon = "✔️"
                        quality_text = "Très bon"
                        quality_msg = "Le split est bien équilibré. Différences mineures acceptables."
                    elif max_diff < 10:
                        quality_color = "#f39c12"
                        quality_icon = "⚠️"
                        quality_text = "Acceptable"
                        quality_msg = "Le split présente quelques différences. Surveiller les performances."
                    else:
                        quality_color = "#e74c3c"
                        quality_icon = "❌"
                        quality_text = "À améliorer"
                        quality_msg = "Le split est déséquilibré. Considérez l'utilisation du Stratified Split."
                    
                    with ui.card().classes("w-full").style(
                        f"background:{quality_color}15 !important; padding:20px !important; "
                        f"border-radius:12px !important; border-left:4px solid {quality_color} !important; "
                        "box-shadow:none !important;"
                    ):
                        with ui.row().classes("items-center gap-3"):
                            ui.label(quality_icon).style("font-size:32px;")
                            with ui.column().classes("gap-2"):
                                ui.label(f"Qualité du split : {quality_text}").style(
                                    f"font-weight:700 !important; font-size:18px !important; color:{quality_color} !important;"
                                )
                                ui.label(quality_msg).style(
                                    "font-size:14px !important; color:#2c3e50 !important; line-height:1.6 !important;"
                                )
                                ui.label(f"Différence maximale : ±{max_diff:.1f}%").style(
                                    "font-size:13px !important; color:#636e72 !important; font-family:monospace !important; margin-top:4px !important;"
                                )
                    
                    # Recommandations
                    with ui.expansion("💡 Comment interpréter ces résultats ?", icon="help").classes("w-full mt-4"):
                        ui.markdown("""
<div style="color:#2c3e50; font-size:14px; line-height:1.8; padding:12px;">

**Différence maximale :** Écart maximum entre la distribution originale et un ensemble

**Seuils de qualité :**
- **< 2%** : Excellent - Split parfaitement stratifié
- **2-5%** : Très bon - Variations naturelles acceptables
- **5-10%** : Acceptable - Peut affecter légèrement les performances
- **> 10%** : Problématique - Risque de biais dans l'évaluation

**Que faire si le split est déséquilibré ?**
1. Utiliser le **Stratified Split** (recommandé)
2. Augmenter la taille du dataset si possible
3. Vérifier le random seed (essayer plusieurs valeurs)

</div>
                        """)

                # Bouton de navigation
                with ui.row().classes("w-full justify-center mt-6"):
                    ui.button(
                        " Analyse Univariée",
                        on_click=lambda: ui.run_javascript("window.location.href='/supervised/univariate_analysis'"),
                    ).style(
                        "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white !important; "
                        "font-weight:600 !important; border-radius:12px !important; height:50px !important; "
                        "width:280px !important; font-size:16px !important; "
                        "box-shadow:0 6px 20px rgba(1,51,90,0.3) !important; transition:all 0.3s ease !important;"
                    ).props('icon-right="arrow_forward"')

            ui.notify("✅ Split effectué avec succès !", color="positive")

        ui.button("Effectuer le split", on_click=do_split).style(
            "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white; font-weight:600; "
            "border-radius:12px; height:50px; width:280px; margin-top:20px; font-size:16px; "
            "box-shadow:0 6px 20px rgba(1,51,90,0.3); transition:all 0.3s ease;"
        ).classes("hover:shadow-xl")
        
        ui.button(
                " Analyse Univariée",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/univariate_analysis'"),
        ).style(
                "background:#01335A !important; color:white; font-weight:600; border-radius:8px; height:40px; width:280px; margin-top:12px;"
        )

        
@ui.page('/supervised/univariate_analysis')
def univariate_analysis_page():
    import plotly.graph_objects as go
    from scipy.stats import skew

    # Récupération des données train
    split = state.get("split")
    if split is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Aucun split trouvé. Veuillez d'abord effectuer le split.").style(
                "font-size:18px !important; color:#c0392b !important; font-weight:600 !important;"
            )
            ui.button(
                "⬅ Retour au Preprocessing", 
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/split'")
            ).style(
                "margin-top:20px !important; background:#01335A !important; color:white !important; "
                "font-weight:600 !important; padding:12px 32px !important; border-radius:8px !important;"
            )
        return

    X_train = split["X_train"]
    y_train = split["y_train"]
    
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    
    # Statistiques globales
    total_features = len(X_train.columns)
    n_numeric = len(numeric_cols)
    n_categorical = len(categorical_cols)
    n_observations = len(X_train)

    # ---------- UI PRINCIPALE ----------
    with ui.column().classes("w-full items-center").style(
        "background:#f0f2f5 !important; min-height:100vh !important; padding:48px 24px !important; "
        "font-family:'Inter', sans-serif !important;"
    ):
        # HEADER
        ui.label("Analyse Univariée").style(
            "font-weight:700 !important; font-size:36px !important; color:#01335A !important; "
            "margin-bottom:8px !important; text-align:center !important; letter-spacing:-0.5px !important;"
        )
        ui.label("Exploration détaillée de chaque variable (dataset d'entraînement)").style(
            "font-size:16px !important; color:#636e72 !important; margin-bottom:48px !important; "
            "text-align:center !important; font-weight:400 !important;"
        )

        # --- OVERVIEW COMPACTE ---
        with ui.row().classes("w-full max-w-6xl items-center justify-around mb-6").style(
            "background:linear-gradient(135deg, #01335A 0%, #09538C 100%) !important; "
            "padding:20px !important; border-radius:16px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            with ui.row().classes("items-center gap-3"):
                ui.icon("analytics", size="md").classes("text-white")
                with ui.column().classes("gap-0"):
                    ui.label(str(n_numeric)).style(
                        "font-weight:700 !important; font-size:24px !important; color:white !important; "
                        "line-height:1 !important;"
                    )
                    ui.label(f"numériques ({round(n_numeric/total_features*100)}%)").style(
                        "font-size:12px !important; color:rgba(255,255,255,0.8) !important; margin-top:4px !important;"
                    )
            
            with ui.row().classes("items-center gap-3"):
                ui.icon("category", size="md").classes("text-white")
                with ui.column().classes("gap-0"):
                    ui.label(str(n_categorical)).style(
                        "font-weight:700 !important; font-size:24px !important; color:white !important; "
                        "line-height:1 !important;"
                    )
                    ui.label(f"catégorielles ({round(n_categorical/total_features*100)}%)").style(
                        "font-size:12px !important; color:rgba(255,255,255,0.8) !important; margin-top:4px !important;"
                    )
            
            with ui.row().classes("items-center gap-3"):
                ui.icon("dataset", size="md").classes("text-white")
                with ui.column().classes("gap-0"):
                    ui.label(f"{n_observations:,}").style(
                        "font-weight:700 !important; font-size:24px !important; color:white !important; "
                        "line-height:1 !important;"
                    )
                    ui.label("observations").style(
                        "font-size:12px !important; color:rgba(255,255,255,0.8) !important; margin-top:4px !important;"
                    )
            
            with ui.row().classes("items-center gap-3"):
                ui.icon("table_chart", size="md").classes("text-white")
                with ui.column().classes("gap-0"):
                    ui.label(str(total_features)).style(
                        "font-weight:700 !important; font-size:24px !important; color:white !important; "
                        "line-height:1 !important;"
                    )
                    ui.label("features totales").style(
                        "font-size:12px !important; color:rgba(255,255,255,0.8) !important; margin-top:4px !important;"
                    )

        # --- FILTRES & TRI ---
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white !important; border-radius:16px !important; padding:20px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            with ui.row().classes("w-full gap-4 items-end"):
                filter_type = ui.select(
                    options={
                        "all": "Toutes les variables",
                        "numeric": "Numériques uniquement",
                        "categorical": "Catégorielles uniquement"
                    },
                    value="all",
                    label="Type de variable"
                ).props("outlined dense").classes("flex-1")
                
                sort_by = ui.select(
                    options={
                        "name": "Nom (A-Z)",
                        "name_desc": "Nom (Z-A)",
                        "type": "Type",
                        "variance": "Variance (numériques)"
                    },
                    value="name",
                    label="Trier par"
                ).props("outlined dense").classes("flex-1")
                
                search_input = ui.input(
                    label="Rechercher",
                    placeholder="Nom de colonne..."
                ).props("outlined dense clearable").classes("flex-1")

        # --- CONTENEUR DES CARTES ---
        cards_container = ui.column().classes("w-full max-w-6xl gap-4")

        def create_numeric_plot(series, col_name):
            """Crée un histogramme pour une variable numérique"""
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=series.dropna(),
                name="Distribution",
                marker_color='#01335A',
                opacity=0.7,
                nbinsx=30
            ))
            
            fig.update_layout(
                height=220,
                margin=dict(l=30, r=10, t=20, b=30),
                showlegend=False,
                paper_bgcolor='white',
                plot_bgcolor='#f8f9fa',
                font=dict(family="Inter, sans-serif", size=11, color="#2c3e50"),
                xaxis=dict(title=None),
                yaxis=dict(title=None)
            )
            
            return fig

        def create_categorical_plot(series, col_name):
            """Crée un bar chart pour une variable catégorielle"""
            counts = series.value_counts().head(10)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=counts.index.astype(str),
                y=counts.values,
                marker_color='#01335A',
                text=counts.values,
                textposition='outside'
            ))
            
            fig.update_layout(
                height=220,
                margin=dict(l=30, r=10, t=20, b=50),
                showlegend=False,
                paper_bgcolor='white',
                plot_bgcolor='#f8f9fa',
                font=dict(family="Inter, sans-serif", size=11, color="#2c3e50"),
                xaxis=dict(title=None, tickangle=-45),
                yaxis=dict(title=None)
            )
            
            return fig

        def create_boxplot(series, col_name):
            """Crée un boxplot pour détecter les outliers"""
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=series.dropna(),
                name=col_name,
                marker_color='#01335A',
                boxmean='sd'
            ))
            
            fig.update_layout(
                height=220,
                margin=dict(l=30, r=10, t=20, b=30),
                showlegend=False,
                paper_bgcolor='white',
                plot_bgcolor='#f8f9fa',
                font=dict(family="Inter, sans-serif", size=11, color="#2c3e50")
            )
            
            return fig

        def add_numeric_card(col):
            """Carte COMPACTE pour variable numérique"""
            series = X_train[col].dropna()
            
            # Statistiques
            mean_val = round(series.mean(), 3)
            median_val = round(series.median(), 3)
            std_val = round(series.std(), 3)
            min_val = round(series.min(), 3)
            max_val = round(series.max(), 3)
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr_val = q3 - q1
            skewness = round(skew(series), 3)
            
            # Outliers
            lower_bound = q1 - 1.5 * iqr_val
            upper_bound = q3 + 1.5 * iqr_val
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            n_outliers = len(outliers)
            pct_outliers = round(n_outliers / len(series) * 100, 1) if len(series) > 0 else 0
            
            with cards_container:
                with ui.card().classes("w-full").style(
                    "background:white !important; border-radius:16px !important; padding:20px !important; "
                    "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important; border-left:4px solid #01335A !important;"
                ):
                    # Header compact
                    with ui.row().classes("w-full items-center justify-between mb-3"):
                        ui.label(col).style(
                            "font-weight:700 !important; font-size:18px !important; color:#01335A !important;"
                        )
                        with ui.badge().style(
                            "background:#e3f2fd !important; color:#01335A !important; "
                            "padding:4px 10px !important; border-radius:6px !important;"
                        ):
                            ui.label("Numérique").style(
                                "font-size:11px !important; font-weight:600 !important;"
                            )
                    
                    # Stats compactes (2 lignes de 4)
                    with ui.grid(columns=4).classes("w-full gap-2 mb-3"):
                        for stat_label, stat_value in [
                            ("Moyenne", mean_val),
                            ("Médiane", median_val),
                            ("Écart-type", std_val),
                            ("Min", min_val),
                            ("Max", max_val),
                            ("IQR", round(iqr_val, 3)),
                            ("Skewness", skewness),
                            ("Outliers", f"{n_outliers} ({pct_outliers}%)")
                        ]:
                            with ui.card().classes("p-2").style(
                                "background:#f8f9fa !important; border-radius:6px !important; "
                                "box-shadow:none !important;"
                            ):
                                ui.label(stat_label).style(
                                    "font-size:10px !important; color:#636e72 !important; "
                                    "margin-bottom:2px !important; text-transform:uppercase !important;"
                                )
                                ui.label(str(stat_value)).style(
                                    "font-weight:700 !important; font-size:14px !important; "
                                    "color:#01335A !important; font-family:monospace !important;"
                                )
                    
                    # Alertes compactes (bleu uniquement)
                    alerts = []
                    if abs(skewness) > 1:
                        alerts.append(f"Asymétrie forte (skewness = {skewness})")
                    elif abs(skewness) > 0.5:
                        alerts.append(f"Asymétrie modérée (skewness = {skewness})")
                    
                    if pct_outliers > 10:
                        alerts.append(f"Nombreux outliers ({pct_outliers}%)")
                    elif pct_outliers > 5:
                        alerts.append(f"Outliers détectés ({pct_outliers}%)")
                    
                    if series.isna().sum() > 0:
                        pct_missing = round(series.isna().sum() / len(X_train) * 100, 1)
                        alerts.append(f"Valeurs manquantes: {pct_missing}%")
                    
                    if alerts:
                        with ui.column().classes("w-full gap-1 mb-3"):
                            for msg in alerts:
                                with ui.card().classes("w-full p-2").style(
                                    "background:#e3f2fd !important; border-radius:6px !important; "
                                    "border-left:3px solid #2196f3 !important; box-shadow:none !important;"
                                ):
                                    ui.label(f"⚠️ {msg}").style(
                                        "color:#01335A !important; font-size:12px !important; font-weight:500 !important;"
                                    )
                    
                    # Visualisations côte à côte
                    with ui.row().classes("w-full gap-3"):
                        with ui.column().classes("flex-1"):
                            hist_fig = create_numeric_plot(series, col)
                            ui.plotly(hist_fig).style("width:100% !important;")
                        
                        with ui.column().classes("flex-1"):
                            box_fig = create_boxplot(series, col)
                            ui.plotly(box_fig).style("width:100% !important;")

        def add_categorical_card(col):
            """Carte COMPACTE pour variable catégorielle"""
            series = X_train[col]
            
            n_unique = series.nunique()
            mode_val = series.mode()[0] if len(series.mode()) > 0 else "N/A"
            counts = series.value_counts()
            total = len(series)
            
            with cards_container:
                with ui.card().classes("w-full").style(
                    "background:white !important; border-radius:16px !important; padding:20px !important; "
                    "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important; border-left:4px solid #2196f3 !important;"
                ):
                    # Header
                    with ui.row().classes("w-full items-center justify-between mb-3"):
                        ui.label(col).style(
                            "font-weight:700 !important; font-size:18px !important; color:#01335A !important;"
                        )
                        with ui.badge().style(
                            "background:#e3f2fd !important; color:#01335A !important; "
                            "padding:4px 10px !important; border-radius:6px !important;"
                        ):
                            ui.label("Catégorielle").style(
                                "font-size:11px !important; font-weight:600 !important;"
                            )
                    
                    # Stats
                    with ui.row().classes("w-full gap-3 mb-3"):
                        with ui.card().classes("flex-1 p-2").style(
                            "background:#f8f9fa !important; border-radius:6px !important; box-shadow:none !important;"
                        ):
                            ui.label("CATÉGORIES").style(
                                "font-size:10px !important; color:#636e72 !important; margin-bottom:2px !important;"
                            )
                            ui.label(str(n_unique)).style(
                                "font-weight:700 !important; font-size:16px !important; color:#01335A !important;"
                            )
                        
                        with ui.card().classes("flex-1 p-2").style(
                            "background:#f8f9fa !important; border-radius:6px !important; box-shadow:none !important;"
                        ):
                            ui.label("MODE").style(
                                "font-size:10px !important; color:#636e72 !important; margin-bottom:2px !important;"
                            )
                            ui.label(str(mode_val)[:20]).style(
                                "font-weight:700 !important; font-size:16px !important; color:#01335A !important;"
                            )
                    
                    # Alerte cardinalité
                    if n_unique > 50:
                        with ui.card().classes("w-full p-2 mb-3").style(
                            "background:#e3f2fd !important; border-radius:6px !important; "
                            "border-left:3px solid #2196f3 !important; box-shadow:none !important;"
                        ):
                            ui.label(f"⚠️ Cardinalité élevée: {n_unique} catégories").style(
                                "color:#01335A !important; font-size:12px !important; font-weight:500 !important;"
                            )
                    
                    # Distribution Top 5 (pas 10)
                    with ui.column().classes("w-full gap-2 mb-3"):
                        ui.label(f"Top {min(5, len(counts))} catégories").style(
                            "font-weight:600 !important; font-size:13px !important; color:#636e72 !important;"
                        )
                        
                        for idx, (cat, cnt) in enumerate(counts.head(5).items()):
                            pct = round(cnt / total * 100, 1)
                            
                            with ui.row().classes("w-full items-center gap-2"):
                                ui.label(str(cat)[:20]).style(
                                    "width:120px !important; font-size:12px !important; "
                                    "color:#2c3e50 !important; font-weight:500 !important;"
                                )
                                
                                with ui.column().classes("flex-1"):
                                    ui.linear_progress(value=pct/100).props(
                                        'color="primary"'
                                    ).classes("h-2 rounded")
                                
                                ui.label(f"{cnt} ({pct}%)").style(
                                    "width:80px !important; text-align:right !important; "
                                    "font-size:11px !important; color:#636e72 !important; font-weight:600 !important;"
                                )
                        
                        if len(counts) > 5:
                            ui.label(f"... et {len(counts) - 5} autres catégories").style(
                                "font-size:11px !important; color:#7f8c8d !important; "
                                "font-style:italic !important; margin-top:4px !important;"
                            )
                    
                    # Graphique
                    with ui.column().classes("w-full"):
                        cat_fig = create_categorical_plot(series, col)
                        ui.plotly(cat_fig).style("width:100% !important;")

        def update_display():
            """Met à jour l'affichage selon les filtres"""
            cards_container.clear()
            
            cols_to_show = []
            filter_val = filter_type.value
            search_val = search_input.value.lower() if search_input.value else ""
            
            if filter_val == "numeric":
                cols_to_show = [c for c in numeric_cols if search_val in c.lower()]
            elif filter_val == "categorical":
                cols_to_show = [c for c in categorical_cols if search_val in c.lower()]
            else:
                cols_to_show = [c for c in X_train.columns if search_val in c.lower()]
            
            # Tri
            if sort_by.value == "name_desc":
                cols_to_show = sorted(cols_to_show, reverse=True)
            elif sort_by.value == "name":
                cols_to_show = sorted(cols_to_show)
            elif sort_by.value == "type":
                cols_to_show = sorted(cols_to_show, key=lambda x: x in categorical_cols)
            elif sort_by.value == "variance":
                numeric_vars = {c: X_train[c].var() for c in numeric_cols if c in cols_to_show}
                sorted_numeric = sorted(numeric_vars.items(), key=lambda x: -x[1])
                remaining = [c for c in cols_to_show if c not in numeric_cols]
                cols_to_show = [c for c, _ in sorted_numeric] + remaining
            
            if not cols_to_show:
                with cards_container:
                    with ui.card().classes("w-full p-8").style(
                        "background:white !important; border-radius:16px !important; text-align:center !important;"
                    ):
                        ui.label("🔍").style("font-size:48px !important; margin-bottom:12px !important;")
                        ui.label("Aucune variable ne correspond aux filtres").style(
                            "font-size:15px !important; color:#636e72 !important;"
                        )
            else:
                for col in cols_to_show:
                    if col in numeric_cols:
                        add_numeric_card(col)
                    else:
                        add_categorical_card(col)
        
        filter_type.on_value_change(lambda: update_display())
        sort_by.on_value_change(lambda: update_display())
        search_input.on_value_change(lambda: update_display())
        
        update_display()

        # --- NAVIGATION ---
        with ui.row().classes("w-full max-w-6xl justify-between gap-4 mt-8"):
            ui.button(
                "← Précédent",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/split'")
            ).style(
                "background:white !important; color:#01335A !important; font-weight:500 !important; "
                "border:1px solid #e1e8ed !important; border-radius:8px !important; height:48px !important; "
                "min-width:140px !important; font-size:14px !important; text-transform:none !important; "
                "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
            )
            
            ui.button(
                "Suivant →",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/outliers_analysis'")
            ).style(
                "background:#01335A !important; color:white !important; font-weight:600 !important; "
                "border-radius:8px !important; height:48px !important; min-width:140px !important; "
                "font-size:14px !important; text-transform:none !important;"
            )

global_state = state

# ----------------- PAGE 3.4 : GESTION DES OUTLIERS (VERSION COMPLÈTE) -----------------


@ui.page('/supervised/outliers_analysis')
def outliers_analysis_page():
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from scipy import stats

    # ---------- CONTEXTE ----------
    df = state.get("raw_df", None)
    split = state.get("split", None)
    columns_exclude = state.get("columns_exclude", {}) or {}
    target_col = state.get("target_column", None)

    if df is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Aucun dataset chargé.").style("font-size:18px; color:#c0392b; font-weight:600;")
            ui.button(" Retour à l'Upload", on_click=lambda: ui.run_javascript("window.location.href='/supervised/upload'")).style(
                "background:#01335A; color:white; padding:12px 32px; border-radius:8px; margin-top:20px;"
            )
        return

    # Sauvegarder l'original si pas déjà fait
    if "df_original_outliers" not in state:
        state["df_original_outliers"] = df.copy()

    df_current = state["raw_df"].copy()

    # Récupérer données d'entraînement si split existe
    df_train = None
    if split and "X_train" in split:
        df_train = split["X_train"].copy()
        if target_col and "y_train" in split and target_col not in df_train.columns:
            df_train[target_col] = split["y_train"]
    else:
        df_train = df_current.copy()

    active_cols = [c for c in df_current.columns if not columns_exclude.get(c, False)]
    numeric_cols = [c for c in active_cols if pd.api.types.is_numeric_dtype(df_current[c])]

    state.setdefault("outliers_strategy", {})
    state.setdefault("outliers_applied", False)

    # ---------- HELPERS ROBUSTES (identiques) ----------
    
    def analyze_variable_type(data_series):
        """Détermine si une variable est continue ou discrète"""
        n_unique = data_series.nunique()
        n_total = len(data_series.dropna())
        
        if n_unique <= 1:
            return 'constant'
        
        if n_unique == 2:
            return 'binary'
        
        uniqueness_ratio = n_unique / n_total
        
        if n_unique <= 20 or uniqueness_ratio < 0.05:
            return 'discrete'
        
        if data_series.dropna().apply(lambda x: float(x).is_integer()).all():
            value_counts = data_series.value_counts()
            if (value_counts > 1).sum() / len(value_counts) > 0.5:
                return 'discrete'
        
        return 'continuous'
    
    def detect_outliers_adaptive(data_series, var_type='continuous'):
        """Détection d'outliers adaptée au type de variable"""
        data_clean = data_series.dropna()
        
        if var_type == 'constant':
            return pd.Series(False, index=data_series.index), []
        
        if var_type == 'binary':
            return pd.Series(False, index=data_series.index), []
        
        if var_type == 'discrete':
            if len(data_clean) < 10:
                return pd.Series(False, index=data_series.index), []
            
            try:
                median = data_clean.median()
                mad = np.median(np.abs(data_clean - median))
                
                if mad == 0:
                    return pd.Series(False, index=data_series.index), []
                
                modified_z_scores = 0.6745 * (data_clean - median) / mad
                mask = pd.Series(False, index=data_series.index)
                mask[data_clean.index] = np.abs(modified_z_scores) > 3.5
                
            except Exception as e:
                print(f"Erreur MAD pour {data_series.name}: {e}")
                return pd.Series(False, index=data_series.index), []
        
        else:
            Q1 = data_clean.quantile(0.25)
            Q3 = data_clean.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                return pd.Series(False, index=data_series.index), []
            
            cv = data_clean.std() / data_clean.mean() if data_clean.mean() != 0 else 0
            
            if cv > 1:
                threshold = 2.0
            elif cv > 0.5:
                threshold = 1.5
            else:
                threshold = 3.0
            
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            mask = (data_series < lower) | (data_series > upper)
        
        outlier_indices = data_series[mask].index.tolist()
        return mask, outlier_indices

    def get_correlation_pairs(df_data, threshold=0.7):
        """Trouve les paires de variables fortement corrélées"""
        if len(numeric_cols) < 2:
            return []
        
        try:
            corr = df_data[numeric_cols].corr()
            pairs = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    corr_val = corr.iloc[i, j]
                    if pd.notna(corr_val) and abs(corr_val) >= threshold:
                        pairs.append({
                            'col_a': corr.columns[i],
                            'col_b': corr.columns[j],
                            'corr': round(corr_val, 3),
                            'abs_corr': round(abs(corr_val), 3)
                        })
            return sorted(pairs, key=lambda x: x['abs_corr'], reverse=True)
        except Exception as e:
            print(f"Erreur corrélation: {e}")
            return []

    def treat_outliers(df_data, col, method, indices_to_treat):
        """Applique le traitement des outliers - CORRECTION TYPE SAFE"""
        df_result = df_data.copy()
        
        if method == 'none' or not indices_to_treat:
            return df_result
        
        valid_indices = [idx for idx in indices_to_treat if idx in df_result.index]
        
        if not valid_indices:
            return df_result
        
        # Stocker le type d'origine
        original_dtype = df_result[col].dtype
        is_integer_type = pd.api.types.is_integer_dtype(original_dtype)
        
        if method == 'remove':
            df_result = df_result.drop(valid_indices, errors='ignore')
        
        elif method == 'cap':
            Q1 = df_result[col].quantile(0.25)
            Q3 = df_result[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            # Appliquer le capping
            clipped_values = df_result.loc[valid_indices, col].clip(lower, upper)
            
            # Si type entier, arrondir
            if is_integer_type:
                clipped_values = clipped_values.round().astype(original_dtype)
            
            df_result.loc[valid_indices, col] = clipped_values
        
        elif method == 'median':
            median_val = df_result[col].median()
            
            # Si type entier, arrondir la médiane
            if is_integer_type:
                median_val = round(median_val)
                df_result[col] = df_result[col].astype('float64')  # Conversion temporaire
                df_result.loc[valid_indices, col] = median_val
                df_result[col] = df_result[col].astype(original_dtype)  # Reconversion
            else:
                df_result.loc[valid_indices, col] = median_val
        
        elif method == 'mean':
            mean_val = df_result[col].mean()
            
            # Si type entier, arrondir la moyenne
            if is_integer_type:
                mean_val = round(mean_val)
                df_result[col] = df_result[col].astype('float64')  # Conversion temporaire
                df_result.loc[valid_indices, col] = mean_val
                df_result[col] = df_result[col].astype(original_dtype)  # Reconversion
            else:
                df_result.loc[valid_indices, col] = mean_val
        
        return df_result
    def create_adaptive_visualization(col_name, data_series, outlier_indices, var_type):
        """Crée des visualisations adaptées au type de variable"""
        
        if var_type == 'constant':
            fig = go.Figure()
            fig.add_annotation(
                text=f"Variable constante<br>Valeur unique: {data_series.iloc[0]}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="#7f8c8d")
            )
            fig.update_layout(
                title=f"{col_name} - Variable constante",
                height=300,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                paper_bgcolor='white'
            )
            return fig
        
        fig = go.Figure()
        normal_idx = [idx for idx in data_series.index if idx not in outlier_indices]
        
        if len(normal_idx) > 0:
            fig.add_trace(go.Scatter(
                x=list(range(len(normal_idx))),
                y=data_series.loc[normal_idx].values,
                mode='markers',
                marker=dict(
                    color='#01335A',
                    size=6 if len(normal_idx) < 1000 else 4,
                    opacity=0.6
                ),
                name='Valeurs normales',
                hovertemplate='Index: %{x}<br>Valeur: %{y:.2f}<extra></extra>'
            ))
        
        if outlier_indices:
            outlier_data = data_series.loc[outlier_indices]
            fig.add_trace(go.Scatter(
                x=list(range(len(outlier_indices))),
                y=outlier_data.values,
                mode='markers',
                marker=dict(
                    color='#e74c3c',
                    size=12 if len(outlier_indices) < 100 else 8,
                    symbol='x',
                    line=dict(width=2, color='darkred')
                ),
                name=f'Outliers ({len(outlier_indices)})',
                hovertemplate='Outlier<br>Valeur: %{y:.2f}<extra></extra>'
            ))
        
        type_label = {
            'continuous': 'Continue',
            'discrete': 'Discrète',
            'binary': 'Binaire'
        }.get(var_type, var_type)
        
        fig.update_layout(
            title=f"{col_name} - Distribution ({type_label})<br><sub>Outliers: {len(outlier_indices)} / {len(data_series)} ({len(outlier_indices)/len(data_series)*100:.1f}%)</sub>",
            xaxis_title="Observations",
            yaxis_title=col_name,
            height=400,
            showlegend=True,
            paper_bgcolor='white',
            plot_bgcolor='#f8f9fa',
            hovermode='closest',
            font=dict(family="Inter, sans-serif", size=12, color="#2c3e50")
        )
        
        return fig

    def open_column_modal(col_name):
        """Ouvre modal de configuration pour une colonne - STYLE MODERNE"""
        data_series = df_train[col_name]
        var_type = analyze_variable_type(data_series)
        
        mask, outlier_indices = detect_outliers_adaptive(data_series, var_type)
        n_outliers = len(outlier_indices)
        pct = round(n_outliers / len(df_train) * 100, 2) if len(df_train) > 0 else 0
        
        current_strategy = state.get("outliers_strategy", {}).get(col_name, {})
        current_method = current_strategy.get("method", "none")
        
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-6xl").style(
            "padding:0; border-radius:16px; box-shadow:0 10px 40px rgba(0,0,0,0.15);"
        ):
            # Header avec fond coloré
            with ui.column().classes("w-full").style(
                "background:linear-gradient(135deg, #01335A 0%, #09538C 100%); "
                "padding:24px 32px; border-radius:16px 16px 0 0;"
            ):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label(f"Configuration : {col_name}").style(
                        "font-weight:700; font-size:24px; color:white; font-family:'Inter', sans-serif;"
                    )
                    
                    type_badge_color = {
                        'continuous': '#01335A',
                        'discrete': '#9b59b6',
                        'binary': '#01335A',
                        'constant': '#01335A !important'
                    }.get(var_type, '#7f8c8d')
                    
                    type_label = {
                        'continuous': 'Continue',
                        'discrete': 'Discrète',
                        'binary': 'Binaire',
                        'constant': 'Constante'
                    }.get(var_type, var_type)
                    
                    ui.label(f"Type: {type_label}").style(
                        f"background:{type_badge_color}; color:white; padding:8px 20px; "
                        f"border-radius:24px; font-weight:600; font-size:13px;"
                    )
                
                outlier_color = '#e74c3c' if pct > 5 else ('#01335A' if pct > 1 else '#01335A')
                ui.label(f"Outliers détectés: {n_outliers} ({pct}%)").style(
                    f"color:white; margin-top:12px; font-weight:500; font-size:16px; opacity:0.95;"
                )
            
            # Contenu du modal
            with ui.column().classes("w-full").style("padding:32px;"):
                # Visualisation
                with ui.column().classes("w-full mb-6"):
                    fig_adaptive = create_adaptive_visualization(col_name, data_series, outlier_indices, var_type)
                    ui.plotly(fig_adaptive).style("width:100%;")
                
                # Graphiques côte à côte
                with ui.row().classes("gap-4 w-full mb-6"):
                    with ui.column().classes("flex-1"):
                        fig_box = go.Figure()
                        fig_box.add_trace(go.Box(
                            y=data_series.dropna(),
                            name=col_name,
                            marker_color='#01335A',
                            boxmean='sd'
                        ))
                        fig_box.update_layout(
                            title=f"Boxplot",
                            height=300,
                            showlegend=False,
                            paper_bgcolor='white',
                            plot_bgcolor='#f8f9fa',
                            font=dict(family="Inter, sans-serif", color="#2c3e50")
                        )
                        ui.plotly(fig_box).style("width:100%;")
                    
                    with ui.column().classes("flex-1"):
                        fig_hist = go.Figure()
                        
                        if var_type in ['discrete', 'binary']:
                            value_counts = data_series.value_counts().sort_index()
                            fig_hist.add_trace(go.Bar(
                                x=value_counts.index.astype(str),
                                y=value_counts.values,
                                marker_color='#01335A'
                            ))
                        else:
                            fig_hist.add_trace(go.Histogram(
                                x=data_series.dropna(),
                                marker_color='#01335A',
                                nbinsx=min(30, data_series.nunique())
                            ))
                        
                        fig_hist.update_layout(
                            title=f"Distribution",
                            height=300,
                            showlegend=False,
                            paper_bgcolor='white',
                            plot_bgcolor='#f8f9fa',
                            font=dict(family="Inter, sans-serif", color="#2c3e50")
                        )
                        ui.plotly(fig_hist).style("width:100%;")
                
                # Statistiques
                with ui.card().classes("w-full").style(
                    "background:#f8f9fa; padding:20px; border-radius:12px; border:1px solid #e1e8ed;"
                ):
                    ui.label(" Statistiques").style(
                        "font-weight:600; margin-bottom:12px; color:#01335A; font-size:15px;"
                    )
                    
                    col_stats = data_series.describe()
                    
                    with ui.row().classes("w-full gap-8"):
                        with ui.column():
                            ui.label(f"N valeurs: {int(col_stats['count'])}").style("font-size:13px; color:#2c3e50;")
                            ui.label(f"Min: {col_stats['min']:.2f}").style("font-size:13px; color:#2c3e50;")
                            ui.label(f"Q1: {col_stats['25%']:.2f}").style("font-size:13px; color:#2c3e50;")
                        
                        with ui.column():
                            ui.label(f"Médiane: {col_stats['50%']:.2f}").style("font-size:13px; color:#2c3e50;")
                            ui.label(f"Moyenne: {col_stats['mean']:.2f}").style("font-size:13px; color:#2c3e50;")
                            ui.label(f"Q3: {col_stats['75%']:.2f}").style("font-size:13px; color:#2c3e50;")
                        
                        with ui.column():
                            ui.label(f"Max: {col_stats['max']:.2f}").style("font-size:13px; color:#2c3e50;")
                            ui.label(f"Écart-type: {col_stats['std']:.2f}").style("font-size:13px; color:#2c3e50;")
                            ui.label(f"Valeurs uniques: {data_series.nunique()}").style("font-size:13px; color:#2c3e50;")
                
                # Avertissement
                if var_type in ['discrete', 'binary', 'constant']:
                    with ui.card().classes("w-full mt-4").style(
                        "background:#fff9e6; padding:16px; border-radius:12px; border-left:4px solid #01335A;"
                    ):
                        ui.label("💡 Recommandation").style("font-weight:600; margin-bottom:8px; color:#856404;")
                        
                        if var_type == 'constant':
                            ui.label("Variable constante - Considérez l'exclusion").style("font-size:13px; color:#856404;")
                        elif var_type == 'binary':
                            ui.label("Variable binaire - Pas d'outliers à traiter").style("font-size:13px; color:#856404;")
                        else:
                            ui.label("Variable discrète - Détection adaptative appliquée").style("font-size:13px; color:#856404;")
                
                # Sélection méthode
                ui.label("Méthode de traitement").style(
                    "font-weight:600; margin-top:24px; font-size:15px; color:#01335A;"
                )
                
                method_select = ui.select(
                    options={
                        "none": "Aucun traitement",
                        "remove": "Supprimer les lignes",
                        "cap": "Capping (IQR)",
                        "median": "Remplacer par médiane",
                        "mean": "Remplacer par moyenne"
                    },
                    value=current_method,
                    label="Choisir"
                ).props("outlined").classes("w-full").style(
                    "margin-top:8px;"
                )
                
                # Boutons
                with ui.row().classes("w-full justify-end gap-3 mt-8"):
                    ui.button("Annuler", on_click=dialog.close).props("flat").style(
                        "color:#7f8c8d; font-weight:500; text-transform:none; font-size:14px;"
                    )
                    
                    def save_strategy():
                        state.setdefault("outliers_strategy", {})[col_name] = {
                            "method": method_select.value,
                            "indices": outlier_indices,
                            "var_type": var_type
                        }
                        ui.notify(f" Stratégie sauvegardée", color="positive")
                        dialog.close()
                        ui.run_javascript("setTimeout(() => window.location.reload(), 500);")
                    
                    ui.button("Sauvegarder", on_click=save_strategy).style(
                        "background:#01335A; color:white; border-radius:8px; padding:10px 32px; "
                        "font-weight:600; text-transform:none; font-size:14px;"
                    )
        
        dialog.open()

    def apply_outliers_treatment(navigate_after=False):
        """Applique le traitement (fonction identique)"""
        try:
            strategies = state.get("outliers_strategy", {})
            
            if not strategies:
                ui.notify(" Aucune stratégie configurée", color="warning")
                return False
            
            print("\n" + "="*60)
            print("🔧 DÉBUT DU TRAITEMENT DES OUTLIERS")
            print("="*60)
            
            df_treated = state["raw_df"].copy()
            indices_to_remove = set()
            
            for col, strat in strategies.items():
                method = strat.get("method", "none")
                indices = strat.get("indices", [])
                
                if method == 'remove':
                    indices_to_remove.update(indices)
                elif method != 'none':
                    df_treated = treat_outliers(df_treated, col, method, indices)
            
            if indices_to_remove:
                df_treated = df_treated.drop(list(indices_to_remove), errors='ignore')
            
            state["raw_df"] = df_treated
            state["outliers_applied"] = True
            
            # Mise à jour du split
            split_data = state.get("split", {})
            if split_data and "X_train" in split_data:
                from sklearn.model_selection import train_test_split
                
                target_col = state.get("target_column")
                if target_col and target_col in df_treated.columns:
                    X = df_treated.drop(columns=[target_col])
                    y = df_treated[target_col]
                    
                    test_size = state.get("test_size", 0.2)
                    val_size = state.get("val_size", 0.2)
                    random_state = state.get("random_state", 42)
                    
                    stratify_param = y if y.nunique() <= 20 and y.nunique() >= 2 else None
                    
                    X_temp, X_test, y_temp, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
                    )
                    
                    val_size_adjusted = val_size / (1 - test_size)
                    stratify_param_val = y_temp if y_temp.nunique() <= 20 and y_temp.nunique() >= 2 else None
                    
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_temp, y_temp, test_size=val_size_adjusted,
                        random_state=random_state, stratify=stratify_param_val
                    )
                    
                    state["split"] = {
                        "X_train": X_train,
                        "X_val": X_val,
                        "X_test": X_test,
                        "y_train": y_train,
                        "y_val": y_val,
                        "y_test": y_test
                    }
            
            ui.notify(f" Traitement appliqué!", color="positive", timeout=3000)
            
            if navigate_after:
                ui.run_javascript("setTimeout(() => window.location.href='/supervised/multivariate_analysis', 1500);")
            else:
                ui.run_javascript("setTimeout(() => window.location.reload(), 1500);")
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERREUR: {str(e)}")
            import traceback
            traceback.print_exc()
            ui.notify(f"❌ Erreur : {str(e)}", color="negative")
            return False

    # ---------- UI PRINCIPALE - NOUVEAU STYLE ----------
    with ui.column().classes("w-full items-center").style(
        "background:#f0f2f5; min-height:100vh; padding:48px 24px; font-family:'Inter', sans-serif;"
    ):
        # HEADER ÉLÉGANT
        ui.label("Gestion des Outliers").style(
            "font-weight:700; font-size:36px; color:#01335A; margin-bottom:8px; text-align:center; letter-spacing:-0.5px;"
        )
        ui.label("Détection intelligente et traitement adaptatif des valeurs aberrantes").style(
            "font-size:16px; color:#636e72; margin-bottom:48px; text-align:center; font-weight:400;"
        )
        
        # Indicateur statut
        if state.get("outliers_applied", False):
            with ui.card().classes("w-full max-w-6xl mb-6").style(
                "background:white; border-radius:16px; padding:20px; box-shadow:0 2px 8px rgba(0,0,0,0.08); "
                "border-left:4px solid #01335A;"
            ):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label(" Traitement des outliers appliqué au dataset").style(
                        "font-weight:600; color:#01335A; font-size:15px;"
                    )
                    ui.button(
                        "Réinitialiser",
                        on_click=lambda: (
                            state.update({
                                "raw_df": state.get("df_original_outliers").copy(),
                                "outliers_applied": False,
                                "outliers_strategy": {}
                            }),
                            ui.notify("Dataset restauré", color="info"),
                            ui.run_javascript("setTimeout(() => window.location.reload(), 800);")
                        )
                    ).props("flat").style(
                        "color:#01335A; text-transform:none; font-weight:500;"
                    )

        # COMPARAISON (si appliqué)
        if state.get("outliers_applied", False):
            with ui.card().classes("w-full max-w-6xl mb-6").style(
                "background:white; border-radius:16px; padding:32px; box-shadow:0 2px 8px rgba(0,0,0,0.08);"
            ):
                ui.label("Impact du traitement").style(
                    "font-weight:700; font-size:22px; color:#01335A; margin-bottom:24px;"
                )
                
                df_original = state.get("df_original_outliers")
                df_treated = state["raw_df"]
                
                with ui.row().classes("w-full gap-6"):
                    with ui.card().classes("flex-1").style(
                        "background:#f8f9fa; padding:24px; border-radius:12px; border:1px solid #e1e8ed;"
                    ):
                        ui.label("Original").style("font-weight:600; font-size:14px; color:#7f8c8d; margin-bottom:12px;")
                        ui.label(f"{df_original.shape[0]:,}").style("font-size:32px; font-weight:700; color:#01335A;")
                        ui.label("lignes").style("font-size:14px; color:#7f8c8d;")
                    
                    with ui.card().classes("flex-1").style(
                        "background:#f8f9fa; padding:24px; border-radius:12px; border:1px solid #c8e6c9;"
                    ):
                        ui.label("Après traitement").style("font-weight:600; font-size:14px; color:#7f8c8d; margin-bottom:12px;")
                        ui.label(f"{df_treated.shape[0]:,}").style("font-size:32px; font-weight:700; color:#01335A;")
                        rows_removed = df_original.shape[0] - df_treated.shape[0]
                        if rows_removed > 0:
                            ui.label(f"-{rows_removed} lignes ({rows_removed/df_original.shape[0]*100:.1f}%)").style(
                                "font-size:14px; color:#e74c3c; font-weight:600;"
                            )
                        else:
                            ui.label("lignes").style("font-size:14px; color:#7f8c8d;")

        # MATRICE CORRÉLATION
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white; border-radius:16px; padding:32px; box-shadow:0 2px 8px rgba(0,0,0,0.08);"
        ):
            ui.label("Matrice de corrélation").style(
                "font-weight:700; font-size:22px; color:#01335A; margin-bottom:24px;"
            )
            
            if len(numeric_cols) >= 2:
                try:
                    corr_matrix = df_current[numeric_cols].corr()
                    
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu',
                        zmid=0,
                        text=np.round(corr_matrix.values, 2),
                        texttemplate='%{text}',
                        textfont={"size": 9},
                        colorbar=dict(title="Corrélation")
                    ))
                    
                    fig_corr.update_layout(
                        height=min(700, len(numeric_cols) * 35 + 150),
                        xaxis={'side': 'bottom'},
                        paper_bgcolor='white',
                        plot_bgcolor='#f8f9fa',
                        font=dict(family="Inter, sans-serif", color="#2c3e50")
                    )
                    
                    ui.plotly(fig_corr).style("width:100%;")
                    
                    pairs = get_correlation_pairs(df_current, threshold=0.7)
                    if pairs:
                        with ui.card().classes("w-full mt-4").style(
                            "background:#fff9e6; padding:16px; border-radius:12px; border-left:4px solid #01335A;"
                        ):
                            ui.label(f" {len(pairs)} paires fortement corrélées").style(
                                "font-weight:600; margin-bottom:8px; color:#856404;"
                            )
                            for p in pairs[:8]:
                                ui.label(f"• {p['col_a']} ↔ {p['col_b']}: r = {p['corr']}").style(
                                    "font-size:13px; margin-left:12px; color:#856404;"
                                )
                except Exception as e:
                    ui.label(f"Erreur: {str(e)}").style("color:#e74c3c;")
            else:
                ui.label("Moins de 2 colonnes numériques").style("color:#7f8c8d;")

        # TABLEAU DÉTECTION
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white; border-radius:16px; padding:32px; box-shadow:0 2px 8px rgba(0,0,0,0.08);"
        ):
            ui.label("Analyse des variables").style(
                "font-weight:700; font-size:22px; color:#01335A; margin-bottom:24px;"
            )
            
            if not numeric_cols:
                ui.label("Aucune colonne numérique détectée").style("color:#7f8c8d;")
            else:
                rows = []
                for col in numeric_cols:
                    try:
                        data_series = df_train[col]
                        var_type = analyze_variable_type(data_series)
                        mask, indices = detect_outliers_adaptive(data_series, var_type)
                        n_outliers = len(indices)
                        pct = round(n_outliers / len(df_train) * 100, 2) if len(df_train) > 0 else 0
                        
                        if pct > 5:
                            niveau = "🔴 Critique"
                        elif pct > 1:
                            niveau = "🟡 Modéré"
                        elif n_outliers > 0:
                            niveau = "🟢 Faible"
                        else:
                            niveau = " Aucun"
                        
                        type_label = {
                            'continuous': 'Continue',
                            'discrete': 'Discrète',
                            'binary': 'Binaire',
                            'constant': 'Constante'
                        }.get(var_type, var_type)
                        
                        current_strat = state.get("outliers_strategy", {}).get(col, {})
                        method = current_strat.get("method", "none")
                        
                        rows.append({
                            "Feature": col,
                            "Type": type_label,
                            "Outliers": n_outliers,
                            "Pourcentage": f"{pct}%",
                            "Niveau": niveau,
                            "Traitement": method if method != "none" else "Aucun"
                        })
                    except Exception as e:
                        print(f"Erreur {col}: {e}")
                
                table = ui.table(
                    columns=[
                        {"name": "Feature", "label": "Feature", "field": "Feature", "align": "left", "sortable": True},
                        {"name": "Type", "label": "Type", "field": "Type", "align": "center"},
                        {"name": "Outliers", "label": "Outliers", "field": "Outliers", "align": "center", "sortable": True},
                        {"name": "Pourcentage", "label": "%", "field": "Pourcentage", "align": "center", "sortable": True},
                        {"name": "Niveau", "label": "Niveau", "field": "Niveau", "align": "center"},
                        {"name": "Traitement", "label": "Traitement", "field": "Traitement", "align": "center"}
                    ],
                    rows=rows,
                    row_key="Feature",
                    pagination={"rowsPerPage": 20}
                ).style("width:100%;").props("flat bordered").classes("cursor-pointer")
                
                table.on('row-click', lambda e: open_column_modal(e.args[1]['Feature']))
                
                with ui.card().classes("w-full mt-4").style(
                    "background:#01335A; padding:16px; border-radius:12px; border-left:4px solid#01335A;"
                ):
                    ui.label("💡 Cliquez sur une ligne pour configurer").style(
                        "font-weight:500; color:white; font-size:14px;"
                    )

        # BOUTON APPLIQUER
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:linear-gradient(135deg, #01335A 0%, #09538C 100%); "
            "border-radius:16px; padding:40px; box-shadow:0 4px 16px rgba(1,51,90,0.2);"
        ):
            ui.label("Appliquer le traitement").style(
                "font-weight:700; font-size:24px; color:white; margin-bottom:16px; text-align:center;"
            )
            
            strategies = state.get("outliers_strategy", {})
            n_strategies = len([s for s in strategies.values() if s.get("method") != "none"])
            
            if n_strategies > 0:
                ui.label(f"{n_strategies} stratégie(s) configurée(s)").style(
                    "color:white; font-size:16px; text-align:center; margin-bottom:24px; opacity:0.9;"
                )
                
                with ui.row().classes("w-full justify-center gap-4"):
                    ui.button(
                        "Appliquer",
                        on_click=lambda: apply_outliers_treatment(navigate_after=False)
                    ).style(
                        "background:white; color:#01335A; font-weight:600; border-radius:8px; "
                        "height:50px; min-width:180px; font-size:15px; text-transform:none;"
                    )
                    
                    ui.button(
                        "Appliquer et continuer",
                        on_click=lambda: apply_outliers_treatment(navigate_after=True)
                    ).style(
                        "background:rgba(255,255,255,0.2); color:white; font-weight:600; border-radius:8px; "
                        "height:50px; min-width:200px; font-size:15px; text-transform:none; backdrop-filter:blur(10px);"
                    )
            else:
                ui.label("Configurez d'abord les traitements").style(
                    "color:white; font-size:16px; text-align:center; opacity:0.8;"
                )

        # NAVIGATION
        with ui.row().classes("w-full max-w-6xl justify-between gap-4 mt-8"):
            ui.button(
                " Précédent",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/split'")
            ).style(
                "background:#01335A !important; color:white; font-weight:500; border-radius:8px; "
                "height:48px; min-width:140px; font-size:14px; text-transform:none; box-shadow:0 2px 8px rgba(0,0,0,0.08);"
            )
            
            ui.button(
                "Suivant ",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/multivariate_analysis'")
            ).style(
                "background:#01335A !important; color:white; font-weight:600; border-radius:8px; "
                "height:48px; min-width:140px; font-size:14px; text-transform:none;"
            )



# ----------------- PAGE 3.5 : ANALYSE MULTIVARIÉE ----------------

@ui.page('/supervised/multivariate_analysis')
def multivariate_analysis_page():
    """
    Analyse multivariée - Corrélations et redondance
    Design moderne unifié avec #01335A
    
    Cette page permet d'analyser les corrélations entre variables numériques pour :
    - Détecter la redondance (features hautement corrélées)
    - Calculer le VIF (Variance Inflation Factor) pour la multicolinéarité
    - Créer des features engineered (combinaisons de features existantes)
    - Optimiser le dataset pour les algorithmes (notamment Naive Bayes)
    """
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    df = state.get("raw_df", None)
    target_col = state.get("target_column", None)
    columns_exclude = state.get("columns_exclude", {}) or {}

    if df is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Aucun dataset chargé.").style(
                "font-size:18px !important; color:#c0392b !important; font-weight:600 !important;"
            )
            ui.button(
                " Retour à l'Upload",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/upload'")
            ).style(
                "background:#01335A !important; color:white !important; padding:12px 32px !important; "
                "border-radius:8px !important; margin-top:20px !important;"
            )
        return

    # Préparations : identifier les colonnes numériques actives
    numeric_cols = df.select_dtypes(include=[int, float, "number"]).columns.tolist()
    active_numeric = [c for c in numeric_cols if not columns_exclude.get(c, False) and c != target_col]

    if "engineered_features" not in state:
        state["engineered_features"] = []

    # Calculer les paires corrélées une seule fois (optimisation)
    pairs = []
    corr = None
    if len(active_numeric) >= 2:
        corr = df[active_numeric].corr()
        for i, a in enumerate(corr.index):
            for j, b in enumerate(corr.columns):
                if j <= i:
                    continue
                r = corr.iloc[i, j]
                if pd.notna(r) and abs(r) >= 0.8:
                    pairs.append((a, b, float(r)))
        pairs = sorted(pairs, key=lambda x: -abs(x[2]))

    # ----------------- FONCTIONS INTERNES -----------------
    
    def make_histogram_plot(series, name):
        """Crée un histogramme Plotly pour visualiser la distribution d'une feature"""
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=series,
            nbinsx=30,
            marker_color='#01335A',
            opacity=0.8,
            name=name
        ))
        fig.update_layout(
            margin=dict(l=40, r=10, t=20, b=40),
            height=220,
            showlegend=False,
            paper_bgcolor='white',
            plot_bgcolor='#f8f9fa',
            font=dict(family="Inter, sans-serif", color="#2c3e50")
        )
        return fig

    def open_pair_modal(feat_a, feat_b, r_val):
        """
        Modal détaillé pour une paire de features corrélées
        Affiche : scatter plot + formulaire de création de feature combinée
        """
        with ui.dialog() as dlg, ui.card().classes("w-full max-w-4xl").style(
            "padding:0 !important; border-radius:16px !important; box-shadow:0 10px 40px rgba(0,0,0,0.15) !important;"
        ):
            # Header avec gradient
            with ui.column().classes("w-full").style(
                "background:linear-gradient(135deg, #01335A 0%, #09538C 100%) !important; "
                "padding:24px 32px !important; border-radius:16px 16px 0 0 !important;"
            ):
                ui.label(f"Analyse de paire : {feat_a} ↔ {feat_b}").style(
                    "font-weight:700 !important; font-size:22px !important; color:white !important;"
                )
                ui.label(f"Corrélation : {r_val:.3f}").style(
                    "color:white !important; font-size:16px !important; margin-top:4px !important; opacity:0.9 !important;"
                )
            
            # Contenu
            with ui.column().classes("w-full").style("padding:32px !important;"):
                # Scatter plot pour visualiser la relation
                scatter = go.Figure()
                scatter.add_trace(go.Scatter(
                    x=df[feat_a],
                    y=df[feat_b],
                    mode='markers',
                    marker=dict(color='#01335A', opacity=0.5, size=6),
                    name="points"
                ))
                scatter.update_layout(
                    height=400,
                    xaxis_title=feat_a,
                    yaxis_title=feat_b,
                    paper_bgcolor='white',
                    plot_bgcolor='#f8f9fa',
                    font=dict(family="Inter, sans-serif", color="#2c3e50")
                )
                ui.plotly(scatter).style("width:100% !important;")
                
                # Formulaire création feature combinée
                with ui.card().classes("w-full mt-6").style(
                    "background:#f8f9fa !important; padding:20px !important; "
                    "border-radius:12px !important; border:1px solid #e1e8ed !important;"
                ):
                    ui.label("Créer une nouvelle feature combinée").style(
                        "font-weight:600 !important; font-size:16px !important; "
                        "color:#01335A !important; margin-bottom:16px !important;"
                    )
                    
                    with ui.row().classes("w-full gap-4 items-end"):
                        name_input = ui.input(
                            label="Nom de la nouvelle feature",
                            placeholder=f"{feat_b}_par_{feat_a}"
                        ).props("outlined").classes("flex-1")
                        
                        formula_select = ui.select(
                            options={
                                f"{feat_b} / {feat_a}": f"{feat_b} ÷ {feat_a}",
                                f"{feat_b} - {feat_a}": f"{feat_b} − {feat_a}",
                                f"{feat_b} + {feat_a}": f"{feat_b} + {feat_a}",
                                f"{feat_b} * {feat_a}": f"{feat_b} × {feat_a}"
                            },
                            label="Formule",
                            value=f"{feat_b} / {feat_a}"
                        ).props("outlined").classes("flex-1")
                
                # Boutons
                with ui.row().classes("w-full justify-end gap-3 mt-6"):
                    ui.button("Annuler", on_click=lambda: dlg.close()).props("flat").style(
                        "color:#7f8c8d !important; text-transform:none !important;"
                    )
                    
                    ui.button(
                        "Créer la feature",
                        on_click=lambda: create_engineered_feature(
                            feat_a, feat_b, name_input.value, formula_select.value, dlg
                        )
                    ).style(
                        "background:#01335A !important; color:white !important; border-radius:8px !important; "
                        "padding:10px 24px !important; text-transform:none !important; font-weight:600 !important;"
                    )
        
        dlg.open()

    def create_engineered_feature(a, b, new_name, formula_str, dialog_obj):
        """
        Crée une nouvelle feature en combinant deux features existantes
        Formules possibles : division, soustraction, addition, multiplication
        """
        if not new_name:
            ui.notify("⚠️ Donnez un nom à la nouvelle feature", color="warning")
            return
        
        if new_name in df.columns:
            ui.notify("⚠️ Une colonne avec ce nom existe déjà", color="warning")
            return
        
        try:
            s_a = df[a].astype(float)
            s_b = df[b].astype(float)
            
            if "/" in formula_str:
                new_series = s_b / s_a.replace({0: np.nan})  # Éviter division par zéro
            elif "-" in formula_str:
                new_series = s_b - s_a
            elif "+" in formula_str:
                new_series = s_b + s_a
            elif "*" in formula_str:
                new_series = s_b * s_a
            else:
                ui.notify("❌ Formule non reconnue", color="negative")
                return
            
            # Ajouter au DataFrame
            df[new_name] = new_series
            state["raw_df"] = df
            state.setdefault("engineered_features", []).append(
                (new_name, f"{formula_str} of {b} and {a}")
            )
            
            ui.notify(f"✅ Feature '{new_name}' créée avec succès", color="positive")
            dialog_obj.close()
            
        except Exception as e:
            ui.notify(f"❌ Erreur : {str(e)}", color="negative")

    def keep_feature(a, b, keep="both"):
        """
        Gère l'exclusion de features redondantes
        Options : garder A, garder B, ou garder les deux
        """
        if keep == a:
            state.setdefault("columns_exclude", {})[b] = True
            ui.notify(f"✅ {a} conservée, {b} exclue", color="positive")
        elif keep == b:
            state.setdefault("columns_exclude", {})[a] = True
            ui.notify(f"✅ {b} conservée, {a} exclue", color="positive")
        else:
            state.setdefault("columns_exclude", {})[a] = False
            state.setdefault("columns_exclude", {})[b] = False
            ui.notify(f"✅ Les deux features sont conservées", color="positive")
        
        ui.run_javascript("setTimeout(() => window.location.reload(), 800);")

    def apply_naivebayes_prune():
        """
        Applique une stratégie d'optimisation pour Naive Bayes
        Pour chaque paire corrélée, garde la feature la plus corrélée avec la target
        Naive Bayes assume l'indépendance des features : les corrélations élevées violent cette hypothèse
        """
        if not pairs:
            ui.notify("⚠️ Aucune paire corrélée à traiter", color="warning")
            return
        
        excluded_count = 0
        for a, b, r in pairs:
            if target_col and target_col in df.columns:
                try:
                    corr_a = abs(df[a].corr(df[target_col]))
                    corr_b = abs(df[b].corr(df[target_col]))
                except:
                    corr_a = corr_b = 0
                
                # Garder celle avec la plus forte corrélation à la target
                if corr_a >= corr_b:
                    state.setdefault("columns_exclude", {})[b] = True
                    excluded_count += 1
                else:
                    state.setdefault("columns_exclude", {})[a] = True
                    excluded_count += 1
            else:
                # Si pas de target, exclure arbitrairement B
                state.setdefault("columns_exclude", {})[b] = True
                excluded_count += 1
        
        ui.notify(f"✅ {excluded_count} features exclues (optimisation Naive Bayes)", color="positive")
        ui.run_javascript("setTimeout(() => window.location.reload(), 1000);")

    def open_bulk_engineer_modal():
        """Modal pour créer une feature combinée de manière manuelle (choix libre des features)"""
        with ui.dialog() as dlg, ui.card().classes("w-full max-w-2xl").style(
            "padding:0 !important; border-radius:16px !important; box-shadow:0 10px 40px rgba(0,0,0,0.15) !important;"
        ):
            # Header
            with ui.column().classes("w-full").style(
                "background:linear-gradient(135deg, #01335A 0%, #09538C 100%) !important; "
                "padding:24px 32px !important; border-radius:16px 16px 0 0 !important;"
            ):
                ui.label("Créer une feature combinée").style(
                    "font-weight:700 !important; font-size:22px !important; color:white !important;"
                )
            
            # Contenu
            with ui.column().classes("w-full").style("padding:32px !important;"):
                col_a = ui.select(
                    options=active_numeric,
                    label="Feature A"
                ).props("outlined").classes("w-full mb-4")
                
                col_b = ui.select(
                    options=active_numeric,
                    label="Feature B"
                ).props("outlined").classes("w-full mb-4")
                
                name_input = ui.input(
                    label="Nom de la nouvelle feature",
                    placeholder="nouvelle_feature"
                ).props("outlined").classes("w-full mb-4")
                
                formula_select = ui.select(
                    options={
                        "A / B": "A ÷ B",
                        "B / A": "B ÷ A",
                        "A - B": "A − B",
                        "B - A": "B − A",
                        "A + B": "A + B",
                        "A * B": "A × B"
                    },
                    label="Formule",
                    value="B / A"
                ).props("outlined").classes("w-full")
                
                def create_btn_action():
                    a = col_a.value
                    b = col_b.value
                    
                    if not a or not b or not name_input.value:
                        ui.notify("⚠️ Sélectionnez A, B et donnez un nom", color="warning")
                        return
                    
                    mapping = {
                        "A / B": f"{a} / {b}",
                        "B / A": f"{b} / {a}",
                        "A - B": f"{a} - {b}",
                        "B - A": f"{b} - {a}",
                        "A + B": f"{a} + {b}",
                        "A * B": f"{a} * {b}",
                    }
                    formula = mapping.get(formula_select.value, f"{b} / {a}")
                    create_engineered_feature(a, b, name_input.value, formula, dlg)
                
                # Boutons
                with ui.row().classes("w-full justify-end gap-3 mt-6"):
                    ui.button("Annuler", on_click=lambda: dlg.close()).props("flat").style(
                        "color:#7f8c8d !important; text-transform:none !important;"
                    )
                    
                    ui.button("Créer", on_click=create_btn_action).style(
                        "background:#01335A !important; color:white !important; border-radius:8px !important; "
                        "padding:10px 24px !important; text-transform:none !important; font-weight:600 !important;"
                    )
        
        dlg.open()

    # ---------- UI PRINCIPALE ----------
    with ui.column().classes("w-full items-center").style(
        "background:#f0f2f5 !important; min-height:100vh !important; padding:48px 24px !important; "
        "font-family:'Inter', sans-serif !important;"
    ):
        # HEADER
        ui.label("Analyse Multivariée").style(
            "font-weight:700 !important; font-size:36px !important; color:#01335A !important; "
            "margin-bottom:8px !important; text-align:center !important; letter-spacing:-0.5px !important;"
        )
        ui.label("Corrélations et gestion de la redondance entre variables").style(
            "font-size:16px !important; color:#636e72 !important; margin-bottom:48px !important; "
            "text-align:center !important; font-weight:400 !important;"
        )
        
        # EXPLICATION DE LA PAGE
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:#e3f2fd !important; border-radius:16px !important; padding:24px !important; "
            "border-left:4px solid #2196f3 !important; box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            ui.label("💡 À quoi sert cette étape ?").style(
                "font-weight:700 !important; font-size:18px !important; color:#01335A !important; "
                "margin-bottom:12px !important;"
            )
            
            ui.label(
                "Cette page analyse les corrélations entre vos features numériques pour :"
            ).style(
                "font-size:14px !important; color:#2c3e50 !important; margin-bottom:8px !important;"
            )
            
            points = [
                "🔍 Détecter la redondance : deux features très corrélées apportent une information similaire",
                "📊 Calculer le VIF : mesure la multicolinéarité (problématique pour certains modèles)",
                " Créer des features combinées : nouvelles features en combinant des existantes (ex: ratio, différence)",
                "🎯 Optimiser pour vos algorithmes : Naive Bayes est sensible aux corrélations, les arbres de décision non"
            ]
            
            for point in points:
                ui.label(f"• {point}").style(
                    "font-size:13px !important; color:#01335A !important; margin-left:12px !important; "
                    "margin-bottom:4px !important; line-height:1.6 !important;"
                )
        
        # SECTION A : HEATMAP
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white !important; border-radius:16px !important; padding:32px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            ui.label("📊 Matrice de corrélation").style(
                "font-weight:700 !important; font-size:22px !important; color:#01335A !important; "
                "margin-bottom:16px !important;"
            )
            
            ui.label("Cliquez sur une cellule pour analyser la paire en détail").style(
                "font-size:14px !important; color:#636e72 !important; margin-bottom:20px !important;"
            )

            if len(active_numeric) < 2:
                ui.label("⚠️ Moins de 2 colonnes numériques disponibles").style(
                    "color:#7f8c8d !important; font-size:15px !important;"
                )
            else:
                try:
                    # Heatmap interactive
                    heatmap_fig = go.Figure(data=go.Heatmap(
                        z=corr.values,
                        x=corr.columns,
                        y=corr.columns,
                        colorscale='RdBu',
                        zmid=0,
                        colorbar=dict(title="Corrélation"),
                        hovertemplate='%{y} ↔ %{x}<br>Corrélation: %{z:.3f}<extra></extra>'
                    ))
                    
                    heatmap_fig.update_layout(
                        title="Matrice de corrélation (variables numériques actives)",
                        height=min(700, len(active_numeric) * 35 + 150),
                        xaxis={'side': 'bottom'},
                        paper_bgcolor='white',
                        plot_bgcolor='#f8f9fa',
                        font=dict(family="Inter, sans-serif", color="#2c3e50")
                    )

                    plot = ui.plotly(heatmap_fig).style("width:100% !important;")
                    
                    # Click handler
                    def on_heatmap_click(e):
                        try:
                            point = e["points"][0]
                            feat_b = point["x"]
                            feat_a = point["y"]
                            r_val = float(point["z"])
                            if feat_a != feat_b:
                                open_pair_modal(feat_a, feat_b, r_val)
                        except Exception as exc:
                            print(f"Erreur click heatmap: {exc}")
                    
                    plot.on("plotly_click", on_heatmap_click)
                    
                    # Afficher les paires fortement corrélées
                    if len(pairs) > 0:
                        with ui.card().classes("w-full mt-4").style(
                            "background:#fff9e6 !important; padding:16px !important; "
                            "border-radius:12px !important; border-left:4px solid #f39c12 !important;"
                        ):
                            ui.label(f"⚠️ {len(pairs)} paire(s) fortement corrélée(s) (|r| ≥ 0.8)").style(
                                "font-weight:600 !important; margin-bottom:8px !important; color:#856404 !important;"
                            )
                            for p_a, p_b, p_r in pairs[:8]:
                                ui.label(f"• {p_a} ↔ {p_b}: r = {p_r:.3f}").style(
                                    "font-size:13px !important; margin-left:12px !important; color:#856404 !important;"
                                )
                            if len(pairs) > 8:
                                ui.label(f"... et {len(pairs)-8} autres paires").style(
                                    "font-size:13px !important; margin-left:12px !important; "
                                    "font-style:italic !important; color:#7f8c8d !important;"
                                )
                
                except Exception as e:
                    ui.label(f"⚠️ Erreur lors du calcul de la corrélation: {str(e)}").style(
                        "color:#e74c3c !important;"
                    )

        # SECTION B : PAIRES CORRÉLÉES
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white !important; border-radius:16px !important; padding:32px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            ui.label("🔗 Paires fortement corrélées").style(
                "font-weight:700 !important; font-size:22px !important; color:#01335A !important; "
                "margin-bottom:16px !important;"
            )
            
            ui.label("Détection automatique des corrélations élevées (|r| ≥ 0.8)").style(
                "font-size:14px !important; color:#636e72 !important; margin-bottom:20px !important;"
            )

            if len(active_numeric) < 2:
                ui.label("⚠️ Pas assez de variables numériques").style("color:#7f8c8d !important;")
            elif len(pairs) == 0:
                with ui.card().classes("w-full").style(
                    "background:#e8f5e9 !important; padding:16px !important; "
                    "border-radius:12px !important; border-left:4px solid #4caf50 !important;"
                ):
                    ui.label("✅ Aucune corrélation élevée détectée").style(
                        "color:#1b5e20 !important; font-weight:500 !important;"
                    )
            else:
                rows = []
                for a, b, r in pairs:
                    if abs(r) > 0.9:
                        mark = "🔴"
                    elif abs(r) >= 0.8:
                        mark = "🟡"
                    else:
                        mark = "🟢"
                    
                    # Impact sur les algorithmes
                    impact_nb = "🔴" if abs(r) > 0.85 else "🟡"  # Naive Bayes très sensible
                    impact_knn = "🟡" if abs(r) >= 0.8 else "🟢"  # KNN modérément sensible
                    impact_c45 = "🟢"  # Decision Tree robuste
                    
                    rows.append({
                        "Feature A": a,
                        "Feature B": b,
                        "Corrélation": f"{r:.3f} {mark}",
                        "Impact NB": impact_nb,
                        "Impact KNN": impact_knn,
                        "Impact C4.5": impact_c45
                    })
                
                ui.table(
                    columns=[
                        {"name": "Feature A", "label": "Feature A", "field": "Feature A", "align": "left"},
                        {"name": "Feature B", "label": "Feature B", "field": "Feature B", "align": "left"},
                        {"name": "Corrélation", "label": "Corrélation", "field": "Corrélation", "align": "center"},
                        {"name": "Impact NB", "label": "Naive Bayes", "field": "Impact NB", "align": "center"},
                        {"name": "Impact KNN", "label": "KNN", "field": "Impact KNN", "align": "center"},
                        {"name": "Impact C4.5", "label": "C4.5", "field": "Impact C4.5", "align": "center"}
                    ],
                    rows=rows,
                    row_key="Feature A"
                ).props("flat bordered").style("width:100% !important;")

        # SECTION C : VIF (CORRIGÉE)
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white !important; border-radius:16px !important; padding:32px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            ui.label("📈 VIF (Variance Inflation Factor)").style(
                "font-weight:700 !important; font-size:22px !important; color:#01335A !important; "
                "margin-bottom:16px !important;"
            )
            
            ui.label("Indicateur de multicolinéarité : VIF > 5 suggère une redondance").style(
                "font-size:14px !important; color:#636e72 !important; margin-bottom:20px !important;"
            )
            
            if len(active_numeric) < 2:
                ui.label("⚠️ Pas assez de colonnes numériques").style(
                    "color:#7f8c8d !important; font-size:15px !important;"
                )
            else:
                vif_df = []
                try:
                    # Nettoyage et conversion des données
                    X_vif = df[active_numeric].copy()
                    X_vif = X_vif.dropna()
                    
                    if len(X_vif) < 10:
                        ui.label("⚠️ Pas assez de lignes valides après suppression des NaN").style(
                            "color:#7f8c8d !important; font-size:15px !important;"
                        )
                    else:
                        # Convertir en float64 (Int64 cause des erreurs)
                        for col in X_vif.columns:
                            X_vif[col] = X_vif[col].astype('float64')
                        
                        # Supprimer les colonnes avec variance nulle
                        variance = X_vif.var()
                        non_constant_cols = variance[variance > 1e-10].index.tolist()
                        
                        if len(non_constant_cols) < 2:
                            ui.label("⚠️ Pas assez de colonnes avec variance non-nulle").style(
                                "color:#7f8c8d !important; font-size:15px !important;"
                            )
                        else:
                            X_vif = X_vif[non_constant_cols]
                            
                            # Supprimer les valeurs infinies
                            X_vif = X_vif.replace([np.inf, -np.inf], np.nan).dropna()
                            
                            if len(X_vif) < 10:
                                ui.label("⚠️ Pas assez de lignes valides après nettoyage").style(
                                    "color:#7f8c8d !important; font-size:15px !important;"
                                )
                            else:
                                # Ajouter constante
                                try:
                                    X_vif_const = sm.add_constant(X_vif, has_constant='add')
                                except Exception as e:
                                    print(f"⚠️ add_constant failed: {e}")
                                    X_vif_const = X_vif
                                
                                # Calculer VIF pour chaque colonne
                                calculated_vifs = []
                                for i, col in enumerate(non_constant_cols):
                                    try:
                                        if X_vif_const.shape[1] > X_vif.shape[1]:
                                            vif_val = variance_inflation_factor(X_vif_const.values, i + 1)
                                        else:
                                            vif_val = variance_inflation_factor(X_vif_const.values, i)
                                        
                                        if np.isnan(vif_val) or np.isinf(vif_val):
                                            vif_val = 999.99
                                            level = "🔴 Colinéarité parfaite"
                                        else:
                                            if vif_val <= 5:
                                                level = "🟢 OK"
                                            elif vif_val <= 10:
                                                level = "🟡 Modéré"
                                            else:
                                                level = "🔴 Élevé"
                                        
                                        calculated_vifs.append({
                                            "Feature": col,
                                            "VIF": f"{vif_val:.2f}" if vif_val < 999 else "> 999",
                                            "Niveau": level
                                        })
                                    
                                    except Exception as e:
                                        print(f"⚠️ VIF calculation failed for {col}: {e}")
                                        calculated_vifs.append({
                                            "Feature": col,
                                            "VIF": "N/A",
                                            "Niveau": "⚠️ Erreur calcul"
                                        })
                                
                                if len(calculated_vifs) > 0:
                                    # Trier par VIF décroissant
                                    try:
                                        calculated_vifs = sorted(
                                            calculated_vifs, 
                                            key=lambda x: float(x["VIF"].replace("> 999", "999").replace("N/A", "0")),
                                            reverse=True
                                        )
                                    except:
                                        pass
                                    
                                    ui.table(
                                        columns=[
                                            {"name": "Feature", "label": "Feature", "field": "Feature", "align": "left"},
                                            {"name": "VIF", "label": "VIF", "field": "VIF", "align": "center"},
                                            {"name": "Niveau", "label": "Niveau", "field": "Niveau", "align": "center"}
                                        ],
                                        rows=calculated_vifs,
                                        row_key="Feature"
                                    ).props("flat bordered").style("width:100% !important;")
                                    
                                    # Avertissement si VIF élevé
                                    high_vif = [x for x in calculated_vifs if "Élevé" in x["Niveau"] or "parfaite" in x["Niveau"]]
                                    if high_vif:
                                        with ui.card().classes("w-full mt-4").style(
                                            "background:#fff3cd !important; padding:16px !important; "
                                            "border-radius:12px !important; border-left:4px solid #f39c12 !important;"
                                        ):
                                            ui.label(f"⚠️ {len(high_vif)} feature(s) avec VIF élevé détectée(s)").style(
                                                "font-weight:600 !important; color:#856404 !important; margin-bottom:8px !important;"
                                            )
                                            ui.label(
                                                "Recommandation : Exclure une des features corrélées ou créer une combinaison"
                                            ).style(
                                                "font-size:13px !important; color:#856404 !important;"
                                            )
                                else:
                                    ui.label("⚠️ Impossible de calculer le VIF").style(
                                        "color:#e74c3c !important; font-size:15px !important;"
                                    )
                
                except Exception as e:
                    with ui.card().classes("w-full").style(
                        "background:#ffe6e6 !important; padding:16px !important; "
                        "border-radius:12px !important; border-left:4px solid #e74c3c !important;"
                    ):
                        ui.label("❌ Erreur lors du calcul du VIF").style(
                            "font-weight:600 !important; color:#c0392b !important; margin-bottom:8px !important;"
                        )
                        ui.label(f"Détail : {str(e)}").style(
                            "font-size:13px !important; color:#c0392b !important; margin-bottom:8px !important;"
                        )
                        ui.label(
                            "Causes probables : Colonnes avec variance nulle, valeurs infinies, ou types incompatibles"
                        ).style(
                            "font-size:13px !important; color:#c0392b !important;"
                        )

        # SECTION D : ACTIONS
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white !important; border-radius:16px !important; padding:32px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            ui.label(" Actions et recommandations").style(
                "font-weight:700 !important; font-size:22px !important; color:#01335A !important; "
                "margin-bottom:16px !important;"
            )

            # Boutons globaux
            with ui.row().classes("gap-3 mb-6"):
                ui.button(
                    "🎯 Optimiser pour Naive Bayes",
                    on_click=apply_naivebayes_prune
                ).style(
                    "background:#01335A !important; color:white !important; border-radius:8px !important; "
                    "padding:10px 20px !important; text-transform:none !important; font-weight:500 !important;"
                )
                
                ui.button(
                    "➕ Créer feature combinée",
                    on_click=open_bulk_engineer_modal
                ).style(
                    "background:#3498db !important; color:white !important; border-radius:8px !important; "
                    "padding:10px 20px !important; text-transform:none !important; font-weight:500 !important;"
                )

            # Cartes comparatives pour chaque paire
            if len(pairs) > 0:
                ui.label(f"{len(pairs)} paire(s) à analyser").style(
                    "font-weight:600 !important; color:#636e72 !important; margin-bottom:16px !important;"
                )
                
                for idx, (a, b, r) in enumerate(pairs[:5]):  # Limiter à 5 pour performance
                    with ui.expansion(
                        f"{a} ↔ {b} (r = {r:.3f})", 
                        icon="compare_arrows"
                    ).classes("w-full mb-3"):
                        with ui.row().classes("w-full gap-4"):
                            # Feature A
                            with ui.card().classes("flex-1").style(
                                "background:#f8f9fa !important; padding:16px !important; border-radius:12px !important;"
                            ):
                                ui.label(a).style(
                                    "font-weight:700 !important; font-size:15px !important; "
                                    "margin-bottom:8px !important; color:#01335A !important;"
                                )
                                hist_a = make_histogram_plot(df[a].dropna(), a)
                                ui.plotly(hist_a).style("width:100% !important;")
                                ui.label(f"Moyenne: {float(df[a].dropna().mean()):.3f}").style(
                                    "font-size:13px !important; color:#636e72 !important;"
                                )
                                
                                if target_col and pd.api.types.is_numeric_dtype(df[target_col]):
                                    try:
                                        corr_a = df[a].corr(df[target_col])
                                        ui.label(f"Corrélation avec target: {corr_a:.3f}").style(
                                            "font-size:13px !important; color:#01335A !important; font-weight:600 !important;"
                                        )
                                    except:
                                        pass

                            # Feature B
                            with ui.card().classes("flex-1").style(
                                "background:#f8f9fa !important; padding:16px !important; border-radius:12px !important;"
                            ):
                                ui.label(b).style(
                                    "font-weight:700 !important; font-size:15px !important; "
                                    "margin-bottom:8px !important; color:#01335A !important;"
                                )
                                hist_b = make_histogram_plot(df[b].dropna(), b)
                                ui.plotly(hist_b).style("width:100% !important;")
                                ui.label(f"Moyenne: {float(df[b].dropna().mean()):.3f}").style(
                                    "font-size:13px !important; color:#636e72 !important;"
                                )
                                
                                if target_col and pd.api.types.is_numeric_dtype(df[target_col]):
                                    try:
                                        corr_b = df[b].corr(df[target_col])
                                        ui.label(f"Corrélation avec target: {corr_b:.3f}").style(
                                            "font-size:13px !important; color:#01335A !important; font-weight:600 !important;"
                                        )
                                    except:
                                        pass
                        
                        # Actions pour cette paire
                        with ui.row().classes("w-full gap-2 mt-4"):
                            ui.button(
                                f"Garder {a}",
                                on_click=lambda a=a, b=b: keep_feature(a, b, keep=a)
                            ).props("flat").style("color:#01335A !important; text-transform:none !important;")
                            
                            ui.button(
                                f"Garder {b}",
                                on_click=lambda a=a, b=b: keep_feature(a, b, keep=b)
                            ).props("flat").style("color:#01335A !important; text-transform:none !important;")
                            
                            ui.button(
                                "Garder les deux",
                                on_click=lambda a=a, b=b: keep_feature(a, b, keep="both")
                            ).props("flat").style("color:#01335A !important; text-transform:none !important;")
                            
                            ui.button(
                                "Créer feature combinée",
                                on_click=lambda a=a, b=b, r=r: open_pair_modal(a, b, r)
                            ).style(
                                "background:#01335A !important; color:white !important; border-radius:8px !important; "
                                "padding:8px 16px !important; text-transform:none !important;"
                            )
                
                if len(pairs) > 5:
                    ui.label(f"... et {len(pairs) - 5} autres paires (limitées pour performance)").style(
                        "font-size:13px !important; color:#7f8c8d !important; "
                        "font-style:italic !important; text-align:center !important; margin-top:12px !important;"
                    )
            else:
                with ui.card().classes("w-full").style(
                    "background:#e8f5e9 !important; padding:16px !important; "
                    "border-radius:12px !important; border-left:4px solid #4caf50 !important;"
                ):
                    ui.label("✅ Aucune paire corrélée à analyser").style(
                        "color:#1b5e20 !important; font-weight:500 !important;"
                    )

        # NAVIGATION
        with ui.row().classes("w-full max-w-6xl justify-between gap-4 mt-8"):
            ui.button(
                " Précédent",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/split'")
            ).style(
                "background:white !important; color:#01335A !important; font-weight:500 !important; "
                "border-radius:8px !important; height:48px !important; min-width:140px !important; "
                "font-size:14px !important; text-transform:none !important; "
                "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
            )
            
            ui.button(
                "Suivant ",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/missing_values'")
            ).style(
                "background:#01335A !important; color:white !important; font-weight:600 !important; "
                "border-radius:8px !important; height:48px !important; min-width:140px !important; "
                "font-size:14px !important; text-transform:none !important;"
            )





# ----------------- PAGE 3.6 : GESTION DES VALEURS MANQUANTES (VERSION FINALE) -----------------

@ui.page('/supervised/missing_values')
def missing_values_page():
    """
    Page complète pour gestion des valeurs manquantes avec visualisation before/after
    Gestion robuste des features créées dynamiquement + vérification exhaustive
    """
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

    # ---------- CONTEXTE ----------
    df = state.get("raw_df", None)
    split = state.get("split", None)
    columns_exclude = state.get("columns_exclude", {}) or {}
    target_col = state.get("target_column", None)

    if df is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Aucun dataset chargé.").style(
                "font-size:18px !important; color:#c0392b !important; font-weight:600 !important;"
            )
            ui.button(
                " Retour à l'Upload",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/upload'")
            ).style(
                "background:#01335A !important; color:white !important; padding:12px 32px !important; "
                "border-radius:8px !important; margin-top:20px !important;"
            )
        return

    # CORRECTION : Synchroniser le split avec raw_df pour inclure les nouvelles features
    if split and "X_train" in split:
        try:
            new_cols = [c for c in df.columns if c not in split["X_train"].columns and c != target_col]
            
            if new_cols:
                print(f"🔄 Synchronisation des nouvelles features : {new_cols}")
                
                for key in ["X_train", "X_val", "X_test"]:
                    if key in split and isinstance(split[key], pd.DataFrame):
                        indices = split[key].index
                        for col in new_cols:
                            if col in df.columns:
                                try:
                                    split[key][col] = df[col].reindex(indices)
                                except Exception as e:
                                    print(f"⚠️ Impossible de synchroniser {col} pour {key}: {e}")
                
                state["split"] = split
                print(f"✅ {len(new_cols)} nouvelles features synchronisées dans le split")
        
        except Exception as e:
            print(f"❌ Erreur lors de la synchronisation : {e}")

    if "df_original" not in state:
        state["df_original"] = df.copy()

    df_train = None
    if split and "X_train" in split:
        df_train = split["X_train"].copy()
        if target_col and "y_train" in split and target_col not in df_train.columns:
            df_train[target_col] = split["y_train"]
    else:
        df_train = df.copy()

    active_cols = [c for c in df.columns if not columns_exclude.get(c, False)]
    miss_counts = df[active_cols].isna().sum()
    miss_pct = (miss_counts / len(df) * 100).round(2)
    affected = (miss_counts > 0).sum()
    total_missing = int(miss_counts.sum())
    total_pct = round(total_missing / (df.shape[0] * df.shape[1]) * 100, 2) if df.shape[0] * df.shape[1] > 0 else 0

    state.setdefault("missing_strategy", {})
    state.setdefault("fitted_imputers", {})

    # ---------- HELPERS ----------
    def fit_imputers(strategies: dict, df_train_local: pd.DataFrame):
        """Fit les imputers sur df_train"""
        fitted = {}
        
        for col, strat in strategies.items():
            method = strat.get("method", "none")
            params = strat.get("params", {})
            
            if method == "none" or method == "drop":
                continue
            
            if col not in df_train_local.columns:
                print(f"⚠️ Colonne {col} introuvable dans df_train, ignorée")
                continue
            
            try:
                if method in ["mean", "median", "mode", "constant"]:
                    if method == "mode":
                        strategy = "most_frequent"
                    else:
                        strategy = method
                    
                    imputer = SimpleImputer(strategy=strategy, **params)
                    imputer.fit(df_train_local[[col]])
                    fitted[col] = {"imputer": imputer, "method": method, "params": params}
                
                elif method == "knn":
                    num_cols = df_train_local.select_dtypes(include=[np.number]).columns.tolist()
                    if col in num_cols:
                        imputer = KNNImputer(n_neighbors=params.get("n_neighbors", 5))
                        imputer.fit(df_train_local[num_cols])
                        fitted[col] = {"imputer": imputer, "method": method, "params": params, "num_cols": num_cols}
                
                elif method == "iterative":
                    num_cols = df_train_local.select_dtypes(include=[np.number]).columns.tolist()
                    if col in num_cols:
                        imputer = IterativeImputer(max_iter=params.get("max_iter", 10), random_state=42)
                        imputer.fit(df_train_local[num_cols])
                        fitted[col] = {"imputer": imputer, "method": method, "params": params, "num_cols": num_cols}
                
                elif method == "forward_fill":
                    fitted[col] = {"method": "forward_fill", "params": params}
                
                elif method == "backward_fill":
                    fitted[col] = {"method": "backward_fill", "params": params}
            
            except Exception as e:
                print(f"❌ Erreur fit imputer pour {col}: {e}")
        
        return fitted

    def apply_fitted_imputers(df_target: pd.DataFrame, fitted: dict, cols_filter: list = None):
        """Applique les imputers fitted sur un dataframe"""
        df_result = df_target.copy()
        
        for col, info in fitted.items():
            if cols_filter and col not in cols_filter:
                continue
            
            if col not in df_result.columns:
                print(f"⚠️ Colonne {col} introuvable dans df_target, ignorée")
                continue
            
            method = info.get("method")
            imputer = info.get("imputer")
            
            try:
                if method in ["mean", "median", "mode", "constant"]:
                    df_result[[col]] = imputer.transform(df_result[[col]])
                
                elif method in ["knn", "iterative"]:
                    num_cols = info.get("num_cols", [])
                    available_cols = [c for c in num_cols if c in df_result.columns]
                    if available_cols:
                        df_result[available_cols] = imputer.transform(df_result[available_cols])
                
                elif method == "forward_fill":
                    df_result[col] = df_result[col].fillna(method='ffill')
                
                elif method == "backward_fill":
                    df_result[col] = df_result[col].fillna(method='bfill')
            
            except Exception as e:
                print(f"❌ Erreur apply imputer pour {col}: {e}")
        
        return df_result

    def serialize_fitted_imputers(fitted: dict):
        """Sérialise les imputers pour sauvegarde"""
        return {
            col: {
                "method": info.get("method"),
                "params": info.get("params", {}),
            } 
            for col, info in fitted.items()
        }

    def get_rows_with_missing(df_data: pd.DataFrame, cols: list, max_rows: int = 20):
        """Récupère les indices des lignes contenant des valeurs manquantes"""
        valid_cols = [c for c in cols if c in df_data.columns]
        if not valid_cols:
            return []
        
        mask = df_data[valid_cols].isna().any(axis=1)
        indices = df_data[mask].index.tolist()
        return indices[:max_rows]

    def verify_missing_values_after_imputation():
        """
        Vérifie exhaustivement s'il reste des valeurs manquantes après imputation
        Retourne un rapport détaillé
        """
        report = {
            "has_remaining_missing": False,
            "total_remaining": 0,
            "affected_columns": {},
            "datasets": {}
        }
        
        # Vérification raw_df
        raw_df = state.get("raw_df")
        if raw_df is not None:
            missing_in_raw = raw_df[active_cols].isna().sum()
            cols_with_missing = missing_in_raw[missing_in_raw > 0]
            
            if len(cols_with_missing) > 0:
                report["has_remaining_missing"] = True
                report["total_remaining"] += int(missing_in_raw.sum())
                report["datasets"]["raw_df"] = {
                    "total": int(missing_in_raw.sum()),
                    "columns": cols_with_missing.to_dict()
                }
        
        # Vérification splits
        split = state.get("split", {})
        for dataset_name in ["X_train", "X_val", "X_test"]:
            if dataset_name in split and isinstance(split[dataset_name], pd.DataFrame):
                df_split = split[dataset_name]
                split_cols = [c for c in active_cols if c in df_split.columns]
                
                if split_cols:
                    missing_in_split = df_split[split_cols].isna().sum()
                    cols_with_missing = missing_in_split[missing_in_split > 0]
                    
                    if len(cols_with_missing) > 0:
                        report["has_remaining_missing"] = True
                        report["total_remaining"] += int(missing_in_split.sum())
                        report["datasets"][dataset_name] = {
                            "total": int(missing_in_split.sum()),
                            "columns": cols_with_missing.to_dict()
                        }
        
        # Consolidation par colonne
        for dataset_info in report["datasets"].values():
            for col, count in dataset_info["columns"].items():
                if col not in report["affected_columns"]:
                    report["affected_columns"][col] = 0
                report["affected_columns"][col] += int(count)
        
        return report

    def check_unconfigured_or_none_columns():
        """
        Vérifie s'il y a des colonnes non configurées ou avec méthode 'none'
        Retourne un rapport avec les colonnes problématiques
        """
        strategies = state.get("missing_strategy", {})
        unconfigured = []
        none_method = []
        
        # Colonnes avec valeurs manquantes
        cols_with_missing = [c for c in active_cols if df[c].isna().sum() > 0]
        
        for col in cols_with_missing:
            if col not in strategies:
                unconfigured.append(col)
            elif strategies[col].get("method", "none") == "none":
                none_method.append(col)
        
        return {
            "has_issues": len(unconfigured) > 0 or len(none_method) > 0,
            "unconfigured": unconfigured,
            "none_method": none_method,
            "total_issues": len(unconfigured) + len(none_method)
        }

    def show_unconfigured_warning_dialog(report):
        """Affiche un dialog d'avertissement pour colonnes non traitées"""
        
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-3xl").style(
            "padding:0 !important; border-radius:16px !important; "
            "box-shadow:0 10px 40px rgba(0,0,0,0.15) !important;"
        ):
            # Header avec alerte orange
            with ui.column().classes("w-full").style(
                "background:linear-gradient(135deg, #f39c12 0%, #e67e22 100%) !important; "
                "padding:24px 32px !important; border-radius:16px 16px 0 0 !important;"
            ):
                ui.label("⚠️ Colonnes non traitées détectées").style(
                    "font-weight:700 !important; font-size:20px !important; color:white !important;"
                )
                ui.label(
                    f"{report['total_issues']} colonne(s) avec valeurs manquantes non traitées"
                ).style(
                    "color:white !important; font-size:14px !important; "
                    "margin-top:4px !important; opacity:0.9 !important;"
                )
            
            # Contenu
            with ui.column().classes("w-full").style("padding:32px !important;"):
                # Avertissement principal
                with ui.card().classes("w-full mb-6").style(
                    "background:#fff3cd !important; padding:20px !important; "
                    "border-radius:12px !important; border-left:4px solid #f39c12 !important; "
                    "box-shadow:none !important;"
                ):
                    ui.markdown("""
<div style="color:#856404; font-size:14px; line-height:1.8;">

### ⚠️ Impact sur les algorithmes de Machine Learning

Les valeurs manquantes non traitées peuvent **gravement affecter** les performances de vos modèles :

**Problèmes potentiels :**
- ❌ **Erreurs d'exécution** : Certains algorithmes (KNN, SVM, Réseaux de neurones) refusent les NaN
- 📉 **Biais dans les prédictions** : Les algorithmes gèrent mal les données incomplètes
- 🔻 **Perte de données** : Suppression automatique des lignes avec NaN (perte d'information)
- ⚠️ **Résultats imprévisibles** : Comportement inconsistant selon les librairies

**Recommandation :** Configurez une stratégie d'imputation pour **toutes** les colonnes avec valeurs manquantes.

</div>
                    """)
                
                # Liste des colonnes non configurées
                if report["unconfigured"]:
                    ui.separator().classes("my-4")
                    ui.label(f"📋 Colonnes non configurées ({len(report['unconfigured'])}) :").style(
                        "font-weight:700 !important; font-size:16px !important; "
                        "color:#2c3e50 !important; margin-bottom:12px !important;"
                    )
                    
                    with ui.column().classes("w-full gap-2"):
                        for col in report["unconfigured"][:10]:
                            n_missing = int(df[col].isna().sum())
                            pct = round(n_missing / len(df) * 100, 1)
                            
                            with ui.card().classes("w-full").style(
                                "background:#f8f9fa !important; padding:12px 16px !important; "
                                "border-radius:8px !important; box-shadow:none !important; "
                                "border-left:3px solid #e74c3c !important;"
                            ):
                                with ui.row().classes("items-center justify-between w-full"):
                                    ui.label(f"• {col}").style(
                                        "font-weight:600 !important; color:#2c3e50 !important;"
                                    )
                                    ui.label(f"{n_missing} NaN ({pct}%)").style(
                                        "font-size:12px !important; color:#e74c3c !important; "
                                        "font-family:monospace !important;"
                                    )
                        
                        if len(report["unconfigured"]) > 10:
                            ui.label(f"... et {len(report['unconfigured']) - 10} autre(s)").style(
                                "font-size:12px !important; color:#7f8c8d !important; "
                                "font-style:italic !important; margin-top:8px !important;"
                            )
                
                # Liste des colonnes avec méthode "none"
                if report["none_method"]:
                    ui.separator().classes("my-4")
                    ui.label(f"🚫 Colonnes avec méthode 'none' ({len(report['none_method'])}) :").style(
                        "font-weight:700 !important; font-size:16px !important; "
                        "color:#2c3e50 !important; margin-bottom:12px !important;"
                    )
                    
                    with ui.column().classes("w-full gap-2"):
                        for col in report["none_method"][:10]:
                            n_missing = int(df[col].isna().sum())
                            pct = round(n_missing / len(df) * 100, 1)
                            
                            with ui.card().classes("w-full").style(
                                "background:#f8f9fa !important; padding:12px 16px !important; "
                                "border-radius:8px !important; box-shadow:none !important; "
                                "border-left:3px solid #f39c12 !important;"
                            ):
                                with ui.row().classes("items-center justify-between w-full"):
                                    ui.label(f"• {col}").style(
                                        "font-weight:600 !important; color:#2c3e50 !important;"
                                    )
                                    ui.label(f"{n_missing} NaN ({pct}%)").style(
                                        "font-size:12px !important; color:#f39c12 !important; "
                                        "font-family:monospace !important;"
                                    )
                        
                        if len(report["none_method"]) > 10:
                            ui.label(f"... et {len(report['none_method']) - 10} autre(s)").style(
                                "font-size:12px !important; color:#7f8c8d !important; "
                                "font-style:italic !important; margin-top:8px !important;"
                            )
                
                # Recommandations
                ui.separator().classes("my-4")
                with ui.card().classes("w-full").style(
                    "background:linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%) !important; "
                    "padding:20px !important; border-radius:12px !important; "
                    "border-left:4px solid #2196f3 !important; box-shadow:none !important;"
                ):
                    ui.label("💡 Que faire ?").style(
                        "font-weight:700 !important; font-size:16px !important; "
                        "color:#01335A !important; margin-bottom:12px !important;"
                    )
                    
                    ui.markdown("""
<div style="color:#01335A; font-size:13px; line-height:1.8;">

**Option 1 : Configuration manuelle**
- Retournez à la configuration et choisissez une méthode pour chaque colonne

**Option 2 : Stratégie globale**
- Utilisez "Balanced" pour appliquer automatiquement median/mode
- Utilisez "Aggressive" pour appliquer KNN sur les colonnes numériques

**Option 3 : Exclusion**
- Si la colonne est peu importante, excluez-la de l'analyse

</div>
                    """)
                
                # Boutons d'action
                with ui.row().classes("w-full justify-end gap-3 mt-6"):
                    ui.button(
                        " Retour à la configuration",
                        on_click=dialog.close
                    ).style(
                        "background:#01335A !important; color:white !important; "
                        "border-radius:8px !important; padding:10px 24px !important; "
                        "text-transform:none !important; font-weight:600 !important;"
                    ).props('icon="arrow_back"')
                    
                    def continue_anyway():
                        dialog.close()
                        ui.notify(
                            "⚠️ Continuer avec des valeurs manquantes peut affecter vos modèles",
                            color="warning",
                            timeout=5000
                        )
                    
                    ui.button(
                        "Continuer quand même ",
                        on_click=continue_anyway
                    ).style(
                        "background:#f39c12 !important; color:white !important; "
                        "border-radius:8px !important; padding:10px 24px !important; "
                        "text-transform:none !important; font-weight:600 !important;"
                    ).props('icon-right="warning"')
        
        dialog.open()

    def show_verification_dialog(verification_report):
        """Affiche un dialog détaillé avec les valeurs manquantes restantes"""
        
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-4xl").style(
            "padding:0 !important; border-radius:16px !important; "
            "box-shadow:0 10px 40px rgba(0,0,0,0.15) !important;"
        ):
            # Header avec alerte
            with ui.column().classes("w-full").style(
                "background:linear-gradient(135deg, #e74c3c 0%, #c0392b 100%) !important; "
                "padding:24px 32px !important; border-radius:16px 16px 0 0 !important;"
            ):
                ui.label("⚠️ Valeurs manquantes détectées").style(
                    "font-weight:700 !important; font-size:20px !important; color:white !important;"
                )
                ui.label(
                    f"{verification_report['total_remaining']} valeurs manquantes restantes"
                ).style(
                    "color:white !important; font-size:14px !important; "
                    "margin-top:4px !important; opacity:0.9 !important;"
                )
            
            # Contenu détaillé
            with ui.column().classes("w-full").style("padding:32px !important;"):
                # Par dataset
                ui.label("📊 Répartition par dataset :").style(
                    "font-weight:600 !important; font-size:16px !important; "
                    "color:#2c3e50 !important; margin-bottom:12px !important;"
                )
                
                for dataset_name, info in verification_report["datasets"].items():
                    with ui.expansion(dataset_name, icon="dataset").classes("w-full mb-2"):
                        ui.label(f"Total : {info['total']} valeurs manquantes").style(
                            "font-weight:600 !important; margin-bottom:8px !important;"
                        )
                        
                        # Tableau des colonnes affectées
                        rows = []
                        for col, count in info["columns"].items():
                            current_strat = state.get("missing_strategy", {}).get(col, {})
                            method = current_strat.get("method", "none")
                            
                            rows.append({
                                "Colonne": col,
                                "Valeurs manquantes": int(count),
                                "Stratégie actuelle": method
                            })
                        
                        ui.table(
                            columns=[
                                {"name": "Colonne", "label": "Colonne", "field": "Colonne", "align": "left"},
                                {"name": "Valeurs manquantes", "label": "Manquantes", 
                                 "field": "Valeurs manquantes", "align": "center"},
                                {"name": "Stratégie actuelle", "label": "Stratégie", 
                                 "field": "Stratégie actuelle", "align": "center"}
                            ],
                            rows=rows,
                            row_key="Colonne"
                        ).props("flat dense").style("width:100% !important;")
                
                # Recommandations
                ui.separator().classes("my-4")
                ui.label("💡 Recommandations :").style(
                    "font-weight:600 !important; font-size:16px !important; "
                    "color:#2c3e50 !important; margin-bottom:12px !important;"
                )
                
                with ui.column().classes("w-full gap-2"):
                    for col in verification_report["affected_columns"].keys():
                        current_strat = state.get("missing_strategy", {}).get(col, {})
                        method = current_strat.get("method", "none")
                        
                        if method == "none":
                            ui.label(f"• {col} : Aucune stratégie configurée  Configurez une méthode").style(
                                "color:#e74c3c !important; font-size:13px !important;"
                            )
                        elif method == "forward_fill" or method == "backward_fill":
                            ui.label(
                                f"• {col} : Fill méthode peut laisser des NaN en début/fin  "
                                "Essayez median/mode/constant"
                            ).style(
                                "color:#f39c12 !important; font-size:13px !important;"
                            )
                        else:
                            ui.label(
                                f"• {col} : Vérifiez les paramètres de {method}"
                            ).style(
                                "color:#3498db !important; font-size:13px !important;"
                            )
                
                # Actions
                with ui.row().classes("w-full justify-end gap-3 mt-6"):
                    ui.button(
                        "Retour à la configuration",
                        on_click=dialog.close
                    ).style(
                        "background:#01335A !important; color:white !important; "
                        "border-radius:8px !important; padding:10px 24px !important; "
                        "text-transform:none !important; font-weight:600 !important;"
                    )
        
        dialog.open()

    def apply_and_propagate(navigate_after=False):
        """Applique l'imputation sur raw_df ET tous les splits avec vérification"""
        try:
            print("🚀 Début de apply_and_propagate")  # ✅ DEBUG
            
            strategies = state.get("missing_strategy", {})
            if not strategies:
                ui.notify("⚠️ Aucune stratégie configurée", color="warning")
                return False
            
            print(f"📋 Stratégies à appliquer : {list(strategies.keys())}")  # ✅ DEBUG
            
            # ✅ VÉRIFICATION DES COLONNES NON CONFIGURÉES OU "NONE"
            unconfigured_report = check_unconfigured_or_none_columns()
            print(f"🔍 Rapport non configurées : {unconfigured_report}")  # ✅ DEBUG
            
            if unconfigured_report["has_issues"]:
                print("⚠️ Des colonnes ont des problèmes, affichage du dialog")  # ✅ DEBUG
                show_unconfigured_warning_dialog(unconfigured_report)
                return False
            
            split = state.get("split", {})
            
            if split and "X_train" in split:
                df_train_for_fit = split["X_train"].copy()
                if target_col and "y_train" in split and target_col not in df_train_for_fit.columns:
                    df_train_for_fit[target_col] = split["y_train"]
            else:
                df_train_for_fit = state["raw_df"].copy()
            
            print("🔧 Début du fit des imputers")  # ✅ DEBUG
            
            # Fit des imputers
            fitted = fit_imputers(strategies, df_train_for_fit)
            state["fitted_imputers"] = serialize_fitted_imputers(fitted)
            
            print(f"✅ {len(fitted)} imputers fitted")  # ✅ DEBUG
            
            # Application sur raw_df
            print("📝 Application sur raw_df")  # ✅ DEBUG
            state["raw_df"] = apply_fitted_imputers(state["raw_df"], fitted, active_cols)
            
            # Application sur splits
            if split:
                print("📝 Application sur les splits")  # ✅ DEBUG
                for key in ["X_train", "X_val", "X_test"]:
                    if key in split and isinstance(split[key], pd.DataFrame):
                        print(f"  - Application sur {key}")  # ✅ DEBUG
                        split[key] = apply_fitted_imputers(split[key], fitted, active_cols)
                
                state["split"] = split
            
            print("🔍 Vérification post-imputation")  # ✅ DEBUG
            
            # ✅ VÉRIFICATION POST-IMPUTATION
            verification_report = verify_missing_values_after_imputation()
            
            print(f"📊 Rapport vérification : {verification_report['has_remaining_missing']}")  # ✅ DEBUG
            
            if verification_report["has_remaining_missing"]:
                ui.notify(
                    f"⚠️ Attention : {verification_report['total_remaining']} valeurs manquantes restantes!",
                    color="warning",
                    timeout=5000
                )
                show_verification_dialog(verification_report)
                return False
            else:
                ui.notify("✅ Imputation réussie : Toutes les valeurs manquantes ont été traitées!", color="positive")
            
            print(f"🎯 Navigate after : {navigate_after}")  # ✅ DEBUG
            
            if navigate_after:
                print("🚀 Navigation vers /supervised/encoding")  # ✅ DEBUG
                ui.run_javascript("setTimeout(() => window.location.href='/supervised/encoding', 1500);")
            else:
                print("🔄 Rechargement de la page")  # ✅ DEBUG
                ui.run_javascript("setTimeout(() => window.location.reload(), 1500);")
            
            return True
        
        except Exception as e:
            ui.notify(f"❌ Erreur lors de l'application : {str(e)}", color="negative")
            print(f"❌ Détail erreur: {e}")
            import traceback
            traceback.print_exc()
            return False

    def open_feature_modal(col_name):
        """Ouvre un dialog pour configurer la stratégie d'imputation d'une colonne"""
        current_strategy = state.get("missing_strategy", {}).get(col_name, {})
        current_method = current_strategy.get("method", "none")
        
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-2xl").style(
            "padding:0 !important; border-radius:16px !important; "
            "box-shadow:0 10px 40px rgba(0,0,0,0.15) !important;"
        ):
            # Header avec gradient
            with ui.column().classes("w-full").style(
                "background:linear-gradient(135deg, #01335A 0%, #09538C 100%) !important; "
                "padding:24px 32px !important; border-radius:16px 16px 0 0 !important;"
            ):
                ui.label(f"Configuration : {col_name}").style(
                    "font-weight:700 !important; font-size:20px !important; color:white !important;"
                )
                
                dtype = df[col_name].dtype
                n_missing = int(df[col_name].isna().sum())
                pct = round(n_missing / len(df) * 100, 2)
                
                ui.label(f"Type: {dtype} | Missing: {n_missing} ({pct}%)").style(
                    "color:white !important; font-size:14px !important; margin-top:4px !important; opacity:0.9 !important;"
                )
            
            # Contenu
            with ui.column().classes("w-full").style("padding:32px !important;"):
                # Options d'imputation
                if pd.api.types.is_numeric_dtype(df[col_name]):
                    options = ["none", "drop", "mean", "median", "mode", "constant", "knn", "iterative"]
                else:
                    options = ["none", "drop", "mode", "constant", "forward_fill", "backward_fill"]
                
                method_select = ui.select(
                    options=options, 
                    value=current_method, 
                    label="Méthode"
                ).props("outlined").classes("w-full mb-4")
                
                # Paramètres additionnels
                params_container = ui.column().classes("w-full")
                
                constant_input = None
                knn_input = None
                iter_input = None
                
                def update_params():
                    nonlocal constant_input, knn_input, iter_input
                    params_container.clear()
                    with params_container:
                        if method_select.value == "constant":
                            constant_input = ui.input(
                                label="Valeur constante", 
                                value=str(current_strategy.get("params", {}).get("fill_value", "0"))
                            ).props("outlined")
                        elif method_select.value == "knn":
                            knn_input = ui.number(
                                label="Nombre de voisins", 
                                value=current_strategy.get("params", {}).get("n_neighbors", 5), 
                                min=1, 
                                max=20
                            ).props("outlined")
                        elif method_select.value == "iterative":
                            iter_input = ui.number(
                                label="Max iterations", 
                                value=current_strategy.get("params", {}).get("max_iter", 10), 
                                min=1, 
                                max=100
                            ).props("outlined")
                
                method_select.on_value_change(lambda: update_params())
                update_params()
                
                # Boutons d'action
                with ui.row().classes("w-full justify-end gap-3 mt-6"):
                    ui.button("Annuler", on_click=dialog.close).props("flat").style(
                        "color:#7f8c8d !important; text-transform:none !important;"
                    )
                    
                    def save_strategy():
                        method = method_select.value
                        params = {}
                        
                        if method == "constant" and constant_input:
                            params["fill_value"] = constant_input.value
                        elif method == "knn" and knn_input:
                            params["n_neighbors"] = int(knn_input.value)
                        elif method == "iterative" and iter_input:
                            params["max_iter"] = int(iter_input.value)
                        
                        state.setdefault("missing_strategy", {})[col_name] = {
                            "method": method,
                            "params": params
                        }
                        
                        ui.notify(f"✅ Stratégie sauvegardée pour {col_name}", color="positive")
                        dialog.close()
                        # ✅ CORRECTION : Recharger la page pour mettre à jour le tableau
                        ui.run_javascript("setTimeout(() => window.location.reload(), 500);")
                    
                    ui.button("Sauvegarder", on_click=save_strategy).style(
                        "background:#01335A !important; color:white !important; border-radius:8px !important; "
                        "padding:10px 24px !important; text-transform:none !important; font-weight:600 !important;"
                    )
        
        dialog.open()

    def preview_imputation():
        """Prévisualise l'imputation avec tableaux BEFORE/AFTER des lignes affectées"""
        strategies = state.get("missing_strategy", {})
        
        if not strategies:
            ui.notify("⚠️ Aucune stratégie configurée", color="warning")
            return
        
        try:
            fitted = fit_imputers(strategies, df_train)
            
            cols_with_strategy = list(strategies.keys())
            cols_to_check = [c for c in cols_with_strategy if c in df_train.columns]
            
            if not cols_to_check:
                ui.notify("⚠️ Aucune colonne valide à traiter", color="warning")
                return
            
            missing_indices = get_rows_with_missing(df_train, cols_to_check, max_rows=15)
            
            if not missing_indices:
                ui.notify("ℹ️ Aucune ligne avec valeurs manquantes dans les colonnes sélectionnées", color="info")
                return
            
            df_before = df_train.loc[missing_indices, cols_to_check].copy()
            
            df_preview = df_train.copy()
            df_preview = apply_fitted_imputers(df_preview, fitted, active_cols)
            
            df_after = df_preview.loc[missing_indices, cols_to_check].copy()
            
            before_missing = df_train[active_cols].isna().sum().sum()
            after_missing = df_preview[active_cols].isna().sum().sum()
            
            preview_info.set_text(
                f"📊 Preview généré : {before_missing} valeurs manquantes  {after_missing} après imputation | {len(missing_indices)} lignes affichées"
            )
            
            # Affichage des tableaux BEFORE/AFTER
            table_before_container.clear()
            table_after_container.clear()
            
            with table_before_container:
                ui.label("AVANT Imputation").style(
                    "font-weight:700 !important; font-size:16px !important; color:#01335A !important; "
                    "margin-bottom:12px !important;"
                )
                
                rows_before = []
                for idx in missing_indices:
                    row_dict = {"Index": idx}
                    for col in cols_to_check:
                        val = df_before.loc[idx, col]
                        if pd.isna(val):
                            row_dict[col] = "❌ NaN"
                        else:
                            row_dict[col] = str(val)[:30]
                    rows_before.append(row_dict)
                
                columns_before = [{"name": "Index", "label": "Index", "field": "Index", "align": "center"}]
                columns_before.extend([{"name": c, "label": c, "field": c, "align": "left"} for c in cols_to_check])
                
                ui.table(
                    columns=columns_before,
                    rows=rows_before,
                    row_key="Index"
                ).props("flat bordered dense").style(
                    "width:100% !important; font-size:12px !important;"
                )
            
            with table_after_container:
                ui.label("APRÈS Imputation").style(
                    "font-weight:700 !important; font-size:16px !important; color:#2196f3 !important; "
                    "margin-bottom:12px !important;"
                )
                
                rows_after = []
                for idx in missing_indices:
                    row_dict = {"Index": idx}
                    for col in cols_to_check:
                        val = df_after.loc[idx, col]
                        if pd.isna(val):
                            row_dict[col] = "⚠️ Still NaN"
                        else:
                            was_nan = pd.isna(df_before.loc[idx, col])
                            display_val = str(val)[:30]
                            if was_nan:
                                row_dict[col] = f"✓ {display_val}"
                            else:
                                row_dict[col] = display_val
                    rows_after.append(row_dict)
                
                columns_after = [{"name": "Index", "label": "Index", "field": "Index", "align": "center"}]
                columns_after.extend([{"name": c, "label": c, "field": c, "align": "left"} for c in cols_to_check])
                
                ui.table(
                    columns=columns_after,
                    rows=rows_after,
                    row_key="Index"
                ).props("flat bordered dense").style(
                    "width:100% !important; font-size:12px !important;"
                )
            
            # Histogrammes comparatifs
            num_cols_with_missing = [c for c in cols_to_check if pd.api.types.is_numeric_dtype(df[c]) and df[c].isna().sum() > 0]
            
            if num_cols_with_missing:
                sample_col = num_cols_with_missing[0]
                
                fig_before = go.Figure()
                fig_before.add_trace(go.Histogram(
                    x=df_train[sample_col].dropna(), 
                    name="Before", 
                    marker_color="#64b5f6",
                    opacity=0.7
                ))
                fig_before.update_layout(
                    title=f"{sample_col} - Distribution AVANT",
                    height=280,
                    showlegend=True,
                    paper_bgcolor='white',
                    plot_bgcolor='#f8f9fa',
                    font=dict(family="Inter, sans-serif", color="#2c3e50")
                )
                
                fig_after = go.Figure()
                fig_after.add_trace(go.Histogram(
                    x=df_preview[sample_col].dropna(), 
                    name="After", 
                    marker_color="#01335A",
                    opacity=0.7
                ))
                fig_after.update_layout(
                    title=f"{sample_col} - Distribution APRÈS",
                    height=280,
                    showlegend=True,
                    paper_bgcolor='white',
                    plot_bgcolor='#f8f9fa',
                    font=dict(family="Inter, sans-serif", color="#2c3e50")
                )
                
                chart_before.update_figure(fig_before)
                chart_before.style("display:block !important; width:100% !important;")
                chart_after.update_figure(fig_after)
                chart_after.style("display:block !important; width:100% !important;")
            
        except Exception as e:
            ui.notify(f"❌ Erreur lors du preview : {str(e)}", color="negative")
            print(f"Détail erreur preview: {e}")

    def confirm_and_apply():
        """Confirme et applique l'imputation sur tous les datasets"""
        strategies = state.get("missing_strategy", {})
        
        if not strategies:
            ui.notify("⚠️ Aucune stratégie configurée", color="warning")
            return
        
        # ✅ DEBUG : Afficher les stratégies configurées
        print(f"🔍 Stratégies configurées : {strategies}")
        
        with ui.dialog() as dialog, ui.card().style(
            "padding:0 !important; border-radius:16px !important; "
            "box-shadow:0 10px 40px rgba(0,0,0,0.15) !important;"
        ):
            # Header
            with ui.column().classes("w-full").style(
                "background:linear-gradient(135deg, #01335A 0%, #09538C 100%) !important; "
                "padding:24px 32px !important; border-radius:16px 16px 0 0 !important;"
            ):
                ui.label("⚠️ Confirmation").style(
                    "font-weight:700 !important; font-size:20px !important; color:white !important;"
                )
            
            # Contenu
            with ui.column().classes("w-full").style("padding:32px !important;"):
                ui.label("Voulez-vous appliquer l'imputation sur raw_df et tous les datasets (train/val/test) ?").style(
                    "margin-bottom:12px !important; color:#2c3e50 !important; font-size:14px !important;"
                )
                ui.label("⚠️ Cette action est irréversible (sauf si vous rechargez le fichier)").style(
                    "color:#e74c3c !important; font-size:13px !important;"
                )
                
                with ui.row().classes("w-full justify-end gap-3 mt-6"):
                    ui.button("Annuler", on_click=dialog.close).props("flat").style(
                        "color:#7f8c8d !important; text-transform:none !important;"
                    )
                    
                    def confirm_and_next():
                        print("✅ Bouton 'Confirmer' cliqué")  # ✅ DEBUG
                        dialog.close()
                        result = apply_and_propagate(navigate_after=True)
                        print(f"📊 Résultat apply_and_propagate: {result}")  # ✅ DEBUG
                    
                    ui.button("Confirmer", on_click=confirm_and_next).style(
                        "background:#01335A !important; color:white !important; border-radius:8px !important; "
                        "padding:10px 24px !important; text-transform:none !important; font-weight:600 !important;"
                    )
        
        dialog.open()

    def apply_global_strategy():
        """Applique une stratégie globale de gestion des valeurs manquantes"""
        val = strategy_radio.value
        state["missing_strategy_global"] = val

        if val.startswith("Conservative"):
            excluded_count = 0
            for col in active_cols:
                if miss_pct.get(col, 0.0) > 20:
                    state.setdefault("columns_exclude", {})[col] = True
                    excluded_count += 1
            ui.notify(f"✅ Conservative : {excluded_count} colonnes >20% marquées pour exclusion", color="positive")

        elif val.startswith("Balanced"):
            strategy_count = 0
            for col in active_cols:
                if df[col].isna().sum() == 0:
                    continue
                method = "median" if pd.api.types.is_numeric_dtype(df[col]) else "mode"
                state.setdefault("missing_strategy", {})[col] = {"method": method, "params": {}}
                strategy_count += 1
            ui.notify(f"✅ Balanced : {strategy_count} colonnes configurées (Median/Mode)", color="positive")

        elif val.startswith("Aggressive"):
            knn_count = 0
            for col in active_cols:
                if pd.api.types.is_numeric_dtype(df[col]) and df[col].isna().sum() > 0:
                    state.setdefault("missing_strategy", {})[col] = {"method": "knn", "params": {"n_neighbors": 5}}
                    knn_count += 1
            ui.notify(f"✅ Aggressive : KNN appliqué à {knn_count} colonnes numériques", color="positive")

        elif val.startswith("Custom"):
            ui.notify("ℹ️ Choix Custom : configure colonne par colonne via le tableau", color="info")
        
        # ✅ CORRECTION : Recharger la page après application de la stratégie globale
        ui.run_javascript("setTimeout(() => window.location.reload(), 800);")

    # ---------- UI ----------
    with ui.column().classes("w-full items-center").style(
        "background:#f0f2f5 !important; min-height:100vh !important; padding:48px 24px !important; "
        "font-family:'Inter', sans-serif !important;"
    ):
        # HEADER
        ui.label("Gestion des Valeurs Manquantes").style(
            "font-weight:700 !important; font-size:36px !important; color:#01335A !important; "
            "margin-bottom:8px !important; text-align:center !important; letter-spacing:-0.5px !important;"
        )
        ui.label("Imputation intelligente et visualisation avant/après").style(
            "font-size:16px !important; color:#636e72 !important; margin-bottom:48px !important; "
            "text-align:center !important; font-weight:400 !important;"
        )

        # --- A - Overview COMPACTE ---
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:linear-gradient(135deg, #01335A 0%, #09538C 100%) !important; "
            "border-radius:16px !important; padding:24px 32px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            with ui.row().classes("w-full items-center justify-between"):
                # Total missing
                with ui.row().classes("items-center gap-3"):
                    ui.icon("warning", size="md").classes("text-white")
                    with ui.column().classes("gap-0"):
                        ui.label(f"{total_missing:,}").style(
                            "font-weight:700 !important; font-size:28px !important; color:white !important; "
                            "line-height:1 !important;"
                        )
                        ui.label(f"valeurs manquantes ({total_pct}%)").style(
                            "font-size:12px !important; color:rgba(255,255,255,0.8) !important; margin-top:4px !important;"
                        )
                
                # Features affectées
                with ui.row().classes("items-center gap-3"):
                    ui.icon("table_chart", size="md").classes("text-white")
                    with ui.column().classes("gap-0"):
                        ui.label(f"{affected}/{len(active_cols)}").style(
                            "font-weight:700 !important; font-size:28px !important; color:white !important; "
                            "line-height:1 !important;"
                        )
                        ui.label("features affectées").style(
                            "font-size:12px !important; color:rgba(255,255,255,0.8) !important; margin-top:4px !important;"
                        )
                
                # Lignes
                with ui.row().classes("items-center gap-3"):
                    ui.icon("dataset", size="md").classes("text-white")
                    with ui.column().classes("gap-0"):
                        ui.label(f"{len(df):,}").style(
                            "font-weight:700 !important; font-size:28px !important; color:white !important; "
                            "line-height:1 !important;"
                        )
                        ui.label("observations").style(
                            "font-size:12px !important; color:rgba(255,255,255,0.8) !important; margin-top:4px !important;"
                        )

        # --- B - Badge de statut de configuration ---
        configured_cols = len(state.get("missing_strategy", {}))
        
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white !important; border-radius:16px !important; padding:24px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            with ui.row().classes("w-full items-center justify-between"):
                with ui.column():
                    ui.label("📋 Statut de configuration").style(
                        "font-weight:600 !important; font-size:16px !important; "
                        "color:#2c3e50 !important; margin-bottom:8px !important;"
                    )
                    ui.label(f"{configured_cols}/{affected} colonnes avec valeurs manquantes configurées").style(
                        "font-size:14px !important; color:#636e72 !important;"
                    )
                
                # Badge de statut
                if configured_cols == 0:
                    status_color = "#e74c3c"
                    status_icon = "cancel"
                    status_text = "Non configuré"
                elif configured_cols < affected:
                    status_color = "#f39c12"
                    status_icon = "warning"
                    status_text = "Partiellement configuré"
                else:
                    status_color = "#27ae60"
                    status_icon = "check_circle"
                    status_text = "Prêt"
                
                with ui.row().classes("items-center gap-2").style(
                    f"background:{status_color}15 !important; padding:8px 16px !important; "
                    "border-radius:8px !important;"
                ):
                    ui.icon(status_icon, size="sm").style(f"color:{status_color} !important;")
                    ui.label(status_text).style(
                        f"color:{status_color} !important; font-weight:600 !important; "
                        "font-size:14px !important;"
                    )

        # --- C - Table of columns ---
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white !important; border-radius:16px !important; padding:32px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            ui.label("📋 Détail par colonne").style(
                "font-weight:700 !important; font-size:22px !important; color:#01335A !important; "
                "margin-bottom:16px !important;"
            )
            
            rows = []
            for col in active_cols:
                n_missing = int(miss_counts.get(col, 0))
                pct = float(miss_pct.get(col, 0.0))
                dtype = "Numérique" if pd.api.types.is_numeric_dtype(df[col]) else "Catégoriel"
                
                # Indicateurs en nuances de bleu
                if pct > 20:
                    tag = "🔴"
                elif pct >= 5:
                    tag = "🟡"
                elif pct > 0:
                    tag = "🔵"
                else:
                    tag = ""
                
                current_strat = state.get("missing_strategy", {}).get(col, {})
                method = current_strat.get("method", "none")
                
                rows.append({
                    "Feature": col,
                    "Type": dtype,
                    "Missing": n_missing,
                    "% Missing": f"{pct}%",
                    "Niveau": tag,
                    "Stratégie": method
                })
            
            # ✅ CORRECTION : Créer le tableau PUIS attacher l'événement
            table = ui.table(
                columns=[
                    {"name": "Feature", "label": "Feature", "field": "Feature", "align": "left"},
                    {"name": "Type", "label": "Type", "field": "Type", "align": "left"},
                    {"name": "Missing", "label": "Missing", "field": "Missing", "align": "center"},
                    {"name": "% Missing", "label": "% Missing", "field": "% Missing", "align": "center"},
                    {"name": "Niveau", "label": "Niveau", "field": "Niveau", "align": "center"},
                    {"name": "Stratégie", "label": "Stratégie", "field": "Stratégie", "align": "center"}
                ],
                rows=rows,
                row_key="Feature"
            ).props("flat bordered").style(
                "width:100% !important; cursor:pointer !important;"
            )
            
            # ✅ Gestionnaire de clic - attacher APRÈS création
            def handle_row_click(e):
                """Gestionnaire de clic sur une ligne du tableau"""
                try:
                    print(f"🖱️ Event reçu: {e}")
                    
                    # NiceGUI passe les données dans e.args
                    if hasattr(e, 'args'):
                        print(f"📦 e.args: {e.args}")
                        
                        # Essayer différentes structures possibles
                        if len(e.args) >= 2 and isinstance(e.args[1], dict):
                            row_data = e.args[1]
                        elif len(e.args) >= 1 and isinstance(e.args[0], dict):
                            row_data = e.args[0]
                        else:
                            row_data = e.args
                        
                        print(f"📋 row_data extrait: {row_data}")
                        
                        # Extraire le nom de la feature
                        if isinstance(row_data, dict) and "Feature" in row_data:
                            feature_name = row_data["Feature"]
                            print(f"✅ Feature cliquée: {feature_name}")
                            open_feature_modal(feature_name)
                        else:
                            print(f"⚠️ Structure inattendue - row_data: {row_data}")
                    else:
                        print(f"⚠️ Pas d'attribut args dans l'événement")
                        
                except Exception as err:
                    print(f"❌ Erreur handle_row_click: {err}")
                    import traceback
                    traceback.print_exc()
            
            # ✅ Attacher l'événement avec .on()
            table.on('row-click', handle_row_click)
            
            ui.label("💡 Cliquez sur une ligne pour configurer la stratégie d'imputation").style(
                "font-size:13px !important; color:#636e72 !important; margin-top:12px !important; "
                "font-style:italic !important;"
            )

        # --- D - Global strategy ---
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white !important; border-radius:16px !important; padding:32px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            ui.label("⚡ Stratégie Globale").style(
                "font-weight:700 !important; font-size:22px !important; color:#01335A !important; "
                "margin-bottom:16px !important;"
            )
            ui.label("Applique une stratégie prédéfinie à toutes les colonnes concernées").style(
                "font-size:14px !important; color:#636e72 !important; margin-bottom:20px !important;"
            )
            
            strategy_radio = ui.radio(
                options=[
                    "Conservative (drop cols>20% or rows>50%)",
                    "Balanced (median numeric, mode cat)",
                    "Aggressive (KNN numeric)",
                    "Custom (configure per feature)"
                ],
                value=state.get("missing_strategy_global", "Balanced (median numeric, mode cat)")
            ).classes("mb-4")
            
            ui.button(
                "Appliquer stratégie globale",
                on_click=apply_global_strategy
            ).style(
                "background:#01335A !important; color:white !important; border-radius:8px !important; "
                "padding:10px 20px !important; text-transform:none !important; font-weight:500 !important;"
            )

        # --- E - Preview & Apply ---
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white !important; border-radius:16px !important; padding:32px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            ui.label("🔍 Preview & Application").style(
                "font-weight:700 !important; font-size:22px !important; color:#01335A !important; "
                "margin-bottom:16px !important;"
            )
            
            preview_info = ui.label("Cliquez sur 'Preview' pour visualiser l'impact de l'imputation").style(
                "font-size:14px !important; color:#636e72 !important; margin-bottom:20px !important;"
            )
            
            # Conteneurs pour les tableaux BEFORE/AFTER
            with ui.row().classes("w-full gap-4 mb-4"):
                with ui.column().classes("flex-1"):
                    table_before_container = ui.column().classes("w-full")
                with ui.column().classes("flex-1"):
                    table_after_container = ui.column().classes("w-full")
            
            # Histogrammes comparatifs
            with ui.row().classes("w-full gap-4 mb-6"):
                with ui.column().classes("flex-1"):
                    chart_before = ui.plotly({}).style("width:100% !important; display:none !important;")
                with ui.column().classes("flex-1"):
                    chart_after = ui.plotly({}).style("width:100% !important; display:none !important;")
            
            with ui.row().classes("w-full gap-3"):
                ui.button(
                    "🔍 Preview (train)",
                    on_click=preview_imputation
                ).style(
                    "background:#2196f3 !important; color:white !important; border-radius:8px !important; "
                    "padding:10px 20px !important; text-transform:none !important; font-weight:500 !important;"
                )
                
                ui.button(
                    "✓ Appliquer & Continuer",
                    on_click=confirm_and_apply
                ).style(
                    "background:#01335A !important; color:white !important; border-radius:8px !important; "
                    "padding:10px 20px !important; text-transform:none !important; font-weight:600 !important;"
                )

        # --- Navigation ---
        with ui.row().classes("w-full max-w-6xl justify-between gap-4 mt-8"):
            ui.button(
                " Précédent",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/multivariate_analysis'")
            ).style(
                "background:white !important; color:#01335A !important; font-weight:500 !important; "
                "border:1px solid #e1e8ed !important; border-radius:8px !important; height:48px !important; "
                "min-width:140px !important; font-size:14px !important; text-transform:none !important; "
                "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
            )
            
            ui.button(
                "Suivant ",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/encoding'")
            ).style(
                "background:#01335A !important; color:white !important; font-weight:600 !important; "
                "border-radius:8px !important; height:48px !important; min-width:140px !important; "
                "font-size:14px !important; text-transform:none !important;"
            )




# ----------------- PAGE 3.7 : ENCODAGE DES FEATURES CATÉGORIELLES -----------------


@ui.page('/supervised/encoding')
def encoding_page():
    """
    Page complète pour l'encodage des features catégorielles
    Design moderne avec visualisations et recommandations intelligentes
    """
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from sklearn.preprocessing import LabelEncoder
    
    # ---------- CONTEXTE ----------
    df = state.get("raw_df", None)
    split = state.get("split", None)
    columns_exclude = state.get("columns_exclude", {}) or {}
    target_col = state.get("target_column", None)
    
    if df is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Aucun dataset chargé.").style(
                "font-size:18px !important; color:#c0392b !important; font-weight:600 !important;"
            )
            ui.button(
                " Retour à l'Upload",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/upload'")
            ).style(
                "background:#01335A !important; color:white !important; padding:12px 32px !important; "
                "border-radius:8px !important; margin-top:20px !important;"
            )
        return
    
    # CORRECTION : Synchroniser le split avec raw_df
    if split and "X_train" in split:
        try:
            new_cols = [c for c in df.columns if c not in split["X_train"].columns and c != target_col]
            
            if new_cols:
                print(f"🔄 Synchronisation des nouvelles features : {new_cols}")
                
                for key in ["X_train", "X_val", "X_test"]:
                    if key in split and isinstance(split[key], pd.DataFrame):
                        indices = split[key].index
                        for col in new_cols:
                            if col in df.columns:
                                split[key][col] = df.loc[indices, col]
                
                state["split"] = split
                print(f"✅ {len(new_cols)} nouvelles features synchronisées")
        
        except Exception as e:
            print(f"❌ Erreur synchronisation : {e}")
    
    # Préparer df_train
    df_train = None
    if split:
        Xtr = split.get("X_train")
        ytr = split.get("y_train")
        if isinstance(Xtr, pd.DataFrame) and ytr is not None:
            try:
                df_train = Xtr.copy()
                if target_col is not None and target_col not in df_train.columns:
                    df_train[target_col] = ytr
            except Exception as e:
                print(f"❌ Erreur création df_train : {e}")
                df_train = None
    
    if df_train is None:
        df_train = df.copy()
    
    active_cols = [c for c in df.columns if not columns_exclude.get(c, False) and c != target_col]
    
    # Identifier les colonnes catégorielles
    cat_cols = [c for c in active_cols if df[c].dtype == 'object' or pd.api.types.is_categorical_dtype(df[c])]
    
    state.setdefault("encoding_strategy", {})
    state.setdefault("encoding_params", {})
    
    # Statistiques globales
    total_features = len(active_cols)
    n_categorical = len(cat_cols)
    
    # ---------- HELPERS ----------
    def get_cardinality_level(n):
        """Retourne le niveau de cardinalité"""
        if n <= 10:
            return "🟢", "Low", "#27ae60", "green"
        elif n <= 50:
            return "🟡", "Medium", "#f39c12", "orange"
        else:
            return "🔴", "High", "#e74c3c", "red"
    
    def get_recommended_encoding(col):
        """Recommande une méthode d'encodage"""
        if col not in df.columns:
            return "Label Encoding", "Colonne introuvable", "❓"
        
        n_unique = df[col].nunique()
        
        if n_unique == 2:
            return "Label Encoding", "Binaire - simple et efficace", "✨"
        elif n_unique <= 10:
            return "One-Hot Encoding", "Faible cardinalité - safe et interprétable", "🎯"
        elif n_unique <= 50:
            return "Frequency Encoding", "Cardinalité moyenne - évite l'explosion", "⚡"
        else:
            return "Target Encoding", "Haute cardinalité - capture la relation avec target", "🎲"
    
    def detect_ordinal(col):
        """Détecte si une colonne semble ordinale"""
        common_orders = {
            'education': ['High School', 'Bachelor', 'Master', 'PhD'],
            'level': ['Low', 'Medium', 'High'],
            'grade': ['A', 'B', 'C', 'D', 'F'],
            'size': ['XS', 'S', 'M', 'L', 'XL', 'XXL'],
            'priority': ['Low', 'Medium', 'High', 'Critical']
        }
        
        col_lower = col.lower()
        for key, order in common_orders.items():
            if key in col_lower:
                values = df[col].unique()
                if set(values).issubset(set(order)):
                    return True, order
        
        return False, []
    
    def create_distribution_plot(col):
        """Crée un graphique de distribution pour une variable catégorielle"""
        if col not in df.columns:
            return None
        
        counts = df[col].value_counts().head(15)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=counts.index.astype(str),
            y=counts.values,
            marker=dict(
                color='#01335A',
                line=dict(color='#09538C', width=1)
            ),
            text=counts.values,
            textposition='outside',
            textfont=dict(size=11, color='#2c3e50', family='Inter'),
            opacity=0.9,
            hovertemplate='<b>%{x}</b><br>Fréquence: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=f"Distribution des modalités",
                font=dict(size=16, color='#2c3e50', family='Inter', weight=600)
            ),
            height=300,
            margin=dict(l=40, r=20, t=50, b=100),
            showlegend=False,
            paper_bgcolor='white',
            plot_bgcolor='#f8f9fa',
            font=dict(family="Inter, sans-serif", color="#2c3e50"),
            xaxis=dict(
                title="Modalités",
                tickangle=-45,
                showgrid=False,
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                title="Fréquence",
                showgrid=True,
                gridcolor='#e0e0e0',
                gridwidth=0.5
            )
        )
        
        return fig
    
    def apply_encoding(df_target, strategies, params, fit_on_train=True, fitted_encoders=None):
        """Applique les encodages selon les stratégies définies"""
        df_result = df_target.copy()
        encoders = {} if fit_on_train else fitted_encoders
        
        for col, method in strategies.items():
            if col not in df_result.columns:
                print(f"⚠️ Colonne {col} introuvable dans df_target")
                continue
            
            try:
                if method == "Label Encoding":
                    if fit_on_train:
                        le = LabelEncoder()
                        le.fit(df_train[col].dropna().astype(str))
                        df_result[col] = le.transform(df_result[col].astype(str))
                        encoders[col] = {"encoder": le, "method": method}
                    else:
                        if fitted_encoders and col in fitted_encoders:
                            le = fitted_encoders[col]["encoder"]
                            known_classes = set(le.classes_)
                            df_result[col] = df_result[col].astype(str).apply(
                                lambda x: le.transform([x])[0] if x in known_classes else -1
                            )
                
                elif method == "One-Hot Encoding":
                    drop_first = params.get(col, {}).get("drop_first", True)
                    
                    if fit_on_train:
                        dummies = pd.get_dummies(df_result[col], prefix=col, drop_first=drop_first)
                        df_result = pd.concat([df_result.drop(columns=[col]), dummies], axis=1)
                        encoders[col] = {
                            "method": method, 
                            "columns": dummies.columns.tolist(), 
                            "drop_first": drop_first
                        }
                    else:
                        if fitted_encoders and col in fitted_encoders:
                            expected_cols = fitted_encoders[col]["columns"]
                            drop_first = fitted_encoders[col].get("drop_first", True)
                            dummies = pd.get_dummies(df_result[col], prefix=col, drop_first=drop_first)
                            
                            for exp_col in expected_cols:
                                if exp_col not in dummies.columns:
                                    dummies[exp_col] = 0
                            
                            dummies = dummies[[c for c in expected_cols if c in dummies.columns]]
                            df_result = pd.concat([df_result.drop(columns=[col]), dummies], axis=1)
                
                elif method == "Ordinal Encoding":
                    order = params.get(col, {}).get("order", [])
                    if order:
                        mapping = {val: idx for idx, val in enumerate(order)}
                        df_result[col] = df_result[col].map(mapping).fillna(-1).astype(int)
                        if fit_on_train:
                            encoders[col] = {"method": method, "order": order, "mapping": mapping}
                
                elif method == "Frequency Encoding":
                    if fit_on_train:
                        freq = df_train[col].value_counts(normalize=True).to_dict()
                        df_result[col] = df_result[col].map(freq).fillna(0)
                        encoders[col] = {"method": method, "frequencies": freq}
                    else:
                        if fitted_encoders and col in fitted_encoders:
                            freq = fitted_encoders[col]["frequencies"]
                            df_result[col] = df_result[col].map(freq).fillna(0)
                
                elif method == "Target Encoding":
                    if target_col and target_col in df_train.columns:
                        if fit_on_train:
                            target_means = df_train.groupby(col)[target_col].mean().to_dict()
                            global_mean = df_train[target_col].mean()
                            
                            smoothing = params.get(col, {}).get("smoothing", 10)
                            counts = df_train[col].value_counts().to_dict()
                            
                            smoothed_means = {}
                            for cat, mean in target_means.items():
                                count = counts.get(cat, 0)
                                smoothed_means[cat] = (count * mean + smoothing * global_mean) / (count + smoothing)
                            
                            df_result[col] = df_result[col].map(smoothed_means).fillna(global_mean)
                            encoders[col] = {
                                "method": method, 
                                "target_means": smoothed_means, 
                                "global_mean": global_mean
                            }
                        else:
                            if fitted_encoders and col in fitted_encoders:
                                smoothed_means = fitted_encoders[col]["target_means"]
                                global_mean = fitted_encoders[col]["global_mean"]
                                df_result[col] = df_result[col].map(smoothed_means).fillna(global_mean)
            
            except Exception as e:
                print(f"❌ Erreur encodage {col}: {e}")
                import traceback
                traceback.print_exc()
        
        return df_result, encoders
    
    def open_encoding_modal(col_name):
        """Ouvre un modal pour configurer l'encodage d'une colonne"""
        print(f"🔍 Opening modal for: {col_name}")
        
        if col_name not in df.columns:
            ui.notify(f"⚠️ Colonne {col_name} introuvable", color="warning")
            return
        
        current_method = state.get("encoding_strategy", {}).get(col_name, "")
        current_params = state.get("encoding_params", {}).get(col_name, {})
        
        n_unique = df[col_name].nunique()
        top_values = df[col_name].value_counts().head(5)
        is_ordinal, suggested_order = detect_ordinal(col_name)
        recommended, reason, icon = get_recommended_encoding(col_name)
        
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-4xl").style(
            "padding:0 !important; border-radius:20px !important; "
            "box-shadow:0 20px 60px rgba(1,51,90,0.15) !important; max-height:90vh !important; "
            "overflow-y:auto !important; border:1px solid #e1e8ed !important;"
        ):
            # Header avec gradient amélioré
            with ui.column().classes("w-full").style(
                "background:linear-gradient(135deg, #01335A 0%, #09538C 50%, #0d6eaf 100%) !important; "
                "padding:32px 40px !important; border-radius:20px 20px 0 0 !important; "
                "position:relative !important;"
            ):
                with ui.row().classes("items-center gap-3"):
                    with ui.card().style(
                        "background:rgba(255,255,255,0.15) !important; backdrop-filter:blur(10px) !important; "
                        "padding:12px !important; border-radius:12px !important; box-shadow:none !important;"
                    ):
                        ui.label("🔤").style("font-size:32px !important;")
                    
                    with ui.column().classes("gap-1"):
                        ui.label(f"{col_name}").style(
                            "font-weight:700 !important; font-size:26px !important; color:white !important; "
                            "text-shadow:0 2px 4px rgba(0,0,0,0.1) !important;"
                        )
                        ui.label(f"Variable catégorielle • {n_unique} modalités uniques").style(
                            "color:rgba(255,255,255,0.9) !important; font-size:14px !important; font-weight:400 !important;"
                        )
            
            # Contenu
            with ui.column().classes("w-full").style("padding:40px !important;"):
                # Stats & Distribution
                with ui.row().classes("w-full gap-6 mb-8"):
                    # Carte stats
                    with ui.card().classes("flex-1").style(
                        "background:linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%) !important; "
                        "padding:28px !important; border-radius:16px !important; "
                        "box-shadow:0 4px 12px rgba(0,0,0,0.06) !important; "
                        "border:1px solid #e1e8ed !important;"
                    ):
                        icon_card, level, color, _ = get_cardinality_level(n_unique)
                        
                        with ui.row().classes("items-center gap-3 mb-4"):
                            with ui.card().style(
                                f"background:{color}15 !important; padding:12px !important; "
                                f"border-radius:12px !important; box-shadow:none !important; "
                                f"border:2px solid {color}40 !important;"
                            ):
                                ui.label(icon_card).style("font-size:28px !important;")
                            
                            with ui.column().classes("gap-0"):
                                ui.label(f"Cardinalité: {level}").style(
                                    f"font-weight:700 !important; font-size:18px !important; color:{color} !important;"
                                )
                                ui.label(f"{n_unique} valeurs uniques").style(
                                    "font-size:12px !important; color:#636e72 !important;"
                                )
                        
                        ui.separator().classes("my-4")
                        
                        ui.label("Top 5 modalités :").style(
                            "font-size:14px !important; color:#2c3e50 !important; margin-bottom:16px !important; "
                            "font-weight:600 !important;"
                        )
                        
                        for idx, (val, count) in enumerate(top_values.items()):
                            pct = round(count / len(df) * 100, 1)
                            with ui.row().classes("items-center gap-3 mb-3"):
                                with ui.badge().style(
                                    "background:#01335A !important; color:white !important; "
                                    "min-width:24px !important; height:24px !important; "
                                    "display:flex !important; align-items:center !important; "
                                    "justify-content:center !important; border-radius:6px !important; "
                                    "font-weight:600 !important; font-size:11px !important;"
                                ):
                                    ui.label(str(idx+1))
                                
                                ui.label(str(val)[:25]).style(
                                    "flex:0 0 140px !important; font-size:13px !important; "
                                    "color:#2c3e50 !important; font-weight:500 !important;"
                                )
                                
                                with ui.column().classes("flex-1 gap-1"):
                                    ui.linear_progress(value=pct/100).props(f'color="primary"').style(
                                        "height:8px !important; border-radius:4px !important;"
                                    )
                                
                                ui.label(f"{pct}%").style(
                                    "width:50px !important; text-align:right !important; "
                                    "font-size:13px !important; color:#01335A !important; "
                                    "font-family:monospace !important; font-weight:600 !important;"
                                )
                    
                    # Graphique
                    with ui.column().classes("flex-1"):
                        with ui.card().style(
                            "background:white !important; padding:20px !important; "
                            "border-radius:16px !important; box-shadow:0 4px 12px rgba(0,0,0,0.06) !important; "
                            "border:1px solid #e1e8ed !important;"
                        ):
                            dist_plot = create_distribution_plot(col_name)
                            if dist_plot:
                                ui.plotly(dist_plot).style("width:100% !important;")
                
                # Recommandation
                with ui.card().classes("w-full mb-6").style(
                    "background:linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%) !important; "
                    "padding:24px !important; border-radius:16px !important; "
                    "border-left:5px solid #01335A !important; "
                    "box-shadow:0 4px 12px rgba(1,51,90,0.08) !important;"
                ):
                    with ui.row().classes("items-start gap-4 mb-3"):
                        with ui.card().style(
                            "background:rgba(1,51,90,0.1) !important; padding:12px !important; "
                            "border-radius:12px !important; box-shadow:none !important;"
                        ):
                            ui.label(icon).style("font-size:28px !important;")
                        
                        with ui.column().classes("gap-2"):
                            ui.label(f"Recommandation : {recommended}").style(
                                "font-weight:700 !important; font-size:18px !important; color:#01335A !important;"
                            )
                            ui.label(reason).style(
                                "font-size:14px !important; color:#2c3e50 !important; line-height:1.6 !important;"
                            )
                
                # Sélection méthode
                ui.label("Choisir la méthode d'encodage").style(
                    "font-weight:600 !important; font-size:16px !important; color:#2c3e50 !important; "
                    "margin-bottom:12px !important;"
                )
                
                method_select = ui.select(
                    options=[
                        "Label Encoding",
                        "One-Hot Encoding",
                        "Ordinal Encoding",
                        "Frequency Encoding",
                        "Target Encoding"
                    ],
                    value=current_method or recommended,
                    label="Méthode d'encodage"
                ).props("outlined").classes("w-full mb-6")
                
                # Zone paramètres
                params_container = ui.column().classes("w-full")
                
                drop_first_checkbox = None
                smoothing_input = None
                order_inputs = []
                
                def update_params_ui():
                    nonlocal drop_first_checkbox, smoothing_input, order_inputs
                    params_container.clear()
                    order_inputs = []
                    
                    with params_container:
                        method = method_select.value
                        
                        if method == "Label Encoding":
                            with ui.card().classes("w-full p-6").style(
                                "background:#f8f9fa !important; border-radius:12px !important; "
                                "box-shadow:none !important; border:1px solid #e1e8ed !important;"
                            ):
                                ui.markdown("""
<div style="color:#2c3e50; font-size:14px; line-height:1.8;">

### 📌 Label Encoding

**Principe** : Assigne un entier unique à chaque modalité (0, 1, 2, ...)

✅ **Avantages** :
- Simple et rapide
- Idéal pour variables binaires
- Pas d'explosion dimensionnelle

⚠️ **Attention** :
- Impose un ordre arbitraire
- Peut biaiser les modèles basés sur la distance

</div>
                                """)
                        
                        elif method == "One-Hot Encoding":
                            with ui.card().classes("w-full p-6 mb-4").style(
                                "background:#f8f9fa !important; border-radius:12px !important; "
                                "box-shadow:none !important; border:1px solid #e1e8ed !important;"
                            ):
                                ui.markdown("""
<div style="color:#2c3e50; font-size:14px; line-height:1.8;">

### 📌 One-Hot Encoding

**Principe** : Crée une colonne binaire (0/1) pour chaque modalité

✅ **Avantages** :
- Pas d'ordre imposé
- Très interprétable
- Compatible avec tous algorithmes

❌ **Inconvénients** :
- Explosion dimensionnelle si haute cardinalité

</div>
                                """)
                            
                            drop_first_checkbox = ui.checkbox(
                                "Drop first (éviter multicolinéarité)",
                                value=current_params.get("drop_first", True)
                            )
                        
                        elif method == "Ordinal Encoding":
                            with ui.card().classes("w-full p-6 mb-4").style(
                                "background:#f8f9fa !important; border-radius:12px !important; "
                                "box-shadow:none !important; border:1px solid #e1e8ed !important;"
                            ):
                                ui.markdown("""
<div style="color:#2c3e50; font-size:14px; line-height:1.8;">

### 📌 Ordinal Encoding

**Principe** : Assigne des entiers selon un ordre naturel

✅ Capture l'ordre naturel (Low < Medium < High)

⚠️ Applicable uniquement si ordre naturel existe

</div>
                                """)
                            
                            ui.label("Définir l'ordre (du plus bas au plus haut) :").style(
                                "font-weight:600 !important; margin-bottom:16px !important; color:#01335A !important;"
                            )
                            
                            current_order = current_params.get("order", suggested_order if is_ordinal else list(df[col_name].unique())[:10])
                            
                            with ui.column().classes("w-full gap-3"):
                                for idx, val in enumerate(current_order):
                                    with ui.row().classes("items-center gap-3"):
                                        with ui.badge().style(
                                            "background:linear-gradient(135deg, #01335A, #09538C) !important; "
                                            "color:white !important; min-width:36px !important; height:36px !important; "
                                            "display:flex !important; align-items:center !important; "
                                            "justify-content:center !important; border-radius:8px !important;"
                                        ):
                                            ui.label(str(idx)).style("font-weight:700 !important;")
                                        
                                        inp = ui.input(value=str(val)).props("outlined dense").classes("flex-1")
                                        order_inputs.append(inp)
                        
                        elif method == "Frequency Encoding":
                            with ui.card().classes("w-full p-6").style(
                                "background:#f8f9fa !important; border-radius:12px !important; "
                                "box-shadow:none !important; border:1px solid #e1e8ed !important;"
                            ):
                                ui.markdown("""
<div style="color:#2c3e50; font-size:14px; line-height:1.8;">

### 📌 Frequency Encoding

**Principe** : Remplace chaque modalité par sa fréquence

✅ **Avantages** :
- Simple et efficace
- Pas de data leakage
- Une seule colonne générée

</div>
                                """)
                        
                        elif method == "Target Encoding":
                            with ui.card().classes("w-full p-6 mb-4").style(
                                "background:#fff3cd !important; border-radius:12px !important; "
                                "border-left:4px solid #f39c12 !important; box-shadow:none !important;"
                            ):
                                ui.markdown("""
<div style="color:#856404; font-size:14px; line-height:1.8;">

### 📌 Target Encoding

**Principe** : Remplace chaque modalité par la moyenne de la target

✅ Capture directement la relation avec la target

⚠️ **Risques** : Overfitting et data leakage possibles

</div>
                                """)
                            
                            smoothing_input = ui.number(
                                label="Smoothing (régularisation)",
                                value=current_params.get("smoothing", 10),
                                min=0,
                                max=100
                            ).props("outlined").classes("w-full")
                
                method_select.on_value_change(lambda: update_params_ui())
                update_params_ui()
                
                # Boutons
                with ui.row().classes("w-full justify-end gap-3 mt-8"):
                    ui.button("Annuler", on_click=dialog.close).props("flat")
                    
                    def save_encoding():
                        method = method_select.value
                        params = {}
                        
                        if method == "One-Hot Encoding" and drop_first_checkbox:
                            params["drop_first"] = drop_first_checkbox.value
                        elif method == "Target Encoding" and smoothing_input:
                            params["smoothing"] = int(smoothing_input.value)
                        elif method == "Ordinal Encoding" and order_inputs:
                            order = [inp.value for inp in order_inputs]
                            params["order"] = order
                        
                        state.setdefault("encoding_strategy", {})[col_name] = method
                        state.setdefault("encoding_params", {})[col_name] = params
                        
                        ui.notify(f"✅ Encodage configuré pour {col_name}", color="positive")
                        dialog.close()
                        ui.run_javascript("setTimeout(() => window.location.reload(), 500);")
                    
                    ui.button("Sauvegarder", on_click=save_encoding).style(
                        "background:linear-gradient(135deg, #01335A, #09538C) !important; "
                        "color:white !important; border-radius:10px !important; "
                        "padding:12px 28px !important; text-transform:none !important; "
                        "font-weight:600 !important;"
                    ).props('icon-right="save"')
        
        dialog.open()
    
    def apply_all_encodings():
        """Applique tous les encodages"""
        strategies = state.get("encoding_strategy", {})
        split_data = state.get("split", {})

        if not strategies:
            ui.notify("⚠️ Aucun encodage configuré", color="warning")
            return
        
        with ui.dialog() as dialog, ui.card().style(
            "padding:0 !important; border-radius:20px !important; "
            "box-shadow:0 20px 60px rgba(1,51,90,0.15) !important;"
        ):
            with ui.column().classes("w-full").style(
                "background:linear-gradient(135deg, #01335A 0%, #09538C 100%) !important; "
                "padding:28px 32px !important; border-radius:20px 20px 0 0 !important;"
            ):
                ui.label("⚠️ Confirmation").style(
                    "font-weight:700 !important; font-size:22px !important; color:white !important;"
                )
            
            with ui.column().classes("w-full").style("padding:32px !important;"):
                ui.label("Voulez-vous appliquer les encodages ?").style(
                    "margin-bottom:16px !important; color:#2c3e50 !important;"
                )
                
                ui.label(f"📊 {len(strategies)} encodage(s) configuré(s)").style(
                    "color:#01335A !important; font-weight:600 !important;"
                )
                
                with ui.row().classes("w-full justify-end gap-3 mt-6"):
                    ui.button("Annuler", on_click=dialog.close).props("flat")
                    
                    def confirm_and_next():
                        try:
                            params = state.get("encoding_params", {})
                            
                            if split_data and "X_train" in split_data:
                                df_train_for_fit = split_data["X_train"].copy()
                                if target_col and "y_train" in split_data and target_col not in df_train_for_fit.columns:
                                    df_train_for_fit[target_col] = split_data["y_train"]
                            else:
                                df_train_for_fit = df_train.copy()
                        
                            _, encoders = apply_encoding(df_train_for_fit, strategies, params, fit_on_train=True)
                            state["fitted_encoders"] = encoders
                            
                            df_result, _ = apply_encoding(df.copy(), strategies, params, fit_on_train=False, fitted_encoders=encoders)
                            state["raw_df"] = df_result
                        
                            if split_data:
                                for key in ["X_train", "X_val", "X_test"]:
                                    if key in split_data and isinstance(split_data[key], pd.DataFrame):
                                        split_data[key], _ = apply_encoding(
                                            split_data[key], strategies, params, 
                                            fit_on_train=False, fitted_encoders=encoders
                                        )
                                state["split"] = split_data
                            
                            ui.notify("✅ Encodages appliqués!", color="positive")
                            dialog.close()
                            ui.run_javascript("setTimeout(() => window.location.href='/supervised/distribution_transform', 1000);")
                        
                        except Exception as e:
                            ui.notify(f"❌ Erreur : {str(e)}", color="negative")
                            print(f"Erreur : {e}")
                    
                    ui.button("Confirmer", on_click=confirm_and_next).style(
                        "background:linear-gradient(135deg, #01335A, #09538C) !important; "
                        "color:white !important; border-radius:10px !important; "
                        "padding:12px 24px !important;"
                    )
        
        dialog.open()
    
    def apply_recommended():
        """Applique automatiquement les recommandations"""
        count = 0
        for col in cat_cols:
            recommended, _, _ = get_recommended_encoding(col)
            state.setdefault("encoding_strategy", {})[col] = recommended
            
            params = {}
            if recommended == "One-Hot Encoding":
                params["drop_first"] = True
            elif recommended == "Target Encoding":
                params["smoothing"] = 10
            elif recommended == "Ordinal Encoding":
                is_ordinal, order = detect_ordinal(col)
                params["order"] = order if is_ordinal else list(df[col].unique())[:10]
            
            state.setdefault("encoding_params", {})[col] = params
            count += 1
        
        ui.notify(f"✅ {count} recommandations appliquées", color="positive")
        ui.run_javascript("setTimeout(() => window.location.reload(), 500);")
    
    # ---------- UI PRINCIPALE ----------
    with ui.column().classes("w-full items-center").style(
        "background:linear-gradient(180deg, #f0f4f8 0%, #e8f1f8 100%) !important; "
        "min-height:100vh !important; padding:60px 24px !important;"
    ):
        # HEADER
        with ui.column().classes("items-center mb-12"):
            ui.label("Encodage des Features Catégorielles").style(
                "font-weight:800 !important; font-size:42px !important; "
                "background:linear-gradient(135deg, #01335A 0%, #09538C 100%) !important; "
                "-webkit-background-clip:text !important; -webkit-text-fill-color:transparent !important; "
                "text-align:center !important;"
            )
            ui.label("Conversion des variables catégorielles en format numérique").style(
                "font-size:17px !important; color:#636e72 !important; text-align:center !important;"
            )
        
        # SECTION EXPLICATION
        with ui.card().classes("w-full max-w-6xl mb-8").style(
            "background:white !important; border-radius:24px !important; padding:48px !important; "
            "box-shadow:0 12px 32px rgba(1,51,90,0.1) !important; position:relative !important;"
        ):
            with ui.row().classes("items-start gap-8 w-full"):
                with ui.column().classes("flex-1 gap-6"):
                    ui.label("Pourquoi encoder les variables catégorielles ?").style(
                        "font-weight:800 !important; font-size:28px !important; color:#01335A !important;"
                    )
                    
                    ui.markdown("""
Les algorithmes de Machine Learning travaillent **exclusivement avec des nombres**. 
Les variables catégorielles doivent être **transformées en valeurs numériques**.

**🎯 Objectifs :**
1. **Identifier** toutes les features catégorielles
2. **Choisir** la méthode d'encodage adaptée
3. **Appliquer** l'encodage sur train/validation/test

**⚡ Méthodes disponibles :**
- **Label Encoding** : Variables binaires
- **One-Hot Encoding** : Faible cardinalité (<10)
- **Frequency Encoding** : Cardinalité moyenne (10-50)
- **Target Encoding** : Haute cardinalité (>50)
- **Ordinal Encoding** : Ordre naturel
                    """)
        
        # OVERVIEW METRICS
        with ui.card().classes("w-full max-w-6xl mb-8").style(
            "background:white !important; border-radius:20px !important; padding:40px !important; "
            "box-shadow:0 8px 24px rgba(1,51,90,0.08) !important;"
        ):
            ui.label(" Vue d'ensemble").style(
                "font-weight:700 !important; font-size:26px !important; color:#01335A !important; margin-bottom:28px !important;"
            )
            
            if n_categorical == 0:
                ui.label("✅ Aucune feature catégorielle détectée").style(
                    "font-weight:700 !important; font-size:22px !important; color:#27ae60 !important;"
                )
            else:
                pct_cat = round(n_categorical / total_features * 100) if total_features > 0 else 0
                n_configured = len([c for c in cat_cols if c in state.get("encoding_strategy", {})])
                
                with ui.row().classes("w-full gap-6"):
                    with ui.card().classes("flex-1 text-center p-6").style(
                        "background:linear-gradient(135deg, #01335A15, #01335A05) !important; "
                        "border-radius:16px !important; border:2px solid #01335A30 !important;"
                    ):
                        ui.label(str(n_categorical)).style(
                            "font-weight:800 !important; font-size:36px !important; color:#01335A !important;"
                        )
                        ui.label("Catégorielles").style("color:#2c3e50 !important; font-weight:600 !important;")
                    
                    with ui.card().classes("flex-1 text-center p-6").style(
                        "background:linear-gradient(135deg, #2196f315, #2196f305) !important; "
                        "border-radius:16px !important; border:2px solid #2196f330 !important;"
                    ):
                        ui.label(f"{n_configured}/{n_categorical}").style(
                            "font-weight:800 !important; font-size:36px !important; color:#2196f3 !important;"
                        )
                        ui.label("Configurées").style("color:#2c3e50 !important; font-weight:600 !important;")
        
        # TABLE AVEC BOUTONS INDIVIDUELS
        if n_categorical > 0:
            with ui.card().classes("w-full max-w-6xl mb-8").style(
                "background:white !important; border-radius:20px !important; padding:40px !important; "
                "box-shadow:0 8px 24px rgba(1,51,90,0.08) !important;"
            ):
                ui.label(" Configuration").style(
                    "font-weight:700 !important; font-size:26px !important; color:#01335A !important; margin-bottom:20px !important;"
                )
                
                with ui.card().classes("w-full mb-6").style(
                    "background:linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%) !important; "
                    "padding:20px !important; border-radius:12px !important; "
                    "box-shadow:none !important; border-left:4px solid #01335A !important;"
                ):
                    ui.label("💡 Cliquez sur le bouton 'Configurer' pour chaque variable").style(
                        "font-size:15px !important; color:#01335A !important; font-weight:600 !important;"
                    )
                
                # ✅ SOLUTION : Utiliser des boutons au lieu d'événements de clic sur table
                for col in cat_cols:
                    n_unique = df[col].nunique()
                    icon, level, color, _ = get_cardinality_level(n_unique)
                    recommended, _, rec_icon = get_recommended_encoding(col)
                    current = state.get("encoding_strategy", {}).get(col, "")
                    status = "✅" if current else "⚪"
                    
                    with ui.card().classes("w-full mb-3").style(
                        "background:#f8f9fa !important; padding:20px !important; "
                        "border-radius:12px !important; box-shadow:none !important; "
                        "border:1px solid #e1e8ed !important; transition:all 0.2s ease !important;"
                    ):
                        with ui.row().classes("w-full items-center justify-between gap-4"):
                            # Info colonne
                            with ui.row().classes("items-center gap-4 flex-1"):
                                ui.label(status).style("font-size:20px !important;")
                                
                                with ui.column().classes("gap-1"):
                                    ui.label(col).style(
                                        "font-weight:700 !important; font-size:16px !important; color:#01335A !important;"
                                    )
                                    ui.label(f"{icon} {level} • {n_unique} modalités").style(
                                        "font-size:12px !important; color:#636e72 !important;"
                                    )
                                
                                if current:
                                    with ui.badge().style(
                                        "background:#27ae60 !important; color:white !important; "
                                        "padding:6px 12px !important; border-radius:6px !important;"
                                    ):
                                        ui.label(current).style("font-size:11px !important; font-weight:600 !important;")
                                else:
                                    with ui.badge().style(
                                        "background:#f39c12 !important; color:white !important; "
                                        "padding:6px 12px !important; border-radius:6px !important;"
                                    ):
                                        ui.label(f"Recommandé: {recommended}").style(
                                            "font-size:11px !important; font-weight:600 !important;"
                                        )
                            
                            # Bouton configurer
                            ui.button(
                                "Configurer",
                                on_click=lambda c=col: open_encoding_modal(c)
                            ).style(
                                "background:linear-gradient(135deg, #01335A, #09538C) !important; "
                                "color:white !important; border-radius:8px !important; "
                                "padding:10px 20px !important; text-transform:none !important; "
                                "font-weight:600 !important; font-size:13px !important;"
                            ).props('icon-right="settings"')
                
                ui.separator().classes("my-6")
                
                with ui.row().classes("w-full gap-4"):
                    ui.button("✨ Appliquer recommandations", on_click=apply_recommended).style(
                        "flex:1 !important; background:linear-gradient(135deg, #2196f3, #1976d2) !important; "
                        "color:white !important; border-radius:12px !important; padding:16px !important;"
                    )
                    
                    ui.button("✓ Appliquer & Continuer", on_click=apply_all_encodings).style(
                        "flex:1 !important; background:linear-gradient(135deg, #01335A, #09538C) !important; "
                        "color:white !important; border-radius:12px !important; padding:16px !important;"
                    )
        
        # NAVIGATION
        with ui.row().classes("w-full max-w-6xl justify-between gap-4 mt-12"):
            ui.button(" Précédent", on_click=lambda: ui.run_javascript("window.location.href='/supervised/missing_values'")).style(
                "background:white !important; color:#01335A !important; border:2px solid #01335A !important; "
                "border-radius:12px !important; height:52px !important; min-width:160px !important;"
            )
            
            ui.button("Suivant ", on_click=lambda: ui.run_javascript("window.location.href='/supervised/distribution_transform'")).style(
                "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white !important; "
                "border-radius:12px !important; height:52px !important; min-width:160px !important;"
            )




# ----------------- PAGE 3.8 : TRANSFORMATIONS DE DISTRIBUTIONS -----------------

@ui.page('/supervised/distribution_transform')
def distribution_transform_page():
    """
    Page complète pour les transformations de distributions
    Design moderne unifié avec #01335A
    """
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy import stats
    from scipy.stats import boxcox, yeojohnson
    
    # ---------- CONTEXTE ----------
    df = state.get("raw_df", None)
    split = state.get("split", None)
    columns_exclude = state.get("columns_exclude", {}) or {}
    target_col = state.get("target_column", None)
    selected_algos = state.get("selected_algos", [])
    
    if df is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Aucun dataset chargé.").style(
                "font-size:18px !important; color:#c0392b !important; font-weight:600 !important;"
            )
            ui.button(
                "🔙 Retour à l'Upload",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/upload'")
            ).style(
                "background:#01335A !important; color:white !important; padding:12px 32px !important; "
                "border-radius:8px !important; margin-top:20px !important;"
            )
        return
    
    # 🔄 CORRECTION : Synchroniser le split avec raw_df
    if split and "X_train" in split:
        try:
            new_cols = [c for c in df.columns if c not in split["X_train"].columns and c != target_col]
            
            if new_cols:
                print(f"🔄 Synchronisation des nouvelles features : {new_cols}")
                
                for key in ["X_train", "X_val", "X_test"]:
                    if key in split and isinstance(split[key], pd.DataFrame):
                        indices = split[key].index
                        for col in new_cols:
                            if col in df.columns:
                                split[key][col] = df.loc[indices, col]
                
                state["split"] = split
                print(f"✅ {len(new_cols)} nouvelles features synchronisées")
        
        except Exception as e:
            print(f"❌ Erreur synchronisation : {e}")
    
    # Préparer df_train
    df_train = None
    if split:
        Xtr = split.get("X_train")
        ytr = split.get("y_train")
        if isinstance(Xtr, pd.DataFrame) and ytr is not None:
            try:
                df_train = Xtr.copy()
                if target_col is not None and target_col not in df_train.columns:
                    df_train[target_col] = ytr
            except Exception as e:
                print(f"❌ Erreur création df_train : {e}")
                df_train = None
    
    if df_train is None:
        df_train = df.copy()
    
    active_cols = [c for c in df.columns if not columns_exclude.get(c, False) and c != target_col]
    num_cols = [c for c in active_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    state.setdefault("transform_strategy", {})
    state.setdefault("transform_params", {})
    state.setdefault("fitted_transformers", {})
    
    # ---------- HELPERS ----------
    
    def calculate_skewness_kurtosis(col):
        """Calcule skewness et kurtosis"""
        if col not in df_train.columns:
            return 0, 0
        
        data = df_train[col].dropna()
        if len(data) < 3:
            return 0, 0
        
        try:
            skew = float(stats.skew(data))
            kurt = float(stats.kurtosis(data))
            return skew, kurt
        except Exception as e:
            print(f"❌ Erreur calcul skew/kurt pour {col}: {e}")
            return 0, 0
    
    def get_skewness_level(skew):
        """Retourne le niveau de skewness"""
        abs_skew = abs(skew)
        if abs_skew < 0.5:
            return "🟢", "Normal", "#27ae60"
        elif abs_skew < 1.5:
            return "🟡", "Modéré", "#01335A"
        else:
            return "🔴", "Fort", "#e74c3c"
    
    def get_recommended_transform(col):
        """Recommande une transformation"""
        if col not in df_train.columns:
            return "Aucune", "Colonne introuvable"
        
        skew, _ = calculate_skewness_kurtosis(col)
        abs_skew = abs(skew)
        
        data = df_train[col].dropna()
        if len(data) == 0:
            return "Aucune", "Pas de données"
        
        has_zero = (data == 0).any()
        has_negative = (data < 0).any()
        
        if abs_skew < 0.5:
            return "Aucune", "Distribution acceptable"
        elif abs_skew < 1.5:
            if has_negative:
                return "Yeo-Johnson", "Skew modéré + valeurs négatives"
            else:
                return "Square Root", "Skew modéré"
        else:
            if has_negative:
                return "Yeo-Johnson", "Skew fort + valeurs négatives"
            elif has_zero:
                return "Log Transform", "Skew fort (+ constante pour zéros)"
            else:
                return "Box-Cox", "Skew fort"
    
    def apply_transform(data, method, params=None):
        """Applique une transformation sur les données"""
        if params is None:
            params = {}
        
        # 🔧 CORRECTION : Gérer les différents types de data
        if isinstance(data, pd.Series):
            data = data.values
        
        data = np.array(data).copy()
        
        # Filtrer NaN et Inf
        mask_valid = ~np.isnan(data) & ~np.isinf(data)
        data_clean = data[mask_valid]
        
        if len(data_clean) == 0:
            return data, None
        
        try:
            if method == "Log Transform":
                c = params.get("constant", 1)
                transformed = np.log(data_clean + c)
                return transformed, {"constant": c}
            
            elif method == "Square Root":
                min_val = data_clean.min()
                if min_val < 0:
                    shift = abs(min_val) + 1
                    transformed = np.sqrt(data_clean + shift)
                    return transformed, {"shift": shift}
                else:
                    transformed = np.sqrt(data_clean)
                    return transformed, {}
            
            elif method == "Box-Cox":
                if (data_clean <= 0).any():
                    print(f"⚠️ Box-Cox impossible : valeurs ≤ 0")
                    return data_clean, None
                transformed, lambda_param = boxcox(data_clean)
                return transformed, {"lambda": float(lambda_param)}
            
            elif method == "Yeo-Johnson":
                transformed, lambda_param = yeojohnson(data_clean)
                return transformed, {"lambda": float(lambda_param)}
            
            else:  # Aucune
                return data_clean, {}
        
        except Exception as e:
            print(f"❌ Erreur transformation {method}: {e}")
            return data_clean, None
    
    def calculate_algo_impact(col, skew):
        """Calcule l'impact sur chaque algorithme"""
        abs_skew = abs(skew)
        impacts = {}
        
        # Naive Bayes
        if abs_skew < 0.5:
            impacts["Naive Bayes"] = ("✅", "Distribution normale", "#27ae60")
        elif abs_skew < 1.5:
            impacts["Naive Bayes"] = ("🟡", "Assume normalité (légèrement violée)", "#01335A")
        else:
            impacts["Naive Bayes"] = ("🔴", "Assume normalité (fortement violée)", "#e74c3c")
        
        # KNN
        if abs_skew < 0.5:
            impacts["KNN"] = ("✅", "Distances équilibrées", "#27ae60")
        elif abs_skew < 1.5:
            impacts["KNN"] = ("🟡", "Distances légèrement biaisées", "#01335A")
        else:
            impacts["KNN"] = ("🟡", "Distances biaisées vers extrêmes", "#01335A")
        
        # C4.5
        impacts["Decision Tree"] = ("✅", "Robuste aux distributions", "#27ae60")
        
        return impacts
    
    def create_distribution_plot(col, transform_method="Aucune", params=None):
        """Crée un plot avant/après transformation"""
        if col not in df_train.columns:
            return None, 0, {}
        
        data_original = df_train[col].dropna()
        
        if len(data_original) == 0:
            return None, 0, {}
        
        if transform_method == "Aucune":
            data_transformed = data_original.values
            transform_params = {}
        else:
            data_transformed, transform_params = apply_transform(data_original.values, transform_method, params)
            if data_transformed is None or transform_params is None:
                data_transformed = data_original.values
                transform_params = {}
        
        # Créer subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Distribution Originale', 
                f'Distribution Après {transform_method}',
                'Q-Q Plot Original',
                'Q-Q Plot Transformé'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Histogram original
        fig.add_trace(
            go.Histogram(x=data_original, nbinsx=30, name="Original", marker_color='#01335A', opacity=0.8),
            row=1, col=1
        )
        
        # Histogram transformé
        fig.add_trace(
            go.Histogram(x=data_transformed, nbinsx=30, name="Transformé", marker_color='#01335A', opacity=0.8),
            row=1, col=2
        )
        
        # Q-Q Plot original
        try:
            qq_original = stats.probplot(data_original, dist="norm")
            fig.add_trace(
                go.Scatter(x=qq_original[0][0], y=qq_original[0][1], mode='markers', 
                          name="Original", marker=dict(color='#01335A', size=4)),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=qq_original[0][0], y=qq_original[0][0], mode='lines',
                          name="Théorique", line=dict(color='red', dash='dash')),
                row=2, col=1
            )
        except Exception as e:
            print(f"❌ Erreur Q-Q plot original: {e}")
        
        # Q-Q Plot transformé
        try:
            qq_transformed = stats.probplot(data_transformed, dist="norm")
            fig.add_trace(
                go.Scatter(x=qq_transformed[0][0], y=qq_transformed[0][1], mode='markers',
                          name="Transformé", marker=dict(color='#01335A', size=4)),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=qq_transformed[0][0], y=qq_transformed[0][0], mode='lines',
                          name="Théorique", line=dict(color='red', dash='dash')),
                row=2, col=2
            )
        except Exception as e:
            print(f"❌ Erreur Q-Q plot transformé: {e}")
        
        # Calculs skewness
        try:
            skew_original = float(stats.skew(data_original))
            skew_transformed = float(stats.skew(data_transformed))
        except:
            skew_original = 0
            skew_transformed = 0
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text=f"<b>{col}</b> - Skew Original: {skew_original:.2f}  Transformé: {skew_transformed:.2f}",
            paper_bgcolor='white',
            plot_bgcolor='#f8f9fa',
            font=dict(family="Inter, sans-serif", color="#2c3e50")
        )
        
        fig.update_xaxes(title_text="Valeur", row=1, col=1)
        fig.update_xaxes(title_text="Valeur", row=1, col=2)
        fig.update_xaxes(title_text="Quantiles théoriques", row=2, col=1)
        fig.update_xaxes(title_text="Quantiles théoriques", row=2, col=2)
        
        fig.update_yaxes(title_text="Fréquence", row=1, col=1)
        fig.update_yaxes(title_text="Fréquence", row=1, col=2)
        fig.update_yaxes(title_text="Quantiles observés", row=2, col=1)
        fig.update_yaxes(title_text="Quantiles observés", row=2, col=2)
        
        return fig, skew_transformed, transform_params
    
    def show_transformed_data_preview():
        """Affiche un aperçu (head) des données transformées."""
        try:
            strategies = state.get("transform_strategy", {})
            params_dict = state.get("transform_params", {})
            
            if not strategies:
                ui.notify("⚠️ Aucune transformation configurée.", color="warning")
                return
            
            # Créer un dataframe temporaire avec les transformations appliquées
            df_preview = df_train.copy()
            
            for col, method in strategies.items():
                if method != "Aucune" and col in df_preview.columns:
                    col_params = params_dict.get(col, {})
                    transformed_data, _ = apply_transform(df_preview[col].values, method, col_params)
                    if transformed_data is not None:
                        # Gérer la différence de taille
                        if len(transformed_data) == len(df_preview):
                            df_preview[col] = transformed_data
                        else:
                            new_col = pd.Series(index=df_preview.index, dtype=float)
                            mask = ~df_preview[col].isna() & ~np.isinf(df_preview[col])
                            new_col[mask] = transformed_data
                            df_preview[col] = new_col
            
            # Création de la fenêtre
            with ui.dialog() as dialog, ui.card().style(
                "min-width:700px; max-height:80vh; overflow:auto; padding:20px;"
            ):
                ui.label("📊 Aperçu des données transformées").style(
                    "font-weight:700; font-size:20px; margin-bottom:10px;"
                )
                ui.html(df_preview.head().to_html(classes="table table-striped"), sanitize=False)
                ui.button("Fermer", on_click=dialog.close).classes("mt-4")
            
            dialog.open()

        except Exception as e:
            print("❌ Erreur show_transformed_data_preview:", e)
            import traceback
            traceback.print_exc()
            ui.notify(f"Erreur : {e}", color="negative")
    
    def open_transform_modal(col_name):
        """Ouvre un modal pour configurer la transformation d'une colonne"""
        if col_name not in df_train.columns:
            ui.notify(f"⚠️ Colonne {col_name} introuvable", color="warning")
            return
        
        current_method = state.get("transform_strategy", {}).get(col_name, "")
        current_params = state.get("transform_params", {}).get(col_name, {})
        
        skew, kurt = calculate_skewness_kurtosis(col_name)
        recommended, reason = get_recommended_transform(col_name)
        impacts = calculate_algo_impact(col_name, skew)
        
        data = df_train[col_name].dropna()
        has_zero = (data == 0).any()
        has_negative = (data < 0).any()
        
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-6xl").style(
            "padding:0 !important; border-radius:16px !important; "
            "box-shadow:0 10px 40px rgba(0,0,0,0.15) !important; max-height:90vh !important; "
            "overflow-y:auto !important;"
        ):
            # Header
            with ui.column().classes("w-full").style(
                "background:linear-gradient(135deg, #01335A 0%, #09538C 100%) !important; "
                "padding:24px 32px !important; border-radius:16px 16px 0 0 !important;"
            ):
                ui.label(f"Transformation : {col_name}").style(
                    "font-weight:700 !important; font-size:22px !important; color:white !important;"
                )
            
            # Contenu
            with ui.column().classes("w-full").style("padding:32px !important;"):
                # Infos distribution
                with ui.card().classes("w-full mb-4").style(
                    "background:#f8f9fa !important; padding:20px !important; border-radius:12px !important;"
                ):
                    with ui.row().classes("w-full gap-8 items-center justify-around"):
                        with ui.column().classes("items-center"):
                            ui.label("Skewness").style(
                                "font-size:13px !important; color:#636e72 !important; margin-bottom:4px !important;"
                            )
                            ui.label(f"{skew:.2f}").style(
                                "font-weight:700 !important; font-size:24px !important; color:#01335A !important;"
                            )
                            icon, level, color = get_skewness_level(skew)
                            ui.label(f"{icon} {level}").style(
                                f"color:{color} !important; font-weight:600 !important; margin-top:4px !important;"
                            )
                        
                        with ui.column().classes("items-center"):
                            ui.label("Kurtosis").style(
                                "font-size:13px !important; color:#636e72 !important; margin-bottom:4px !important;"
                            )
                            ui.label(f"{kurt:.2f}").style(
                                "font-weight:700 !important; font-size:24px !important; color:#01335A !important;"
                            )
                        
                        with ui.column().classes("items-center"):
                            ui.label("Valeurs").style(
                                "font-size:13px !important; color:#636e72 !important; margin-bottom:4px !important;"
                            )
                            if has_negative:
                                ui.label("⚠️ Négatives").style(
                                    "color:#e74c3c !important; font-weight:600 !important; font-size:16px !important;"
                                )
                            elif has_zero:
                                ui.label("⚠️ Zéros").style(
                                    "color:#01335A !important; font-weight:600 !important; font-size:16px !important;"
                                )
                            else:
                                ui.label("✅ Toutes > 0").style(
                                    "color:#27ae60 !important; font-weight:600 !important; font-size:16px !important;"
                                )
                
                # Impact algorithmes
                if selected_algos:
                    with ui.card().classes("w-full mb-4").style(
                        "background:#fff9e6 !important; padding:20px !important; "
                        "border-radius:12px !important; border-left:4px solid #01335A !important;"
                    ):
                        ui.label("Impact sur vos algorithmes :").style(
                            "font-weight:700 !important; margin-bottom:12px !important; color:#856404 !important;"
                        )
                        for algo in selected_algos:
                            # Mapper les noms
                            algo_map = {
                                "Gaussian Naive Bayes": "Naive Bayes",
                                "Decision Tree": "Decision Tree",
                                "KNN": "KNN"
                            }
                            algo_key = algo_map.get(algo, algo)
                            
                            if algo_key in impacts:
                                icon, msg, color = impacts[algo_key]
                                ui.label(f"{icon} {algo} : {msg}").style(
                                    f"font-size:14px !important; color:{color} !important; margin-bottom:4px !important;"
                                )
                
                # Recommandation
                with ui.card().classes("w-full mb-4").style(
                    "background:#e3f2fd !important; padding:20px !important; "
                    "border-radius:12px !important; border-left:4px solid #2196f3 !important;"
                ):
                    ui.label(f"💡 Recommandation : {recommended}").style(
                        "font-weight:600 !important; color:#01335A !important; font-size:16px !important;"
                    )
                    ui.label(reason).style(
                        "font-size:14px !important; color:#2c3e50 !important; margin-top:4px !important;"
                    )
                
                # Sélection transformation
                transform_select = ui.select(
                    options=[
                        "Log Transform",
                        "Square Root",
                        "Box-Cox",
                        "Yeo-Johnson",
                        "Aucune"
                    ],
                    value=current_method or recommended,
                    label="Transformation"
                ).props("outlined").classes("w-full mb-4")
                
                # Zone paramètres
                params_container = ui.column().classes("w-full mb-4")
                
                # Zone preview
                preview_container = ui.column().classes("w-full")
                
                constant_input = None
                
                def update_params_and_preview():
                    nonlocal constant_input
                    params_container.clear()
                    preview_container.clear()
                    
                    method = transform_select.value
                    
                    with params_container:
                        if method == "Log Transform":
                            ui.markdown("""
**📌 Log Transform : log(x + c)**

✅ Avantages :
- Réduit skewness fort
- Compresse valeurs extrêmes

❌ Limites :
- Nécessite valeurs > 0 (ajout constante si zéros)
- Interprétabilité réduite
                            """).style(
                                "background:#f8f9fa !important; padding:16px !important; "
                                "border-radius:8px !important; font-size:13px !important;"
                            )
                            
                            constant_input = ui.number(
                                label="Constante c (pour gérer zéros)",
                                value=current_params.get("constant", 1),
                                min=0,
                                max=10,
                                step=0.1
                            ).props("outlined").classes("w-full")
                        
                        elif method == "Square Root":
                            ui.markdown("""
**📌 Square Root : sqrt(x)**

✅ Pour skewness modéré (0.5-1.5)
- Plus douce que log
- Préserve mieux les relations
                            """).style(
                                "background:#f8f9fa !important; padding:16px !important; "
                                "border-radius:8px !important; font-size:13px !important;"
                            )
                        
                        elif method == "Box-Cox":
                            ui.markdown("""
**📌 Box-Cox Transform (automatique)**

✅ Trouve meilleur λ pour normaliser

⚠️ Nécessite valeurs strictement > 0

λ optimal sera calculé automatiquement
                            """).style(
                                "background:#f8f9fa !important; padding:16px !important; "
                                "border-radius:8px !important; font-size:13px !important;"
                            )
                            
                            if has_zero or has_negative:
                                ui.label("❌ Box-Cox impossible (zéros ou valeurs négatives)").style(
                                    "color:#e74c3c !important; font-weight:600 !important; margin-top:8px !important;"
                                )
                        
                        elif method == "Yeo-Johnson":
                            ui.markdown("""
**📌 Yeo-Johnson (Box-Cox généralisé)**

✅ Gère valeurs négatives et zéros

λ optimal sera calculé automatiquement
                            """).style(
                                "background:#f8f9fa !important; padding:16px !important; "
                                "border-radius:8px !important; font-size:13px !important;"
                            )
                        
                        elif method == "Aucune":
                            ui.markdown("""
**📌 Aucune transformation**

✅ Garder distribution originale

Recommandé si : C4.5 uniquement ou distribution déjà normale
                            """).style(
                                "background:#f8f9fa !important; padding:16px !important; "
                                "border-radius:8px !important; font-size:13px !important;"
                            )
                    
                    # Preview
                    with preview_container:
                        ui.label("Aperçu Avant/Après").style(
                            "font-weight:700 !important; font-size:18px !important; "
                            "margin-bottom:16px !important; color:#01335A !important;"
                        )
                        
                        params = {}
                        if method == "Log Transform" and constant_input:
                            params["constant"] = constant_input.value
                        
                        fig, skew_after, _ = create_distribution_plot(col_name, method, params)
                        
                        if fig:
                            ui.plotly(fig).classes("w-full")
                            
                            icon_after, level_after, color_after = get_skewness_level(skew_after)
                            ui.label(f"Skewness après transformation : {skew_after:.2f} {icon_after} {level_after}").style(
                                f"color:{color_after} !important; font-weight:600 !important; "
                                f"font-size:15px !important; margin-top:12px !important;"
                            )
                
                transform_select.on_value_change(lambda: update_params_and_preview())
                if constant_input:
                    constant_input.on_value_change(lambda: update_params_and_preview())
                
                update_params_and_preview()
                
                # Boutons
                with ui.row().classes("w-full justify-end gap-3 mt-6"):
                    ui.button("Annuler", on_click=dialog.close).props("flat").style(
                        "color:#7f8c8d !important; text-transform:none !important;"
                    )
                    
                    def save_transform():
                        method = transform_select.value
                        params = {}
                        
                        if method == "Log Transform" and constant_input:
                            params["constant"] = constant_input.value
                        
                        state.setdefault("transform_strategy", {})[col_name] = method
                        state.setdefault("transform_params", {})[col_name] = params
                        
                        ui.notify(f"✅ Transformation configurée pour {col_name}", color="positive")
                        dialog.close()
                        
                        # Rafraîchir le tableau
                        ui.run_javascript("setTimeout(() => window.location.reload(), 500);")
                    
                    ui.button("Sauvegarder", on_click=save_transform).style(
                        "background:#01335A !important; color:white !important; border-radius:8px !important; "
                        "padding:10px 24px !important; text-transform:none !important; font-weight:600 !important;"
                    )
        
        dialog.open()
    
    def apply_all_transforms():
        """Applique toutes les transformations"""
        strategies = state.get("transform_strategy", {})
        split_data = state.get("split", {})
        
        if not strategies:
            ui.notify("⚠️ Aucune transformation configurée", color="warning")
            return
        
        with ui.dialog() as dialog, ui.card().style(
            "padding:32px !important; border-radius:16px !important; "
            "box-shadow:0 10px 40px rgba(0,0,0,0.15) !important;"
        ):
            ui.label("✅ Confirmation").style(
                "font-weight:700 !important; font-size:20px !important; "
                "color:#01335A !important; margin-bottom:16px !important;"
            )
            ui.label("Appliquer les transformations sur tous les datasets ?").style(
                "margin-bottom:8px !important; color:#2c3e50 !important; font-size:14px !important;"
            )
            ui.label(f"{len(strategies)} transformation(s) configurée(s)").style(
                "color:#636e72 !important; font-size:13px !important;"
            )
            
            with ui.row().classes("w-full justify-end gap-3 mt-6"):
                ui.button("Annuler", on_click=dialog.close).props("flat").style(
                    "color:#7f8c8d !important; text-transform:none !important;"
                )
                
                def confirm_and_next():
                    try:
                        params_dict = state.get("transform_params", {})
                        transformers = {}
                        
                        # 1. FIT sur X_train uniquement
                        if split_data and "X_train" in split_data:
                            df_train_for_fit = split_data["X_train"].copy()
                        else:
                            df_train_for_fit = df_train.copy()
                        
                        # 2. FIT et sauvegarder paramètres
                        for col, method in strategies.items():
                            if method != "Aucune" and col in df_train_for_fit.columns:
                                col_params = params_dict.get(col, {})
                                transformed_data, transform_info = apply_transform(
                                    df_train_for_fit[col].values, 
                                    method, 
                                    col_params
                                )
                                if transformed_data is not None and transform_info is not None:
                                    transformers[col] = {"method": method, "params": transform_info}
                                    print(f"✅ Fitted {method} pour {col}")
                        
                        state["fitted_transformers"] = transformers
                        
                        # 3. TRANSFORM sur raw_df
                        df_result = df.copy()
                        for col, info in transformers.items():
                            if col in df_result.columns:
                                method = info["method"]
                                params = info["params"]
                                transformed_data, _ = apply_transform(df_result[col].values, method, params)
                                if transformed_data is not None:
                                    # 🔧 CORRECTION : Gérer la différence de taille
                                    if len(transformed_data) == len(df_result):
                                        df_result[col] = transformed_data
                                    else:
                                        # Remplir avec NaN les indices manquants
                                        new_col = pd.Series(index=df_result.index, dtype=float)
                                        mask = ~df_result[col].isna() & ~np.isinf(df_result[col])
                                        new_col[mask] = transformed_data
                                        df_result[col] = new_col
                        
                        state["raw_df"] = df_result
                        
                        # 4. TRANSFORM sur les splits
                        if split_data:
                            for key in ["X_train", "X_val", "X_test"]:
                                if key in split_data and isinstance(split_data[key], pd.DataFrame):
                                    df_split = split_data[key].copy()
                                    for col, info in transformers.items():
                                        if col in df_split.columns:
                                            method = info["method"]
                                            params = info["params"]
                                            transformed_data, _ = apply_transform(df_split[col].values, method, params)
                                            if transformed_data is not None:
                                                if len(transformed_data) == len(df_split):
                                                    df_split[col] = transformed_data
                                                else:
                                                    new_col = pd.Series(index=df_split.index, dtype=float)
                                                    mask = ~df_split[col].isna() & ~np.isinf(df_split[col])
                                                    new_col[mask] = transformed_data
                                                    df_split[col] = new_col
                                    split_data[key] = df_split
                            
                            state["split"] = split_data
                        
                        ui.notify("✅ Transformations appliquées avec succès!", color="positive")
                        dialog.close()
                        ui.run_javascript("setTimeout(() => window.location.href='/supervised/scaling', 1000);")
                    
                    except Exception as e:
                        ui.notify(f"❌ Erreur : {str(e)}", color="negative")
                        print(f"Erreur détaillée : {e}")
                        import traceback
                        traceback.print_exc()
                
                ui.button("Confirmer & Continuer", on_click=confirm_and_next).style(
                    "background:#01335A !important; color:white !important; border-radius:8px !important; "
                    "padding:10px 24px !important; text-transform:none !important; font-weight:600 !important;"
                )
        
        dialog.open()
    
    def apply_recommended():
        """Applique automatiquement les recommandations"""
        count = 0
        for col in num_cols:
            skew, _ = calculate_skewness_kurtosis(col)
            if abs(skew) >= 0.5:
                recommended, _ = get_recommended_transform(col)
                state.setdefault("transform_strategy", {})[col] = recommended
                
                params = {}
                if recommended == "Log Transform":
                    data = df_train[col].dropna()
                    if (data == 0).any():
                        params["constant"] = 1
                
                state.setdefault("transform_params", {})[col] = params
                count += 1
        
        ui.notify(f"✅ {count} recommandations appliquées", color="positive")
        ui.run_javascript("setTimeout(() => window.location.reload(), 500);")
    
    # ---------- UI ----------
    with ui.column().classes("w-full items-center").style(
        "background:#f0f2f5 !important; min-height:100vh !important; padding:48px 24px !important; "
        "font-family:'Inter', sans-serif !important;"
    ):
        # HEADER
        ui.label("Transformation des Distributions").style(
            "font-weight:700 !important; font-size:36px !important; color:#01335A !important; "
            "margin-bottom:8px !important; text-align:center !important; letter-spacing:-0.5px !important;"
        )
        ui.label("Normaliser les distributions asymétriques (Log, Box-Cox, Yeo-Johnson)").style(
            "font-size:16px !important; color:#636e72 !important; margin-bottom:48px !important; "
            "text-align:center !important; font-weight:400 !important;"
        )
        
        # Section : Synthèse
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white !important; border-radius:16px !important; padding:32px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            ui.label("📊 Analyse des distributions").style(
                "font-weight:700 !important; font-size:22px !important; color:#01335A !important; "
                "margin-bottom:20px !important;"
            )
            
            skewed_cols = []
            for col in num_cols:
                skew, _ = calculate_skewness_kurtosis(col)
                if abs(skew) >= 0.5:
                    skewed_cols.append(col)
            
            if len(skewed_cols) == 0:
                with ui.card().classes("w-full").style(
                    "background:#e8f5e9 !important; padding:20px !important; "
                    "border-radius:12px !important; border-left:4px solid #4caf50 !important;"
                ):
                    ui.label("✅ Aucune feature nécessitant transformation").style(
                        "color:#1b5e20 !important; font-weight:600 !important; font-size:16px !important;"
                    )
                    ui.label("Vous pouvez passer à l'étape suivante").style(
                        "color:#12344f !important; margin-top:4px !important; font-size:14px !important;"
                    )
            else:
                with ui.row().classes("w-full gap-8 items-center justify-around mb-4"):
                    with ui.column().classes("items-center"):
                        ui.label("Features analysées").style(
                            "font-size:13px !important; color:#636e72 !important; margin-bottom:4px !important;"
                        )
                        ui.label(f"{len(num_cols)}").style(
                            "font-weight:700 !important; font-size:28px !important; color:#01335A !important;"
                        )
                    
                    with ui.column().classes("items-center"):
                        ui.label("Nécessitant transformation").style(
                            "font-size:13px !important; color:#636e72 !important; margin-bottom:4px !important;"
                        )
                        ui.label(f"{len(skewed_cols)}").style(
                            "font-weight:700 !important; font-size:28px !important; color:#e74c3c !important;"
                        )
                    
                    with ui.column().classes("items-center"):
                        ui.label("Configurées").style(
                            "font-size:13px !important; color:#636e72 !important; margin-bottom:4px !important;"
                        )
                        configured_count = len(state.get("transform_strategy", {}))
                        ui.label(f"{configured_count}").style(
                            "font-weight:700 !important; font-size:28px !important; color:#27ae60 !important;"
                        )
                
                # Légende
                with ui.row().classes("w-full gap-6 mt-4").style(
                    "font-size:13px !important; justify-content:center !important;"
                ):
                    ui.label("🔴 Skew > 1.5 : Fortement recommandée").style("color:#e74c3c !important;")
                    ui.label("🟡 Skew 0.5-1.5 : Optionnelle").style("color:#01335A !important;")
                    ui.label("🟢 Skew < 0.5 : Non nécessaire").style("color:#27ae60 !important;")
        
        # Section : Table des features
        if len(num_cols) > 0:
            with ui.card().classes("w-full max-w-6xl mb-6").style(
                "background:white !important; border-radius:16px !important; padding:32px !important; "
                "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
            ):
                ui.label("🔧 Configuration des transformations").style(
                    "font-weight:700 !important; font-size:22px !important; color:#01335A !important; "
                    "margin-bottom:16px !important;"
                )
                
                rows = []
                for col in num_cols:
                    skew, kurt = calculate_skewness_kurtosis(col)
                    icon, level, _ = get_skewness_level(skew)
                    recommended, _ = get_recommended_transform(col)
                    current = state.get("transform_strategy", {}).get(col, "")
                    
                    rows.append({
                        "Feature": col,
                        "Skewness": f"{skew:.2f} {icon}",
                        "Kurtosis": f"{kurt:.2f}",
                        "Niveau": level,
                        "Recommandé": recommended,
                        "Configuré": current or "-"
                    })
                
                # Créer le tableau
                table = ui.table(
                    columns=[
                        {"name": "Feature", "label": "Feature", "field": "Feature", "align": "left"},
                        {"name": "Skewness", "label": "Skewness", "field": "Skewness", "align": "center"},
                        {"name": "Kurtosis", "label": "Kurtosis", "field": "Kurtosis", "align": "center"},
                        {"name": "Niveau", "label": "Niveau", "field": "Niveau", "align": "center"},
                        {"name": "Recommandé", "label": "Recommandé", "field": "Recommandé", "align": "center"},
                        {"name": "Configuré", "label": "Configuré", "field": "Configuré", "align": "center"}
                    ],
                    rows=rows,
                    row_key="Feature"
                ).props("flat bordered").style(
                    "width:100% !important; cursor:pointer !important;"
                )
                
                # 🔧 CORRECTION : Gestionnaire de clic robuste
                def handle_row_click(e):
                    """Gestionnaire de clic sur une ligne du tableau"""
                    try:
                        if e and hasattr(e, 'args') and len(e.args) > 1:
                            row_data = e.args[1]
                            if isinstance(row_data, dict) and "Feature" in row_data:
                                feature_name = row_data["Feature"]
                                print(f"✅ Clic détecté sur : {feature_name}")
                                open_transform_modal(feature_name)
                            else:
                                print(f"⚠️ row_data invalide: {row_data}")
                        else:
                            print(f"⚠️ Structure d'événement inattendue: {e}")
                    except Exception as err:
                        print(f"❌ Erreur handle_row_click: {err}")
                        import traceback
                        traceback.print_exc()
                
                # Attacher l'événement
                table.on('row-click', handle_row_click)
                
                ui.label("💡 Cliquez sur une ligne pour configurer la transformation").style(
                    "font-size:13px !important; color:#636e72 !important; margin-top:12px !important; "
                    "font-style:italic !important;"
                )
                
                # Recommandations par algo
                if selected_algos:
                    with ui.card().classes("w-full mt-4").style(
                        "background:#e3f2fd !important; padding:20px !important; "
                        "border-radius:12px !important; border-left:4px solid #2196f3 !important;"
                    ):
                        ui.label("💡 Impact sur vos algorithmes :").style(
                            "font-weight:700 !important; margin-bottom:12px !important; color:#01335A !important;"
                        )
                        
                        if "Gaussian Naive Bayes" in selected_algos or "Naive Bayes" in selected_algos:
                            ui.label("🔴 Naive Bayes : Transformation CRITIQUE pour features avec skew > 1").style(
                                "font-size:14px !important; color:#e74c3c !important; margin-bottom:4px !important;"
                            )
                        
                        if "KNN" in selected_algos:
                            ui.label("🟡 KNN : Transformation utile (stabilise distances)").style(
                                "font-size:14px !important; color:#f39c12 !important; margin-bottom:4px !important;"
                            )
                        
                        if "Decision Tree" in selected_algos or "C4.5" in selected_algos:
                            ui.label("🟢 Decision Tree : Transformation optionnelle (robuste aux skew)").style(
                                "font-size:14px !important; color:#27ae60 !important;"
                            )
                
                # Boutons actions
                with ui.row().classes("w-full gap-3 mt-6"):
                    ui.button(
                        "📊 Aperçu données transformées",
                        on_click=show_transformed_data_preview
                    ).style(
                        "background:#3498db !important; color:white !important; border-radius:8px !important; "
                        "padding:10px 20px !important; text-transform:none !important; font-weight:500 !important;"
                    )
                    
                    ui.button(
                        "✨ Appliquer recommandations",
                        on_click=apply_recommended
                    ).style(
                        "background:#27ae60 !important; color:white !important; border-radius:8px !important; "
                        "padding:10px 20px !important; text-transform:none !important; font-weight:500 !important;"
                    )
                    
                    ui.button(
                        "✓ Appliquer & Continuer",
                        on_click=apply_all_transforms
                    ).style(
                        "background:#01335A !important; color:white !important; border-radius:8px !important; "
                        "padding:10px 20px !important; text-transform:none !important; font-weight:600 !important;"
                    )
        
        # Navigation
        with ui.row().classes("w-full max-w-6xl justify-between gap-4 mt-8"):
            ui.button(
                " Précédent",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/encoding'")
            ).style(
                "background:white !important; color:#01335A !important; font-weight:500 !important; "
                "border-radius:8px !important; height:48px !important; min-width:140px !important; "
                "font-size:14px !important; text-transform:none !important; "
                "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
            )
            
            ui.button(
                "Suivant ",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/scaling'")
            ).style(
                "background:#01335A !important; color:white !important; font-weight:600 !important; "
                "border-radius:8px !important; height:48px !important; min-width:140px !important; "
                "font-size:14px !important; text-transform:none !important;"
            )




# ----------------- PAGE 3.9 : FEATURE SCALING -----------------


@ui.page('/supervised/scaling')
def feature_scaling_page():
    """
    Page complète pour le Feature Scaling avec détection intelligente des variables continues
    Design moderne unifié avec #01335A
    """
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
    
    # ---------- CONTEXTE ----------
    df = state.get("raw_df", None)
    split = state.get("split", None)
    columns_exclude = state.get("columns_exclude", {}) or {}
    target_col = state.get("target_column", None)
    selected_algos = state.get("selected_algos", [])
    
    if df is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Aucun dataset chargé.").style(
                "font-size:18px !important; color:#c0392b !important; font-weight:600 !important;"
            )
            ui.button(
                "🔙 Retour à l'Upload",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/upload'")
            ).style(
                "background:#01335A !important; color:white !important; padding:12px 32px !important; "
                "border-radius:8px !important; margin-top:20px !important;"
            )
        return
    
    df_train = df.copy()
    active_cols = [c for c in df.columns if not columns_exclude.get(c, False) and c != target_col]
    
    # ---------- DÉTECTION INTELLIGENTE DES VARIABLES CONTINUES ----------
    def detect_continuous_features(df, columns):
        """
        Détecte les vraies variables continues (exclut les catégorielles encodées)
        """
        continuous = []
        categorical_encoded = []
        
        for col in columns:
            try:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                
                unique_values = df[col].nunique()
                data = df[col].dropna()
                
                if len(data) == 0:
                    continue
                
                # Critères pour identifier une variable catégorielle encodée
                is_binary = unique_values == 2
                is_small_discrete = unique_values <= 10
                
                # Vérifier si toutes les valeurs sont des entiers
                try:
                    all_integers = bool((data == data.astype(int)).all())
                except:
                    all_integers = False
                
                # Calculer la range de manière sûre
                try:
                    data_range = float(data.max()) - float(data.min())
                    small_range = data_range < 10
                except:
                    small_range = False
                
                # Si c'est binaire (0/1) ou petit nombre de valeurs discrètes  catégorielle
                if is_binary or (is_small_discrete and all_integers and small_range):
                    categorical_encoded.append(col)
                else:
                    continuous.append(col)
            
            except Exception as e:
                print(f"⚠️ Erreur lors de l'analyse de {col}: {e}")
                # En cas d'erreur, considérer comme continue par défaut
                continuous.append(col)
        
        return continuous, categorical_encoded
    num_cols, cat_encoded = detect_continuous_features(df_train, active_cols)
    
    # ---------- FONCTIONS ----------
    def detect_outliers(col):
        try:
            data = df_train[col].dropna()
            if len(data) < 4:
                return False
            Q1, Q3 = data.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outliers = ((data < (Q1 - 1.5*IQR)) | (data > (Q3 + 1.5*IQR))).sum()
            return outliers > len(data) * 0.05
        except:
            return False
    
    def analyze_features():
        features = []
        for col in num_cols[:10]:
            try:
                data = df_train[col].dropna()
                if len(data) == 0:
                    continue
                features.append({
                    "name": str(col),
                    "range": float(data.max() - data.min()),
                    "mean": float(data.mean()),
                    "std": float(data.std()),
                    "min": float(data.min()),
                    "max": float(data.max()),
                    "outliers": detect_outliers(col)
                })
            except:
                continue
        return features
    
    def get_recommendation():
        features = analyze_features()
        if not features:
            return "none"
        
        has_hetero = any(f["range"] > 1000 for f in features)
        has_outliers = any(f["outliers"] for f in features)
        
        if not has_hetero:
            return "none"
        if has_outliers:
            return "robust"
        return "standard"
    
    def create_comparison_plot(method):
        """Crée un graphique comparatif AVANT/APRÈS scaling"""
        if method == "none" or not num_cols:
            return None
        
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "maxabs": MaxAbsScaler()
        }
        
        if method not in scalers:
            return None
        
        # Sélectionner jusqu'à 4 features continues pour visualisation
        cols_to_show = num_cols[:4]
        
        if not cols_to_show:
            return None
        
        # Créer subplots
        fig = make_subplots(
            rows=2, cols=len(cols_to_show),
            subplot_titles=[f"{col}" for col in cols_to_show] + [f"{col} (Scaled)" for col in cols_to_show],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        scaler = scalers[method]
        
        for idx, col in enumerate(cols_to_show):
            try:
                # Données originales
                original_data = df_train[col].dropna()
                
                # Données scalées
                scaled_data = scaler.fit_transform(df_train[[col]]).flatten()
                
                # AVANT (ligne 1)
                fig.add_trace(
                    go.Box(
                        y=original_data,
                        name="Original",
                        marker_color='#01335A',
                        showlegend=(idx == 0)
                    ),
                    row=1, col=idx+1
                )
                
                # APRÈS (ligne 2)
                fig.add_trace(
                    go.Box(
                        y=scaled_data,
                        name="Scaled",
                        marker_color='#27ae60',
                        showlegend=(idx == 0)
                    ),
                    row=2, col=idx+1
                )
                
            except Exception as e:
                print(f"❌ Erreur visualisation {col}: {e}")
                continue
        
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text=f"<b>Comparaison AVANT/APRÈS - {method.upper()}</b>",
            title_font_size=20,
            title_x=0.5,
            paper_bgcolor='white',
            plot_bgcolor='#f8f9fa',
            font=dict(family="Inter, sans-serif", color="#2c3e50")
        )
        
        return fig
    
    # ---------- ANALYSE ----------
    features_info = analyze_features()
    recommended = get_recommendation()
    
    # ---------- INTERFACE ----------
    with ui.column().classes("w-full items-center").style(
        "background:#f0f2f5 !important; min-height:100vh !important; padding:48px 24px !important; "
        "font-family:'Inter', sans-serif !important;"
    ):
        # HEADER
        ui.label("Feature Scaling").style(
            "font-weight:700 !important; font-size:36px !important; color:#01335A !important; "
            "margin-bottom:8px !important; text-align:center !important; letter-spacing:-0.5px !important;"
        )
        ui.label("Normalisation et standardisation des features continues").style(
            "font-size:16px !important; color:#636e72 !important; margin-bottom:48px !important; "
            "text-align:center !important; font-weight:400 !important;"
        )
        
        # Info détection intelligente
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white !important; border-radius:16px !important; padding:32px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            ui.label("🔍 Détection Intelligente").style(
                "font-weight:700 !important; font-size:22px !important; color:#01335A !important; "
                "margin-bottom:20px !important;"
            )
            with ui.row().classes("w-full gap-8 items-start"):
                with ui.column().classes("flex-1"):
                    ui.label(f"✅ {len(num_cols)} features continues détectées").style(
                        "font-size:15px !important; margin-bottom:8px !important; color:#2c3e50 !important; "
                        "font-weight:600 !important;"
                    )
                    if num_cols:
                        ui.label(f" {', '.join(num_cols[:5])}{'...' if len(num_cols) > 5 else ''}").style(
                            "font-size:13px !important; color:#636e72 !important;"
                        )
                
                with ui.column().classes("flex-1"):
                    ui.label(f"📊 {len(cat_encoded)} variables catégorielles encodées").style(
                        "font-size:15px !important; margin-bottom:8px !important; color:#2c3e50 !important; "
                        "font-weight:600 !important;"
                    )
                    if cat_encoded:
                        ui.label(f" {', '.join(cat_encoded[:5])}{'...' if len(cat_encoded) > 5 else ''}").style(
                            "font-size:13px !important; color:#636e72 !important;"
                        )
                        ui.label(" Seront préservées (pas de scaling)").style(
                            "font-size:12px !important; color:#7f8c8d !important; font-style:italic !important; "
                            "margin-top:4px !important;"
                        )
        
        # Objectif par algorithme
        if selected_algos:
            with ui.card().classes("w-full max-w-6xl mb-6").style(
                "background:#fff9e6 !important; padding:32px !important; "
                "border-radius:16px !important; border-left:4px solid #f39c12 !important; "
                "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
            ):
                ui.label("💡 Impact sur vos algorithmes").style(
                    "font-weight:700 !important; font-size:22px !important; color:#856404 !important; "
                    "margin-bottom:20px !important;"
                )
                
                if "KNN" in selected_algos:
                    ui.label("🔴 KNN : Scaling CRITIQUE").style(
                        "font-size:15px !important; font-weight:600 !important; margin-bottom:4px !important; "
                        "color:#e74c3c !important;"
                    )
                    ui.label(" Les distances sont biaisées par les échelles différentes").style(
                        "font-size:13px !important; color:#636e72 !important; margin-bottom:12px !important;"
                    )
                
                if "Decision Tree" in selected_algos or "C4.5" in selected_algos:
                    ui.label("🟢 Decision Tree : Scaling INUTILE").style(
                        "font-size:15px !important; font-weight:600 !important; margin-bottom:4px !important; "
                        "color:#27ae60 !important;"
                    )
                    ui.label(" Arbres de décision insensibles aux échelles").style(
                        "font-size:13px !important; color:#636e72 !important; margin-bottom:12px !important;"
                    )
                
                if "Gaussian Naive Bayes" in selected_algos or "Naive Bayes" in selected_algos:
                    ui.label("🟡 Naive Bayes : Scaling OPTIONNEL").style(
                        "font-size:15px !important; font-weight:600 !important; margin-bottom:4px !important; "
                        "color:#f39c12 !important;"
                    )
                    ui.label(" Dépend de la distribution des données").style(
                        "font-size:13px !important; color:#636e72 !important;"
                    )
        
        # Analyse des features continues
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white !important; border-radius:16px !important; padding:32px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            ui.label("📊 Analyse des features continues").style(
                "font-weight:700 !important; font-size:22px !important; color:#01335A !important; "
                "margin-bottom:20px !important;"
            )
            
            if features_info:
                # Créer un tableau avec les données
                rows = []
                for f in features_info:
                    rows.append({
                        "Feature": f["name"][:20],
                        "Min": f"{f['min']:.2f}",
                        "Max": f"{f['max']:.2f}",
                        "Range": f"{f['range']:.2f}",
                        "Mean": f"{f['mean']:.2f}",
                        "Std": f"{f['std']:.2f}",
                        "Outliers": "🔴" if f["outliers"] else "🟢"
                    })
                
                table = ui.table(
                    columns=[
                        {"name": "Feature", "label": "Feature", "field": "Feature", "align": "left"},
                        {"name": "Min", "label": "Min", "field": "Min", "align": "right"},
                        {"name": "Max", "label": "Max", "field": "Max", "align": "right"},
                        {"name": "Range", "label": "Range", "field": "Range", "align": "right"},
                        {"name": "Mean", "label": "Mean", "field": "Mean", "align": "right"},
                        {"name": "Std", "label": "Std", "field": "Std", "align": "right"},
                        {"name": "Outliers", "label": "Outliers", "field": "Outliers", "align": "center"}
                    ],
                    rows=rows,
                    row_key="Feature"
                ).props("flat bordered").style("width:100% !important;")
                
                ui.label("💡 Les features avec Range > 1000 bénéficieront fortement du scaling").style(
                    "font-size:13px !important; color:#636e72 !important; margin-top:12px !important; "
                    "font-style:italic !important;"
                )
            else:
                with ui.card().classes("w-full").style(
                    "background:#e8f5e9 !important; padding:20px !important; "
                    "border-radius:12px !important; border-left:4px solid #4caf50 !important;"
                ):
                    ui.label("✅ Aucune feature continue détectée").style(
                        "color:#1b5e20 !important; font-weight:600 !important; font-size:16px !important;"
                    )
                    ui.label("Toutes les features sont catégorielles ou binaires").style(
                        "color:#12344f !important; margin-top:4px !important; font-size:14px !important;"
                    )
        
        # Sélecteur de méthode avec recommandation
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white !important; border-radius:16px !important; padding:32px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            ui.label("🔧 Configuration du scaling").style(
                "font-weight:700 !important; font-size:22px !important; color:#01335A !important; "
                "margin-bottom:20px !important;"
            )
            
            # Recommandation
            method_names = {
                "standard": "StandardScaler (Z-score)",
                "minmax": "MinMaxScaler ([0,1])",
                "robust": "RobustScaler (résistant aux outliers)",
                "maxabs": "MaxAbsScaler ([-1,1])",
                "none": "Aucun scaling"
            }
            
            method_explanations = {
                "standard": "✅ Standardise avec mean=0 et std=1\n✅ Optimal pour la plupart des algorithmes\n❌ Sensible aux outliers",
                "minmax": "✅ Normalise dans l'intervalle [0,1]\n✅ Préserve les relations\n❌ Très sensible aux outliers",
                "robust": "✅ Utilise médiane et IQR\n✅ Résistant aux valeurs extrêmes\n✅ Recommandé si outliers détectés",
                "maxabs": "✅ Normalise dans [-1,1]\n✅ Pour données sparse\n❌ Sensible aux outliers",
                "none": "✅ Conserve les données brutes\n✅ Si échelles déjà similaires"
            }
            
            with ui.card().classes("w-full mb-4").style(
                "background:#e3f2fd !important; padding:20px !important; "
                "border-radius:12px !important; border-left:4px solid #2196f3 !important;"
            ):
                ui.label(f"💡 Recommandation : {method_names[recommended]}").style(
                    "font-weight:600 !important; color:#01335A !important; font-size:16px !important; "
                    "margin-bottom:8px !important;"
                )
                
                # Explication de la recommandation
                if recommended == "robust":
                    outlier_count = len([f for f in features_info if f['outliers']])
                    ui.label(f"🔴 {outlier_count} feature(s) avec outliers détectés").style(
                        "font-size:14px !important; color:#2c3e50 !important; margin-bottom:4px !important;"
                    )
                elif recommended == "standard":
                    ui.label("✅ Pas d'outliers majeurs détectés").style(
                        "font-size:14px !important; color:#2c3e50 !important; margin-bottom:4px !important;"
                    )
                elif recommended == "none":
                    ui.label("✅ Échelles déjà similaires").style(
                        "font-size:14px !important; color:#2c3e50 !important; margin-bottom:4px !important;"
                    )
            
            # Sélecteur
            method_select = ui.select(
                label="Sélectionnez une méthode",
                options={
                    "standard": "📊 StandardScaler (Z-score, mean=0, std=1)",
                    "minmax": "📏 MinMaxScaler (normalise [0,1])",
                    "robust": "🛡️ RobustScaler (robuste aux outliers)",
                    "maxabs": "📐 MaxAbsScaler (sparse data, [-1,1])",
                    "none": "⏭️ Aucun scaling (raw data)"
                },
                value=recommended
            ).props("outlined").classes("w-full mb-4")
            
            # Zone d'explication dynamique
            explanation_container = ui.column().classes("w-full mb-4")
            
            def update_explanation():
                explanation_container.clear()
                method = method_select.value
                with explanation_container:
                    with ui.card().classes("w-full").style(
                        "background:#f8f9fa !important; padding:16px !important; "
                        "border-radius:8px !important;"
                    ):
                        ui.markdown(f"**{method_names.get(method, '')}**\n\n{method_explanations.get(method, '')}").style(
                            "font-size:13px !important; color:#2c3e50 !important;"
                        )
            
            method_select.on_value_change(lambda: update_explanation())
            update_explanation()
            
            # Zone de visualisation
            plot_container = ui.column().classes("w-full")
            
            def preview_scaling():
                """Prévisualise le scaling sans l'appliquer"""
                method = method_select.value
                fig = create_comparison_plot(method)
                plot_container.clear()
                with plot_container:
                    if fig:
                        ui.plotly(fig).classes("w-full")
                    else:
                        ui.label("⚠️ Aucune visualisation disponible pour cette méthode").style(
                            "color:#636e72 !important; font-size:14px !important; text-align:center !important;"
                        )
            
            # Bouton prévisualisation
            ui.button(
                "👁️ Prévisualiser",
                on_click=preview_scaling
            ).style(
                "background:#3498db !important; color:white !important; border-radius:8px !important; "
                "padding:10px 24px !important; text-transform:none !important; font-weight:600 !important; "
                "width:100% !important;"
            )
        
        # Boutons d'action
        with ui.row().classes("w-full max-w-6xl gap-3 justify-center mt-8 mb-8"):
            def apply_scaling():
                """Applique le scaling sélectionné"""
                method = method_select.value
                
                if method == "none":
                    state["scaling_method"] = "none"
                    ui.notify("✅ Aucun scaling appliqué", color="positive")
                    ui.run_javascript("setTimeout(() => window.location.href='/supervised/dimension_reduction', 1000);")
                    return
                
                with ui.dialog() as dialog, ui.card().style(
                    "padding:32px !important; border-radius:16px !important; "
                    "box-shadow:0 10px 40px rgba(0,0,0,0.15) !important;"
                ):
                    ui.label("✅ Confirmation").style(
                        "font-weight:700 !important; font-size:20px !important; "
                        "color:#01335A !important; margin-bottom:16px !important;"
                    )
                    ui.label(f"Appliquer {method_names[method]} sur {len(num_cols)} features continues ?").style(
                        "margin-bottom:8px !important; color:#2c3e50 !important; font-size:14px !important;"
                    )
                    
                    with ui.row().classes("w-full justify-end gap-3 mt-6"):
                        ui.button("Annuler", on_click=dialog.close).props("flat").style(
                            "color:#7f8c8d !important; text-transform:none !important;"
                        )
                        
                        def confirm_and_next():
                            try:
                                scalers = {
                                    "standard": StandardScaler(),
                                    "minmax": MinMaxScaler(),
                                    "robust": RobustScaler(),
                                    "maxabs": MaxAbsScaler()
                                }
                                
                                scaler = scalers[method]
                                
                                # FIT sur X_train uniquement
                                if split and "X_train" in split:
                                    X_train_for_fit = split["X_train"][num_cols].copy()
                                else:
                                    X_train_for_fit = df_train[num_cols].copy()
                                
                                # FIT sur train
                                scaler.fit(X_train_for_fit)
                                
                                # TRANSFORM sur raw_df
                                df[num_cols] = scaler.transform(df[num_cols])
                                state["raw_df"] = df
                                
                                # TRANSFORM sur chaque split
                                if split:
                                    for key in ["X_train", "X_val", "X_test"]:
                                        if key in split and isinstance(split[key], pd.DataFrame):
                                            split[key][num_cols] = scaler.transform(split[key][num_cols])
                                    state["split"] = split
                                
                                state["scaling_method"] = method
                                state["fitted_scaler"] = scaler
                                state["scaled_columns"] = num_cols
                                
                                ui.notify("✅ Scaling appliqué avec succès!", color="positive")
                                dialog.close()
                                ui.run_javascript("setTimeout(() => window.location.href='/supervised/dimension_reduction', 1000);")
                            
                            except Exception as e:
                                ui.notify(f"❌ Erreur : {str(e)}", color="negative")
                                print(f"Erreur détaillée : {e}")
                                import traceback
                                traceback.print_exc()
                        
                        ui.button("Confirmer & Continuer", on_click=confirm_and_next).style(
                            "background:#01335A !important; color:white !important; border-radius:8px !important; "
                            "padding:10px 24px !important; text-transform:none !important; font-weight:600 !important;"
                        )
                
                dialog.open()
            
            ui.button(
                "✨ Appliquer recommandation",
                on_click=lambda: (method_select.set_value(recommended), apply_scaling())
            ).style(
                "background:#27ae60 !important; color:white !important; border-radius:8px !important; "
                "padding:12px 32px !important; text-transform:none !important; font-weight:600 !important; "
                "min-width:200px !important;"
            )
            
            ui.button(
                "✓ Appliquer & Continuer",
                on_click=apply_scaling
            ).style(
                "background:#01335A !important; color:white !important; border-radius:8px !important; "
                "padding:12px 32px !important; text-transform:none !important; font-weight:600 !important; "
                "min-width:200px !important;"
            )
        
        # Navigation
        with ui.row().classes("w-full max-w-6xl justify-between gap-4 mt-8"):
            ui.button(
                " Précédent",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/distribution_transform'")
            ).style(
                "background:white !important; color:#01335A !important; font-weight:500 !important; "
                "border-radius:8px !important; height:48px !important; min-width:140px !important; "
                "font-size:14px !important; text-transform:none !important; "
                "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
            )
            
            ui.button(
                "Suivant ",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/dimension_reduction'")
            ).style(
                "background:#01335A !important; color:white !important; font-weight:600 !important; "
                "border-radius:8px !important; height:48px !important; min-width:140px !important; "
                "font-size:14px !important; text-transform:none !important;"
            )




@ui.page('/supervised/dimension_reduction')
def dimension_reduction_page():
    """
    Page complète pour la Réduction de Dimension (PCA, Feature Selection)
    """
    
    
    # ---------- CONTEXTE ----------
    df = state.get("raw_df", None)
    if df is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Aucun dataset chargé.").style("font-size:18px; color:#c0392b; font-weight:600;")
            ui.button(" Retour à l'Upload", on_click=lambda: ui.run_javascript("window.location.href='/supervised/upload'"))
        return
    
    target_col = state.get("target_column", None)
    columns_exclude = state.get("columns_exclude", {}) or {}
    
    if not target_col:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Aucune variable cible définie.").style("font-size:18px; color:#c0392b; font-weight:600;")
        return
    
    # Préparer X et y
    active_cols = [c for c in df.columns if not columns_exclude.get(c, False) and c != target_col]
    X = df[active_cols].copy()
    y = df[target_col].copy()
    
    n_samples = len(X)
    n_features = len(active_cols)
    ratio = n_samples / n_features if n_features > 0 else 0
    
    # État de la réduction
    reduction_enabled = state.get("reduction_enabled", False)
    reduction_method = state.get("reduction_method", "pca")
    n_components = state.get("n_components", min(10, n_features))
    
    # ---------- FONCTIONS ----------
    def calculate_pca_variance():
        """Calcule variance expliquée par PCA"""
        try:
            pca = PCA()
            pca.fit(X)
            variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(variance_ratio)
            return variance_ratio, cumulative_variance
        except Exception as e:
            print(f"Erreur PCA: {e}")
            return None, None
    
    def create_scree_plot():
        """Crée le scree plot pour PCA"""
        variance_ratio, cumulative_variance = calculate_pca_variance()
        
        if variance_ratio is None:
            return None
        
        n_comp = min(len(variance_ratio), 20)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Variance expliquée par composante", "Variance cumulée"),
            horizontal_spacing=0.12
        )
        
        # Variance individuelle
        fig.add_trace(
            go.Bar(
                x=list(range(1, n_comp + 1)),
                y=variance_ratio[:n_comp] * 100,
                marker_color='#01335A',
                name='Variance individuelle'
            ),
            row=1, col=1
        )
        
        # Variance cumulée
        fig.add_trace(
            go.Scatter(
                x=list(range(1, n_comp + 1)),
                y=cumulative_variance[:n_comp] * 100,
                mode='lines+markers',
                marker=dict(size=8, color='#e74c3c'),
                line=dict(width=3, color='#e74c3c'),
                name='Variance cumulée'
            ),
            row=1, col=2
        )
        
        # Ligne 95%
        fig.add_hline(y=95, line_dash="dash", line_color="green", 
                     annotation_text="95%", row=1, col=2)
        
        fig.update_xaxes(title_text="Composante", row=1, col=1)
        fig.update_xaxes(title_text="Nombre de composantes", row=1, col=2)
        fig.update_yaxes(title_text="Variance (%)", row=1, col=1)
        fig.update_yaxes(title_text="Variance cumulée (%)", row=1, col=2)
        
        fig.update_layout(
            height=400,
            showlegend=False,
            paper_bgcolor='#f8f9fa',
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_pca_2d_plot():
        """Visualisation 2D des 2 premières composantes PCA"""
        try:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            fig = go.Figure()
            
            # Pour chaque classe
            for class_val in y.unique():
                mask = y == class_val
                fig.add_trace(go.Scatter(
                    x=X_pca[mask, 0],
                    y=X_pca[mask, 1],
                    mode='markers',
                    name=f'Classe {class_val}',
                    marker=dict(size=8, opacity=0.7)
                ))
            
            fig.update_layout(
                title="Visualisation PCA 2D (PC1 vs PC2)",
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                height=500,
                paper_bgcolor='#f8f9fa',
                plot_bgcolor='white'
            )
            
            return fig
        except Exception as e:
            print(f"Erreur visualisation PCA: {e}")
            return None
    
    def get_feature_importance():
        """Calcule importance des features via Decision Tree"""
        try:
            clf = DecisionTreeClassifier(max_depth=5, random_state=42)
            clf.fit(X, y)
            importances = clf.feature_importances_
            
            feature_imp = pd.DataFrame({
                'feature': X.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return feature_imp
        except Exception as e:
            print(f"Erreur feature importance: {e}")
            return None
    
    def apply_reduction(method, n_comp):
        """Applique la réduction de dimension"""
        try:
            if method == "pca":
                pca = PCA(n_components=n_comp)
                X_reduced = pca.fit_transform(X)
                
                # Créer DataFrame avec noms de colonnes
                col_names = [f"PC{i+1}" for i in range(n_comp)]
                df_reduced = pd.DataFrame(X_reduced, columns=col_names, index=X.index)
                df_reduced[target_col] = y.values
                
                state["raw_df"] = df_reduced
                state["pca_model"] = pca
                state["original_features"] = X.columns.tolist()
                state["reduction_applied"] = True
                state["reduction_method"] = "pca"
                
                variance_explained = pca.explained_variance_ratio_.sum() * 100
                
                ui.notify(f" PCA appliqué : {n_features}  {n_comp} composantes ({variance_explained:.1f}% variance)", 
                         color="positive", timeout=3000)
                
            elif method == "feature_selection":
                feature_imp = get_feature_importance()
                if feature_imp is None:
                    ui.notify("❌ Erreur lors du calcul d'importance", color="negative")
                    return
                
                top_features = feature_imp.head(n_comp)['feature'].tolist()
                
                df_reduced = X[top_features].copy()
                df_reduced[target_col] = y.values
                
                state["raw_df"] = df_reduced
                state["selected_features"] = top_features
                state["original_features"] = X.columns.tolist()
                state["reduction_applied"] = True
                state["reduction_method"] = "feature_selection"
                
                ui.notify(f" Feature Selection : {n_features}  {n_comp} features", 
                         color="positive", timeout=3000)
            
            # Mettre à jour colonnes exclues
            state["columns_exclude"] = {}
            
            ui.run_javascript("setTimeout(() => window.location.href='/supervised/recap_validation', 1500);")
            
        except Exception as e:
            ui.notify(f"❌ Erreur : {str(e)}", color="negative")
            print(f"Erreur reduction: {e}")
    
    def skip_reduction():
        """Passer l'étape sans réduction"""
        state["reduction_applied"] = False
        ui.notify(" Aucune réduction appliquée", color="info")
        ui.run_javascript("setTimeout(() => window.location.href='/supervised/recap_validation', 1000);")
    
    # ---------- INTERFACE ----------
    with ui.column().classes("w-full items-center p-8").style("background-color:#f8f9fa; min-height:100vh;"):
        
        # Header
        ui.label(" RÉDUCTION DE DIMENSION").style(
            "font-weight:700; font-size:32px; color:#2c3e50; margin-bottom:8px; text-align:center;"
        )
        ui.label("Optionnel - Réduire curse of dimensionality pour KNN").style(
            "font-size:18px; color:#7f8c8d; margin-bottom:32px; text-align:center;"
        )
        
        # Section A : Évaluation
        with ui.card().classes("w-full max-w-6xl p-6 mb-8"):
            ui.label(" État Actuel du Dataset").style(
                "font-weight:700; font-size:20px; color:#2c3e50; margin-bottom:16px;"
            )
            
            with ui.row().classes("w-full gap-8 mb-6"):
                with ui.column().classes("flex-1"):
                    ui.label(f"**Features après preprocessing** : {n_features}").style("font-size:15px; margin-bottom:8px;")
                    ui.label(f"**Samples (train)** : {n_samples}").style("font-size:15px; margin-bottom:8px;")
                    ui.label(f"**Ratio samples/features** : {ratio:.1f}").style(
                        f"font-size:15px; font-weight:700; color:{'#01335A' if ratio > 50 else '#01335A'};"
                    )
            
            # Recommandations
            with ui.card().classes("w-full p-4").style("background:#cde4ff; border-left:4px solid #01335A;"):
                ui.label(" Recommandations par algorithme").style("font-weight:700; font-size:16px; margin-bottom:12px;")
                ui.label(f"• **KNN** : Ratio idéal > 50 {'' if ratio > 50 else ' Curse of dimensionality'}").style("font-size:14px; margin-bottom:6px;")
                ui.label("• **C4.5** : Pas de limite stricte ").style("font-size:14px; margin-bottom:6px;")
                ui.label("• **Naive Bayes** : Assume indépendance (déjà traité) ").style("font-size:14px;")
            
            if ratio < 50:
                ui.label("💡 Réduction recommandée pour améliorer KNN").style(
                    "font-size:16px; color:#01335A; font-weight:600; margin-top:12px; text-align:center;"
                )
        
        # Section B : Options
        with ui.card().classes("w-full max-w-6xl p-8 mb-8"):
            ui.label(" Configuration de la Réduction").style(
                "font-weight:700; font-size:20px; color:#2c3e50; margin-bottom:16px;"
            )
            
            # Toggle activation
            enable_switch = ui.switch("Activer Réduction de Dimension", value=reduction_enabled).style(
                "font-size:16px; font-weight:600; margin-bottom:24px;"
            )
            
            enable_switch.on_value_change(lambda e: state.update({"reduction_enabled": e.value}))
            
            # Container conditionnel
            options_container = ui.column().classes("w-full")
            
            def update_options_visibility():
                options_container.clear()
                if enable_switch.value:
                    with options_container:
                        # Choix de méthode
                        method_radio = ui.radio(
                            ["PCA (Principal Component Analysis)", "Feature Selection (Tree Importance)"],
                            value="PCA (Principal Component Analysis)"
                        ).props("inline").classes("mb-6")
                        
                        method_map = {
                            "PCA (Principal Component Analysis)": "pca",
                            "Feature Selection (Tree Importance)": "feature_selection"
                        }
                        
                        method_radio.on_value_change(lambda e: state.update({"reduction_method": method_map[e.value]}))
                        
                        # Container pour config spécifique
                        config_container = ui.column().classes("w-full")
                        
                        def update_config():
                            config_container.clear()
                            current_method = method_map[method_radio.value]
                            
                            with config_container:
                                if current_method == "pca":
                                    # Configuration PCA
                                    with ui.card().classes("w-full p-6 mb-6").style("background:#e3f2fd; border-left:4px solid #2196f3;"):
                                        ui.label("📌 PCA - Analyse en Composantes Principales").style(
                                            "font-weight:700; font-size:18px; color:#1976d2; margin-bottom:12px;"
                                        )
                                        
                                        ui.label("**Principe** : Transformer features en composantes non-corrélées").style(
                                            "font-size:14px; margin-bottom:8px;"
                                        )
                                        
                                        with ui.row().classes("w-full gap-8 mb-4"):
                                            with ui.column().classes("flex-1"):
                                                ui.label(" **Avantages**").style("font-weight:600; color:#01335A; margin-bottom:4px;")
                                                ui.label("• Linéaire, rapide, déterministe").style("font-size:13px;")
                                                ui.label("• Préserve variance maximale").style("font-size:13px;")
                                                ui.label("• Transformable pour new data").style("font-size:13px;")
                                            
                                            with ui.column().classes("flex-1"):
                                                ui.label("❌ **Inconvénients**").style("font-weight:600; color:#e74c3c; margin-bottom:4px;")
                                                ui.label("• Perd interprétabilité").style("font-size:13px;")
                                                ui.label("• Assume linéarité").style("font-size:13px;")
                                                ui.label("• Sensible au scaling (déjà fait )").style("font-size:13px;")
                                        
                                        # Slider nombre de composantes
                                        ui.label("**Configuration**").style("font-weight:600; font-size:16px; margin:16px 0 12px 0;")
                                        
                                        max_comp = min(n_features, n_samples)
                                        n_comp_slider = ui.slider(
                                            min=1, max=max_comp, value=min(10, max_comp), step=1
                                        ).props("label-always").classes("w-full mb-4")
                                        n_comp_label = ui.label(f"Nombre de composantes : {n_comp_slider.value}").style(
                                            "font-weight:600; margin-bottom:16px;"
                                        )
                                        
                                        n_comp_slider.on_value_change(
                                            lambda e: (
                                                n_comp_label.set_text(f"Nombre de composantes : {int(e.value)}"),
                                                state.update({"n_components": int(e.value)})
                                            )
                                        )
                                    
                                    # Scree Plot
                                    ui.label(" Analyse de variance expliquée").style(
                                        "font-weight:700; font-size:18px; color:#2c3e50; margin-bottom:12px;"
                                    )
                                    
                                    scree_plot = create_scree_plot()
                                    if scree_plot:
                                        ui.plotly(scree_plot).classes("w-full mb-6")
                                    
                                    # Recommandation variance
                                    variance_ratio, cumulative_variance = calculate_pca_variance()
                                    if variance_ratio is not None:
                                        # Trouver n pour 95% variance
                                        n_for_95 = np.argmax(cumulative_variance >= 0.95) + 1
                                        
                                        with ui.card().classes("w-full p-4 mb-6").style("background:#e8f5e8; border-left:4px solid #1c365c;"):
                                            ui.label(f"💡 **Recommandation** : {n_for_95} composantes (≥95% variance)").style(
                                                "font-size:15px; font-weight:600; color:#12344f;"
                                            )
                                            ui.label(f"Impact : {n_features}  {n_for_95} features (réduction {(1-n_for_95/n_features)*100:.0f}%)").style(
                                                "font-size:14px; color:#12344f; margin-top:4px;"
                                            )
                                    
                                    # Visualisation 2D
                                    ui.label("🎨 Prévisualisation PCA 2D").style(
                                        "font-weight:700; font-size:18px; color:#2c3e50; margin:16px 0 12px 0;"
                                    )
                                    
                                    pca_2d_plot = create_pca_2d_plot()
                                    if pca_2d_plot:
                                        ui.plotly(pca_2d_plot).classes("w-full")
                                
                                elif current_method == "feature_selection":
                                    # Configuration Feature Selection
                                    with ui.card().classes("w-full p-6 mb-6").style("background:#cde4ff; border-left:4px solid #ffc107;"):
                                        ui.label("📌 Sélection de Features (sans transformation)").style(
                                            "font-weight:700; font-size:18px; color:#856404; margin-bottom:12px;"
                                        )
                                        
                                        ui.label("**Principe** : Garder uniquement les K features les plus importantes").style(
                                            "font-size:14px; margin-bottom:8px;"
                                        )
                                        
                                        with ui.row().classes("w-full gap-8 mb-4"):
                                            with ui.column().classes("flex-1"):
                                                ui.label(" **Avantages**").style("font-weight:600; color:#01335A; margin-bottom:4px;")
                                                ui.label("• Garde interprétabilité").style("font-size:13px;")
                                                ui.label("• Pas de transformation").style("font-size:13px;")
                                                ui.label("• Features originales").style("font-size:13px;")
                                            
                                            with ui.column().classes("flex-1"):
                                                ui.label("❌ **Inconvénients**").style("font-weight:600; color:#e74c3c; margin-bottom:4px;")
                                                ui.label("• Peut perdre info combinée").style("font-size:13px;")
                                                ui.label("• Dépend du modèle d'évaluation").style("font-size:13px;")
                                        
                                        # Slider top K
                                        ui.label("**Configuration**").style("font-weight:600; font-size:16px; margin:16px 0 12px 0;")
                                        
                                        k_slider = ui.slider(
                                            min=1, max=n_features, value=min(15, n_features), step=1
                                        ).props("label-always").classes("w-full mb-4")
                                        k_label = ui.label(f"Top {k_slider.value} features").style(
                                            "font-weight:600; margin-bottom:16px;"
                                        )
                                        
                                        k_slider.on_value_change(
                                            lambda e: (
                                                k_label.set_text(f"Top {int(e.value)} features"),
                                                state.update({"n_components": int(e.value)})
                                            )
                                        )
                                    
                                    # Affichage importance
                                    feature_imp = get_feature_importance()
                                    if feature_imp is not None:
                                        ui.label(" Features par importance (Decision Tree)").style(
                                            "font-weight:700; font-size:18px; color:#2c3e50; margin-bottom:12px;"
                                        )
                                        
                                        # Bar chart importance
                                        top_15 = feature_imp.head(15)
                                        fig = go.Figure(go.Bar(
                                            x=top_15['importance'],
                                            y=top_15['feature'],
                                            orientation='h',
                                            marker_color='#01335A'
                                        ))
                                        fig.update_layout(
                                            height=400,
                                            xaxis_title="Importance",
                                            yaxis_title="Feature",
                                            yaxis={'categoryorder':'total ascending'},
                                            paper_bgcolor='#f8f9fa',
                                            plot_bgcolor='white'
                                        )
                                        ui.plotly(fig).classes("w-full")
                        
                        method_radio.on_value_change(lambda: update_config())
                        update_config()
            
            enable_switch.on_value_change(lambda: update_options_visibility())
            update_options_visibility()
        
        # Section Décision Finale
        with ui.card().classes("w-full max-w-5xl p-8 mb-12").style(
            "background:#cde4ff !important; "
            "border:4px solid #1c365c; border-radius:20px;"
        ):
            ui.label("💡 Décision Finale").style(
                "font-weight:800; font-size:24px; color:#1c365c; text-align:center; margin-bottom:20px;"
            )
            
            ui.label("**Recommandation pour vos algorithmes** :").style(
                "font-size:16px; color:#12344f; font-weight:600; margin-bottom:12px;"
            )
            
            ui.label("🔵 **KNN** : PCA recommandée (améliore vitesse et précision)").style("font-size:14px; margin-bottom:6px;")
            ui.label("🌳 **C4.5** : Feature Selection optionnelle (garde interprétabilité)").style("font-size:14px; margin-bottom:6px;")
            ui.label(" **Naive Bayes** : Pas de réduction nécessaire").style("font-size:14px; margin-bottom:20px;")
            
            with ui.row().classes("w-full justify-center gap-6 mt-8"):
                ui.button(
                    "⏭️ Passer (pas de réduction)",
                    on_click=skip_reduction
                ).style(
                    "background:#01335A !important; color:white; font-weight:600; "
                    "height:56px; padding:0 32px; border-radius:12px; font-size:16px;"
                )
                
                ui.button(
                    " Appliquer la Réduction",
                    on_click=lambda: apply_reduction(
                        state.get("reduction_method", "pca"),
                        state.get("n_components", 10)
                    )
                ).style(
                    "background:#01335A !important; "
                    "color:white; font-weight:700; height:56px; padding:0 40px; border-radius:12px; font-size:16px;"
                ).bind_enabled_from(enable_switch, 'value')
        
        # Navigation
        with ui.row().classes("w-full max-w-5xl justify-between mt-16"):
            ui.button(
                " Étape précédente",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/scaling'")
            ).style(
                "background:#01335A !important; color:white; font-weight:600; "
                "height:60px; width:260px; border-radius:16px; font-size:16px;"
            )



import asyncio


@ui.page('/supervised/recap_validation')
def recap_validation_page():
    """
    Page complète de récapitulatif et validation finale du preprocessing
    """
    
    
    # ---------- CONTEXTE ----------
    df = state.get("raw_df", None)
    df_original = state.get("df_original", None)  # Dataset avant preprocessing
    split = state.get("split", None)
    target_col = state.get("target_column", None)
    
    if df is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Aucun dataset chargé.").style("font-size:18px; color:#c0392b; font-weight:600;")
            ui.button(" Retour", on_click=lambda: ui.run_javascript("window.location.href='/supervised/upload'"))
        return
    
    # Collecter toutes les transformations
    transformations = {
        "missing_strategy": state.get("missing_strategy", {}),
        "encoding_strategy": state.get("encoding_strategy", {}),
        "transform_strategy": state.get("transform_strategy", {}),
        "scaling_method": state.get("scaling_method", "none"),
        "reduction_applied": state.get("reduction_applied", False),
        "reduction_method": state.get("reduction_method", None),
        "columns_exclude": state.get("columns_exclude", {}),
        "engineered_features": state.get("engineered_features", [])
    }
    
    # ---------- FONCTIONS ----------
    def get_preprocessing_summary():
        """Génère un résumé du preprocessing"""
        summary = []
        
        # 1. Colonnes exclues
        excluded = [k for k, v in transformations["columns_exclude"].items() if v]
        if excluded:
            summary.append({
                "Étape": "Exclusion de colonnes",
                "Action": f"{len(excluded)} colonnes exclues",
                "Détails": ", ".join(excluded[:5]) + ("..." if len(excluded) > 5 else "")
            })
        
        # 2. Valeurs manquantes
        missing_count = len(transformations["missing_strategy"])
        if missing_count > 0:
            methods = {}
            for col, strat in transformations["missing_strategy"].items():
                method = strat.get("method", "none")
                methods[method] = methods.get(method, 0) + 1
            
            details = ", ".join([f"{method}: {count}" for method, count in methods.items()])
            summary.append({
                "Étape": "Valeurs manquantes",
                "Action": f"{missing_count} colonnes traitées",
                "Détails": details
            })
        
        # 3. Encodage
        encoding_count = len(transformations["encoding_strategy"])
        if encoding_count > 0:
            methods = {}
            for col, method in transformations["encoding_strategy"].items():
                methods[method] = methods.get(method, 0) + 1
            
            details = ", ".join([f"{method}: {count}" for method, count in methods.items()])
            summary.append({
                "Étape": "Encodage catégoriel",
                "Action": f"{encoding_count} colonnes encodées",
                "Détails": details
            })
        
        # 4. Transformations de distribution
        transform_count = len([m for m in transformations["transform_strategy"].values() if m != "Aucune"])
        if transform_count > 0:
            methods = {}
            for col, method in transformations["transform_strategy"].items():
                if method != "Aucune":
                    methods[method] = methods.get(method, 0) + 1
            
            details = ", ".join([f"{method}: {count}" for method, count in methods.items()])
            summary.append({
                "Étape": "Transformations distributions",
                "Action": f"{transform_count} colonnes transformées",
                "Détails": details
            })
        
        # 5. Feature Scaling
        scaling = transformations["scaling_method"]
        if scaling != "none":
            summary.append({
                "Étape": "Feature Scaling",
                "Action": f"Méthode: {scaling.upper()}",
                "Détails": f"Appliqué sur colonnes numériques continues"
            })
        
        # 6. Réduction de dimension
        if transformations["reduction_applied"]:
            method = transformations["reduction_method"]
            n_comp = state.get("n_components", "?")
            summary.append({
                "Étape": "Réduction de dimension",
                "Action": f"{method.upper()} - {n_comp} composantes",
                "Détails": f"Features: {len(df.columns)-1} (après réduction)"
            })
        
        # 7. Features engineering
        eng_features = transformations["engineered_features"]
        if eng_features:
            summary.append({
                "Étape": "Feature Engineering",
                "Action": f"{len(eng_features)} features créées",
                "Détails": ", ".join([f[0] for f in eng_features[:3]]) + ("..." if len(eng_features) > 3 else "")
            })
        
        return summary
    
    def create_before_after_comparison():
        """Crée une visualisation avant/après pour les colonnes numériques"""
        if df_original is None:
            return None
        
        # Sélectionner 4 colonnes numériques
        num_cols_original = df_original.select_dtypes(include=[np.number]).columns.tolist()
        num_cols_final = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Trouver colonnes communes
        common_cols = list(set(num_cols_original) & set(num_cols_final))[:4]
        
        if not common_cols:
            return None
        
        # Créer subplots
        fig = make_subplots(
            rows=2, cols=len(common_cols),
            subplot_titles=[f"{col} (Original)" for col in common_cols] + 
                          [f"{col} (Preprocessed)" for col in common_cols],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        for idx, col in enumerate(common_cols):
            # Original
            data_orig = df_original[col].dropna()
            fig.add_trace(
                go.Histogram(x=data_orig, nbinsx=30, name="Original", 
                           marker_color='#e74c3c', showlegend=(idx==0)),
                row=1, col=idx+1
            )
            
            # Preprocessed
            data_prep = df[col].dropna()
            fig.add_trace(
                go.Histogram(x=data_prep, nbinsx=30, name="Preprocessed",
                           marker_color='#01335A', showlegend=(idx==0)),
                row=2, col=idx+1
            )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text=" Comparaison Avant/Après Preprocessing",
            title_font_size=20
        )
        
        return fig
    
    def download_preprocessed_data():
        """Prépare le téléchargement du dataset preprocessé"""
        try:
            # Créer un buffer pour le ZIP
            import zipfile
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # 1. Dataset complet
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                zip_file.writestr('dataset_preprocessed.csv', csv_buffer.getvalue())
                
                # 2. Train/Val/Test si disponibles
                if split:
                    for key in ["X_train", "X_val", "X_test"]:
                        if key in split:
                            csv_buffer = io.StringIO()
                            split[key].to_csv(csv_buffer, index=False)
                            zip_file.writestr(f'{key}.csv', csv_buffer.getvalue())
                    
                    # Target
                    for key in ["y_train", "y_val", "y_test"]:
                        if key in split:
                            csv_buffer = io.StringIO()
                            split[key].to_csv(csv_buffer, index=False, header=[target_col])
                            zip_file.writestr(f'{key}.csv', csv_buffer.getvalue())
                
                # 3. Résumé des transformations
                summary = get_preprocessing_summary()
                summary_df = pd.DataFrame(summary)
                csv_buffer = io.StringIO()
                summary_df.to_csv(csv_buffer, index=False)
                zip_file.writestr('preprocessing_summary.csv', csv_buffer.getvalue())
            
            # Encoder en base64
            zip_buffer.seek(0)
            b64 = base64.b64encode(zip_buffer.read()).decode()
            
            # Créer le lien de téléchargement
            ui.run_javascript(f'''
                const link = document.createElement('a');
                link.href = 'data:application/zip;base64,{b64}';
                link.download = 'preprocessed_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip';
                link.click();
            ''')
            
            ui.notify(" Téléchargement démarré !", color="positive")
        
        except Exception as e:
            ui.notify(f"❌ Erreur lors du téléchargement : {str(e)}", color="negative")
    

    def open_step_selector():
        """Ouvre un dialog pour sélectionner une étape à modifier"""
        steps = [
            ("3.2 - Split Train/Val/Test", "/supervised/split"),
            ("3.3 - Analyse Univariée", "/supervised/univariate_analysis"),
            ("3.4 - Détection Anomalies", "/supervised/outliers_analysis"),
            ("3.5 - Analyse Multivariée", "/supervised/multivariate_analysis"),
            ("3.6 - Gestion Missing", "/supervised/missing_values"),
            ("3.7 - Encodage", "/supervised/encoding"),
            ("3.8 - Transformations", "/supervised/distribution_transform"),
            ("3.9 - Feature Scaling", "/supervised/scaling"),
            ("3.10 - Réduction Dimension", "/supervised/dimension_reduction")
        ]
        
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-2xl p-6"):
            ui.label("Choisir l'étape à modifier").style("font-weight:700; font-size:20px; margin-bottom:16px;")
            
            ui.label(" Attention : Modifier une étape peut invalider les étapes suivantes").style(
                "color:#01335A; font-size:14px; margin-bottom:16px;"
            )
            
            step_radio = ui.radio(
                options=[s[0] for s in steps],
                value=steps[0][0]
            ).classes("mb-4")
            
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Annuler", on_click=dialog.close).props("flat")
                
                def go_to_step():
                    selected = step_radio.value
                    url = next((s[1] for s in steps if s[0] == selected), None)
                    if url:
                        dialog.close()
                        ui.run_javascript(f"window.location.href='{url}'")
                
                ui.button("Modifier", on_click=go_to_step).style("background:#01335A; color:white;")
        
        dialog.open()
    
    def validate_and_continue():
        """Valide le preprocessing et passe à la configuration des algorithmes"""
        
        # VALIDATION PREPROCESSING: Vérifier que les données sont bien préparées
        if split is None:
            ui.notify("❌ Effectuez d'abord le split Train/Val/Test", color="negative")
            return
        
        X_train = split.get("X_train", pd.DataFrame())
        
        if X_train.empty:
            ui.notify("❌ Dataset d'entraînement vide", color="negative")
            return
        
        # Détecter les colonnes non-numériques
        non_numeric_cols = []
        for col in X_train.columns:
            if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
                non_numeric_cols.append(col)
        
        if non_numeric_cols:
            cols_str = ", ".join(non_numeric_cols[:5])
            if len(non_numeric_cols) > 5:
                cols_str += f" (et {len(non_numeric_cols)-5} autres)"
            
            ui.notify(
                f"❌ ERREUR: Colonnes non-numériques détectées: {cols_str}. "
                f"Retournez à la page 'Encodage' pour encoder ces colonnes.",
                color="negative",
                timeout=8000,
                position="top"
            )
            return
        
        # Vérifier colonnes ID potentielles (avertissement uniquement)
        id_pattern_cols = []
        for col in X_train.columns:
            col_lower = col.lower()
            # Cardinalité très élevée = probable identifiant
            if X_train[col].nunique() / len(X_train) > 0.95:
                id_pattern_cols.append(col)
        
        if id_pattern_cols:
            ui.notify(
                f"⚠️ ATTENTION: Colonnes à haute cardinalité détectées: {', '.join(id_pattern_cols[:3])}. "
                f"Ce sont peut-être des identifiants. Retournez à 'Configuration' pour les exclure si nécessaire.",
                color="warning",
                timeout=6000,
                position="top"
            )
            # Ne pas bloquer, juste avertir
        
        # Vérifier valeurs manquantes
        missing_count = X_train.isna().sum().sum()
        if missing_count > 0:
            ui.notify(
                f"⚠️ ATTENTION: {missing_count} valeurs manquantes détectées dans X_train. "
                f"Retournez à 'Gestion Missing' si nécessaire.",
                color="warning",
                timeout=5000
            )
        
        # Tout est OK - Sauvegarder timestamp de validation preprocessing
        state["preprocessing_validated"] = True
        state["preprocessing_validation_timestamp"] = datetime.now().isoformat()
        
        ui.notify("✅ Preprocessing validé ! Passez à la configuration des algorithmes.", color="positive")
        
        # Rediriger vers la page de configuration des algorithmes
        ui.run_javascript("setTimeout(() => window.location.href='/supervised/algorithm_config', 1000);")
    
    
    def reset_preprocessing():
        """Recommencer tout le preprocessing"""
        with ui.dialog() as dialog, ui.card().classes("p-6"):
            ui.label(" Recommencer le Preprocessing ?").style("font-weight:700; font-size:18px;")
            ui.label("Toutes les transformations seront perdues.").style("margin-top:8px; color:#e74c3c;")
            
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Annuler", on_click=dialog.close).props("flat")
                
                def confirm_reset():
                    # Reset state
                    keys_to_reset = [
                        "missing_strategy", "encoding_strategy", "transform_strategy",
                        "scaling_method", "reduction_applied", "fitted_imputers",
                        "fitted_encoders", "fitted_transformers", "engineered_features"
                    ]
                    for key in keys_to_reset:
                        if key in state:
                            del state[key]
                    
                    # Restaurer dataset original
                    if "df_original" in state:
                        state["raw_df"] = state["df_original"].copy()
                    
                    ui.notify("🔄 Preprocessing réinitialisé", color="info")
                    dialog.close()
                    ui.run_javascript("window.location.href='/supervised/user_decisions'")
                
                ui.button("Confirmer", on_click=confirm_reset).style("background:#e74c3c; color:white;")
        
        dialog.open()
    
    # ---------- INTERFACE ----------
    with ui.column().classes("w-full items-center p-8").style("background-color:#f8f9fa; min-height:100vh;"):
        
        # Header
        ui.label(" RÉCAPITULATIF & VALIDATION").style(
            "font-weight:700; font-size:32px; color:#2c3e50; margin-bottom:8px; text-align:center;"
        )
        ui.label("Vue d'ensemble avant lancement des algorithmes").style(
            "font-size:18px; color:#7f8c8d; margin-bottom:32px; text-align:center;"
        )
        
        # Section A : Synthèse des transformations
        with ui.card().classes("w-full max-w-6xl p-6 mb-8"):
            ui.label("📋 Résumé des Transformations Appliquées").style(
                "font-weight:700; font-size:20px; color:#2c3e50; margin-bottom:16px;"
            )
            
            summary = get_preprocessing_summary()
            
            if summary:
                ui.table(
                    columns=[
                        {"name": "Étape", "label": "Étape", "field": "Étape"},
                        {"name": "Action", "label": "Action", "field": "Action"},
                        {"name": "Détails", "label": "Détails", "field": "Détails"}
                    ],
                    rows=summary,
                    row_key="Étape"
                ).style("width:100%;")
            else:
                ui.label(" Aucune transformation appliquée").style("color:#01335A; font-size:16px;")
        
        # Section B : Statistiques comparatives
        with ui.card().classes("w-full max-w-6xl p-6 mb-8"):
            ui.label(" Statistiques Comparatives").style(
                "font-weight:700; font-size:20px; color:#2c3e50; margin-bottom:16px;"
            )
            
            with ui.row().classes("w-full gap-8"):
                # Dataset Original
                with ui.column().classes("flex-1"):
                    with ui.card().classes("p-4").style("background:#ffebee;"):
                        ui.label("Dataset Original").style("font-weight:700; font-size:16px; margin-bottom:8px;")
                        if df_original is not None:
                            ui.label(f"Lignes : {len(df_original):,}").style("font-size:14px;")
                            ui.label(f"Colonnes : {len(df_original.columns)}").style("font-size:14px;")
                            ui.label(f"Missing : {df_original.isna().sum().sum():,}").style("font-size:14px;")
                        else:
                            ui.label("Non disponible").style("color:#636e72;")
                
                # Dataset Preprocessed
                with ui.column().classes("flex-1"):
                    with ui.card().classes("p-4").style("background:#4e88d4;"):
                        ui.label("Dataset Preprocessé").style("font-weight:700; font-size:16px; margin-bottom:8px;")
                        ui.label(f"Lignes : {len(df):,}").style("font-size:14px;")
                        ui.label(f"Colonnes : {len(df.columns)}").style("font-size:14px;")
                        ui.label(f"Missing : {df.isna().sum().sum():,}").style("font-size:14px;")
                
                # Splits
                with ui.column().classes("flex-1"):
                    with ui.card().classes("p-4").style("background:#e3f2fd;"):
                        ui.label("Splits").style("font-weight:700; font-size:16px; margin-bottom:8px;")
                        if split:
                            ui.label(f"Train : {len(split.get('X_train', [])):,}").style("font-size:14px;")
                            ui.label(f"Val : {len(split.get('X_val', [])):,}").style("font-size:14px;")
                            ui.label(f"Test : {len(split.get('X_test', [])):,}").style("font-size:14px;")
                        else:
                            ui.label("Non splitté").style("color:#636e72;")
        
        # Section C : Visualisation Avant/Après
        with ui.card().classes("w-full max-w-6xl p-6 mb-8"):
            ui.label("🎨 Visualisation Avant/Après").style(
                "font-weight:700; font-size:20px; color:#2c3e50; margin-bottom:16px;"
            )
            
            comparison_fig = create_before_after_comparison()
            if comparison_fig:
                ui.plotly(comparison_fig).classes("w-full")
            else:
                ui.label(" Dataset original non disponible pour comparaison").style(
                    "color:#01335A; font-size:14px;"
                )
        
        # Section D : Aperçu des données
        with ui.card().classes("w-full max-w-6xl p-6 mb-8"):
            ui.label("👁️ Aperçu du Dataset Final").style(
                "font-weight:700; font-size:20px; color:#2c3e50; margin-bottom:16px;"
            )
            
            # Afficher les premières lignes
            preview_df = df.head(10)
            
            ui.table(
                columns=[{"name": c, "label": c, "field": c} for c in preview_df.columns],
                rows=preview_df.to_dict('records'),
                row_key=preview_df.columns[0]
            ).style("width:100%; max-height:400px; overflow:auto;")
        
        # Section E : Actions
        with ui.card().classes("w-full max-w-6xl p-6 mb-8"):
            ui.label("📥 Téléchargements & Exports").style(
                "font-weight:700; font-size:20px; color:#2c3e50; margin-bottom:16px;"
            )
            
            with ui.row().classes("gap-4"):
                ui.button(
                    " Télécharger Dataset Preprocessé",
                    on_click=download_preprocessed_data
                ).style(
                    "background:#01335A !important; color:white; font-weight:600; "
                    "height:48px; padding:0 24px; border-radius:10px;"
                )
                
        
        # Section F : Décisions Finales
        with ui.card().classes("w-full max-w-6xl p-8 mb-12").style(
            "background:linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); "
            "border:4px solid #1c365c; border-radius:20px;"
        ):
            ui.label(" Décisions Finales").style(
                "font-weight:800; font-size:24px; color:#1c365c; text-align:center; margin-bottom:20px;"
            )
            
            with ui.row().classes("w-full justify-center gap-4"):
                ui.button(
                    "️ Modifier une Étape",
                    on_click=open_step_selector
                ).style(
                    "background:#01335A !important; color:white; font-weight:600; "
                    "height:56px; padding:0 32px; border-radius:12px; font-size:16px;"
                )
                
                ui.button(
                    "🔄 Recommencer",
                    on_click=reset_preprocessing
                ).style(
                    "background:#01335A !important; color:white; font-weight:600; "
                    "height:56px; padding:0 32px; border-radius:12px; font-size:16px;"
                )
                
                ui.button(
                    " Valider et Continuer",
                    on_click=validate_and_continue
                ).style(
                    "background:#01335A !important; "
                    "color:white; font-weight:700; height:56px; padding:0 40px; border-radius:12px; font-size:16px;"
                )
        
        # Navigation
        with ui.row().classes("w-full max-w-6xl justify-start mt-6"):
            ui.button(
                " Étape précédente",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/dimension_reduction'")
            ).style("background:#01335A !important; color:white; font-weight:600; height:50px; width:220px; border-radius:10px;")





@ui.page('/supervised/algorithm_config')
def algorithm_config_page():
    """
    Page complète de configuration des algorithmes supervisés
    - Sélection des algos à entraîner
    - Configuration des hyperparamètres
    - Validation strategy
    - Récapitulatif avant training
    """
    
    
    # ---------- CONTEXTE ----------
    df = state.get("raw_df", None)
    split = state.get("split", None)
    target_col = state.get("target_column", None)
    
    if df is None or split is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Données non disponibles. Complétez d'abord le preprocessing.").style(
                "font-size:18px; color:#c0392b; font-weight:600;"
            )
            ui.button(" Retour", on_click=lambda: ui.run_javascript("window.location.href='/supervised/recap_validation'"))
        return
    
    # ---------- INITIALISATION STATE ----------
    # Configurations par défaut des algorithmes
    if "algo_configs" not in state:
        state["algo_configs"] = {
            "knn": {
                "n_neighbors": 5,
                "metric": "euclidean",
                "weights": "distance",
                "algorithm": "auto"
            },
            "decision_tree": {
                "criterion": "entropy",
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": None
            },
            "random_forest": {
                "n_estimators": 100,
                "criterion": "gini",
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt"
            },
            "naive_bayes": {
                "var_smoothing": 1e-9
            }
        }
    
    # Algorithmes sélectionnés par défaut
    if "selected_algos" not in state:
        state["selected_algos"] = ["KNN", "Decision Tree", "Naive Bayes"]
    
    # Stratégie de validation
    if "validation_strategy" not in state:
        state["validation_strategy"] = "holdout"
    
    if "cv_folds" not in state:
        state["cv_folds"] = 5
    
    # Métriques à tracker
    if "metrics_to_track" not in state:
        state["metrics_to_track"] = ["accuracy", "precision", "recall", "f1"]
    
    # ---------- FONCTIONS HELPERS ----------
    def get_data_compatibility():
        """Retourne l'état de compatibilité des données"""
        n_samples = len(split.get("X_train", []))
        n_features = len(split.get("X_train", pd.DataFrame()).columns)
        scaling_applied = state.get("scaling_method", "none") != "none"
        reduction_applied = state.get("reduction_applied", False)
        missing_count = split.get("X_train", pd.DataFrame()).isna().sum().sum()
        
        return {
            "n_samples": n_samples,
            "n_features": n_features,
            "scaling": scaling_applied,
            "reduction": reduction_applied,
            "missing": missing_count
        }
    
    def get_recommended_k():
        """Calcule le K recommandé pour KNN"""
        n_samples = get_data_compatibility()["n_samples"]
        return int(np.sqrt(n_samples))
    
    def apply_recommended_config(algo):
        """Applique la configuration recommandée pour un algorithme"""
        compat = get_data_compatibility()
        
        if algo == "knn":
            state["algo_configs"]["knn"] = {
                "n_neighbors": min(get_recommended_k(), 50),
                "metric": "euclidean",
                "weights": "distance",
                "algorithm": "auto"
            }
            ui.notify(" Configuration KNN recommandée appliquée", color="positive")
        
        elif algo == "decision_tree":
            state["algo_configs"]["decision_tree"] = {
                "criterion": "entropy",
                "max_depth": 15,
                "min_samples_split": 5,
                "min_samples_leaf": 5,
                "max_features": "sqrt"
            }
            ui.notify(" Configuration Decision Tree recommandée appliquée", color="positive")
        
        elif algo == "random_forest":
            state["algo_configs"]["random_forest"] = {
                "n_estimators": 100,
                "criterion": "gini",
                "max_depth": 20,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt"
            }
            ui.notify(" Configuration Random Forest recommandée appliquée", color="positive")
        
        elif algo == "naive_bayes":
            state["algo_configs"]["naive_bayes"] = {
                "var_smoothing": 1e-9
            }
            ui.notify(" Configuration Naive Bayes par défaut appliquée", color="positive")
        
        ui.run_javascript("setTimeout(() => location.reload(), 800);")
    
    def reset_config(algo):
        """Réinitialise la configuration d'un algorithme aux valeurs par défaut"""
        defaults = {
            "knn": {
                "n_neighbors": 5,
                "metric": "euclidean",
                "weights": "uniform",
                "algorithm": "auto"
            },
            "decision_tree": {
                "criterion": "gini",
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": None
            },
            "random_forest": {
                "n_estimators": 100,
                "criterion": "gini",
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt"
            },
            "naive_bayes": {
                "var_smoothing": 1e-9
            }
        }
        
        if algo in defaults:
            state["algo_configs"][algo] = defaults[algo]
            ui.notify(f"🔄 Configuration {algo.upper()} réinitialisée", color="info")
            ui.run_javascript("setTimeout(() => location.reload(), 800);")
    
    def validate_and_continue():
        """Valide les configurations et passe à l'entraînement"""
        # Vérifier qu'au moins un algo est sélectionné
        if not state.get("selected_algos", []):
            ui.notify("❌ Sélectionnez au moins un algorithme", color="negative")
            return
        
        # Vérifier qu'au moins une métrique est sélectionnée
        if not state.get("metrics_to_track", []):
            ui.notify("❌ Sélectionnez au moins une métrique", color="negative")
            return
        
        configs = state["algo_configs"]
        selected = state["selected_algos"]
        
        # Validation KNN
        if "KNN" in selected:
            if configs["knn"]["n_neighbors"] < 1:
                ui.notify("❌ KNN: n_neighbors doit être >= 1", color="negative")
                return
        
        # Validation Decision Tree
        if "Decision Tree" in selected:
            if configs["decision_tree"]["max_depth"] is not None and configs["decision_tree"]["max_depth"] < 1:
                ui.notify("❌ Decision Tree: max_depth doit être >= 1 ou None", color="negative")
                return
        
        # Validation Random Forest
        if "Random Forest" in selected:
            if configs["random_forest"]["n_estimators"] < 1:
                ui.notify("❌ Random Forest: n_estimators doit être >= 1", color="negative")
                return
        
        # Validation Naive Bayes
        if "Naive Bayes" in selected:
            if configs["naive_bayes"]["var_smoothing"] <= 0:
                ui.notify("❌ Naive Bayes: var_smoothing doit être > 0", color="negative")
                return
        
        # Sauvegarder timestamp
        state["algo_config_timestamp"] = datetime.now().isoformat()
        
        ui.notify(" Configurations validées !", color="positive")
        
        # Rediriger vers la page d'entraînement
        ui.run_javascript("setTimeout(() => window.location.href='/supervised/feature_importance', 1000);")
    
    # ---------- INTERFACE ----------
    compat = get_data_compatibility()
    
    with ui.column().classes("w-full items-center p-8").style("background-color:#f8f9fa; min-height:100vh;"):
        
        # ==================== HEADER ====================
        ui.label("CONFIGURATION DES ALGORITHMES").style(
            "font-weight:700; font-size:32px; color:#2c3e50; margin-bottom:8px; text-align:center;"
        )
        ui.label("Sélectionner et configurer vos algorithmes avant l'entraînement").style(
            "font-size:18px; color:#7f8c8d; margin-bottom:32px; text-align:center;"
        )
        
        # ==================== INFO DONNÉES ====================
        with ui.card().classes("w-full max-w-6xl p-4 mb-6").style("background:#e3f2fd; border-left:5px solid #2196f3;"):
            with ui.row().classes("w-full gap-8 items-center"):
                ui.label(" État des données :").style("font-weight:700; font-size:16px;")
                ui.label(f"Train: {compat['n_samples']:,} samples").style("font-size:14px;")
                ui.label(f"Features: {compat['n_features']}").style("font-size:14px;")
                ui.label(f"Scaling: {'' if compat['scaling'] else '❌'}").style("font-size:14px;")
                ui.label(f"Reduction: {'' if compat['reduction'] else '❌'}").style("font-size:14px;")
                ui.label(f"Missing: {compat['missing']}").style("font-size:14px;")
        
        # ==================== SÉLECTION ALGORITHMES ====================
        with ui.card().classes("w-full max-w-6xl p-6 mb-8").style("background:#01335A !important; border-radius:16px;"):
            ui.label(" SÉLECTION DES ALGORITHMES À ENTRAÎNER").style(
                "font-weight:700; font-size:24px; color:white; margin-bottom:16px; text-align:center;"
            )
            
            ui.label("Choisissez les algorithmes que vous souhaitez tester sur vos données").style(
                "font-size:16px; color:white; text-align:center; margin-bottom:20px;"
            )
            
            # Conteneur des checkboxes
            with ui.card().classes("w-full p-6").style("background:white;"):
                selected_algos = state.get("selected_algos", [])
                
                with ui.row().classes("w-full gap-8 justify-center"):
                    algo_knn_cb = ui.checkbox(" K-Nearest Neighbors", value="KNN" in selected_algos).classes("text-lg")
                    algo_dt_cb = ui.checkbox("🌳 Decision Tree (C4.5)", value="Decision Tree" in selected_algos).classes("text-lg")
                    algo_nb_cb = ui.checkbox(" Naive Bayes", value="Naive Bayes" in selected_algos).classes("text-lg")
                
                def update_selected_algos():
                    algos = []
                    if algo_knn_cb.value:
                        algos.append("KNN")
                    if algo_dt_cb.value:
                        algos.append("Decision Tree")
                    if algo_nb_cb.value:
                        algos.append("Naive Bayes")
                    state["selected_algos"] = algos
                    
                    # Afficher warning si aucun algo sélectionné
                    if not algos:
                        ui.notify(" Sélectionnez au moins un algorithme", color="warning")
                
                algo_knn_cb.on_value_change(lambda: update_selected_algos())
                algo_dt_cb.on_value_change(lambda: update_selected_algos())
                algo_nb_cb.on_value_change(lambda: update_selected_algos())
                
                # Compteur d'algos sélectionnés
                algo_count = ui.label(f" {len(state.get('selected_algos', []))} algorithme(s) sélectionné(s)").style(
                    "font-weight:700; font-size:16px; color:#01335A; margin-top:16px; text-align:center; width:100%;"
                )
                
                def update_count():
                    count = len(state.get("selected_algos", []))
                    algo_count.set_text(f" {count} algorithme(s) sélectionné(s)")
                    algo_count.style(f"color:{'#01335A' if count > 0 else '#e74c3c'};")
                
                algo_knn_cb.on_value_change(lambda: update_count())
                algo_dt_cb.on_value_change(lambda: update_count())
                algo_nb_cb.on_value_change(lambda: update_count())
        
        # ==================== CARTE 1: KNN ====================
        knn_card = ui.column().classes("w-full max-w-6xl")
        
        if "KNN" in state.get("selected_algos", []):
            with knn_card:
                with ui.card().classes("w-full p-6 mb-6"):
                    ui.label(" K-Nearest Neighbors (KNN)").style(
                        "font-weight:700; font-size:24px; color:#2c3e50; margin-bottom:16px;"
                    )
                    
                    # Description
                    with ui.expansion("📖 Principe & Compatibilité", icon="info").classes("w-full mb-4"):
                        with ui.column().classes("p-4"):
                            ui.label("Principe :").style("font-weight:700; margin-bottom:8px;")
                            ui.label("Classification basée sur les K voisins les plus proches dans l'espace des features").style("font-size:14px; margin-bottom:16px;")
                            
                            with ui.row().classes("w-full gap-8"):
                                with ui.column().classes("flex-1"):
                                    ui.label(" Avantages").style("font-weight:700; color:#01335A; margin-bottom:8px;")
                                    ui.label("• Simple et intuitif").style("font-size:14px;")
                                    ui.label("• Non-paramétrique").style("font-size:14px;")
                                    ui.label("• Adapté frontières non-linéaires").style("font-size:14px;")
                                    ui.label("• Pas de phase d'entraînement").style("font-size:14px;")
                                
                                with ui.column().classes("flex-1"):
                                    ui.label("❌ Inconvénients").style("font-weight:700; color:#e74c3c; margin-bottom:8px;")
                                    ui.label("• Sensible au scaling").style("font-size:14px;")
                                    ui.label("• Coût computationnel élevé").style("font-size:14px;")
                                    ui.label("• Curse of dimensionality").style("font-size:14px;")
                                    ui.label("• Sensible au bruit").style("font-size:14px;")
                            
                            ui.label(" Compatibilité").style("font-weight:700; margin-top:16px; margin-bottom:8px;")
                            ui.label(f"• Scaling : {' Appliqué' if compat['scaling'] else '🔴 CRITIQUE - Non appliqué'}").style("font-size:14px;")
                            ui.label(f"• Dimension : {' Réduite' if compat['reduction'] else ' Peut affecter performance'}").style("font-size:14px;")
                            ui.label(f"• Missing : {' Aucune' if compat['missing'] == 0 else '❌ Présentes'}").style("font-size:14px;")
                    
                    ui.label(" HYPERPARAMÈTRES").style("font-weight:700; font-size:18px; margin-top:16px; margin-bottom:12px;")
                    
                    knn_config = state["algo_configs"]["knn"]
                    
                    # 1. n_neighbors
                    with ui.row().classes("w-full items-center gap-4 mb-4"):
                        ui.label("1. Nombre de voisins (K)").style("font-weight:600; width:250px;")
                        knn_neighbors_slider = ui.slider(min=1, max=50, value=knn_config["n_neighbors"], step=1).classes("flex-1")
                        knn_neighbors_label = ui.label(f"K = {knn_config['n_neighbors']}").style("font-weight:700; width:80px;")
                        
                        def update_knn_neighbors(e):
                            state["algo_configs"]["knn"]["n_neighbors"] = int(e.value)
                            knn_neighbors_label.set_text(f"K = {int(e.value)}")
                        
                        knn_neighbors_slider.on_value_change(update_knn_neighbors)
                    
                    ui.label(f"💡 Règle empirique : K = √n_samples ≈ {get_recommended_k()}").style("font-size:13px; color:#01335A; margin-left:250px; margin-bottom:8px;")
                    ui.label(" K petit  Overfitting | K grand  Underfitting").style("font-size:13px; color:#01335A; margin-left:250px; margin-bottom:16px;")
                    
                    # 2. metric
                    with ui.row().classes("w-full items-start gap-4 mb-4"):
                        ui.label("2. Métrique de distance").style("font-weight:600; width:250px;")
                        with ui.column().classes("flex-1"):
                            knn_metric = ui.radio(
                                options={
                                    "euclidean": "Euclidean (L2) - Distance standard",
                                    "manhattan": "Manhattan (L1) - Robuste aux outliers",
                                    "minkowski": "Minkowski - Généralisation",
                                    "chebyshev": "Chebyshev - Distance max"
                                },
                                value=knn_config["metric"]
                            )
                            knn_metric.on_value_change(lambda e: state["algo_configs"]["knn"].update({"metric": e.value}))
                    
                    # 3. weights
                    with ui.row().classes("w-full items-start gap-4 mb-4"):
                        ui.label("3. Pondération").style("font-weight:600; width:250px;")
                        with ui.column().classes("flex-1"):
                            knn_weights = ui.radio(
                                options={
                                    "uniform": "Uniform - Tous voisins poids égal",
                                    "distance": "Distance - Poids = 1/distance (favorise proches) [Recommandé]"
                                },
                                value=knn_config["weights"]
                            )
                            knn_weights.on_value_change(lambda e: state["algo_configs"]["knn"].update({"weights": e.value}))
                    
                    # 4. algorithm
                    with ui.row().classes("w-full items-start gap-4 mb-4"):
                        ui.label("4. Algorithme recherche").style("font-weight:600; width:250px;")
                        with ui.column().classes("flex-1"):
                            knn_algo = ui.radio(
                                options={
                                    "auto": "Auto - Sélection automatique [Recommandé]",
                                    "ball_tree": "Ball Tree - Rapide, haute dimension",
                                    "kd_tree": "KD Tree - Rapide, basse dimension (<20)",
                                    "brute": "Brute Force - Exhaustif (petit dataset)"
                                },
                                value=knn_config["algorithm"]
                            )
                            knn_algo.on_value_change(lambda e: state["algo_configs"]["knn"].update({"algorithm": e.value}))
                    
                    # Estimation performances
                    with ui.card().classes("w-full p-4 mt-4").style("background:#cde4ff !important;"):
                        ui.label("⏱️ Estimation Performances").style("font-weight:700; margin-bottom:8px;")
                        ui.label(f"• Temps training : < 0.1s (lazy learner - pas d'entraînement réel)").style("font-size:14px;")
                        pred_time = (compat['n_samples'] * compat['n_features']) / 100000
                        ui.label(f"• Temps prédiction : ~{pred_time:.2f}s par échantillon").style("font-size:14px;")
                        mem = (compat['n_samples'] * compat['n_features'] * 8) / 1024
                        ui.label(f"• Mémoire : ~{mem:.0f} KB").style("font-size:14px;")
                    
                    # Boutons
                    with ui.row().classes("w-full justify-end gap-2 mt-4"):
                        ui.button("🔄 Réinitialiser", on_click=lambda: reset_config("knn")).props("flat")
                        ui.button("💡 Appliquer Recommandation", on_click=lambda: apply_recommended_config("knn")).style(
                            "background:#01335A !important; color:white;"
                        )
        
        # ==================== CARTE 2: DECISION TREE ====================
        dt_card = ui.column().classes("w-full max-w-6xl")
        
        if "Decision Tree" in state.get("selected_algos", []):
            with dt_card:
                with ui.card().classes("w-full p-6 mb-6"):
                    ui.label("🌳 C4.5 Decision Tree").style(
                        "font-weight:700; font-size:24px; color:#2c3e50; margin-bottom:16px;"
                    )
                    
                    # Description
                    with ui.expansion("📖 Principe & Compatibilité", icon="info").classes("w-full mb-4"):
                        with ui.column().classes("p-4"):
                            ui.label("Principe :").style("font-weight:700; margin-bottom:8px;")
                            ui.label("Arbre de décision basé sur Information Gain (entropie) - Implémentation sklearn proche de C4.5").style("font-size:14px; margin-bottom:16px;")
                            
                            with ui.row().classes("w-full gap-8"):
                                with ui.column().classes("flex-1"):
                                    ui.label(" Avantages").style("font-weight:700; color:#01335A; margin-bottom:8px;")
                                    ui.label("• Très interprétable (structure arbre)").style("font-size:14px;")
                                    ui.label("• Gère num + catégorielles").style("font-size:14px;")
                                    ui.label("• Peu sensible au scaling").style("font-size:14px;")
                                    ui.label("• Robuste aux outliers").style("font-size:14px;")
                                
                                with ui.column().classes("flex-1"):
                                    ui.label("❌ Inconvénients").style("font-weight:700; color:#e74c3c; margin-bottom:8px;")
                                    ui.label("• Tendance overfitting").style("font-size:14px;")
                                    ui.label("• Instable (petites variations)").style("font-size:14px;")
                                    ui.label("• Biais vers features multi-valeurs").style("font-size:14px;")
                            
                            ui.label(" Compatibilité").style("font-weight:700; margin-top:16px; margin-bottom:8px;")
                            ui.label("• Scaling :  Non nécessaire").style("font-size:14px;")
                            ui.label(f"• Missing : {' OK' if compat['missing'] == 0 else ' sklearn ne gère pas nativement'}").style("font-size:14px;")
                            ui.label("• Outliers :  Robuste").style("font-size:14px;")
                    
                    ui.label(" HYPERPARAMÈTRES").style("font-weight:700; font-size:18px; margin-top:16px; margin-bottom:12px;")
                    
                    dt_config = state["algo_configs"]["decision_tree"]
                    
                    # 1. criterion
                    with ui.row().classes("w-full items-start gap-4 mb-4"):
                        ui.label("1. Critère de split").style("font-weight:600; width:250px;")
                        with ui.column().classes("flex-1"):
                            dt_criterion = ui.radio(
                                options={
                                    "entropy": "Entropy - Information Gain (~ C4.5 original) [Recommandé]",
                                    "gini": "Gini - Gini impurity (default sklearn)",
                                    "log_loss": "Log Loss - Log loss"
                                },
                                value=dt_config["criterion"]
                            )
                            dt_criterion.on_value_change(lambda e: state["algo_configs"]["decision_tree"].update({"criterion": e.value}))
                    
                    # 2. max_depth
                    with ui.row().classes("w-full items-center gap-4 mb-4"):
                        ui.label("2. Profondeur maximale").style("font-weight:600; width:250px;")
                        
                        with ui.column().classes("flex-1"):
                            dt_depth_checkbox = ui.checkbox("Limiter la profondeur (recommandé)", value=dt_config["max_depth"] is not None)
                            dt_depth_slider = ui.slider(min=1, max=30, value=15 if dt_config["max_depth"] is None else dt_config["max_depth"], step=1).classes("w-full")
                            dt_depth_label = ui.label(f"max_depth = {dt_config['max_depth'] if dt_config['max_depth'] is not None else 'None (illimité)'}").style("font-weight:700;")
                            
                            if dt_config["max_depth"] is None:
                                dt_depth_slider.disable()
                            
                            def update_dt_depth_checkbox(e):
                                if e.value:
                                    state["algo_configs"]["decision_tree"]["max_depth"] = int(dt_depth_slider.value)
                                    dt_depth_slider.enable()
                                    dt_depth_label.set_text(f"max_depth = {int(dt_depth_slider.value)}")
                                else:
                                    state["algo_configs"]["decision_tree"]["max_depth"] = None
                                    dt_depth_slider.disable()
                                    dt_depth_label.set_text("max_depth = None (illimité)")
                            
                            def update_dt_depth_slider(e):
                                if dt_depth_checkbox.value:
                                    state["algo_configs"]["decision_tree"]["max_depth"] = int(e.value)
                                    dt_depth_label.set_text(f"max_depth = {int(e.value)}")
                            
                            dt_depth_checkbox.on_value_change(update_dt_depth_checkbox)
                            dt_depth_slider.on_value_change(update_dt_depth_slider)
                            
                            ui.label(" None = Pas de limite (fort risque overfitting)").style("font-size:13px; color:#01335A; margin-top:4px;")
                    
                    # 3. min_samples_split
                    with ui.row().classes("w-full items-center gap-4 mb-4"):
                        ui.label("3. Min samples split").style("font-weight:600; width:250px;")
                        dt_split_slider = ui.slider(min=2, max=100, value=dt_config["min_samples_split"], step=1).classes("flex-1")
                        dt_split_label = ui.label(f"{dt_config['min_samples_split']}").style("font-weight:700; width:80px;")
                        
                        def update_dt_split(e):
                            state["algo_configs"]["decision_tree"]["min_samples_split"] = int(e.value)
                            dt_split_label.set_text(f"{int(e.value)}")
                        
                        dt_split_slider.on_value_change(update_dt_split)
                    
                    ui.label("Minimum d'échantillons pour splitter un nœud").style("font-size:13px; color:#7f8c8d; margin-left:250px; margin-bottom:16px;")
                    
                    # 4. min_samples_leaf
                    with ui.row().classes("w-full items-center gap-4 mb-4"):
                        ui.label("4. Min samples leaf").style("font-weight:600; width:250px;")
                        dt_leaf_slider = ui.slider(min=1, max=50, value=dt_config["min_samples_leaf"], step=1).classes("flex-1")
                        dt_leaf_label = ui.label(f"{dt_config['min_samples_leaf']}").style("font-weight:700; width:80px;")
                        
                        def update_dt_leaf(e):
                            state["algo_configs"]["decision_tree"]["min_samples_leaf"] = int(e.value)
                            dt_leaf_label.set_text(f"{int(e.value)}")
                        
                        dt_leaf_slider.on_value_change(update_dt_leaf)
                    
                    ui.label("Minimum d'échantillons dans une feuille").style("font-size:13px; color:#7f8c8d; margin-left:250px; margin-bottom:16px;")
                    
                    # 5. max_features
                    with ui.row().classes("w-full items-start gap-4 mb-4"):
                        ui.label("5. Max features").style("font-weight:600; width:250px;")
                        with ui.column().classes("flex-1"):
                            dt_features = ui.radio(
                                options={
                                    "None": f"None - Toutes features ({compat['n_features']})",
                                    "sqrt": f"Sqrt - √features ≈ {int(np.sqrt(compat['n_features']))}",
                                    "log2": f"Log2 - log₂(features) ≈ {int(np.log2(compat['n_features']))}",
                                },
                                value="None" if dt_config["max_features"] is None else dt_config["max_features"]
                            )
                            
                            def update_dt_features(e):
                                state["algo_configs"]["decision_tree"]["max_features"] = None if e.value == "None" else e.value
                            
                            dt_features.on_value_change(update_dt_features)
                    
                    # Boutons
                    with ui.row().classes("w-full justify-end gap-2 mt-4"):
                        ui.button("🔄 Réinitialiser", on_click=lambda: reset_config("decision_tree")).props("flat")
                        ui.button("💡 Appliquer Recommandation", on_click=lambda: apply_recommended_config("decision_tree")).style(
                            "background:#01335A !important; color:white;"
                        )
        
        
        # ==================== CARTE 4: NAIVE BAYES ====================
        nb_card = ui.column().classes("w-full max-w-6xl")
        
        if "Naive Bayes" in state.get("selected_algos", []):
            with nb_card:
                with ui.card().classes("w-full p-6 mb-6"):
                    ui.label(" Naive Bayes").style(
                        "font-weight:700; font-size:24px; color:#2c3e50; margin-bottom:16px;"
                    )
                    
                    # Description
                    with ui.expansion("📖 Principe & Compatibilité", icon="info").classes("w-full mb-4"):
                        with ui.column().classes("p-4"):
                            ui.label("Principe :").style("font-weight:700; margin-bottom:8px;")
                            ui.label("Classifieur probabiliste simple basé sur le théorème de Bayes avec hypothèse d'indépendance des features").style("font-size:14px; margin-bottom:16px;")

                            with ui.row().classes("w-full gap-8"):
                                with ui.column().classes("flex-1"):
                                    ui.label(" Avantages").style("font-weight:700; color:#01335A; margin-bottom:8px;")
                                    ui.label("• Très rapide (training + prédiction)").style("font-size:14px;")
                                    ui.label("• Nécessite peu de données").style("font-size:14px;")
                                    ui.label("• Robuste au bruit").style("font-size:14px;")
                                    ui.label("• Donne des probabilités").style("font-size:14px;")
                                    ui.label("• Performant texte/spam").style("font-size:14px;")
                                
                                with ui.column().classes("flex-1"):
                                    ui.label("❌ Inconvénients").style("font-weight:700; color:#e74c3c; margin-bottom:8px;")
                                    ui.label("• Hypothèse indépendance rarement vraie").style("font-size:14px;")
                                    ui.label("• Assume distribution gaussienne").style("font-size:14px;")
                                    ui.label("• Sensible features corrélées").style("font-size:14px;")
                                    ui.label("• Moins performant que RF/Tree").style("font-size:14px;")
                            
                            ui.label(" Compatibilité").style("font-weight:700; margin-top:16px; margin-bottom:8px;")
                            ui.label("• Transformations : 🔴 CRITIQUE (Log, Box-Cox pour normaliser)").style("font-size:14px;")
                            ui.label(f"• Scaling : {' Appliqué' if compat['scaling'] else '🟡 Recommandé'}").style("font-size:14px;")
                            ui.label("• Corrélations :  Viole hypothèse d'indépendance").style("font-size:14px;")
                            ui.label(f"• Missing : {' OK' if compat['missing'] == 0 else '❌ À gérer avant'}").style("font-size:14px;")
                    
                    ui.label(" HYPERPARAMÈTRES").style("font-weight:700; font-size:18px; margin-top:16px; margin-bottom:12px;")
                    
                    nb_config = state["algo_configs"]["naive_bayes"]
                    
                    # 1. var_smoothing
                    with ui.row().classes("w-full items-start gap-4 mb-4"):
                        ui.label("1. Var Smoothing").style("font-weight:600; width:250px;")
                        with ui.column().classes("flex-1"):
                            ui.label("Portion de variance ajoutée pour stabilité numérique").style("font-size:14px; color:#7f8c8d; margin-bottom:8px;")
                            
                            # Radio pour choisir entre valeurs prédéfinies ou custom
                            nb_smoothing_radio = ui.radio(
                                options={
                                    "1e-9": "1e-9 (défaut sklearn) [Recommandé]",
                                    "1e-10": "1e-10 (plus sensible)",
                                    "1e-8": "1e-8 (plus stable)",
                                    "custom": "Custom"
                                },
                                value="1e-9" if nb_config["var_smoothing"] == 1e-9 else "custom"
                            )
                            
                            nb_custom_input = ui.number(
                                label="Valeur custom (format: 1e-9)",
                                value=nb_config["var_smoothing"],
                                format="%.1e",
                                min=1e-12,
                                max=1e-6
                            ).classes("w-full mt-2")
                            
                            if nb_smoothing_radio.value != "custom":
                                nb_custom_input.disable()
                            
                            def update_nb_smoothing_radio(e):
                                if e.value == "custom":
                                    nb_custom_input.enable()
                                    state["algo_configs"]["naive_bayes"]["var_smoothing"] = float(nb_custom_input.value)
                                else:
                                    nb_custom_input.disable()
                                    state["algo_configs"]["naive_bayes"]["var_smoothing"] = float(e.value)
                            
                            def update_nb_custom_input(e):
                                if nb_smoothing_radio.value == "custom":
                                    state["algo_configs"]["naive_bayes"]["var_smoothing"] = float(e.value)
                            
                            nb_smoothing_radio.on_value_change(update_nb_smoothing_radio)
                            nb_custom_input.on_value_change(update_nb_custom_input)
                            
                            ui.label("💡 Augmenter si underfitting, diminuer si overfitting (mais 1e-9 optimal généralement)").style(
                                "font-size:13px; color:#01335A; margin-top:8px;"
                            )
                    
                    # Info sur les hypothèses
                    with ui.card().classes("w-full p-4 mt-4").style("background:#ffe5e5; border-left:4px solid #e74c3c;"):
                        ui.label(" HYPOTHÈSES CRITIQUES").style("font-weight:700; color:#c0392b; margin-bottom:8px;")
                        ui.label("1. Features indépendantes conditionnellement à la classe").style("font-size:14px; margin-bottom:4px;")
                        ui.label("2. Distribution gaussienne de chaque feature").style("font-size:14px; margin-bottom:12px;")
                        ui.label("💡 Si vos données violent ces hypothèses (features corrélées, distributions skewed), les performances seront limitées.").style(
                            "font-size:13px; color:#01335A; font-style:italic;"
                        )
                    
                    # Estimation performances
                    with ui.card().classes("w-full p-4 mt-4").style("background:;"):
                        ui.label("⚡ Performances").style("font-weight:700; marg#cde4ff !importantin-bottom:8px;")
                        ui.label("• Temps training : < 0.1s (le plus rapide de tous)").style("font-size:14px;")
                        ui.label("• Temps prédiction : < 0.01s (très rapide)").style("font-size:14px;")
                        ui.label("• Mémoire : Très faible (~quelques KB)").style("font-size:14px;")
                        ui.label("• Idéal pour : Baseline rapide, données textuelles, peu de données").style("font-size:14px; margin-top:8px; font-weight:600; color:#01335A;")
                    
                    # Boutons
                    with ui.row().classes("w-full justify-end gap-2 mt-4"):
                        ui.button("🔄 Réinitialiser", on_click=lambda: reset_config("naive_bayes")).props("flat")
                        ui.button("💡 Config par défaut", on_click=lambda: apply_recommended_config("naive_bayes")).style(
                            "background:#01335A !important; color:white;"
                        )
        
        # ==================== SECTION: STRATÉGIE DE VALIDATION ====================
        with ui.card().classes("w-full max-w-6xl p-6 mb-6").style("background:#cde4ff !important; border:2px solid #1c365c;"):
            ui.label(" STRATÉGIE DE VALIDATION").style(
                "font-weight:700; font-size:24px; color:#2c3e50; margin-bottom:16px;"
            )
            
            # Info splits
            with ui.card().classes("w-full p-4 mb-4").style("background:white;"):
                ui.label(" Vos données :").style("font-weight:700; margin-bottom:8px;")
                ui.label(f"• Train : {len(split.get('X_train', [])):,} samples").style("font-size:14px;")
                ui.label(f"• Validation : {len(split.get('X_val', [])):,} samples").style("font-size:14px;")
                ui.label(f"• Test : {len(split.get('X_test', [])):,} samples (réservé pour évaluation finale)").style("font-size:14px;")
            
            # Méthode validation
            ui.label("Méthode de validation :").style("font-weight:600; font-size:16px; margin-bottom:8px;")
            
            validation_strategy = ui.radio(
                options={
                    "holdout": "Hold-out (Train  Val) - Rapide, déjà splitté  [Recommandé]",
                    "cv": "Cross-Validation K-Fold sur Train - Plus robuste mais plus lent"
                },
                value=state.get("validation_strategy", "holdout")
            ).classes("mb-4")
            
            validation_strategy.on_value_change(lambda e: state.update({"validation_strategy": e.value}))
            
            # Si CV, montrer options
            cv_options_container = ui.column().classes("w-full")
            
            def update_cv_display():
                cv_options_container.clear()
                if state.get("validation_strategy") == "cv":
                    with cv_options_container:
                        with ui.card().classes("w-full p-4").style("background:#cde4ff;"):
                            ui.label(" Paramètres Cross-Validation :").style("font-weight:600; margin-bottom:8px;")
                            
                            with ui.row().classes("w-full items-center gap-4"):
                                ui.label("Nombre de folds :").style("width:200px;")
                                cv_folds_slider = ui.slider(min=3, max=10, value=state.get("cv_folds", 5), step=1).classes("flex-1")
                                cv_folds_label = ui.label(f"K = {state.get('cv_folds', 5)}").style("font-weight:700; width:80px;")
                                
                                def update_cv_folds(e):
                                    state["cv_folds"] = int(e.value)
                                    cv_folds_label.set_text(f"K = {int(e.value)}")
                                
                                cv_folds_slider.on_value_change(update_cv_folds)
                            
                            ui.label(f"💡 Chaque fold : ~{len(split.get('X_train', []))//state.get('cv_folds', 5):,} samples").style(
                                "font-size:13px; color:#7f8c8d; margin-top:8px;"
                            )
                            ui.label(" Le temps d'entraînement sera multiplié par K").style(
                                "font-size:13px; color:#01335A; margin-top:4px;"
                            )
            
            update_cv_display()
            validation_strategy.on_value_change(lambda e: update_cv_display())
            
            # Métriques à tracker
            ui.label("Métriques à suivre :").style("font-weight:600; font-size:16px; margin-top:16px; margin-bottom:8px;")
            
            metrics_selected = state.get("metrics_to_track", ["accuracy", "precision", "recall", "f1"])
            
            with ui.row().classes("gap-4"):
                metric_accuracy = ui.checkbox("Accuracy", value="accuracy" in metrics_selected)
                metric_precision = ui.checkbox("Precision", value="precision" in metrics_selected)
                metric_recall = ui.checkbox("Recall", value="recall" in metrics_selected)
                metric_f1 = ui.checkbox("F1-Score", value="f1" in metrics_selected)
            
            def update_metrics():
                metrics = []
                if metric_accuracy.value:
                    metrics.append("accuracy")
                if metric_precision.value:
                    metrics.append("precision")
                if metric_recall.value:
                    metrics.append("recall")
                if metric_f1.value:
                    metrics.append("f1")
                state["metrics_to_track"] = metrics
            
            metric_accuracy.on_value_change(lambda: update_metrics())
            metric_precision.on_value_change(lambda: update_metrics())
            metric_recall.on_value_change(lambda: update_metrics())
            metric_f1.on_value_change(lambda: update_metrics())
            
            with ui.card().classes("w-full p-4 mt-4").style("background:#e3f2fd;"):
                ui.label("📖 Définitions des métriques :").style("font-weight:700; margin-bottom:8px;")
                ui.label("• Accuracy : (TP + TN) / Total - Précision globale").style("font-size:13px;")
                ui.label("• Precision : TP / (TP + FP) - Fiabilité des prédictions positives").style("font-size:13px;")
                ui.label("• Recall : TP / (TP + FN) - Capacité à détecter les vrais positifs").style("font-size:13px;")
                ui.label("• F1-Score : 2×(Precision×Recall)/(Precision+Recall) - Moyenne harmonique").style("font-size:13px;")
        
        # ==================== RÉCAPITULATIF FINAL ====================
        with ui.card().classes("w-full max-w-6xl p-6 mb-8").style(
            " border-radius:16px;"
        ):
            ui.label("📋 RÉCAPITULATIF CONFIGURATION").style(
                "font-weight:700; font-size:24px; color:white; margin-bottom:16px; text-align:center;"
            )
            
            selected = state.get("selected_algos", [])
            
            if not selected:
                ui.label(" Aucun algorithme sélectionné").style(
                    "font-size:18px; color:#cde4ff; text-align:center;"
                )
            else:
                with ui.column().classes("w-full gap-4"):
                    # KNN
                    if "KNN" in selected:
                        with ui.card().classes("p-4"):
                            ui.label(" KNN").style("font-weight:700; font-size:16px; margin-bottom:8px;")
                            knn_cfg = state["algo_configs"]["knn"]
                            ui.label(f"• n_neighbors: {knn_cfg['n_neighbors']}").style("font-size:14px;")
                            ui.label(f"• metric: {knn_cfg['metric']}").style("font-size:14px;")
                            ui.label(f"• weights: {knn_cfg['weights']}").style("font-size:14px;")
                            ui.label(f"• algorithm: {knn_cfg['algorithm']}").style("font-size:14px;")
                    
                    # Decision Tree
                    if "Decision Tree" in selected:
                        with ui.card().classes("p-4"):
                            ui.label("🌳 Decision Tree").style("font-weight:700; font-size:16px; margin-bottom:8px;")
                            dt_cfg = state["algo_configs"]["decision_tree"]
                            ui.label(f"• criterion: {dt_cfg['criterion']}").style("font-size:14px;")
                            ui.label(f"• max_depth: {dt_cfg['max_depth'] if dt_cfg['max_depth'] is not None else 'None'}").style("font-size:14px;")
                            ui.label(f"• min_samples_split: {dt_cfg['min_samples_split']}").style("font-size:14px;")
                            ui.label(f"• min_samples_leaf: {dt_cfg['min_samples_leaf']}").style("font-size:14px;")
                            ui.label(f"• max_features: {dt_cfg['max_features'] if dt_cfg['max_features'] is not None else 'None'}").style("font-size:14px;")
                    
                    
                    # Naive Bayes
                    if "Naive Bayes" in selected:
                        with ui.card().classes("p-4"):
                            ui.label(" Naive Bayes").style("font-weight:700; font-size:16px; margin-bottom:8px;")
                            nb_cfg = state["algo_configs"]["naive_bayes"]
                            ui.label(f"• var_smoothing: {nb_cfg['var_smoothing']:.1e}").style("font-size:14px;")
                    
                    # Validation
                    with ui.card().classes("p-4"):
                        ui.label(" Validation").style("font-weight:700; font-size:16px; margin-bottom:8px;")
                        ui.label(f"• Stratégie: {state.get('validation_strategy', 'holdout').upper()}").style("font-size:14px;")
                        if state.get("validation_strategy") == "cv":
                            ui.label(f"• K-Folds: {state.get('cv_folds', 5)}").style("font-size:14px;")
                        metrics_str = ", ".join(state.get("metrics_to_track", []))
                        ui.label(f"• Métriques: {metrics_str}").style("font-size:14px;")
        
        # ==================== BOUTONS FINAUX ====================
        with ui.row().classes("w-full max-w-6xl justify-between gap-4 mt-8"):
            ui.button(
                " Retour au Récapitulatif",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/recap_validation'")
            ).style(
                "background:#01335A !important; color:white; font-weight:600; height:56px; padding:0 32px; border-radius:12px; font-size:16px;"
            )
            
            ui.button(
                " Valider et Lancer l'Entraînement",
                on_click=validate_and_continue
            ).style(
                "background:#01335A !important "
                "color:white; font-weight:700; height:56px; padding:0 40px; border-radius:12px; font-size:16px;"
            )




@ui.page('/supervised/feature_importance')
def feature_importance_page():
    """
    Page d'analyse de l'importance des features AVANT training complet
    - Analyse univariée (corrélation + tests statistiques adaptés au type)
    - Analyse multivariée (Mutual Information)
    - Recommandations de sélection de features
    - Visualisations interactives
    
     CORRECTION : Gère correctement variables catégorielles/binaires
    """
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.stats import chi2_contingency, f_oneway, pearsonr
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.preprocessing import LabelEncoder
    
    # ---------- CONTEXTE ----------
    df = state.get("raw_df", None)
    split = state.get("split", None)
    target_col = state.get("target_column", None)
    columns_exclude = state.get("columns_exclude", {}) or {}
    
    if df is None or split is None or target_col is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Données non disponibles. Complétez d'abord les étapes précédentes.").style(
                "font-size:18px !important; color:#c0392b !important; font-weight:600 !important;"
            )
            ui.button(" Retour", on_click=lambda: ui.run_javascript("window.location.href='/supervised/algorithm_config'"))
        return
    
    # Préparer données
    X_train = split.get("X_train")
    y_train = split.get("y_train")
    
    if X_train is None or y_train is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Split non disponible.").style("font-size:18px !important; color:#c0392b !important; font-weight:600 !important;")
        return
    
    active_features = [c for c in X_train.columns if not columns_exclude.get(c, False)]
    
    # Déterminer si classification ou régression
    is_classification = y_train.nunique() <= 20
    task_type = "Classification" if is_classification else "Régression"
    
    # ---------- FONCTIONS D'ANALYSE (CORRIGÉES) ----------
    def calculate_univariate_importance():
        """
         CORRIGÉ : Calcule importance univariée avec tests adaptés au type de feature
        - Chi² + Cramér's V pour features catégorielles/binaires
        - ANOVA F + Eta² pour features continues (classification)
        - Pearson pour régression
        """
        results = []
        
        for feature in active_features:
            try:
                X_feature = X_train[feature].dropna()
                y_aligned = y_train.loc[X_feature.index]
                
                if len(X_feature) == 0 or len(y_aligned) == 0:
                    results.append({
                        "feature": feature,
                        "corr": 0,
                        "test_name": "N/A",
                        "test_stat": 0,
                        "p_value": 1,
                        "signif": "",
                        "signif_color": "#01335A !important"
                    })
                    continue
                
                # Détecter le type de feature
                n_unique = X_feature.nunique()
                is_binary = n_unique == 2
                is_categorical = n_unique <= 10 and n_unique > 0
                
                # Choisir la bonne méthode selon le type
                if is_classification:
                    # CLASSIFICATION
                    if is_binary or is_categorical:
                        #  Test Chi² pour features catégorielles
                        try:
                            contingency_table = pd.crosstab(X_feature, y_aligned)
                            
                            # Vérifier que la table n'est pas vide
                            if contingency_table.size == 0 or contingency_table.sum().sum() == 0:
                                raise ValueError("Contingency table vide")
                            
                            chi2_stat, p_val_test, dof, expected = chi2_contingency(contingency_table)
                            
                            # Cramér's V (mesure force association 0-1)
                            n = contingency_table.sum().sum()
                            min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
                            
                            if min_dim > 0 and n > 0:
                                cramers_v = np.sqrt(chi2_stat / (n * min_dim))
                            else:
                                cramers_v = 0
                            
                            corr = cramers_v  # Utiliser Cramér's V comme "corrélation"
                            test_stat = chi2_stat
                            test_name = "Chi²"
                            
                        except Exception as e:
                            print(f" Erreur Chi² pour {feature}: {e}")
                            corr, test_stat, p_val_test = 0, 0, 1
                            test_name = "Chi²"
                    
                    else:
                        #  ANOVA F-test pour features continues
                        try:
                            groups = [X_feature[y_aligned == c].dropna() for c in y_aligned.unique()]
                            groups = [g for g in groups if len(g) > 0]
                            
                            if len(groups) > 1:
                                f_stat, p_val_test = f_oneway(*groups)
                                test_stat = f_stat
                                
                                # Eta squared (effect size)
                                overall_mean = X_feature.mean()
                                ss_between = sum(len(g) * (g.mean() - overall_mean)**2 for g in groups)
                                ss_total = ((X_feature - overall_mean)**2).sum()
                                
                                if ss_total > 0:
                                    eta_squared = ss_between / ss_total
                                    corr = np.sqrt(eta_squared)  # Correlation ratio
                                else:
                                    corr = 0
                            else:
                                f_stat, p_val_test = 0, 1
                                corr = 0
                            
                            test_name = "ANOVA F"
                            
                        except Exception as e:
                            print(f" Erreur ANOVA pour {feature}: {e}")
                            corr, test_stat, p_val_test = 0, 0, 1
                            test_name = "ANOVA F"
                
                else:
                    # RÉGRESSION
                    try:
                        # Pearson correlation pour toutes features
                        corr, p_val_test = pearsonr(X_feature, y_aligned)
                        corr = abs(corr)
                        test_stat = corr
                        test_name = "Pearson"
                        
                    except Exception as e:
                        print(f" Erreur corrélation pour {feature}: {e}")
                        corr, test_stat, p_val_test = 0, 0, 1
                        test_name = "Pearson"
                
                # Significativité
                if p_val_test < 0.001:
                    signif = "***"
                    signif_color = "#01335A"
                elif p_val_test < 0.01:
                    signif = "**"
                    signif_color = "#01335A"
                elif p_val_test < 0.05:
                    signif = "*"
                    signif_color = "#01335A"
                else:
                    signif = ""
                    signif_color = "#666666"
                
                results.append({
                    "feature": feature,
                    "corr": float(corr) if not np.isnan(corr) else 0,
                    "test_name": test_name,
                    "test_stat": float(test_stat) if not np.isnan(test_stat) else 0,
                    "p_value": float(p_val_test) if not np.isnan(p_val_test) else 1,
                    "signif": signif,
                    "signif_color": signif_color
                })
            
            except Exception as e:
                print(f"❌ Erreur globale pour {feature}: {e}")
                results.append({
                    "feature": feature,
                    "corr": 0,
                    "test_name": "ERROR",
                    "test_stat": 0,
                    "p_value": 1,
                    "signif": "",
                    "signif_color": "#666666"
                })
        
        # Normaliser importance relative (0-100%)
        df_results = pd.DataFrame(results)
        
        if len(df_results) > 0:
            max_corr = df_results["corr"].max()
            if max_corr > 0:
                df_results["importance"] = (df_results["corr"] / max_corr * 100).round(0).astype(int)
            else:
                df_results["importance"] = 0
        else:
            df_results["importance"] = 0
        
        return df_results.sort_values("importance", ascending=False)
    
    def calculate_mutual_information():
        """Calcule Mutual Information (capture relations non-linéaires)"""
        try:
            if is_classification:
                mi_scores = mutual_info_classif(X_train[active_features], y_train, random_state=42)
            else:
                mi_scores = mutual_info_regression(X_train[active_features], y_train, random_state=42)
            
            df_mi = pd.DataFrame({
                "feature": active_features,
                "mi_score": mi_scores
            }).sort_values("mi_score", ascending=False)
            
            # Importance relative
            max_mi = df_mi["mi_score"].max()
            if max_mi > 0:
                df_mi["importance"] = (df_mi["mi_score"] / max_mi * 100).round(0).astype(int)
            else:
                df_mi["importance"] = 0
            
            return df_mi
        
        except Exception as e:
            print(f"❌ Erreur MI: {e}")
            return pd.DataFrame()
    
    def create_importance_barplot(df_importance, title="Feature Importance"):
        """Crée un barplot horizontal des importances"""
        if len(df_importance) == 0:
            return go.Figure()
        
        fig = go.Figure()
        
        # Gradient de nuances de #01335A
        colors = []
        for imp in df_importance["importance"]:
            if imp > 60:
                colors.append('#01335A')
            elif imp > 30:
                colors.append('#023d6b')
            else:
                colors.append('#04507c')
        
        fig.add_trace(go.Bar(
            y=df_importance["feature"],
            x=df_importance["importance"],
            orientation='h',
            marker=dict(color=colors),
            text=df_importance["importance"].astype(str) + "%",
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Importance: %{x}%<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance Relative (%)",
            yaxis_title="",
            height=max(400, len(df_importance) * 30),
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False,
            paper_bgcolor='#f5f7fa',
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_scatter_feature_vs_target(feature_name):
        """Crée scatter plot feature vs target"""
        fig = go.Figure()
        
        X_feature = X_train[feature_name]
        
        if is_classification:
            # Pour classification : boxplot par classe
            for class_val in sorted(y_train.unique()):
                mask = y_train == class_val
                fig.add_trace(go.Box(
                    y=X_feature[mask],
                    name=f"Classe {class_val}",
                    boxmean='sd',
                    marker_color='#01335A'
                ))
            
            fig.update_layout(
                title=f"{feature_name} par Classe",
                yaxis_title=feature_name,
                xaxis_title="Classe",
                height=400,
                paper_bgcolor='#f5f7fa',
                plot_bgcolor='white'
            )
        else:
            # Pour régression : scatter plot
            fig.add_trace(go.Scatter(
                x=X_feature,
                y=y_train,
                mode='markers',
                marker=dict(size=5, opacity=0.6, color='#01335A')
            ))
            
            fig.update_layout(
                title=f"{feature_name} vs Target",
                xaxis_title=feature_name,
                yaxis_title=target_col,
                height=400,
                paper_bgcolor='#f5f7fa',
                plot_bgcolor='white'
            )
        
        return fig
    
    def get_feature_icon(importance):
        """Retourne icône selon importance"""
        if importance > 60:
            return "🟢"
        elif importance > 30:
            return "🟡"
        else:
            return "🔴"
    
    def create_progress_bar_text(importance):
        """Crée barre de progression ASCII"""
        filled = int(importance / 10)
        empty = 10 - filled
        return "█" * filled + "▌" * min(1, empty)
    
    def skip_to_training():
        """Passer directement au training sans modifier features"""
        state["feature_selection_applied"] = False
        ui.notify(" Passage au training avec toutes les features", color="info")
        ui.run_javascript("setTimeout(() => window.location.href='/supervised/training', 1000);")
    
    def apply_feature_selection(top_n):
        """Applique sélection de features et continue"""
        univariate_results = calculate_univariate_importance()
        selected_features = univariate_results.head(top_n)["feature"].tolist()
        
        if len(selected_features) == 0:
            ui.notify("❌ Aucune feature à sélectionner", color="negative")
            return
        
        # Sauvegarder dans state
        state["selected_features_importance"] = selected_features
        state["feature_selection_applied"] = True
        state["n_features_selected"] = top_n
        
        # Appliquer sur les splits
        split = state.get("split", {})
        for key in ["X_train", "X_val", "X_test"]:
            if key in split and isinstance(split[key], pd.DataFrame):
                # Garder seulement les features sélectionnées qui existent
                valid_features = [f for f in selected_features if f in split[key].columns]
                split[key] = split[key][valid_features]
        
        state["split"] = split
        
        ui.notify(f" Sélection appliquée : {len(selected_features)} features conservées", color="positive")
        ui.run_javascript("setTimeout(() => window.location.href='/supervised/training', 1500);")
    
    # ---------- CALCULS ----------
    ui.notify("🔄 Calcul des importances en cours...", color="info", timeout=2000)
    
    univariate_results = calculate_univariate_importance()
    mi_results = calculate_mutual_information()
    
    # ---------- INTERFACE ----------
    with ui.column().classes("w-full items-center p-8").style("background-color:#f5f7fa !important; min-height:100vh !important;"):
        
        # ==================== HEADER ====================
        ui.label(" ANALYSE D'IMPORTANCE DES FEATURES").style(
            "font-weight:700 !important; font-size:32px !important; color:#01335A !important; margin-bottom:8px !important; text-align:center !important;"
        )
        ui.label(f"Analyse préliminaire pour guider la modélisation ({task_type})").style(
            "font-size:18px !important; color:#023d6b !important; margin-bottom:32px !important; text-align:center !important;"
        )
        
        # Info dataset
        with ui.card().classes("w-full max-w-6xl p-4 mb-6").style("background:#ffffff !important; border:2px solid #01335A !important; border-radius:12px !important;"):
            with ui.row().classes("w-full gap-8 items-center"):
                ui.label(" Dataset :").style("font-weight:700 !important; font-size:16px !important; color:#01335A !important;")
                ui.label(f"Features: {len(active_features)}").style("font-size:14px !important; color:#023d6b !important;")
                ui.label(f"Samples: {len(X_train):,}").style("font-size:14px !important; color:#023d6b !important;")
                ui.label(f"Target: {target_col}").style("font-size:14px !important; color:#023d6b !important;")
                ui.label(f"Type: {task_type}").style("font-size:14px !important; color:#023d6b !important;")
        
        # ==================== SECTION A : IMPORTANCE UNIVARIÉE ====================
        with ui.card().classes("w-full max-w-6xl p-6 mb-8").style("background:#ffffff !important; border-radius:12px !important; box-shadow:0 2px 8px rgba(1,51,90,0.1) !important;"):
            ui.label(" SECTION A : IMPORTANCE UNIVARIÉE").style(
                "font-weight:700 !important; font-size:24px !important; color:#01335A !important; margin-bottom:16px !important;"
            )
            
            ui.label("Méthode : Tests Statistiques Adaptés au Type de Feature").style(
                "font-size:16px !important; color:#023d6b !important; margin-bottom:12px !important;"
            )
            
            # Info méthode
            with ui.card().classes("w-full p-3 mb-4").style("background:#e8f0f7 !important; border-left:3px solid #01335A !important; border-radius:8px !important;"):
                ui.label(" Méthodes utilisées :").style("font-weight:600 !important; color:#01335A !important; margin-bottom:6px !important;")
                ui.label("• Chi² + Cramér's V : Variables catégorielles/binaires").style("font-size:13px !important; color:#023d6b !important;")
                ui.label("• ANOVA F + Eta² : Variables continues (classification)").style("font-size:13px !important; color:#023d6b !important;")
                ui.label("• Pearson : Variables continues (régression)").style("font-size:13px !important; color:#023d6b !important;")
            
            # Tableau détaillé
            if len(univariate_results) > 0:
                # Créer tableau HTML stylisé
                table_html = """
                <div style="background:#01335A !important; border-radius:12px !important; padding:20px !important; overflow-x:auto !important; margin-bottom:24px !important;">
                <table style="width:100% !important; color:#ffffff !important; font-family:monospace !important; font-size:13px !important; border-collapse:collapse !important;">
                    <thead>
                        <tr style="border-bottom:2px solid #ffffff !important;">
                            <th style="text-align:left !important; padding:12px !important; color:#ffffff !important;">Feature</th>
                            <th style="text-align:center !important; padding:12px !important; color:#ffffff !important;">Assoc. Target</th>
                            <th style="text-align:center !important; padding:12px !important; color:#ffffff !important;">Test</th>
                            <th style="text-align:center !important; padding:12px !important; color:#ffffff !important;">Test Stat</th>
                            <th style="text-align:center !important; padding:12px !important; color:#ffffff !important;">Signif.</th>
                            <th style="text-align:left !important; padding:12px !important; color:#ffffff !important;">Importance</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                
                for _, row in univariate_results.iterrows():
                    icon = get_feature_icon(row["importance"])
                    bar = create_progress_bar_text(row["importance"])
                    
                    # Tronquer nom feature si trop long
                    feature_display = row['feature'][:25] + "..." if len(row['feature']) > 25 else row['feature']
                    
                    table_html += f"""
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.2) !important;">
                            <td style="padding:10px !important; font-weight:600 !important; color:#ffffff !important;">{feature_display}</td>
                            <td style="text-align:center !important; padding:10px !important; color:#ffffff !important;">{row['corr']:.3f} {icon}</td>
                            <td style="text-align:center !important; padding:10px !important; color:#d4e1ed !important;">{row['test_name']}</td>
                            <td style="text-align:center !important; padding:10px !important; color:#ffffff !important;">{row['test_stat']:.2f}</td>
                            <td style="text-align:center !important; padding:10px !important; color:{row['signif_color']} !important; font-weight:700 !important;">{row['signif']}</td>
                            <td style="padding:10px !important; color:#ffffff !important;">{bar} {row['importance']}%</td>
                        </tr>
                    """
                
                table_html += """
                    </tbody>
                </table>
                </div>
                """
                
                ui.html(table_html, sanitize=False)
                
                # Légende significativité
                with ui.card().classes("w-full p-4 mb-4").style("background:#e8f0f7 !important; border-radius:8px !important;"):
                    ui.label("📖 Légende Significativité (p-value) :").style("font-weight:700 !important; color:#01335A !important; margin-bottom:8px !important;")
                    ui.label("*** p < 0.001 (Très significatif) - Forte évidence d'association").style("font-size:14px !important; color:#023d6b !important;")
                    ui.label("** p < 0.01 (Significatif) - Bonne évidence d'association").style("font-size:14px !important; color:#023d6b !important;")
                    ui.label("* p < 0.05 (Marginalement significatif) - Évidence modérée").style("font-size:14px !important; color:#023d6b !important;")
                    ui.label("(vide) p >= 0.05 (Non significatif) - Pas d'évidence d'association").style("font-size:14px !important; color:#666666 !important;")
                
                # Observations
                high_importance = univariate_results[univariate_results["importance"] > 60]
                medium_importance = univariate_results[(univariate_results["importance"] >= 30) & (univariate_results["importance"] <= 60)]
                low_importance = univariate_results[univariate_results["importance"] < 20]
                
                with ui.card().classes("w-full p-4 mb-4").style("background:#e8f0f7 !important; border-left:4px solid #01335A !important; border-radius:8px !important;"):
                    ui.label("💡 Observations :").style("font-weight:700 !important; color:#01335A !important; margin-bottom:8px !important;")
                    
                    if len(high_importance) > 0:
                        features_str = ", ".join(high_importance["feature"].head(5).tolist())
                        if len(high_importance) > 5:
                            features_str += f" (+ {len(high_importance)-5} autres)"
                        ui.label(f"• 🟢 {len(high_importance)} features très prédictives (importance > 60%) : {features_str}").style("font-size:14px !important; color:#023d6b !important; margin-bottom:4px !important;")
                    
                    if len(medium_importance) > 0:
                        ui.label(f"• 🟡 {len(medium_importance)} features moyennement prédictives (30-60%)").style("font-size:14px !important; color:#023d6b !important; margin-bottom:4px !important;")
                    
                    if len(low_importance) > 0:
                        ui.label(f"• 🔴 {len(low_importance)} features faiblement prédictives (< 20%)").style("font-size:14px !important; color:#023d6b !important; margin-bottom:4px !important;")
                    
                    # Recommandation nombre features
                    recommended_n = len(univariate_results[univariate_results["importance"] >= 30])
                    if recommended_n > 0:
                        ui.label(f"• 💡 Recommandation : Conserver les {recommended_n} features (importance >= 30%) pour un bon compromis").style(
                            "font-size:14px !important; font-weight:600 !important; color:#01335A !important; margin-top:8px !important;"
                        )
                
                # Barplot horizontal
                ui.label(" Visualisation des Importances").style(
                    "font-weight:700 !important; font-size:20px !important; color:#01335A !important; margin-top:20px !important; margin-bottom:12px !important;"
                )
                
                barplot_fig = create_importance_barplot(univariate_results, "Importance Univariée des Features")
                barplot_plot = ui.plotly(barplot_fig).classes("w-full")
                
                # Container pour scatter plot interactif
                scatter_container = ui.column().classes("w-full mt-4")
                
                # Callback sur click barre
                def handle_bar_click(e):
                    try:
                        if e and "points" in e and len(e["points"]) > 0:
                            point = e["points"][0]
                            feature_clicked = point["y"]
                            
                            scatter_container.clear()
                            with scatter_container:
                                ui.label(f" Relation : {feature_clicked} vs {target_col}").style(
                                    "font-weight:700 !important; font-size:18px !important; color:#01335A !important; margin-bottom:12px !important;"
                                )
                                
                                scatter_fig = create_scatter_feature_vs_target(feature_clicked)
                                ui.plotly(scatter_fig).classes("w-full")
                    except Exception as exc:
                        print(f"❌ Erreur click: {exc}")
                
                barplot_plot.on("plotly_click", handle_bar_click)
                
                ui.label("💡 Cliquez sur une barre pour voir la relation avec la target").style(
                    "font-size:13px !important; color:#023d6b !important; margin-top:8px !important; text-align:center !important;"
                )
            
            else:
                ui.label(" Aucune feature disponible pour l'analyse").style("color:#666666 !important; font-size:16px !important;")
        
        # ==================== SECTION B : MUTUAL INFORMATION ====================
        if len(mi_results) > 0:
            with ui.card().classes("w-full max-w-6xl p-6 mb-8").style("background:#ffffff !important; border-radius:12px !important; box-shadow:0 2px 8px rgba(1,51,90,0.1) !important;"):
                ui.label("🔄 SECTION B : IMPORTANCE MULTIVARIÉE (MUTUAL INFORMATION)").style(
                    "font-weight:700 !important; font-size:24px !important; color:#01335A !important; margin-bottom:16px !important;"
                )
                
                ui.label("Méthode : Mutual Information (capture relations non-linéaires)").style(
                    "font-size:16px !important; color:#023d6b !important; margin-bottom:20px !important;"
                )
                
                # Tableau MI
                table_html_mi = """
                <div style="background:#01335A !important; border-radius:12px !important; padding:20px !important; overflow-x:auto !important; margin-bottom:24px !important;">
                <table style="width:100% !important; color:#ffffff !important; font-family:monospace !important; font-size:13px !important; border-collapse:collapse !important;">
                    <thead>
                        <tr style="border-bottom:2px solid #ffffff !important;">
                            <th style="text-align:left !important; padding:12px !important; color:#ffffff !important;">Feature</th>
                            <th style="text-align:center !important; padding:12px !important; color:#ffffff !important;">MI Score</th>
                            <th style="text-align:left !important; padding:12px !important; color:#ffffff !important;">Importance Relative</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                
                for _, row in mi_results.iterrows():
                    bar = create_progress_bar_text(row["importance"])
                    feature_display = row['feature'][:25] + "..." if len(row['feature']) > 25 else row['feature']
                    
                    table_html_mi += f"""
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.2) !important;">
                            <td style="padding:10px !important; font-weight:600 !important; color:#ffffff !important;">{feature_display}</td>
                            <td style="text-align:center !important; padding:10px !important; color:#ffffff !important;">{row['mi_score']:.4f}</td>
                            <td style="padding:10px !important; color:#ffffff !important;">{bar} {row['importance']}%</td>
                        </tr>
                    """
                
                table_html_mi += """
                    </tbody>
                </table>
                </div>
                """
                
                ui.html(table_html_mi, sanitize=False)
                
                # Info MI
                with ui.card().classes("w-full p-4").style("background:#e8f0f7 !important; border-radius:8px !important;"):
                    ui.label("💡 Mutual Information (Avantages)").style(
                        "font-weight:700 !important; color:#01335A !important; margin-bottom:8px !important;"
                    )
                    ui.label(" Capture relations non-linéaires (complète l'analyse univariée)").style("font-size:14px !important; color:#023d6b !important;")
                    ui.label(" Pas d'hypothèse sur la distribution des données").style("font-size:14px !important; color:#023d6b !important;")
                    ui.label(" Fonctionne avec variables continues ET catégorielles").style("font-size:14px !important; color:#023d6b !important;")
                    ui.label(" Détecte interactions complexes avec la target").style("font-size:14px !important; color:#023d6b !important;")
                
                # Barplot MI
                ui.label(" Visualisation MI Scores").style(
                    "font-weight:700 !important; font-size:20px !important; color:#01335A !important; margin-top:20px !important; margin-bottom:12px !important;"
                )
                
                mi_barplot = create_importance_barplot(mi_results, "Mutual Information Scores")
                ui.plotly(mi_barplot).classes("w-full")
        
        # ==================== SECTION C : RECOMMANDATIONS ====================
        with ui.card().classes("w-full max-w-6xl p-6 mb-8").style(
            "background:#01335A !important; border-radius:16px !important;"
        ):
            ui.label("💡 RECOMMANDATIONS AVANT TRAINING").style(
                "font-weight:700 !important; font-size:24px !important; color:#ffffff !important; margin-bottom:16px !important; text-align:center !important;"
            )
            
            ui.label("Basé sur l'analyse d'importance :").style(
                "font-size:16px !important; color:#d4e1ed !important; text-align:center !important; margin-bottom:20px !important;"
            )
            
            if len(univariate_results) > 0:
                # Options
                with ui.column().classes("w-full gap-4"):
                    # Option 1 : Garder toutes
                    with ui.card().classes("p-6").style("background:#ffffff !important; border-radius:12px !important;"):
                        ui.label(f"Option 1 : Garder toutes les features ({len(active_features)})").style(
                            "font-weight:700 !important; font-size:18px !important; color:#01335A !important; margin-bottom:12px !important;"
                        )
                        
                        with ui.row().classes("gap-6 mb-4"):
                            with ui.column().classes("flex-1"):
                                ui.label(" Avantages").style("font-weight:600 !important; color:#01335A !important; margin-bottom:6px !important;")
                                ui.label("• Maximise information disponible").style("font-size:14px !important; color:#023d6b !important;")
                                ui.label("• Pas de risque de perte d'info").style("font-size:14px !important; color:#023d6b !important;")
                                ui.label("• Laisse l'algo choisir").style("font-size:14px !important; color:#023d6b !important;")
                            
                            with ui.column().classes("flex-1"):
                                ui.label("❌ Inconvénients").style("font-weight:600 !important; color:#666666 !important; margin-bottom:6px !important;")
                                ui.label("• Risque léger de bruit").style("font-size:14px !important; color:#666666 !important;")
                                ui.label("• Training plus lent").style("font-size:14px !important; color:#666666 !important;")
                                ui.label("• Overfitting possible").style("font-size:14px !important; color:#666666 !important;")
                        
                        ui.button(
                            " Continuer avec toutes les features",
                            on_click=skip_to_training
                        ).style(
                            "background:#01335A !important; color:#ffffff !important; font-weight:600 !important; height:48px !important; width:100% !important; border-radius:8px !important;"
                        )
                    
                    # Option 2 : Réduire Top N
                    recommended_n = len(univariate_results[univariate_results["importance"] >= 30])
                    
                    if recommended_n > 0 and recommended_n < len(active_features):
                        with ui.card().classes("p-6").style("background:#ffffff !important; border-radius:12px !important;"):
                            ui.label(f"Option 2 : Réduire à Top {recommended_n} features (importance >= 30%)").style(
                                "font-weight:700 !important; font-size:18px !important; color:#01335A !important; margin-bottom:12px !important;"
                            )
                            
                            top_features = univariate_results[univariate_results["importance"] >= 30]["feature"].tolist()
                            ui.label(f"Features conservées : {', '.join(top_features[:5])}{'...' if len(top_features) > 5 else ''}").style(
                                "font-size:13px !important; color:#023d6b !important; margin-bottom:12px !important;"
                            )
                            
                            with ui.row().classes("gap-6 mb-4"):
                                with ui.column().classes("flex-1"):
                                    ui.label(" Avantages").style("font-weight:600 !important; color:#01335A !important; margin-bottom:6px !important;")
                                    ui.label("• Élimine bruit potentiel").style("font-size:14px !important; color:#023d6b !important;")
                                    ui.label("• Accélère training").style("font-size:14px !important; color:#023d6b !important;")
                                    ui.label("• Réduit overfitting").style("font-size:14px !important; color:#023d6b !important;")
                                
                                with ui.column().classes("flex-1"):
                                    ui.label("❌ Inconvénients").style("font-weight:600 !important; color:#666666 !important; margin-bottom:6px !important;")
                                    low_features = univariate_results[univariate_results["importance"] < 30]
                                    n_removed = len(low_features)
                                    ui.label(f"• Supprime {n_removed} features").style("font-size:14px !important; color:#666666 !important;")
                                    ui.label("• Risque léger de sous-apprentissage").style("font-size:14px !important; color:#666666 !important;")
                            
                            ui.button(
                                f"✂️ Réduire à Top {recommended_n} features [Recommandé]",
                                on_click=lambda: apply_feature_selection(recommended_n)
                            ).style(
                                "background:#01335A !important; color:#ffffff !important; font-weight:600 !important; height:48px !important; width:100% !important; border-radius:8px !important;"
                            )
                    
                    # Option 3 : Custom
                    with ui.card().classes("p-6").style("background:#ffffff !important; border-radius:12px !important;"):
                        ui.label("Option 3 : Sélection personnalisée").style(
                            "font-weight:700 !important; font-size:18px !important; color:#01335A !important; margin-bottom:12px !important;"
                        )
                        
                        with ui.row().classes("w-full items-center gap-4 mb-4"):
                            ui.label("Nombre de features à conserver :").style("font-weight:600 !important; color:#01335A !important; width:250px !important;")
                            custom_n_slider = ui.slider(
                                min=1, 
                                max=len(active_features), 
                                value=min(recommended_n if recommended_n > 0 else len(active_features), len(active_features)), 
                                step=1
                            ).classes("flex-1")
                            custom_n_label = ui.label(f"{custom_n_slider.value}").style("font-weight:700 !important; color:#01335A !important; width:60px !important;")
                            
                            def update_custom_n(e):
                                custom_n_label.set_text(f"{int(e.value)}")
                            
                            custom_n_slider.on_value_change(update_custom_n)
                        
                        ui.button(
                            "✂️ Appliquer sélection personnalisée",
                            on_click=lambda: apply_feature_selection(int(custom_n_slider.value))
                        ).style(
                            "background:#023d6b !important; color:#ffffff !important; font-weight:600 !important; height:48px !important; width:100% !important; border-radius:8px !important;"
                        )
        
        # ==================== COMPARAISON MÉTHODES ====================
        if len(mi_results) > 0 and len(univariate_results) > 0:
            with ui.card().classes("w-full max-w-6xl p-6 mb-8").style("background:#ffffff !important; border-radius:12px !important; box-shadow:0 2px 8px rgba(1,51,90,0.1) !important;"):
                ui.label(" COMPARAISON DES MÉTHODES").style(
                    "font-weight:700 !important; font-size:24px !important; color:#01335A !important; margin-bottom:16px !important;"
                )
                
                # Merge univariate et MI
                comparison_df = univariate_results[["feature", "importance"]].copy()
                comparison_df.columns = ["feature", "univariate"]
                
                mi_scores = mi_results[["feature", "importance"]].copy()
                mi_scores.columns = ["feature", "multivariate"]
                
                comparison_df = comparison_df.merge(mi_scores, on="feature", how="left")
                comparison_df["multivariate"] = comparison_df["multivariate"].fillna(0)
                
                # Graphique comparatif
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=comparison_df["feature"],
                    y=comparison_df["univariate"],
                    name="Univariée (Chi²/ANOVA)",
                    marker_color='#01335A'
                ))
                
                fig.add_trace(go.Bar(
                    x=comparison_df["feature"],
                    y=comparison_df["multivariate"],
                    name="Multivariée (MI)",
                    marker_color='#023d6b'
                ))
                
                fig.update_layout(
                    title="Comparaison Importance Univariée vs Multivariée",
                    xaxis_title="Features",
                    yaxis_title="Importance (%)",
                    barmode='group',
                    height=500,
                    showlegend=True,
                    paper_bgcolor='#f5f7fa',
                    plot_bgcolor='white'
                )
                
                ui.plotly(fig).classes("w-full")
                
                # Interprétation
                with ui.card().classes("w-full p-4 mt-4").style("background:#e8f0f7 !important; border-radius:8px !important;"):
                    ui.label("💡 Interprétation :").style("font-weight:700 !important; color:#01335A !important; margin-bottom:8px !important;")
                    ui.label("• Si les deux méthodes donnent des résultats similaires  Relations linéaires/monotones dominantes").style("font-size:14px !important; color:#023d6b !important;")
                    ui.label("• Si MI > Analyse univariée  Relations non-linéaires importantes").style("font-size:14px !important; color:#023d6b !important;")
                    ui.label("• Features avec MI élevé mais score univarié faible  Interactions complexes").style("font-size:14px !important; color:#023d6b !important;")
                    ui.label("• Consensus entre les deux méthodes  Features robustement importantes").style("font-size:14px !important; color:#023d6b !important;")
        
        # ==================== NAVIGATION ====================
        with ui.row().classes("w-full max-w-6xl justify-between gap-4 mt-8"):
            ui.button(
                " Retour Config Algos",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/algorithm_config'")
            ).style(
                "background:#01335A !important; color:#ffffff !important; font-weight:600 !important; height:56px !important; padding:0 32px !important; border-radius:12px !important; font-size:16px !important;"
            )



@ui.page('/supervised/training')
def training_page():
    """
    ENTRAÎNEMENT INITIAL (BASELINE)
    
    - Entraînement des modèles sélectionnés
    - Progress bars en temps réel
    - Validation sur hold-out ou CV
    - Métriques de performance
    - Comparaison des modèles
    - Sauvegarde des meilleurs modèles
    """
    
    
    # ---------- CONTEXTE ----------
    split = state.get("split", None)
    target_col = state.get("target_column", None)
    algo_configs = state.get("algo_configs", {})
    selected_algos = state.get("selected_algos", [])
    validation_strategy = state.get("validation_strategy", "holdout")
    cv_folds = state.get("cv_folds", 5)
    metrics_to_track = state.get("metrics_to_track", ["accuracy", "precision", "recall", "f1"])
    
    if split is None or target_col is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Données non disponibles. Complétez les étapes précédentes.").style(
                "font-size:18px; color:#c0392b; font-weight:600;"
            )
            ui.button(" Retour", on_click=lambda: ui.run_javascript("window.location.href='/supervised/feature_importance'"))
        return
    
    X_train = split.get("X_train")
    y_train = split.get("y_train")
    X_val = split.get("X_val")
    y_val = split.get("y_val")
    X_test = split.get("X_test")
    y_test = split.get("y_test")
    
    if X_train is None or y_train is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Données d'entraînement introuvables.").style("font-size:18px; color:#c0392b; font-weight:600;")
        return
    
    if not selected_algos:
        selected_algos = ["KNN", "Decision Tree", "Naive Bayes"]
    
    # ---------- MAPPAGE ALGORITHMES ----------
    algo_mapping = {
        "KNN": {
            "name": "K-Nearest Neighbors",
            "class": KNeighborsClassifier,
            "icon": "",
            "color": "#01335A"
        },
        "Decision Tree": {
            "name": "C4.5 Decision Tree",
            "class": DecisionTreeClassifier,
            "icon": "🌳",
            "color": "#01335A"
        },
        
        "Naive Bayes": {
            "name": "Naive Bayes",
            "class": GaussianNB,
            "icon": "",
            "color": "#e74c3c"
        }
    }
    
    # ---------- FONCTIONS D'ENTRAÎNEMENT ----------
    def create_model(algo_name):
        """Crée une instance du modèle avec sa configuration"""
        if algo_name == "KNN":
            config = algo_configs.get("knn", {})
            return KNeighborsClassifier(
                n_neighbors=config.get("n_neighbors", 5),
                metric=config.get("metric", "euclidean"),
                weights=config.get("weights", "distance"),
                algorithm=config.get("algorithm", "auto")
            )
        
        elif algo_name == "Decision Tree":
            config = algo_configs.get("decision_tree", {})
            return DecisionTreeClassifier(
                criterion=config.get("criterion", "entropy"),
                max_depth=config.get("max_depth", None),
                min_samples_split=config.get("min_samples_split", 2),
                min_samples_leaf=config.get("min_samples_leaf", 1),
                max_features=config.get("max_features", None),
                random_state=42
            )
        
        
        elif algo_name == "Naive Bayes":
            config = algo_configs.get("naive_bayes", {})
            return GaussianNB(
                var_smoothing=config.get("var_smoothing", 1e-9)
            )
        
        return None
    
    def calculate_metrics(y_true, y_pred, y_pred_proba=None):
        """
         CORRIGÉ : Calcule toutes les métriques demandées
        Gère correctement les labels catégoriels (ex: 'N', 'Y')
        """
        results = {}
        
        # Déterminer le nombre de classes uniques
        classes_true = np.unique(y_true)
        classes_pred = np.unique(y_pred)
        all_classes = np.unique(np.concatenate([classes_true, classes_pred]))
        n_classes = len(all_classes)
        
        #  CORRECTION : Déterminer la bonne méthode d'average
        # Pour binaire : utiliser 'binary' seulement si labels sont numériques 0/1
        # Sinon utiliser 'weighted' même pour binaire
        is_numeric_binary = (
            n_classes == 2 and 
            all(isinstance(c, (int, np.integer)) for c in all_classes) and
            set(all_classes).issubset({0, 1})
        )
        
        if is_numeric_binary:
            average_method = 'binary'
            pos_label = 1
        elif n_classes == 2:
            # Binaire mais labels non-numériques (ex: 'N', 'Y')
            average_method = 'binary'
            pos_label = sorted(all_classes)[1]  # Prendre le 2ème label alphabétiquement
        else:
            # Multi-classe
            average_method = 'weighted'
            pos_label = None
        
        try:
            if "accuracy" in metrics_to_track:
                results["accuracy"] = accuracy_score(y_true, y_pred)
            
            if "precision" in metrics_to_track:
                if pos_label is not None:
                    results["precision"] = precision_score(
                        y_true, y_pred, 
                        average=average_method, 
                        pos_label=pos_label,
                        zero_division=0
                    )
                else:
                    results["precision"] = precision_score(
                        y_true, y_pred, 
                        average=average_method,
                        zero_division=0
                    )
            
            if "recall" in metrics_to_track:
                if pos_label is not None:
                    results["recall"] = recall_score(
                        y_true, y_pred, 
                        average=average_method,
                        pos_label=pos_label,
                        zero_division=0
                    )
                else:
                    results["recall"] = recall_score(
                        y_true, y_pred, 
                        average=average_method,
                        zero_division=0
                    )
            
            if "f1" in metrics_to_track:
                if pos_label is not None:
                    results["f1"] = f1_score(
                        y_true, y_pred, 
                        average=average_method,
                        pos_label=pos_label,
                        zero_division=0
                    )
                else:
                    results["f1"] = f1_score(
                        y_true, y_pred, 
                        average=average_method,
                        zero_division=0
                    )
            
            # AUC-ROC (si probabilités disponibles et binaire)
            if y_pred_proba is not None and n_classes == 2:
                try:
                    # Pour labels catégoriels, on doit mapper aux indices
                    if not is_numeric_binary:
                        # Créer mapping labels -> indices
                        label_to_idx = {label: idx for idx, label in enumerate(sorted(all_classes))}
                        y_true_numeric = np.array([label_to_idx[label] for label in y_true])
                        
                        # Prendre proba de la classe positive (2ème classe alphabétiquement)
                        results["auc_roc"] = roc_auc_score(y_true_numeric, y_pred_proba[:, 1])
                    else:
                        results["auc_roc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
                except Exception as e:
                    print(f" Impossible de calculer AUC-ROC: {e}")
                    results["auc_roc"] = None
        
        except Exception as e:
            print(f"❌ Erreur calcul métriques: {e}")
            # Retourner au moins accuracy si possible
            try:
                results["accuracy"] = accuracy_score(y_true, y_pred)
            except:
                results["accuracy"] = 0.0
        
        return results
    
    def train_single_model(algo_name, progress_callback=None, status_callback=None):
        """Entraîne un seul modèle et retourne les résultats"""
        try:
            if status_callback:
                status_callback(f"Initialisation {algo_name}...")
            
            # Créer le modèle
            model = create_model(algo_name)
            
            if model is None:
                return None
            
            if status_callback:
                status_callback(f"Entraînement {algo_name}...")
            
            # Training
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            if progress_callback:
                progress_callback(0.5)
            
            if status_callback:
                status_callback(f"Validation {algo_name}...")
            
            # Prédictions
            if validation_strategy == "holdout" and X_val is not None and y_val is not None:
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
                
                # Métriques validation
                val_metrics = calculate_metrics(y_val, y_pred, y_pred_proba)
                
                # Métriques train (pour détecter overfitting)
                y_train_pred = model.predict(X_train)
                train_metrics = calculate_metrics(y_train, y_train_pred)
            
            elif validation_strategy == "cv":
                # Cross-validation
                cv_scores = {}
                for metric in metrics_to_track:
                    scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=metric)
                    cv_scores[metric] = scores.mean()
                
                val_metrics = cv_scores
                train_metrics = {}
                
                # Re-fit sur tout le train pour avoir le modèle final
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                train_metrics = calculate_metrics(y_train, y_train_pred)
            
            else:
                # Pas de validation séparée
                y_train_pred = model.predict(X_train)
                train_metrics = calculate_metrics(y_train, y_train_pred)
                val_metrics = train_metrics
            
            if progress_callback:
                progress_callback(1.0)
            
            if status_callback:
                status_callback(f" {algo_name} terminé")
            
            # Confusion matrix
            if validation_strategy == "holdout" and X_val is not None:
                cm = confusion_matrix(y_val, y_pred)
            else:
                cm = confusion_matrix(y_train, y_train_pred)
            
            return {
                "model": model,
                "algo_name": algo_name,
                "train_time": train_time,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "confusion_matrix": cm,
                "y_pred": y_pred if validation_strategy == "holdout" else y_train_pred,
                "y_true": y_val if validation_strategy == "holdout" else y_train,
                "y_pred_proba": y_pred_proba if validation_strategy == "holdout" else None
            }
        
        except Exception as e:
            print(f"❌ Erreur training {algo_name}: {e}")
            import traceback
            traceback.print_exc()
            
            if status_callback:
                status_callback(f"❌ Erreur {algo_name}")
            
            return None
    
    async def run_training():
        """Lance l'entraînement de tous les modèles sélectionnés"""
        results = {}
        total_algos = len(selected_algos)
        
        for idx, algo in enumerate(selected_algos):
            # Update global progress
            global_progress.set_value((idx) / total_algos)
            global_status.set_text(f"Entraînement {idx+1}/{total_algos} : {algo}")
            
            # Progress bar et status pour cet algo
            algo_info = algo_mapping.get(algo, {})
            algo_icon = algo_info.get("icon", "")
            algo_color = algo_info.get("color", "#01335A !important")
            
            # Créer carte de training
            with training_cards_container:
                with ui.card().classes("w-full p-4 mb-3").style(f"border-left:5px solid {algo_color};"):
                    with ui.row().classes("w-full items-center gap-3"):
                        ui.label(f"{algo_icon} {algo}").style("font-weight:700; font-size:16px; width:200px;")
                        
                        algo_progress = ui.linear_progress(value=0, show_value=False).classes("flex-1")
                        algo_status = ui.label("En attente...").style("font-size:13px; color:#7f8c8d; width:150px;")
            
            # Callbacks pour update
            def update_progress(value):
                algo_progress.set_value(value)
            
            def update_status(text):
                algo_status.set_text(text)
            
            # Lancer training
            await asyncio.sleep(0.1)  # Pour que l'UI s'update
            
            result = train_single_model(algo, update_progress, update_status)
            
            if result:
                results[algo] = result
                algo_status.set_text(f" Terminé ({result['train_time']:.2f}s)")
                algo_status.style("color:#01335A;")
            else:
                algo_status.set_text("❌ Erreur")
                algo_status.style("color:#e74c3c;")
            
            algo_progress.set_value(1.0)
        
        # Update global
        global_progress.set_value(1.0)
        total_time = sum(r["train_time"] for r in results.values())
        global_status.set_text(f" Training Baseline Complété ({total_time:.2f}s total)")
        
        # Sauvegarder résultats
        state["training_results"] = results
        state["training_timestamp"] = datetime.now().isoformat()
        
        # Afficher boutons résultats
        await asyncio.sleep(0.5)
        show_results_section()
    
    def show_results_section():
        """Affiche la section des résultats après training"""
        results_section.clear()
        
        with results_section:
            with ui.row().classes("w-full justify-center gap-4 mt-6"):
                ui.button(
                    " Voir les Résultats Détaillés",
                    on_click=lambda: ui.run_javascript("window.location.href='/supervised/results'")
                ).style(
                    "background:linear-gradient(135deg, #01335A 0%, #229954 100%); "
                    "color:white; font-weight:700; height:56px; padding:0 40px; border-radius:12px; font-size:16px;"
                )
                
                ui.button(
                    "🔄 Relancer avec autres paramètres",
                    on_click=lambda: ui.run_javascript("window.location.href='/supervised/algorithm_config'")
                ).style(
                    "background:#01335A; color:white; font-weight:600; height:56px; padding:0 32px; border-radius:12px; font-size:16px;"
                )
    
    # ---------- INTERFACE ----------
    with ui.column().classes("w-full items-center p-8").style("background-color:#f8f9fa; min-height:100vh;"):
        
        # ==================== HEADER ====================
        ui.label(" ENTRAÎNEMENT BASELINE").style(
            "font-weight:700; font-size:32px; color:#2c3e50; margin-bottom:8px; text-align:center;"
        )
        ui.label("Établir une baseline avec configurations initiales").style(
            "font-size:18px; color:#7f8c8d; margin-bottom:32px; text-align:center;"
        )
        
        # ==================== CONFIGURATION ====================
        with ui.card().classes("w-full max-w-6xl p-6 mb-6").style("background:#e3f2fd; border-left:5px solid #2196f3;"):
            ui.label("📋 Configuration").style("font-weight:700; font-size:20px; margin-bottom:12px;")
            
            with ui.column().classes("gap-2"):
                ui.label(f"• Dataset : Train ({len(X_train):,} samples, {X_train.shape[1]} features)").style("font-size:14px;")
                
                if validation_strategy == "holdout" and X_val is not None:
                    ui.label(f"• Validation : Hold-out ({len(X_val):,} samples)").style("font-size:14px;")
                elif validation_strategy == "cv":
                    ui.label(f"• Validation : {cv_folds}-Fold Cross-Validation").style("font-size:14px;")
                
                ui.label("• Random State : 42 (reproductibilité)").style("font-size:14px;")
                ui.label(f"• Métriques : {', '.join(metrics_to_track)}").style("font-size:14px;")
        
        # ==================== MODÈLES À ENTRAÎNER ====================
        with ui.card().classes("w-full max-w-6xl p-6 mb-6"):
            ui.label("🤖 Modèles à Entraîner").style("font-weight:700; font-size:20px; margin-bottom:12px;")
            
            for algo in selected_algos:
                algo_info = algo_mapping.get(algo, {})
                icon = algo_info.get("icon", "")
                name = algo_info.get("name", algo)
                
                # Afficher config
                if algo == "KNN":
                    config = algo_configs.get("knn", {})
                    config_str = f"K={config.get('n_neighbors', 5)}, metric={config.get('metric', 'euclidean')}, weights={config.get('weights', 'distance')}"
                elif algo == "Decision Tree":
                    config = algo_configs.get("decision_tree", {})
                    config_str = f"criterion={config.get('criterion', 'entropy')}, max_depth={config.get('max_depth', 'None')}"
                elif algo == "Naive Bayes":
                    config = algo_configs.get("naive_bayes", {})
                    config_str = f"var_smoothing={config.get('var_smoothing', 1e-9):.1e}"
                else:
                    config_str = ""
                
                ui.label(f"  {icon} {name}").style("font-size:15px; font-weight:600; margin-bottom:2px;")
                ui.label(f"      {config_str}").style("font-size:13px; color:#7f8c8d; margin-bottom:8px;")
        
        # ==================== PROGRESS SECTION ====================
        with ui.card().classes("w-full max-w-6xl p-6 mb-6"):
            ui.label("🔄 Progression").style("font-weight:700; font-size:20px; margin-bottom:12px;")
            
            # Progress global
            global_status = ui.label("En attente du lancement...").style("font-size:14px; color:#7f8c8d; margin-bottom:8px;")
            global_progress = ui.linear_progress(value=0, show_value=False).classes("w-full mb-4")
            
            # Container pour les cartes de training individuelles
            training_cards_container = ui.column().classes("w-full")
        
        # ==================== BOUTON LANCER ====================
        launch_button = ui.button(
            " Lancer l'Entraînement",
            on_click=lambda: (launch_button.disable(), ui.timer(0.1, run_training, once=True))
        ).style(
            "background:linear-gradient(135deg, #01335A 0%, #229954 100%); "
            "color:white; font-weight:700; height:56px; padding:0 40px; border-radius:12px; font-size:18px; "
            "margin:20px auto; display:block;"
        )
        
        # ==================== RÉSULTATS (caché au début) ====================
        results_section = ui.column().classes("w-full max-w-6xl")
        
        # ==================== NAVIGATION ====================
        with ui.row().classes("w-full max-w-6xl justify-start gap-4 mt-8"):
            ui.button(
                " Retour Feature Importance",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/feature_importance'")
            ).style(
                "background:#01335A !important; color:white; font-weight:600; height:50px; padding:0 24px; border-radius:10px;"
            )



@ui.page('/supervised/results')
def results_page():
    """
     VALIDATION & MÉTRIQUES
    
    - Tableau comparatif de performance
    - Métriques détaillées par classe
    - Matrices de confusion interactives
    - Courbes ROC
    - Courbes Precision-Recall
    - Feature importance (pour modèles compatibles)
    - Analyse d'erreurs
    """
  
    
    # ---------- CONTEXTE ----------
    training_results = state.get("training_results", {})
    split = state.get("split", {})
    target_col = state.get("target_column", None)
    
    if not training_results:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Aucun résultat d'entraînement disponible.").style(
                "font-size:18px; color:#c0392b; font-weight:600;"
            )
            ui.button(" Retour au Training", on_click=lambda: ui.run_javascript("window.location.href='/supervised/training'"))
        return
    
    # Déterminer classes
    y_val = split.get("y_val")
    if y_val is not None:
        all_classes = sorted(np.unique(y_val))
    else:
        # Fallback
        first_result = list(training_results.values())[0]
        all_classes = sorted(np.unique(first_result["y_true"]))
    
    n_classes = len(all_classes)
    
    # ---------- FONCTIONS HELPERS ----------
    def get_medal(rank):
        """Retourne médaille selon rang"""
        if rank == 1:
            return "🥇"
        elif rank == 2:
            return "🥈"
        elif rank == 3:
            return "🥉"
        else:
            return ""
    
    def create_comparison_table():
        """Crée tableau comparatif des performances"""
        rows = []
        
        for algo_name, result in training_results.items():
            val_metrics = result.get("val_metrics", {})
            train_time = result.get("train_time", 0)
            
            rows.append({
                "Modèle": algo_name,
                "Accuracy": val_metrics.get("accuracy", 0),
                "Precision": val_metrics.get("precision", 0),
                "Recall": val_metrics.get("recall", 0),
                "F1-Score": val_metrics.get("f1", 0),
                "AUC-ROC": val_metrics.get("auc_roc", 0) if val_metrics.get("auc_roc") is not None else 0,
                "Time": train_time
            })
        
        df = pd.DataFrame(rows)
        
        # Trier par Accuracy
        df = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
        
        # Ajouter rangs
        df["Rank"] = range(1, len(df) + 1)
        
        return df
    
    def create_confusion_matrix_plot(algo_name):
        """Crée heatmap de la matrice de confusion"""
        result = training_results.get(algo_name)
        if not result:
            return None
        
        cm = result.get("confusion_matrix")
        if cm is None:
            return None
        
        # Normaliser
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm_normalized,
            x=[f"Pred {c}" for c in all_classes],
            y=[f"Actual {c}" for c in all_classes],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16},
            hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<br>Normalized: %{z:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"{algo_name} - Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400,
            paper_bgcolor='#f8f9fa',
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_roc_curves():
        """Crée graphique avec toutes les courbes ROC"""
        if n_classes != 2:
            return None
        
        fig = go.Figure()
        
        # Ligne diagonale (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Random (AUC=0.5)',
            showlegend=True
        ))
        
        colors = ['#01335A', '#01335A', '#e74c3c', '#01335A', '#9b59b6']
        
        for idx, (algo_name, result) in enumerate(training_results.items()):
            y_true = result.get("y_true")
            y_pred_proba = result.get("y_pred_proba")
            
            if y_pred_proba is None:
                continue
            
            # Convertir labels catégoriels en numériques si nécessaire
            if not all(isinstance(c, (int, np.integer)) for c in all_classes):
                label_to_idx = {label: idx for idx, label in enumerate(sorted(all_classes))}
                y_true_numeric = np.array([label_to_idx[label] for label in y_true])
            else:
                y_true_numeric = y_true
            
            try:
                fpr, tpr, _ = roc_curve(y_true_numeric, y_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                
                fig.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f'{algo_name} (AUC={roc_auc:.3f})',
                    line=dict(color=colors[idx % len(colors)], width=2)
                ))
            except Exception as e:
                print(f"Erreur ROC pour {algo_name}: {e}")
        
        fig.update_layout(
            title=' Courbes ROC (Receiver Operating Characteristic)',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500,
            paper_bgcolor='#f8f9fa',
            plot_bgcolor='white',
            hovermode='x unified'
        )
        
        return fig
    
    def create_precision_recall_curves():
        """Crée graphique Precision-Recall"""
        if n_classes != 2:
            return None
        
        fig = go.Figure()
        
        colors = ['#01335A', '#01335A', '#e74c3c', '#01335A', '#9b59b6']
        
        for idx, (algo_name, result) in enumerate(training_results.items()):
            y_true = result.get("y_true")
            y_pred_proba = result.get("y_pred_proba")
            
            if y_pred_proba is None:
                continue
            
            # Convertir labels catégoriels
            if not all(isinstance(c, (int, np.integer)) for c in all_classes):
                label_to_idx = {label: idx for idx, label in enumerate(sorted(all_classes))}
                y_true_numeric = np.array([label_to_idx[label] for label in y_true])
            else:
                y_true_numeric = y_true
            
            try:
                precision, recall, _ = precision_recall_curve(y_true_numeric, y_pred_proba[:, 1])
                ap = average_precision_score(y_true_numeric, y_pred_proba[:, 1])
                
                fig.add_trace(go.Scatter(
                    x=recall,
                    y=precision,
                    mode='lines',
                    name=f'{algo_name} (AP={ap:.3f})',
                    line=dict(color=colors[idx % len(colors)], width=2)
                ))
            except Exception as e:
                print(f"Erreur PR pour {algo_name}: {e}")
        
        fig.update_layout(
            title=' Courbes Precision-Recall',
            xaxis_title='Recall',
            yaxis_title='Precision',
            height=500,
            paper_bgcolor='#f8f9fa',
            plot_bgcolor='white',
            hovermode='x unified'
        )
        
        return fig
    
    def create_metrics_comparison_chart():
        """Crée graphique radar de comparaison"""
        df_comparison = create_comparison_table()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig = go.Figure()
        
        colors = ['#01335A', '#01335A', '#e74c3c', '#01335A', '#9b59b6']
        
        for idx, row in df_comparison.iterrows():
            values = [row[m] for m in metrics]
            values.append(values[0])  # Fermer le radar
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=row['Modèle'],
                line=dict(color=colors[idx % len(colors)])
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=' Comparaison Multi-Métriques (Radar Chart)',
            height=500,
            paper_bgcolor='#f8f9fa',
            showlegend=True
        )
        
        return fig
    
    def get_classification_report_df(algo_name):
        """Récupère le classification report sous forme de DataFrame"""
        result = training_results.get(algo_name)
        if not result:
            return None
        
        y_true = result.get("y_true")
        y_pred = result.get("y_pred")
        
        if y_true is None or y_pred is None:
            return None
        
        # Générer report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Convertir en DataFrame
        df = pd.DataFrame(report).transpose()
        
        return df
    
    # ---------- INTERFACE ----------
    with ui.column().classes("w-full items-center p-8").style("background-color:#f8f9fa; min-height:100vh;"):
        
        # ==================== HEADER ====================
        ui.label(" RÉSULTATS BASELINE (VALIDATION)").style(
            "font-weight:700; font-size:32px; color:#2c3e50; margin-bottom:8px; text-align:center;"
        )
        ui.label("Analyse comparative des performances").style(
            "font-size:18px; color:#7f8c8d; margin-bottom:32px; text-align:center;"
        )
        
        # ==================== SECTION A : TABLEAU COMPARATIF ====================
        with ui.card().classes("w-full max-w-6xl p-6 mb-8"):
            ui.label(" SECTION A : TABLEAU COMPARATIF DE PERFORMANCE").style(
                "font-weight:700; font-size:24px; color:#2c3e50; margin-bottom:16px;"
            )
            
            df_comparison = create_comparison_table()
            
            # Créer tableau HTML stylisé
            table_html = """
            <div style="background:#1a1a1a; border-radius:12px; padding:20px; overflow-x:auto; margin-bottom:24px;">
            <table style="width:100%; color:#00ff88; font-family:monospace; font-size:14px; border-collapse:collapse;">
                <thead>
                    <tr style="border-bottom:2px solid #00ff88;">
                        <th style="text-align:left; padding:12px; color:#00ffff;">Modèle</th>
                        <th style="text-align:center; padding:12px;">Accuracy</th>
                        <th style="text-align:center; padding:12px;">Precision</th>
                        <th style="text-align:center; padding:12px;">Recall</th>
                        <th style="text-align:center; padding:12px;">F1-Score</th>
                        <th style="text-align:center; padding:12px;">AUC-ROC</th>
                        <th style="text-align:center; padding:12px;">Time</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for idx, row in df_comparison.iterrows():
                medal = get_medal(row["Rank"])
                
                # Couleur selon rang
                if row["Rank"] == 1:
                    row_color = "#ffd700"
                elif row["Rank"] == 2:
                    row_color = "#c0c0c0"
                elif row["Rank"] == 3:
                    row_color = "#cd7f32"
                else:
                    row_color = "#00ff88"
                
                # Formatter AUC-ROC (éviter erreur si 0 ou None)
                auc_value = row['AUC-ROC']
                if auc_value is None or auc_value == 0:
                    auc_display = 'N/A'
                else:
                    auc_display = f"{auc_value:.3f}"
                
                table_html += f"""
                    <tr style="border-bottom:1px solid #333;">
                        <td style="padding:10px; font-weight:600; color:{row_color};">{medal} {row['Modèle']}</td>
                        <td style="text-align:center; padding:10px;">{row['Accuracy']:.3f}</td>
                        <td style="text-align:center; padding:10px;">{row['Precision']:.3f}</td>
                        <td style="text-align:center; padding:10px;">{row['Recall']:.3f}</td>
                        <td style="text-align:center; padding:10px;">{row['F1-Score']:.3f}</td>
                        <td style="text-align:center; padding:10px;">{auc_display}</td>
                        <td style="text-align:center; padding:10px;">{row['Time']:.2f}s</td>
                    </tr>
                """
            
            table_html += """
                </tbody>
            </table>
            </div>
            """
            
            ui.html(table_html, sanitize=False)
            
            # Meilleur modèle
            best_model = df_comparison.iloc[0]
            with ui.card().classes("w-full p-4").style("background:#01335A !important;"):
                ui.label(f"🥇 Meilleur modèle baseline : {best_model['Modèle']}").style(
                    "font-weight:700; font-size:18px; color:white; margin-bottom:4px;"
                )
                ui.label(f"Accuracy: {best_model['Accuracy']:.1%} | F1-Score: {best_model['F1-Score']:.3f} | Time: {best_model['Time']:.2f}s").style(
                    "font-size:14px; color:white;"
                )
            
            # Légende
            with ui.card().classes("w-full p-3 mt-4").style("background:#cde4ff;"):
                ui.label("📖 Légende :").style("font-weight:700; margin-bottom:6px;")
                ui.label("🥇 Meilleur  🥈 Second  🥉 Troisième").style("font-size:14px;")
        
        # ==================== SECTION B : MÉTRIQUES DÉTAILLÉES ====================
        with ui.card().classes("w-full max-w-6xl p-6 mb-8"):
            ui.label("📋 SECTION B : MÉTRIQUES DÉTAILLÉES PAR CLASSE").style(
                "font-weight:700; font-size:24px; color:#2c3e50; margin-bottom:16px;"
            )
            
            # Onglets pour chaque modèle
            with ui.tabs().classes('w-full') as tabs:
                for algo_name in training_results.keys():
                    ui.tab(algo_name)
            
            with ui.tab_panels(tabs, value=list(training_results.keys())[0]).classes('w-full'):
                for algo_name in training_results.keys():
                    with ui.tab_panel(algo_name):
                        result = training_results[algo_name]
                        
                        # Icon selon algo
                        if "KNN" in algo_name:
                            icon = ""
                        elif "Tree" in algo_name:
                            icon = "🌳"
                        elif "Forest" in algo_name:
                            icon = "🌲"
                        elif "Bayes" in algo_name:
                            icon = ""
                        else:
                            icon = "🤖"
                        
                        ui.label(f"{icon} {algo_name.upper()} - MÉTRIQUES DÉTAILLÉES").style(
                            "font-weight:700; font-size:20px; margin-bottom:16px;"
                        )
                        
                        # Classification Report
                        report_df = get_classification_report_df(algo_name)
                        
                        if report_df is not None:
                            # Formater et afficher
                            report_html = """
                            <div style="background:#1a1a1a; border-radius:12px; padding:20px; overflow-x:auto; margin-bottom:24px;">
                            <div style="color:#00ffff; font-family:monospace; font-size:14px; margin-bottom:12px; font-weight:700;">
                                Classification Report :
                            </div>
                            <table style="width:100%; color:#00ff88; font-family:monospace; font-size:13px; border-collapse:collapse;">
                                <thead>
                                    <tr style="border-bottom:2px solid #00ff88;">
                                        <th style="text-align:left; padding:10px; color:#00ffff;">Class</th>
                                        <th style="text-align:center; padding:10px;">Precision</th>
                                        <th style="text-align:center; padding:10px;">Recall</th>
                                        <th style="text-align:center; padding:10px;">F1-Score</th>
                                        <th style="text-align:center; padding:10px;">Support</th>
                                    </tr>
                                </thead>
                                <tbody>
                            """
                            
                            for class_name in all_classes:
                                if str(class_name) in report_df.index:
                                    row = report_df.loc[str(class_name)]
                                    report_html += f"""
                                        <tr style="border-bottom:1px solid #333;">
                                            <td style="padding:10px; font-weight:600;">Class {class_name}</td>
                                            <td style="text-align:center; padding:10px;">{row['precision']:.2f}</td>
                                            <td style="text-align:center; padding:10px;">{row['recall']:.2f}</td>
                                            <td style="text-align:center; padding:10px;">{row['f1-score']:.2f}</td>
                                            <td style="text-align:center; padding:10px;">{int(row['support'])}</td>
                                        </tr>
                                    """
                            
                            # Ajouter moyennes
                            if 'accuracy' in report_df.index:
                                acc_row = report_df.loc['accuracy']
                                report_html += f"""
                                    <tr style="border-top:2px solid #00ff88; border-bottom:1px solid #333;">
                                        <td style="padding:10px; font-weight:700; color:#ffd700;">Accuracy</td>
                                        <td style="text-align:center; padding:10px;" colspan="3">{acc_row['precision']:.2f}</td>
                                        <td style="text-align:center; padding:10px;">{int(acc_row['support'])}</td>
                                    </tr>
                                """
                            
                            for avg_type in ['macro avg', 'weighted avg']:
                                if avg_type in report_df.index:
                                    avg_row = report_df.loc[avg_type]
                                    report_html += f"""
                                        <tr style="border-bottom:1px solid #333;">
                                            <td style="padding:10px; font-weight:600; color:#ffff00;">{avg_type.title()}</td>
                                            <td style="text-align:center; padding:10px;">{avg_row['precision']:.2f}</td>
                                            <td style="text-align:center; padding:10px;">{avg_row['recall']:.2f}</td>
                                            <td style="text-align:center; padding:10px;">{avg_row['f1-score']:.2f}</td>
                                            <td style="text-align:center; padding:10px;">{int(avg_row['support'])}</td>
                                        </tr>
                                    """
                            
                            report_html += """
                                </tbody>
                            </table>
                            </div>
                            """
                            
                            ui.html(report_html, sanitize=False)
                        
                        # Observations
                        y_true = result.get("y_true")
                        if y_true is not None:
                            class_counts = pd.Series(y_true).value_counts()
                            minority_class = class_counts.idxmin()
                            minority_pct = (class_counts.min() / class_counts.sum() * 100)
                            
                            with ui.card().classes("w-full p-4").style("background:#cde4ff;"):
                                ui.label(" Observations :").style("font-weight:700; margin-bottom:8px;")
                                
                                if minority_pct < 30:
                                    ui.label(f"• Classe minoritaire ({minority_class}) : {minority_pct:.1f}% des données").style("font-size:14px;")
                                    ui.label("• Dataset déséquilibré - Considérer class_weight='balanced' ou SMOTE").style("font-size:14px;")
                                
                                if report_df is not None and str(minority_class) in report_df.index:
                                    minority_recall = report_df.loc[str(minority_class), 'recall']
                                    if minority_recall < 0.7:
                                        ui.label(f"• Recall faible pour classe {minority_class} ({minority_recall:.2f}) - Modèle peine à détecter cette classe").style("font-size:14px; color:#01335A;")
        
        # ==================== SECTION C : MATRICES DE CONFUSION ====================
        with ui.card().classes("w-full max-w-6xl p-6 mb-8"):
            ui.label(" SECTION C : MATRICES DE CONFUSION").style(
                "font-weight:700; font-size:24px; color:#2c3e50; margin-bottom:16px;"
            )
            
            ui.label("💡 Cliquez sur une cellule pour analyser les erreurs").style(
                "font-size:14px; color:#7f8c8d; margin-bottom:16px;"
            )
            
            # Grid de matrices
            n_models = len(training_results)
            cols_per_row = 3
            
            with ui.column().classes("w-full gap-4"):
                row_container = None
                
                for idx, algo_name in enumerate(training_results.keys()):
                    if idx % cols_per_row == 0:
                        row_container = ui.row().classes("w-full gap-4")
                    
                    with row_container:
                        with ui.column().classes("flex-1"):
                            cm_fig = create_confusion_matrix_plot(algo_name)
                            if cm_fig:
                                ui.plotly(cm_fig).classes("w-full")
        
        # ==================== SECTION D : COURBES ROC ====================
        if n_classes == 2:
            with ui.card().classes("w-full max-w-6xl p-6 mb-8"):
                ui.label(" SECTION D : COURBES ROC").style(
                    "font-weight:700; font-size:24px; color:#2c3e50; margin-bottom:16px;"
                )
                
                roc_fig = create_roc_curves()
                if roc_fig:
                    ui.plotly(roc_fig).classes("w-full")
                    
                    with ui.card().classes("w-full p-4 mt-4").style("background:#e3f2fd;"):
                        ui.label("💡 Interprétation ROC :").style("font-weight:700; margin-bottom:8px;")
                        ui.label("• AUC = 1.0 : Classificateur parfait").style("font-size:14px;")
                        ui.label("• AUC = 0.5 : Classificateur aléatoire (pas mieux que le hasard)").style("font-size:14px;")
                        ui.label("• AUC > 0.8 : Bonne performance").style("font-size:14px;")
                        ui.label("• Courbe plus proche du coin supérieur gauche = meilleur modèle").style("font-size:14px;")
        
        # ==================== SECTION E : COURBES PRECISION-RECALL ====================
        if n_classes == 2:
            with ui.card().classes("w-full max-w-6xl p-6 mb-8"):
                ui.label(" SECTION E : COURBES PRECISION-RECALL").style(
                    "font-weight:700; font-size:24px; color:#2c3e50; margin-bottom:16px;"
                )
                
                ui.label("Important pour classes déséquilibrées").style(
                    "font-size:16px; color:#7f8c8d; margin-bottom:16px;"
                )
                
                pr_fig = create_precision_recall_curves()
                if pr_fig:
                    ui.plotly(pr_fig).classes("w-full")
                    
                    # Calculer % classe minoritaire
                    first_result = list(training_results.values())[0]
                    y_true = first_result.get("y_true")
                    if y_true is not None:
                        class_counts = pd.Series(y_true).value_counts()
                        minority_pct = (class_counts.min() / class_counts.sum() * 100)
                        
                        with ui.card().classes("w-full p-4 mt-4").style("background:#cde4ff;"):
                            ui.label("💡 Rappel :").style("font-weight:700; margin-bottom:8px;")
                            ui.label(f"• Classe minoritaire représente {minority_pct:.1f}% des données").style("font-size:14px;")
                            ui.label("• Precision-Recall plus informative que ROC pour classes déséquilibrées").style("font-size:14px;")
                            ui.label("• AP (Average Precision) résume la courbe en un seul score").style("font-size:14px;")
        
        # ==================== SECTION F : COMPARAISON RADAR ====================
        with ui.card().classes("w-full max-w-6xl p-6 mb-8"):
            ui.label(" SECTION F : COMPARAISON MULTI-MÉTRIQUES").style(
                "font-weight:700; font-size:24px; color:#2c3e50; margin-bottom:16px;"
            )
            
            radar_fig = create_metrics_comparison_chart()
            if radar_fig:
                ui.plotly(radar_fig).classes("w-full")
        
        # ==================== RECOMMANDATIONS ====================
        with ui.card().classes("w-full max-w-6xl p-6 mb-8").style(
            "background:#01335A !important;"
        ):
            ui.label("💡 RECOMMANDATIONS POUR AMÉLIORER").style(
                "font-weight:700; font-size:24px; color:white; margin-bottom:16px; text-align:center;"
            )
            
            with ui.column().classes("w-full gap-4"):
                # Analyser résultats pour recommandations
                best_model = df_comparison.iloc[0]
                best_acc = best_model["Accuracy"]
                
                with ui.card().classes("p-4"):
                    ui.label(" Pistes d'Amélioration").style("font-weight:700; font-size:18px; margin-bottom:12px;")
                    
                    if best_acc < 0.85:
                        ui.label("🔴 Performance modérée (< 85%)").style("font-weight:600; color:#e74c3c; margin-bottom:8px;")
                        ui.label(" Essayer Grid Search pour optimiser hyperparamètres").style("font-size:14px;")
                        ui.label(" Ajouter plus de features engineering").style("font-size:14px;")
                        ui.label(" Collecter plus de données si possible").style("font-size:14px;")
                    
                    elif best_acc < 0.92:
                        ui.label("🟡 Bonne performance (85-92%)").style("font-weight:600; color:#01335A; margin-bottom:8px;")
                        ui.label(" Hyperparameter tuning peut encore améliorer").style("font-size:14px;")
                        ui.label(" Tester ensemble methods (stacking/blending)").style("font-size:14px;")
                    
                    else:
                        ui.label("🟢 Excellente performance (> 92%)").style("font-weight:600; color:#01335A; margin-bottom:8px;")
                        ui.label(" Valider sur test set pour confirmer").style("font-size:14px;")
                        ui.label(" Surveiller overfitting (comparer train vs val)").style("font-size:14px;")
                
                with ui.card().classes("p-4"):
                    ui.label(" Prochaines Étapes").style("font-weight:700; font-size:18px; margin-bottom:12px;")
                    ui.label("1. Hyperparameter Tuning (Grid Search / Random Search)").style("font-size:14px;")
                    ui.label("2. Validation sur Test Set (évaluation finale)").style("font-size:14px;")
                    ui.label("3. Analyse d'Erreurs (comprendre les faux positifs/négatifs)").style("font-size:14px;")
                    ui.label("4. Déploiement du meilleur modèle").style("font-size:14px;")
        
        # ==================== NAVIGATION ====================
        with ui.row().classes("w-full max-w-6xl justify-between gap-4 mt-8"):
            ui.button(
                " Retour au Training",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/training'")
            ).style(
                "background:#01335A !important; color:white; font-weight:600; height:56px; padding:0 32px; border-radius:12px; font-size:16px;"
            )
            
            ui.button(
                "🔬 Hyperparameter Tuning",
                on_click=lambda: ui.notify("🚧 Fonctionnalité en développement", color="info")
            ).style(
                "background:#01335A; color:white; font-weight:600; height:56px; padding:0 32px; border-radius:12px; font-size:16px;"
            )
            
            ui.button(
                " Évaluation Finale (Test Set)",
                on_click=lambda: ui.notify("🚧 Fonctionnalité en développement", color="info")
            ).style(
                "background:linear-gradient(135deg, #01335A 0%, #229954 100%); "
                "color:white; font-weight:700; height:56px; padding:0 40px; border-radius:12px; font-size:16px;"
            )





# ----------------- UNSUPERVISED-----------------



# ----------------- PAGE UPLOAD -----------------


@ui.page('/unsupervised/upload')
def unsupervised_upload_page():



    # Conteneur principal
    with ui.column() as main_col:
        main_col.style(
            """
            width: 100% !important;
            min-height: 100vh !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: flex-start !important;
            background-color: #f5f6fa !important;
            padding-top: 60px !important;
            font-family: 'Inter', sans-serif !important;
            """
        )

        # Titre
        ui.label("Phase 2 : Chargement et Validation des Données").style(
            """
            font-weight: 700 !important;
            font-size: 32px !important;
            color: #01335A !important;
            margin-bottom: 24px !important;
            text-align: center !important;
            """
        )

        # Carte centrale
        with ui.card() as card:
            card.style(
                """
                padding: 32px !important;
                width: 800px !important;
                border-radius: 12px !important;
                box-shadow: 0 4px 20px rgba(0,0,0,0.12) !important;
                display: flex !important;
                flex-direction: column !important;
                align-items: center !important;
                background-color: white !important;
                """
            )

            ui.label("📂 Importer un Dataset CSV").style(
                """
                font-weight: 700 !important;
                font-size: 24px !important;
                color: #01335A !important;
                text-align: center !important;
                margin-bottom: 10px !important;
                """
            )

            ui.label("Glissez-déposez un fichier CSV ou cliquez pour parcourir").style(
                """
                color: #09538C !important;
                font-size: 15px !important;
                text-align: center !important;
                margin-bottom: 24px !important;
                """
            )

            status_label = ui.label("Aucun fichier chargé").style(
                """
                color: #e74c3c !important;
                font-size: 14px !important;
                margin-bottom: 18px !important;
                font-weight: 600 !important;
                """
            )

            table_placeholder = ui.column().style(
                """
                width: 100% !important;
                margin-top: 20px !important;
                border-top: 1px solid #ecf0f1 !important;
                padding-top: 20px !important;
                """
            )

            async def on_upload(e):
                try:
                    content = await e.file.read()
                    df = pd.read_csv(io.BytesIO(content))
                    state["raw_df"] = df

                    # Mise à jour du statut
                    status_label.text = f"Fichier chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes"
                    status_label.style(
                        """
                        color: #01335A !important;
                        font-size: 14px !important;
                        margin-bottom: 18px !important;
                        font-weight: 600 !important;
                        """
                    )

                    # Affichage aperçu
                    table_placeholder.clear()
                    with table_placeholder:
                        ui.label("Aperçu des 10 premières lignes :").style(
                            """
                            font-weight: 600 !important;
                            color: #01335A !important;
                            font-size: 16px !important;
                            margin-bottom: 8px !important;
                            """
                        )
                        ui.table(rows=df.head(10).to_dict(orient="records")).style(
                            """
                            width: 100% !important;
                            font-size: 14px !important;
                            border: 1px solid #dfe6e9 !important;
                            border-radius: 6px !important;
                            background-color: #fafafa !important;
                            """
                        )

                    btn_next.enable()
                    ui.notify("Dataset chargé avec succès !", color='positive')

                except Exception as err:
                    ui.notify(f"Erreur lors de l'import : {err}", color='negative')

            # Zone de drag & drop
            ui.upload(
                on_upload=on_upload,
                label="Glissez-déposez un fichier CSV ou cliquez ici"
            ).style(
                """
                width: 100% !important;
                padding: 24px !important;
                margin-bottom: 24px !important;
                border: 2px dashed #09538C !important;
                border-radius: 10px !important;
                text-align: center !important;
                font-size: 15px !important;
                color: #01335A !important;
                cursor: pointer !important;
                background-color: #fdfdfd !important;
                transition: all 0.2s ease !important;
                """
            ).props('accept=".csv"')

            # Boutons navigation
            with ui.row() as buttons_row:
                buttons_row.style(
                    """
                    width: 100% !important;
                    display: flex !important;
                    gap: 16px !important;
                    margin-top: 12px !important;
                    """
                )

                btn_next = ui.button("Continuer ")
                btn_next.disable()
                btn_next.style(
                    """
                    width: 100% !important;
                    height: 48px !important;
                    margin-top: 14px !important;
                    border-radius: 8px !important;
                    background: linear-gradient(0deg,rgba(1, 51, 90, 1) 0%, rgba(15, 50, 102, 1) 100%) !important;
                    color: white !important;
                    font-weight: 600 !important;
                    font-size: 15px !important;
                    border: none !important;
                    cursor: pointer !important;
                    """
                )

                ui.button(
                    " Retour",
                    on_click=lambda: ui.run_javascript("window.location.href='/'")
                ).style(
                    """
                    width: 100% !important;
                    height: 48px !important;
                    border-radius: 8px !important;
                    background: #dfe6e9 !important;
                    color: #2c3e50 !important;
                    font-weight: 600 !important;
                    border: none !important;
                    cursor: pointer !important;
                    """
                )

                btn_next.on_click(lambda: ui.run_javascript("window.location.href='/unsupervised/preprocessing'"))



# ----------------- PAGE /unsupervised/preprocessing -----------------



@ui.page('/unsupervised/preprocessing')
def unsupervised_preprocessing_page():

    df = state.get("raw_df", None)

    if df is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Aucun dataset chargé. Veuillez importer un fichier avant de continuer.").style(
                "font-size:18px !important; color:#c0392b !important; font-weight:600 !important;"
            )
            ui.button(" Retour à l'Upload",
                      on_click=lambda: ui.run_javascript("window.location.href='/upload'")).style(
                "margin-top:20px !important; background:#01335A !important; color:white !important; font-weight:600 !important;"
            )
        return

    # ----------- STYLES -----------
    with ui.column().classes("w-full h-auto items-center p-10").style(
        "background-color:#f5f6fa !important; font-family:'Inter', sans-serif !important;"
    ):

        ui.label("Phase 3.1 : Data Understanding & Schema Definition").style(
            "font-weight:700 !important; font-size:32px !important; color:#01335A !important; margin-bottom:32px !important; text-align:center !important;"
        )

        # ---------- SECTION A : VUE D'ENSEMBLE ----------
        with ui.card().classes("w-full max-w-5xl p-6 mb-8").style(
            "background-color:white !important; border-radius:12px !important; box-shadow:0 4px 15px rgba(0,0,0,0.08) !important;"
        ):
            ui.label("📊 Vue d’Ensemble du Dataset").style(
                "font-weight:700 !important; font-size:22px !important; color:#01335A !important; margin-bottom:12px !important;"
            )

            n_rows, n_cols = df.shape
            mem_mb = round(df.memory_usage(deep=True).sum() / 1e6, 2)

            with ui.row().classes("justify-around w-full mt-2"):
                def metric(label, value, color):
                    with ui.column().classes("items-center"):
                        ui.label(label).style("font-size:15px !important; color:#636e72 !important;")
                        ui.label(value).style(f"font-weight:700 !important; font-size:20px !important; color:{color} !important;")
                metric("Nombre de lignes", f"{n_rows:,}", "#01335A")
                metric("Nombre de colonnes", f"{n_cols:,}", "#01335A")
                metric("Taille mémoire", f"{mem_mb} MB", "#09538C")

        # ---------- SECTION B : TABLEAU DE SCHÉMA ----------
        with ui.card().classes("w-full max-w-6xl p-6").style(
            "background-color:white !important; border-radius:12px !important; box-shadow:0 4px 15px rgba(0,0,0,0.08) !important;"
        ):
            ui.label("🧾 Schéma du Dataset (Colonnes)").style(
                "font-weight:700 !important; font-size:22px !important; color:#01335A !important; margin-bottom:16px !important;"
            )

            columns_info = []

            for col in df.columns:
                series = df[col]
                series_clean = series.replace('', np.nan).replace('?', np.nan)
                conv = pd.to_numeric(series_clean.dropna(), errors='coerce')
                detected_type = "Texte"

                if conv.notna().sum() / max(1, len(series_clean.dropna())) >= 0.9:
                    n_unique = conv.nunique(dropna=True)
                    ratio_unique = n_unique / max(1, len(series_clean))
                    integer_like = np.all(np.abs(conv - conv.round()) < 1e-6)

                    if integer_like:
                        detected_type = "Numérique Discrète"
                    else:
                        if n_unique > 20 or ratio_unique > 0.05:
                            detected_type = "Numérique Continue"
                        else:
                            detected_type = "Numérique Discrète"

                missing_pct = round(series_clean.isna().mean() * 100, 2)
                cardinality = int(series_clean.nunique(dropna=True))
                unique_vals = series_clean.dropna().astype(str).unique()
                examples = ", ".join(unique_vals[:3]) if len(unique_vals) > 0 else ""

                columns_info.append({
                    "Colonne": col,
                    "Type Détecté": detected_type,
                    "Type Réel": detected_type,
                    "Rôle": "Feature",
                    "% Missing": f"{missing_pct}%",
                    "Cardinalité": cardinality,
                    "Exemples": examples
                })

            # ✅ Unsupervised: pas de target
            state["columns_info"] = columns_info

            ui.table(
                columns=[
                    {"name": "Colonne", "label": "Colonne", "field": "Colonne", "sortable": True},
                    {"name": "Type Détecté", "label": "Type Détecté", "field": "Type Détecté"},
                    {"name": "Type Réel", "label": "Type Réel", "field": "Type Réel"},
                    {"name": "Rôle", "label": "Rôle", "field": "Rôle"},
                    {"name": "% Missing", "label": "% Missing", "field": "% Missing"},
                    {"name": "Cardinalité", "label": "Cardinalité", "field": "Cardinalité"},
                    {"name": "Exemples", "label": "Exemples", "field": "Exemples"},
                ],
                rows=columns_info,
                row_key="Colonne",
            ).style(
                "width:100% !important; font-size:14px !important; background:#fafafa !important; border-radius:8px !important;"
            )

        # ---------- NAV ----------
        with ui.row().classes("justify-between w-full max-w-6xl mt-8"):
            ui.button(" Retour",
                      on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/upload'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )

            ui.button(" Étape suivante",
                      on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/user_decisions'")).style(
                "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )

# ----------------- PAGE /unsupervised/user_decisions -----------------


def map_detected_type(detected_type):
    if detected_type in ["Texte", "Catégorielle / Texte"]:
        return "Texte"
    elif detected_type == "Numérique Discrète":
        return "Numérique Discrète"
    elif detected_type == "Numérique Continue":
        return "Numérique Continue"
    else:
        return "Texte"

@ui.page('/unsupervised/user_decisions')
def unsupervised_user_decisions_page():
    import pandas as pd
    import numpy as np
    
    df = state.get("raw_df", None)
    columns_info = state.get("columns_info", None)

    if df is None or columns_info is None:
        with ui.column().classes("w-full h-screen items-center justify-center"):
            ui.label("❌ Aucun dataset chargé ou informations de colonnes manquantes.").style(
                "font-size:18px !important; color:#c0392b !important; font-weight:600 !important;"
            )
            ui.button(
                "⬅ Retour à l'Upload",
                on_click=lambda: ui.run_javascript("window.location.href='/upload'")
            ).style(
                "margin-top:20px !important; background:#01335A !important; color:white !important; "
                "font-weight:600 !important; padding:12px 32px !important; border-radius:8px !important;"
            )
        return

    # ---------------------- CORRECTIONS NUMÉRIQUES ----------------------
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            non_null = df[col].dropna()
            if len(non_null) > 0 and (non_null % 1 == 0).all():
                df[col] = df[col].astype('Int64')
    
    state["raw_df"] = df

    # ---------------------- DÉTECTION TYPE ----------------------
    def detect_actual_type(col_info, col_name):
        col_data = df[col_name]
        n_unique = col_data.nunique()
        n_total = len(col_data.dropna())

        if n_unique == 1:
            return "Identifiant"
        if col_name.lower().startswith('id') or col_name.lower().endswith('_id'):
            return "Identifiant"
        if n_total > 0 and n_unique / n_total > 0.95:
            return "Identifiant"

        if pd.api.types.is_numeric_dtype(col_data):
            non_null = col_data.dropna()
            if len(non_null) > 0:
                is_integer = (non_null % 1 == 0).all()
                if is_integer:
                    if n_unique <= 20:
                        return "Numérique Discrète"
                    else:
                        unique_sorted = sorted(non_null.unique())
                        if len(unique_sorted) > 1:
                            gaps = np.diff(unique_sorted)
                            avg_gap = np.mean(gaps)
                            if avg_gap <= 5 and np.std(gaps) < avg_gap:
                                return "Numérique Continue"
                            else:
                                return "Numérique Discrète"
                        return "Numérique Discrète"
                else:
                    return "Numérique Continue"

        if pd.api.types.is_datetime64_any_dtype(col_data):
            return "Date/Datetime"

        if col_data.dtype == 'object' or pd.api.types.is_categorical_dtype(col_data):
            values_lower = [str(v).lower() for v in col_data.unique()[:20]]
            ordinal_patterns = [
                ['low', 'medium', 'high'],
                ['bad', 'good', 'excellent'],
                ['small', 'medium', 'large'],
                ['never', 'sometimes', 'often', 'always'],
                ['poor', 'fair', 'good', 'excellent'],
                ['xs', 's', 'm', 'l', 'xl', 'xxl']
            ]
            for pattern in ordinal_patterns:
                if any(p in values_lower for p in pattern):
                    return "Catégorielle Ordinale"
            return "Catégorielle Nominale"

        return "Texte"

    # ---------------------- FONCTIONS ----------------------
    def on_confirm():
        selected_features = feature_dropdown.value
        if not selected_features or len(selected_features) < 2:
            feature_warning.text = "⚠️ Sélectionnez au moins 2 colonnes pour le clustering"
            ui.notify("⚠️ Sélectionnez au moins 2 colonnes", color="warning")
            return

        state["selected_features"] = selected_features

        for col_name, widget in column_type_widgets.items():
            state.setdefault("columns_types", {})[col_name] = widget.value

        for col_name, cb in column_exclude_widgets.items():
            state.setdefault("columns_exclude", {})[col_name] = cb.value

        ui.notify("✅ Décisions enregistrées avec succès !", color="positive")

    def save_and_go():
        on_confirm()
        ui.run_javascript("setTimeout(() => window.location.href='/unsupervised/univariate_analysis', 500);")

    # ---------------------- UI ----------------------
    with ui.column().classes("w-full items-center").style(
        "min-height:100vh !important; background:#f0f2f5 !important; padding:48px 24px !important; "
        "font-family:'Inter', sans-serif !important;"
    ):
        # HEADER
        ui.label("Configuration Clustering").style(
            "font-weight:700 !important; font-size:36px !important; color:#01335A !important; "
            "margin-bottom:8px !important; text-align:center !important; letter-spacing:-0.5px !important;"
        )
        ui.label("Sélection des features et configuration des types").style(
            "font-size:16px !important; color:#636e72 !important; margin-bottom:48px !important; "
            "text-align:center !important; font-weight:400 !important;"
        )

        # ---------- APERÇU DES DONNÉES ----------
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white !important; border-radius:16px !important; padding:32px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            ui.label("👁️ Aperçu des données").style(
                "font-weight:700 !important; font-size:22px !important; color:#01335A !important; "
                "margin-bottom:16px !important;"
            )
            
            ui.label(
                f"Visualisation des 10 premières lignes du dataset ({len(df)} lignes × {len(df.columns)} colonnes)"
            ).style(
                "font-size:14px !important; color:#636e72 !important; margin-bottom:20px !important;"
            )

            # Créer un aperçu des 10 premières lignes
            df_preview = df.head(10).copy()
            
            columns_for_table = []
            rows_for_table = []
            
            for col in df_preview.columns:
                columns_for_table.append({
                    "name": col,
                    "label": col,
                    "field": col,
                    "align": "left",
                    "sortable": True
                })
            
            for idx, row in df_preview.iterrows():
                row_dict = {"_index": str(idx)}
                for col in df_preview.columns:
                    val = row[col]
                    if pd.isna(val):
                        row_dict[col] = "NaN"
                    elif isinstance(val, (int, np.integer)):
                        row_dict[col] = str(val)
                    elif isinstance(val, (float, np.floating)):
                        row_dict[col] = f"{val:.2f}"
                    else:
                        row_dict[col] = str(val)[:50]
                rows_for_table.append(row_dict)
            
            columns_for_table.insert(0, {
                "name": "_index",
                "label": "Index",
                "field": "_index",
                "align": "center",
                "sortable": False
            })
            
            ui.table(
                columns=columns_for_table,
                rows=rows_for_table,
                row_key="_index"
            ).props("flat bordered dense").style(
                "width:100% !important; font-size:12px !important; max-height:400px !important; "
                "overflow-y:auto !important;"
            )
            
            # Statistiques compactes
            with ui.row().classes("w-full items-center justify-around mt-6").style(
                "background:linear-gradient(135deg, #01335A 0%, #09538C 100%) !important; "
                "padding:20px !important; border-radius:12px !important;"
            ):
                with ui.row().classes("items-center gap-3"):
                    ui.icon("dataset", size="md").classes("text-white")
                    with ui.column().classes("gap-0"):
                        ui.label(f"{len(df):,}").style(
                            "font-weight:700 !important; font-size:24px !important; color:white !important; "
                            "line-height:1 !important;"
                        )
                        ui.label("lignes").style(
                            "font-size:12px !important; color:rgba(255,255,255,0.8) !important; margin-top:4px !important;"
                        )
                
                with ui.row().classes("items-center gap-3"):
                    ui.icon("table_chart", size="md").classes("text-white")
                    with ui.column().classes("gap-0"):
                        ui.label(f"{len(df.columns)}").style(
                            "font-weight:700 !important; font-size:24px !important; color:white !important; "
                            "line-height:1 !important;"
                        )
                        ui.label("colonnes").style(
                            "font-size:12px !important; color:rgba(255,255,255,0.8) !important; margin-top:4px !important;"
                        )
                
                with ui.row().classes("items-center gap-3"):
                    ui.icon("analytics", size="md").classes("text-white")
                    with ui.column().classes("gap-0"):
                        n_numeric = len(df.select_dtypes(include=[np.number]).columns)
                        ui.label(f"{n_numeric}").style(
                            "font-weight:700 !important; font-size:24px !important; color:white !important; "
                            "line-height:1 !important;"
                        )
                        ui.label("numériques").style(
                            "font-size:12px !important; color:rgba(255,255,255,0.8) !important; margin-top:4px !important;"
                        )
                
                with ui.row().classes("items-center gap-3"):
                    ui.icon("category", size="md").classes("text-white")
                    with ui.column().classes("gap-0"):
                        n_categorical = len(df.select_dtypes(include=['object', 'category']).columns)
                        ui.label(f"{n_categorical}").style(
                            "font-weight:700 !important; font-size:24px !important; color:white !important; "
                            "line-height:1 !important;"
                        )
                        ui.label("catégorielles").style(
                            "font-size:12px !important; color:rgba(255,255,255,0.8) !important; margin-top:4px !important;"
                        )

        # ---------- SÉLECTION DES FEATURES ----------
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white !important; border-radius:16px !important; padding:32px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            ui.label("🧩 Sélection des features pour le clustering").style(
                "font-weight:700 !important; font-size:22px !important; color:#01335A !important; "
                "margin-bottom:16px !important;"
            )
            
            ui.label("Choisissez les colonnes qui seront utilisées pour créer les clusters").style(
                "font-size:14px !important; color:#636e72 !important; margin-bottom:20px !important;"
            )

            all_columns = [col["Colonne"] for col in columns_info]
            
            # Boutons de sélection rapide
            with ui.row().classes("w-full justify-end gap-2 mb-4"):
                ui.button(
                    "Tout sélectionner",
                    on_click=lambda: feature_dropdown.set_value(all_columns)
                ).props("flat").style(
                    "color:#01335A !important; font-weight:500 !important; text-transform:none !important;"
                )
                ui.button(
                    "Tout désélectionner",
                    on_click=lambda: feature_dropdown.set_value([])
                ).props("flat").style(
                    "color:#636e72 !important; font-weight:500 !important; text-transform:none !important;"
                )

            feature_dropdown = ui.select(
                options=all_columns,
                multiple=True,
                label="Colonnes sélectionnées",
                value=all_columns  # Toutes sélectionnées par défaut
            ).props("outlined").classes("w-full")

            feature_warning = ui.label("").style(
                "color:#e74c3c !important; font-weight:600 !important; margin-top:8px !important; font-size:14px !important;"
            )
            
            # Info box
            with ui.card().classes("w-full mt-4").style(
                "background:#e3f2fd !important; padding:16px !important; border-radius:12px !important; "
                "border-left:4px solid #2196f3 !important; box-shadow:none !important;"
            ):
                ui.label("💡 Conseil").style(
                    "font-weight:700 !important; color:#01335A !important; margin-bottom:8px !important;"
                )
                ui.label("Pour un clustering efficace, sélectionnez des features numériques pertinentes. Les colonnes identifiants et à variance nulle seront automatiquement exclues.").style(
                    "font-size:13px !important; color:#01335A !important; line-height:1.6 !important;"
                )

        # ---------- CONFIGURATION DES TYPES (2-3 PAR LIGNE) ----------
        column_type_widgets = {}
        column_exclude_widgets = {}

        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white !important; border-radius:16px !important; padding:32px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            ui.label("🛠 Configuration des types de colonnes").style(
                "font-weight:700 !important; font-size:22px !important; color:#01335A !important; "
                "margin-bottom:16px !important;"
            )
            ui.label("Vérifiez et corrigez les types détectés automatiquement").style(
                "font-size:14px !important; color:#636e72 !important; margin-bottom:20px !important;"
            )

            # ✅ DISPOSITION EN GRILLE : 2-3 COLONNES PAR LIGNE
            for i in range(0, len(columns_info), 2):  # 2 colonnes par ligne
                with ui.row().classes("w-full gap-4 mb-4"):
                    # Colonne 1
                    col1 = columns_info[i]
                    col1_name = col1["Colonne"]
                    actual_type1 = detect_actual_type(col1, col1_name)
                    
                    with ui.card().classes("flex-1").style(
                        "background:#f8f9fa !important; padding:16px !important; "
                        "border-radius:12px !important; border:1px solid #e1e8ed !important;"
                    ):
                        # Nom de la colonne
                        ui.label(col1_name).style(
                            "font-weight:700 !important; font-size:15px !important; "
                            "color:#01335A !important; margin-bottom:12px !important;"
                        )
                        
                        # Type selector
                        col_type1 = ui.select(
                            options=[
                                "Numérique Continue", "Numérique Discrète",
                                "Catégorielle Nominale", "Catégorielle Ordinale",
                                "Date/Datetime", "Texte", "Identifiant"
                            ],
                            value=actual_type1,
                            label="Type"
                        ).props("outlined dense").classes("w-full")
                        
                        column_type_widgets[col1_name] = col_type1
                        
                        # Exclusion automatique
                        auto_exclude1 = False
                        if col1["Cardinalité"] == 1:
                            auto_exclude1 = True
                        if "%" in col1["% Missing"]:
                            try:
                                missing_pct = float(col1["% Missing"].replace("%", "").strip())
                                if missing_pct >= 100:
                                    auto_exclude1 = True
                            except:
                                pass
                        if actual_type1 == "Identifiant":
                            auto_exclude1 = True
                        
                        exclude_cb1 = ui.checkbox("Exclure cette colonne", value=auto_exclude1).classes("mt-2")
                        column_exclude_widgets[col1_name] = exclude_cb1
                    
                    # Colonne 2 (si elle existe)
                    if i + 1 < len(columns_info):
                        col2 = columns_info[i + 1]
                        col2_name = col2["Colonne"]
                        actual_type2 = detect_actual_type(col2, col2_name)
                        
                        with ui.card().classes("flex-1").style(
                            "background:#f8f9fa !important; padding:16px !important; "
                            "border-radius:12px !important; border:1px solid #e1e8ed !important;"
                        ):
                            ui.label(col2_name).style(
                                "font-weight:700 !important; font-size:15px !important; "
                                "color:#01335A !important; margin-bottom:12px !important;"
                            )
                            
                            col_type2 = ui.select(
                                options=[
                                    "Numérique Continue", "Numérique Discrète",
                                    "Catégorielle Nominale", "Catégorielle Ordinale",
                                    "Date/Datetime", "Texte", "Identifiant"
                                ],
                                value=actual_type2,
                                label="Type"
                            ).props("outlined dense").classes("w-full")
                            
                            column_type_widgets[col2_name] = col_type2
                            
                            auto_exclude2 = False
                            if col2["Cardinalité"] == 1:
                                auto_exclude2 = True
                            if "%" in col2["% Missing"]:
                                try:
                                    missing_pct = float(col2["% Missing"].replace("%", "").strip())
                                    if missing_pct >= 100:
                                        auto_exclude2 = True
                                except:
                                    pass
                            if actual_type2 == "Identifiant":
                                auto_exclude2 = True
                            
                            exclude_cb2 = ui.checkbox("Exclure cette colonne", value=auto_exclude2).classes("mt-2")
                            column_exclude_widgets[col2_name] = exclude_cb2

            # Info exclusions
            with ui.card().classes("w-full mt-6").style(
                "background:#fff9e6 !important; padding:20px !important; border-radius:12px !important; "
                "border-left:4px solid #f39c12 !important; box-shadow:none !important;"
            ):
                ui.label("💡 Exclusions automatiques détectées :").style(
                    "font-weight:700 !important; margin-bottom:12px !important; color:#856404 !important;"
                )
                
                exclusions = [
                    "• Colonnes avec cardinalité = 1 (valeur unique)",
                    "• Colonnes avec 100% de valeurs manquantes",
                    "• Colonnes identifiants (détection automatique)"
                ]
                
                for excl in exclusions:
                    ui.label(excl).style(
                        "font-size:13px !important; color:#856404 !important; margin-bottom:4px !important;"
                    )

        # ---------- BOUTONS NAVIGATION ----------
        with ui.row().classes("w-full max-w-6xl justify-between gap-4 mt-8"):
            ui.button(
                "← Retour",
                on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/preprocessing'")
            ).style(
                "background:white !important; color:#01335A !important; font-weight:500 !important; "
                "border:1px solid #e1e8ed !important; border-radius:8px !important; height:50px !important; "
                "min-width:200px !important; font-size:14px !important; text-transform:none !important; "
                "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
            )
            
            ui.button(
                "Suivant →",
                on_click=save_and_go
            ).style(
                "background:#01335A !important; color:white !important; font-weight:600 !important; "
                "border-radius:8px !important; height:50px !important; min-width:200px !important; "
                "font-size:14px !important; text-transform:none !important;"
            )

# ----------------- PAGE /unsupervised/missing_values ----------------- 


def get_recommendation(missing_pct):
    """Recommandation automatique basée sur le % de valeurs manquantes"""
    if missing_pct == 0:
        return "Aucune action", "green"
    elif missing_pct < 5:
        return "Imputation recommandée", "blue"
    elif missing_pct < 30:
        return "Imputation ou suppression", "orange"
    else:
        return "Suppression recommandée", "red"

@ui.page('/unsupervised/missing_values')
def missing_values_page():
    df = state.get("raw_df")
    features = state.get("selected_features")
    univariate_decisions = state.get("univariate_decisions", {})
    anomaly_decisions = state.get("anomaly_decisions", {})
    
    if df is None or features is None:
        with ui.column().style("width:100%; height:100vh; display:flex; align-items:center; justify-content:center;"):
            ui.label("❌ Données manquantes").style("font-size:20px; color:#e74c3c; font-weight:600; margin-bottom:12px;")
            ui.button(" Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/multivariate_analysis'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )
        return
    
    # Appliquer les décisions précédentes
    df_proc = df[features].copy()
    
    # Convertir les "?" en NaN pour toutes les colonnes
    df_proc = df_proc.replace('?', np.nan)
    df_proc = df_proc.replace('', np.nan)
    df_proc = df_proc.replace(' ', np.nan)
    
    # Appliquer suppressions univariées
    for col, decision in univariate_decisions.items():
        if decision == "Supprimer" and col in df_proc.columns:
            df_proc.drop(columns=col, inplace=True)
    
    # Appliquer décisions anomalies (winsorisation, suppression)
    for col, decision in anomaly_decisions.items():
        if col not in df_proc.columns:
            continue
        if decision == "Supprimer":
            df_proc.drop(columns=col, inplace=True)
        elif decision == "Winsoriser":
            series = pd.to_numeric(df_proc[col], errors="coerce")
            if series.notna().sum() > 0:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                df_proc[col] = series.clip(lower=lower, upper=upper)
    
    # Calculer statistiques missing
    missing_stats = []
    for col in df_proc.columns:
        missing_count = df_proc[col].isna().sum()
        missing_pct = (missing_count / len(df_proc)) * 100
        recommendation, color = get_recommendation(missing_pct)
        missing_stats.append({
            'Colonne': col,
            'Valeurs manquantes': missing_count,
            '% Manquant': missing_pct,
            'Recommandation': recommendation,
            'Color': color
        })
    
    missing_df = pd.DataFrame(missing_stats).sort_values('% Manquant', ascending=False)
    
    # Stockage des décisions
    column_decisions = {}
    row_threshold = {'value': 50}
    global_method = {'value': None}  # Pour stocker la méthode globale
    
    with ui.column().style("width:100%; min-height:100vh; padding:40px; background:#f5f6fa; font-family:'Inter', sans-serif; display:flex; align-items:center;"):
        
        # Header moderne
        with ui.column().style("margin-bottom:48px; text-align:center;"):
            ui.label("💧 Gestion des Valeurs Manquantes").style("font-weight:800; font-size:42px; background:linear-gradient(135deg, #01335A, #09538C); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:12px; letter-spacing:-1px;")
            
            # Résumé global
            total_missing = df_proc.isna().sum().sum()
            total_cells = df_proc.shape[0] * df_proc.shape[1]
            global_missing_pct = (total_missing / total_cells) * 100
            
            ui.label(f"{df_proc.shape[0]} lignes × {df_proc.shape[1]} colonnes | {global_missing_pct:.2f}% de valeurs manquantes").style(
                "color:#7f8c8d; font-size:16px;"
            )
        
       
        
        # Heatmap des valeurs manquantes
        if total_missing > 0:
            with ui.column().style("width:100%; max-width:900px; margin-bottom:32px;"):
                with ui.card().style("width:100%; padding:24px; background:white; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                    
                    with ui.column().style("margin-bottom:20px;"):
                        ui.label("🔥 Pattern des Valeurs Manquantes").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                        ui.separator().style("width:60px; height:3px; background:#e74c3c; border-radius:2px; margin:0;")
                    
                    with ui.pyplot(figsize=(12, 6), close=True):
                        missing_matrix = df_proc.isna().astype(int)
                        sample_size = min(50, len(missing_matrix))
                        missing_sample = missing_matrix.head(sample_size)
                        
                        sns.heatmap(missing_sample.T, cmap='RdYlGn_r', cbar=True, 
                                   yticklabels=missing_sample.columns, xticklabels=False,
                                   cbar_kws={'label': 'Valeur manquante'})
                        plt.title(f"Pattern ({sample_size} premières lignes)", fontsize=12, fontweight='bold')
                        plt.xlabel("Index des lignes", fontsize=10)
                        plt.ylabel("Colonnes", fontsize=10)
                        plt.tight_layout()
         # 🆕 SECTION MÉTHODE GLOBALE
        with ui.column().style("width:100%; max-width:900px; margin-bottom:32px;"):
            with ui.card().style("width:100%; padding:24px; background:white; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08); border:2px solid #3498db;"):
                
                with ui.column().style("margin-bottom:20px;"):
                    ui.label("⚡ Méthode Globale (Appliquer à toutes les colonnes)").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                    ui.separator().style("width:60px; height:3px; background:#3498db; border-radius:2px; margin:0;")
                
                ui.label("Choisissez une méthode à appliquer automatiquement pour toutes les colonnes avec valeurs manquantes :").style(
                    "font-size:14px; color:#7f8c8d; margin-bottom:16px;"
                )
                
                global_method_select = ui.select(
                    options=["Mode personnalisé (par colonne)", "Supprimer toutes les colonnes", 
                            "Imputer Mean (tout)", "Imputer Median (tout)", "Imputer Mode (tout)", "Imputer KNN (tout)"],
                    value="Mode personnalisé (par colonne)",
                    label="Méthode globale"
                ).style("width:100%; border:2px solid #3498db; border-radius:8px; font-weight:600;")
                
                with ui.card().style("padding:12px 16px; background:#e8f4f8; border-left:4px solid #3498db; border-radius:6px; margin-top:12px;"):
                    ui.label("💡 Astuce : Sélectionnez 'Mode personnalisé' pour configurer chaque colonne individuellement.").style(
                        "font-size:13px; color:#2c3e50;"
                    )
                
                def apply_global_method():
                    method = global_method_select.value
                    global_method['value'] = method
                    
                    if method == "Mode personnalisé (par colonne)":
                        ui.notify("Mode personnalisé activé. Configurez chaque colonne ci-dessous.", color="info")
                    else:
                        # Appliquer la méthode globale à toutes les colonnes
                        action_map = {
                            "Supprimer toutes les colonnes": "Supprimer colonne",
                            "Imputer Mean (tout)": "Imputer Mean",
                            "Imputer Median (tout)": "Imputer Median",
                            "Imputer Mode (tout)": "Imputer Mode",
                            "Imputer KNN (tout)": "Imputer KNN"
                        }
                        
                        action = action_map.get(method, "Garder tel quel")
                        
                        for col, widget in column_decisions.items():
                            if widget is not None:
                                widget.set_value(action)
                        
                        ui.notify(f"✅ Méthode '{method}' appliquée à toutes les colonnes !", color="positive")
                
                ui.button("✓ Appliquer cette méthode", on_click=apply_global_method).style(
                    "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white; font-weight:600; border-radius:8px; padding:10px 20px; margin-top:12px; width:100%;"
                )                
        
        # Statistiques par colonne
        with ui.column().style("width:100%; max-width:900px; margin-bottom:32px;"):
            with ui.card().style("width:100%; padding:24px; background:white; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                
                with ui.column().style("margin-bottom:20px;"):
                    ui.label("📊 Statistiques par Colonne").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                    ui.separator().style("width:60px; height:3px; background:#3498db; border-radius:2px; margin:0;")
                
                # Table rows
                for _, row in missing_df.iterrows():
                    col = row['Colonne']
                    
                    # Déterminer la couleur de bordure selon le pourcentage
                    if row['% Manquant'] > 50:
                        border_color = "#e74c3c"
                        bg_color = "#fadbd8"
                    elif row['% Manquant'] > 20:
                        border_color = "#e67e22"
                        bg_color = "#fdebd0"
                    elif row['% Manquant'] > 0:
                        border_color = "#f39c12"
                        bg_color = "#fef5e7"
                    else:
                        border_color = "#27ae60"
                        bg_color = "#d5f4e6"
                    
                    with ui.card().style(f"padding:20px; margin-bottom:16px; background:#fafbfc; border-radius:8px; border:1px solid {border_color}; width:100%; transition:all 0.2s;"):
                        
                        # Nom de colonne et stats
                        with ui.row().style("display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; width:100%;"):
                            ui.label(col).style("font-weight:600; color:#2c3e50; font-size:18px;")
                            with ui.row().style("display:flex; gap:12px; align-items:center;"):
                                with ui.card().style(f"padding:6px 12px; background:{bg_color}; border-radius:6px; border:1px solid {border_color};"):
                                    ui.label(f"{row['Valeurs manquantes']} manquantes").style(f"color:{border_color}; font-size:14px; font-weight:600;")
                                with ui.card().style(f"padding:6px 12px; background:{bg_color}; border-radius:6px; border:1px solid {border_color};"):
                                    ui.label(f"{row['% Manquant']:.1f}%").style(f"color:{border_color}; font-weight:700; font-size:16px;")
                        
                        # Recommandation
                        with ui.card().style("padding:10px 14px; background:#f8f9fa; border-radius:6px; margin-bottom:16px;"):
                            ui.label(f"💡 {row['Recommandation']}").style(f"color:{row['Color']}; font-size:14px; font-weight:600;")
                        
                        # Action
                        if row['% Manquant'] > 0:
                            with ui.row().style("display:flex; align-items:center; gap:12px;"):
                                ui.label("Action:").style("color:#7f8c8d; font-size:14px; font-weight:600;")
                                column_decisions[col] = ui.select(
                                    options=["Garder tel quel", "Supprimer colonne", "Imputer Mean", 
                                            "Imputer Median", "Imputer Mode", "Imputer KNN"],
                                    value="Garder tel quel"
                                ).style("width:300px; border:2px solid #e1e8ed; border-radius:6px;")
                        else:
                            column_decisions[col] = None
                            with ui.card().style("padding:8px 14px; background:#d5f4e6; border-radius:6px; display:inline-block;"):
                                ui.label("✓ Aucune action requise").style("color:#27ae60; font-size:14px; font-weight:600;")
        
        # Gestion des lignes
        with ui.column().style("width:100%; max-width:900px; margin-bottom:32px;"):
            with ui.card().style("width:100%; padding:24px; background:white; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                
                with ui.column().style("margin-bottom:20px;"):
                    ui.label("🗑️ Gestion des Lignes").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                    ui.separator().style("width:60px; height:3px; background:#e67e22; border-radius:2px; margin:0;")
                
                ui.label("Supprimer les lignes ayant plus de X% de valeurs manquantes :").style("font-size:15px; color:#2c3e50; font-weight:600; margin-bottom:16px;")
                
                threshold_slider = ui.slider(min=0, max=100, value=50, step=5).props('label-always').style("width:100%; margin-bottom:16px;")
                
                rows_missing_pct = (df_proc.isna().sum(axis=1) / df_proc.shape[1]) * 100
                initial_rows_to_drop = (rows_missing_pct > threshold_slider.value).sum()
                
                with ui.card().style("padding:16px; background:#fdebd0; border-left:3px solid #e67e22; border-radius:6px;"):
                    impact_label = ui.label(
                        f"📌 Impact : {initial_rows_to_drop} lignes supprimées ({(initial_rows_to_drop/len(df_proc)*100):.1f}%)"
                    ).style("font-size:15px; color:#e67e22; font-weight:600;")
                
                def update_threshold():
                    row_threshold['value'] = threshold_slider.value
                    rows_to_drop = (rows_missing_pct > threshold_slider.value).sum()
                    impact_label.set_text(
                        f"📌 Impact : {rows_to_drop} lignes supprimées ({(rows_to_drop/len(df_proc)*100):.1f}%)"
                    )
                
                threshold_slider.on('update:model-value', update_threshold)
        
        # Prévisualisation
        with ui.column().style("width:100%; max-width:900px; margin-bottom:32px;"):
            with ui.card().style("width:100%; padding:24px; background:white; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                
                with ui.column().style("margin-bottom:20px;"):
                    ui.label("👀 Prévisualisation de l'Impact").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                    ui.separator().style("width:60px; height:3px; background:#9b59b6; border-radius:2px; margin:0;")
                
                preview_container = ui.column()
                
                def update_preview():
                    preview_container.clear()
                    with preview_container:
                        df_preview = df_proc.copy()
                        cols_to_drop = []
                        
                        for col, widget in column_decisions.items():
                            if widget is None:
                                continue
                            decision = widget.value
                            if decision == "Supprimer colonne":
                                cols_to_drop.append(col)
                            elif decision == "Imputer Mean":
                                df_preview[col] = pd.to_numeric(df_preview[col], errors="coerce")
                                df_preview[col] = df_preview[col].fillna(df_preview[col].mean())
                            elif decision == "Imputer Median":
                                df_preview[col] = pd.to_numeric(df_preview[col], errors="coerce")
                                df_preview[col] = df_preview[col].fillna(df_preview[col].median())
                            elif decision == "Imputer Mode":
                                mode_value = df_preview[col].mode().iloc[0] if df_preview[col].mode().size else None
                                if mode_value is not None:
                                    df_preview[col] = df_preview[col].fillna(mode_value)
                            elif decision == "Imputer KNN":
                                try:
                                    series = pd.to_numeric(df_preview[col], errors="coerce")
                                    imputer = KNNImputer(n_neighbors=5)
                                    df_preview[col] = imputer.fit_transform(series.to_frame()).ravel()
                                except:
                                    pass
                        
                        if cols_to_drop:
                            df_preview = df_preview.drop(columns=[c for c in cols_to_drop if c in df_preview.columns])
                        
                        rows_missing_pct = (df_preview.isna().sum(axis=1) / df_preview.shape[1]) * 100
                        rows_to_keep = rows_missing_pct <= row_threshold['value']
                        df_preview = df_preview[rows_to_keep]
                        
                        # Résumé transformation
                        with ui.row().style("display:flex; gap:16px; align-items:center; margin-bottom:20px;"):
                            with ui.card().style("padding:12px 20px; background:#f8f9fa; border-radius:8px; border:2px dashed #bdc3c7;"):
                                ui.label(f"Original: {df_proc.shape[0]} × {df_proc.shape[1]}").style("color:#7f8c8d; font-size:14px; font-weight:600;")
                            
                            ui.label("").style("font-size:24px; color:#01335A;")
                            
                            with ui.card().style("padding:12px 20px; background:#d5f4e6; border-radius:8px; border:2px solid #27ae60;"):
                                ui.label(f"Après: {df_preview.shape[0]} × {df_preview.shape[1]}").style("color:#27ae60; font-weight:700; font-size:14px;")
                            
                            rows_lost = df_proc.shape[0] - df_preview.shape[0]
                            cols_lost = df_proc.shape[1] - df_preview.shape[1]
                            
                            if rows_lost > 0 or cols_lost > 0:
                                with ui.card().style("padding:12px 20px; background:#fadbd8; border-radius:8px; border:2px solid #e74c3c;"):
                                    ui.label(f"Supprimé: {rows_lost} lignes, {cols_lost} colonnes").style("color:#e74c3c; font-size:14px; font-weight:700;")
                
                ui.button("🔄 Mettre à jour", on_click=update_preview).style(
                    "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white; font-weight:600; border-radius:8px; padding:12px 24px; border:none; cursor:pointer; transition:all 0.2s;"
                )
        
        # Boutons de navigation
        def save_and_next():
            decisions = {
                'column_decisions': {col: widget.value for col, widget in column_decisions.items() if widget is not None},
                'row_threshold': row_threshold['value'],
                'global_method': global_method['value']
            }
            state["missing_decisions"] = decisions
            
            df_final = df_proc.copy()
            
            for col, action in decisions['column_decisions'].items():
                if action == "Supprimer colonne" and col in df_final.columns:
                    df_final.drop(columns=col, inplace=True)
                elif action == "Imputer Mean":
                    df_final[col] = pd.to_numeric(df_final[col], errors="coerce")
                    df_final[col] = df_final[col].fillna(df_final[col].mean())
                elif action == "Imputer Median":
                    df_final[col] = pd.to_numeric(df_final[col], errors="coerce")
                    df_final[col] = df_final[col].fillna(df_final[col].median())
                elif action == "Imputer Mode":
                    mode = df_final[col].mode()
                    if not mode.empty:
                        df_final[col] = df_final[col].fillna(mode.iloc[0])
                elif action == "Imputer KNN":
                    try:
                        series = pd.to_numeric(df_final[col], errors="coerce")
                        imputer = KNNImputer(n_neighbors=5)
                        df_final[col] = imputer.fit_transform(series.to_frame()).ravel()
                    except:
                        pass
            
            rows_missing_pct = (df_final.isna().sum(axis=1) / df_final.shape[1]) * 100
            df_final = df_final[rows_missing_pct <= decisions['row_threshold']]
            
            state["cleaned_data"] = df_final
            ui.run_javascript("window.location.href='/unsupervised/encoding'")
        
        with ui.row().style("display:flex; justify-content:space-between; width:100%; max-width:900px; margin-top:32px;"):
            ui.button(" Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/multivariate_analysis'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important; transition:all 0.2s !important;"
            )
            ui.button("Sauvegarder et Continuer ", on_click=save_and_next).style(
                "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:250px !important; transition:all 0.2s !important;"
            )
#----------------- PAGE /unsupervised/encoding -----------------



@ui.page('/unsupervised/encoding')
def encoding_page():

    df = state.get("cleaned_data")

    if df is None:
        with ui.column().style("width:100%; height:100vh; display:flex; align-items:center; justify-content:center;"):
            ui.label("❌ Données manquantes").style("font-size:20px; color:#e74c3c; font-weight:600; margin-bottom:12px;")
            ui.button(" Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/missing_values'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )
        return

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    encoding_decisions = {}
    numeric_decisions = {}

    with ui.column().style("width:100%; min-height:100vh; padding:40px; background:#f5f6fa; font-family:'Inter', sans-serif;"):
        
        # Header avec animation hover
        with ui.column().style("margin-bottom:48px; text-align:center;"):
            ui.label("🎨 Encodage & Transformations").style("font-weight:800; font-size:42px; background:linear-gradient(135deg, #01335A, #09538C); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:12px; letter-spacing:-1px;")
    
        
        # VARIABLES CATEGORIELLES
        with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:24px;"):
            with ui.card().style("width:100%; max-width:900px; padding:24px; background:white; display:flex; align-items:center; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                
                # En-tête de section avec ligne décorative
                with ui.column().style("width:600px; margin-bottom:20px;"):
                    ui.label("Variables Catégorielles").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                    ui.separator().style("width:60px; height:3px; background:#3498db; border-radius:2px; margin:0;")
                
                if categorical_cols:
                    for col in categorical_cols:
                        unique_count = df[col].nunique()
                        
                        with ui.card().style("padding:20px; margin-bottom:16px; display:flex; align-items:center; background:#fafbfc; border-radius:8px; min-height:100px; width:700px; border:1px solid #e1e8ed; transition:all 0.2s;"):
                            with ui.row().style("display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; width:100%;"):
                                ui.label(col).style("font-weight:600; color:#2c3e50; font-size:18px;")
                                with ui.row().style("display:flex; align-items:center; gap:8px;"):
                                    ui.label(f"{unique_count}").style("font-size:18px; font-weight:700; color:#3498db;")
                                    ui.label("valeurs uniques").style("color:#7f8c8d; font-weight:500; font-size:14px;")
                            
                            with ui.row().style("display:flex; align-items:center; justify-content:center; width:100%;"):
                                encoding_decisions[col] = ui.select(
                                    options=["One-Hot Encoding", "Ordinal Encoding"],
                                    value="One-Hot Encoding"
                                ).style("width:300px; border:2px solid #e1e8ed; border-radius:6px;")
                else:
                    ui.label("✓ Aucune variable catégorielle").style("color:#27ae60; font-size:14px;")
        
        # VARIABLES NUMERIQUES
        with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:24px;"):
            with ui.card().style("width:100%; max-width:900px; padding:24px; background:white; display:flex; align-items:center; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                
                # En-tête de section
                with ui.column().style("margin-bottom:20px;"):
                    ui.label("Correction de l'Asymétrie (Skewness)").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                    ui.separator().style("width:60px; height:3px; background:#e67e22; border-radius:2px; margin:0;")
                
                if numeric_cols:
                    skewness = df[numeric_cols].skew().sort_values(ascending=False)
                    
                    for col, skew in skewness.items():
                        if abs(skew) > 1.0:
                            skew_color = "#e74c3c"
                            skew_bg = "#fadbd8"
                            skew_label = "Forte asymétrie"
                        elif abs(skew) > 0.5:
                            skew_color = "#e67e22"
                            skew_bg = "#fdebd0"
                            skew_label = "Asymétrie modérée"
                        else:
                            skew_color = "#27ae60"
                            skew_bg = "#d5f4e6"
                            skew_label = "Distribution normale"
                        
                        with ui.card().style("padding:20px; margin-bottom:16px; display:flex; align-items:center; background:#fafbfc; border-radius:8px; min-height:100px; width:700px; border:1px solid #e1e8ed; transition:all 0.2s;"):
                            with ui.row().style("display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; width:100%;"):
                                ui.label(col).style("font-weight:600; color:#2c3e50; font-size:18px;")
                                with ui.row().style("display:flex; gap:12px; align-items:center;"):
                                    with ui.card().style(f"padding:6px 12px; background:{skew_bg}; border-radius:6px; border:1px solid {skew_color};"):
                                        ui.label(f"Skew: {skew:.3f}").style(f"color:{skew_color}; font-weight:700; font-size:14px;")
                                    ui.label(skew_label).style("color:#7f8c8d; font-size:14px; font-weight:500;")
                            
                            with ui.row().style("display:flex; align-items:center; justify-content:center; width:100%;"):
                                numeric_decisions[col] = ui.select(
                                    options=["Aucun traitement", "Log Transform", "Yeo-Johnson"],
                                    value="Aucun traitement"
                                ).style("width:300px; border:2px solid #e1e8ed; border-radius:6px;")
                else:
                    ui.label("✓ Aucune variable numérique").style("color:#27ae60; font-size:14px;")
        
        # PREVISUALISATION
        with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:24px;"):
            with ui.card().style("width:100%; max-width:900px; padding:24px; background:white; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                
                # En-tête
                with ui.column().style("margin-bottom:20px;"):
                    ui.label("Prévisualisation de l'Impact").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                    ui.separator().style("width:60px; height:3px; background:#9b59b6; border-radius:2px; margin:0;")
                
                preview_container = ui.column()
                
                def update_preview():
                    preview_container.clear()
                    with preview_container:
                        temp = df.copy()
                        
                        for col, widget in encoding_decisions.items():
                            if col not in temp.columns:
                                continue
                            
                            method = widget.value
                            if method == "One-Hot Encoding":
                                dummies = pd.get_dummies(temp[col], prefix=col, dtype=int)
                                temp = temp.drop(columns=[col])
                                temp = pd.concat([temp, dummies], axis=1)
                            elif method == "Ordinal Encoding":
                                uniques = temp[col].dropna().unique().tolist()
                                mapping = {v: i for i, v in enumerate(uniques)}
                                temp[col] = temp[col].map(mapping)
                        
                        for col, widget in numeric_decisions.items():
                            if col not in temp.columns:
                                continue
                            
                            method = widget.value
                            if method == "Log Transform":
                                min_val = temp[col].min()
                                if min_val <= 0:
                                    temp[col] = np.log1p(temp[col] - min_val + 1)
                                else:
                                    temp[col] = np.log1p(temp[col])
                            elif method == "Yeo-Johnson":
                                col_filled = temp[col].fillna(temp[col].mean())
                                temp[col] = stats.yeojohnson(col_filled)[0]
                        
                        # Résumé de transformation avec design moderne
                        with ui.row().style("display:flex; gap:16px; align-items:center; margin-bottom:20px;"):
                            with ui.card().style("padding:12px 20px; background:#f8f9fa; border-radius:8px; border:2px dashed #bdc3c7;"):
                                ui.label(f"Original: {df.shape[0]} × {df.shape[1]}").style("color:#7f8c8d; font-size:14px; font-weight:600;")
                            
                            ui.label("").style("font-size:24px; color:#01335A;")
                            
                            with ui.card().style("padding:12px 20px; background:#d5f4e6; border-radius:8px; border:2px solid #27ae60;"):
                                ui.label(f"Après: {temp.shape[0]} × {temp.shape[1]}").style("color:#27ae60; font-weight:700; font-size:14px;")
                            
                            cols_added = temp.shape[1] - df.shape[1]
                            if cols_added > 0:
                                with ui.card().style("padding:12px 20px; background:#d6eaf8; border-radius:8px; border:2px solid #3498db;"):
                                    ui.label(f"+{cols_added} colonnes").style("color:#3498db; font-size:14px; font-weight:700;")
                        
                        ui.separator().style("margin:20px 0; background:#ecf0f1;")
                        
                        ui.label("📋 Premières lignes").style("font-size:16px; font-weight:600; margin-bottom:12px; color:#2c3e50;")
                        with ui.card().style("width:100%; overflow:auto; background:#fafbfc; padding:16px; border-radius:8px; border:1px solid #e1e8ed;"):
                            ui.table.from_pandas(temp.head(10)).style("width:100%;")
                        
                        ui.separator().style("margin:20px 0; background:#ecf0f1;")
                        
                        ui.label("🏷️ Types de colonnes").style("font-size:16px; font-weight:600; margin-bottom:12px; color:#2c3e50;")
                        types_df = pd.DataFrame({
                            'Colonne': temp.columns[:20],
                            'Type': temp.dtypes[:20].astype(str)
                        })
                        with ui.card().style("width:100%; overflow:auto; background:#fafbfc; padding:16px; border-radius:8px; border:1px solid #e1e8ed;"):
                            ui.table.from_pandas(types_df).style("width:100%;")
                
                ui.button("🔄 Mettre à jour", on_click=update_preview).style(
                    "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white; font-weight:600; border-radius:8px; padding:12px 24px; margin-top:16px; border:none; cursor:pointer; transition:all 0.2s;"
                )
        
        # BOUTONS DE NAVIGATION
        def save_and_next():
            temp = df.copy()
            
            for col, widget in encoding_decisions.items():
                if col not in temp.columns:
                    continue
                
                method = widget.value
                if method == "One-Hot Encoding":
                    dummies = pd.get_dummies(temp[col], prefix=col, dtype=int)
                    temp = temp.drop(columns=[col])
                    temp = pd.concat([temp, dummies], axis=1)
                elif method == "Ordinal Encoding":
                    uniques = temp[col].dropna().unique().tolist()
                    mapping = {v: i for i, v in enumerate(uniques)}
                    temp[col] = temp[col].map(mapping)
            
            for col, widget in numeric_decisions.items():
                if col not in temp.columns:
                    continue
                
                method = widget.value
                if method == "Log Transform":
                    min_val = temp[col].min()
                    if min_val <= 0:
                        temp[col] = np.log1p(temp[col] - min_val + 1)
                    else:
                        temp[col] = np.log1p(temp[col])
                elif method == "Yeo-Johnson":
                    col_filled = temp[col].fillna(temp[col].mean())
                    temp[col] = stats.yeojohnson(col_filled)[0]
            
            state["encoded_data"] = temp
            state["encoding_decisions"] = {
                "categorical": {col: widget.value for col, widget in encoding_decisions.items()},
                "numeric": {col: widget.value for col, widget in numeric_decisions.items()},
            }
            
            ui.run_javascript("window.location.href='/unsupervised/anomalies'")
        
        with ui.row().style("display:flex; justify-content:space-between; width:100%; margin-top:32px;"):
            ui.button(" Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/missing_values'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important; transition:all 0.2s !important;"
            )
            ui.button("Sauvegarder et Continuer ", on_click=save_and_next).style(
                "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:250px !important; transition:all 0.2s !important;"
            ) 
# 
#   ----------------- PAGE /unsupervised/univariate_analysis -----------------




@ui.page('/unsupervised/univariate_analysis')
def unsupervised_univariate_page():
    import pandas as pd
    import numpy as np
    from scipy.stats import skew

    df = state.get("raw_df")
    features = state.get("selected_features")

    if df is None or features is None:
        with ui.column().classes("w-full h-screen items-center justify-center"):
            ui.label("❌ Données manquantes").style(
                "font-size:20px !important; color:#c0392b !important; font-weight:600 !important; margin-bottom:12px !important;"
            )
            ui.label("Veuillez revenir en arrière et sélectionner les features").style(
                "color:#7f8c8d !important; margin-bottom:20px !important;"
            )
            ui.button(
                "⬅ Retour",
                on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/user_decisions'")
            ).style(
                "background:#01335A !important; color:white !important; font-weight:600 !important; "
                "border-radius:8px !important; height:46px !important; width:200px !important;"
            )
        return

    # Page principale
    with ui.column().classes("w-full items-center").style(
        "min-height:100vh !important; background:#f0f2f5 !important; padding:48px 24px !important; "
        "font-family:'Inter', sans-serif !important;"
    ):
        # HEADER
        ui.label("Analyse Univariée").style(
            "font-weight:700 !important; font-size:36px !important; color:#01335A !important; "
            "margin-bottom:8px !important; text-align:center !important; letter-spacing:-0.5px !important;"
        )
        ui.label(f"Explorez et transformez vos {len(features)} variables").style(
            "font-size:16px !important; color:#636e72 !important; margin-bottom:48px !important; "
            "text-align:center !important; font-weight:400 !important;"
        )

        decisions = {}

        # Séparer les variables par type
        numeric_features = []
        categorical_features = []
        
        for col in features:
            if col not in df.columns:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)
            else:
                categorical_features.append(col)

        # ==========================================================
        # VARIABLES NUMÉRIQUES
        # ==========================================================
        if numeric_features:
            with ui.card().classes("w-full max-w-6xl mb-6").style(
                "background:white !important; border-radius:16px !important; padding:32px !important; "
                "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
            ):
                # En-tête de section
                ui.label("📊 Variables Numériques").style(
                    "font-weight:700 !important; font-size:22px !important; color:#01335A !important; "
                    "margin-bottom:20px !important;"
                )
                
                for col in numeric_features:
                    series = pd.to_numeric(df[col], errors="coerce").dropna()

                    if len(series) == 0:
                        continue

                    # Calculs statistiques
                    sk = round(skew(series), 2)
                    mean_val = round(series.mean(), 2)
                    std_val = round(series.std(), 2)
                    min_val = round(series.min(), 2)
                    max_val = round(series.max(), 2)

                    q1, q3 = np.percentile(series, [25, 75])
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outliers = ((series < lower) | (series > upper)).sum()

                    # Déterminer le badge selon l'asymétrie (bleu uniquement)
                    if abs(sk) > 1.0:
                        skew_label = "Forte asymétrie"
                        skew_bg = "#bbdefb"
                    elif abs(sk) > 0.5:
                        skew_label = "Asymétrie modérée"
                        skew_bg = "#e3f2fd"
                    else:
                        skew_label = "Distribution symétrique"
                        skew_bg = "#e8f5e9"
                    
                    # Carte variable COMPACTE
                    with ui.card().classes("w-full mb-4").style(
                        "background:#f8f9fa !important; border-radius:12px !important; padding:16px !important; "
                        "border:1px solid #e1e8ed !important; box-shadow:none !important;"
                    ):
                        # En-tête avec nom et badge
                        with ui.row().classes("w-full items-center justify-between mb-3"):
                            ui.label(col).style(
                                "font-weight:700 !important; color:#01335A !important; font-size:16px !important;"
                            )
                            
                            with ui.row().classes("items-center gap-2"):
                                with ui.badge().style(
                                    f"background:{skew_bg} !important; color:#01335A !important; "
                                    "padding:4px 10px !important; border-radius:6px !important;"
                                ):
                                    ui.label(f"Skew: {sk}").style(
                                        "font-size:11px !important; font-weight:600 !important;"
                                    )
                                ui.label(skew_label).style(
                                    "color:#636e72 !important; font-size:12px !important; font-weight:500 !important;"
                                )
                        
                        # Statistiques en grille compacte
                        with ui.grid(columns=5).classes("w-full gap-2 mb-3"):
                            for stat_label, stat_value in [
                                ("Moyenne", mean_val),
                                ("Écart-type", std_val),
                                ("Min", min_val),
                                ("Max", max_val),
                                ("Outliers", outliers)
                            ]:
                                with ui.card().classes("p-2").style(
                                    "background:white !important; border-radius:6px !important; "
                                    "box-shadow:none !important; border:1px solid #e1e8ed !important;"
                                ):
                                    ui.label(stat_label).style(
                                        "font-size:10px !important; color:#636e72 !important; "
                                        "margin-bottom:2px !important; text-transform:uppercase !important;"
                                    )
                                    ui.label(str(stat_value)).style(
                                        "font-weight:700 !important; font-size:14px !important; "
                                        "color:#01335A !important; font-family:monospace !important;"
                                    )
                        
                        # Alerte outliers (bleu uniquement)
                        if outliers > 0:
                            pct_outliers = round((outliers / len(series)) * 100, 1)
                            with ui.card().classes("w-full p-2 mb-3").style(
                                "background:#e3f2fd !important; border-radius:6px !important; "
                                "border-left:3px solid #2196f3 !important; box-shadow:none !important;"
                            ):
                                ui.label(f"⚠️ {outliers} outliers détectés ({pct_outliers}% des données)").style(
                                    "color:#01335A !important; font-size:12px !important; font-weight:500 !important;"
                                )
                        
                        # Sélecteur d'action
                        with ui.row().classes("w-full items-center gap-3"):
                            ui.label("Action :").style(
                                "color:#636e72 !important; font-size:13px !important; font-weight:600 !important;"
                            )
                            action_select = ui.select(
                                options=[
                                    "Garder tel quel",
                                    "Transformer (log)",
                                    "Transformer (sqrt)",
                                    "Supprimer"
                                ],
                                value="Garder tel quel"
                            ).props("outlined dense").classes("flex-1")
                            
                            decisions[col] = action_select

        # ==========================================================
        # VARIABLES CATÉGORIELLES
        # ==========================================================
        if categorical_features:
            with ui.card().classes("w-full max-w-6xl mb-6").style(
                "background:white !important; border-radius:16px !important; padding:32px !important; "
                "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
            ):
                # En-tête de section
                ui.label("🏷️ Variables Catégorielles").style(
                    "font-weight:700 !important; font-size:22px !important; color:#01335A !important; "
                    "margin-bottom:20px !important;"
                )
                
                for col in categorical_features:
                    series = df[col].dropna()
                    
                    if len(series) == 0:
                        continue

                    # Calculs statistiques
                    n_unique = series.nunique()
                    most_common = series.value_counts().head(5)  # Top 5 au lieu de 3
                    missing = df[col].isna().sum()
                    missing_pct = round((missing / len(df)) * 100, 1)

                    # Carte variable COMPACTE
                    with ui.card().classes("w-full mb-4").style(
                        "background:#f8f9fa !important; border-radius:12px !important; padding:16px !important; "
                        "border:1px solid #e1e8ed !important; box-shadow:none !important;"
                    ):
                        # En-tête avec nom et modalités
                        with ui.row().classes("w-full items-center justify-between mb-3"):
                            ui.label(col).style(
                                "font-weight:700 !important; color:#01335A !important; font-size:16px !important;"
                            )
                            
                            with ui.row().classes("items-center gap-2"):
                                ui.label(f"{n_unique}").style(
                                    "font-size:20px !important; font-weight:700 !important; color:#2196f3 !important;"
                                )
                                ui.label("modalités").style(
                                    "color:#636e72 !important; font-weight:500 !important; font-size:12px !important;"
                                )
                        
                        # Statistiques
                        with ui.row().classes("w-full gap-2 mb-3"):
                            with ui.card().classes("flex-1 p-2").style(
                                "background:white !important; border-radius:6px !important; "
                                "box-shadow:none !important; border:1px solid #e1e8ed !important;"
                            ):
                                ui.label("OBSERVATIONS").style(
                                    "font-size:10px !important; color:#636e72 !important; margin-bottom:2px !important;"
                                )
                                ui.label(str(len(series))).style(
                                    "font-size:14px !important; font-weight:700 !important; color:#01335A !important;"
                                )
                            
                            if missing > 0:
                                with ui.card().classes("flex-1 p-2").style(
                                    "background:#e3f2fd !important; border-radius:6px !important; "
                                    "border-left:3px solid #2196f3 !important; box-shadow:none !important;"
                                ):
                                    ui.label("MANQUANTES").style(
                                        "font-size:10px !important; color:#01335A !important; "
                                        "margin-bottom:2px !important; font-weight:600 !important;"
                                    )
                                    ui.label(f"{missing} ({missing_pct}%)").style(
                                        "font-size:14px !important; font-weight:700 !important; color:#2196f3 !important;"
                                    )
                        
                        # Alerte cardinalité
                        if n_unique > 50:
                            with ui.card().classes("w-full p-2 mb-3").style(
                                "background:#e3f2fd !important; border-radius:6px !important; "
                                "border-left:3px solid #2196f3 !important; box-shadow:none !important;"
                            ):
                                ui.label(f"⚠️ Cardinalité élevée: {n_unique} catégories").style(
                                    "color:#01335A !important; font-size:12px !important; font-weight:500 !important;"
                                )
                        
                        # Top 5 modalités avec barres de progression
                        if len(most_common) > 0:
                            with ui.card().classes("w-full p-3 mb-3").style(
                                "background:white !important; border-radius:8px !important; "
                                "box-shadow:none !important; border:1px solid #e1e8ed !important;"
                            ):
                                ui.label(f"Top {len(most_common)} modalités :").style(
                                    "color:#636e72 !important; font-size:12px !important; "
                                    "font-weight:600 !important; margin-bottom:8px !important;"
                                )
                                
                                for category, count in most_common.items():
                                    pct = round((count / len(series)) * 100, 1)
                                    
                                    with ui.row().classes("w-full items-center gap-2 mb-2"):
                                        ui.label(str(category)[:30]).style(
                                            "width:120px !important; font-size:12px !important; "
                                            "color:#2c3e50 !important; font-weight:500 !important;"
                                        )
                                        
                                        with ui.column().classes("flex-1"):
                                            ui.linear_progress(value=pct/100).props(
                                                'color="primary"'
                                            ).classes("h-2 rounded")
                                        
                                        ui.label(f"{count} ({pct}%)").style(
                                            "width:80px !important; text-align:right !important; "
                                            "font-size:11px !important; color:#636e72 !important; "
                                            "font-weight:600 !important;"
                                        )
                        
                        # Sélecteur d'action
                        with ui.row().classes("w-full items-center gap-3"):
                            ui.label("Action :").style(
                                "color:#636e72 !important; font-size:13px !important; font-weight:600 !important;"
                            )
                            action_select = ui.select(
                                options=[
                                    "Garder tel quel",
                                    "Encoder (One-Hot)",
                                    "Encoder (Ordinal)",
                                    "Supprimer"
                                ],
                                value="Garder tel quel"
                            ).props("outlined dense").classes("flex-1")
                            
                            decisions[col] = action_select

        # ==========================================================
        # BOUTONS DE NAVIGATION
        # ==========================================================
        def save_and_next():
            state["univariate_decisions"] = {
                col: widget.value for col, widget in decisions.items()
            }
            ui.notify("✅ Décisions sauvegardées", color="positive")
            ui.run_javascript("setTimeout(() => window.location.href='/unsupervised/multivariate_analysis', 500);")

        with ui.row().classes("w-full max-w-6xl justify-between gap-4 mt-8"):
            ui.button(
                "← Retour",
                on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/user_decisions'")
            ).style(
                "background:white !important; color:#01335A !important; font-weight:500 !important; "
                "border:1px solid #e1e8ed !important; border-radius:8px !important; height:50px !important; "
                "min-width:200px !important; font-size:14px !important; text-transform:none !important; "
                "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
            )
            
            ui.button(
                "Suivant →",
                on_click=save_and_next
            ).style(
                "background:#01335A !important; color:white !important; font-weight:600 !important; "
                "border-radius:8px !important; height:50px !important; min-width:200px !important; "
                "font-size:14px !important; text-transform:none !important;"
            )
 
 
# ----------------- PAGE /unsupervised/multivariate_analysis -----------------

@ui.page('/unsupervised/multivariate_analysis')
def multivariate_analysis_page():
    
    df = state.get("raw_df")
    features = state.get("selected_features")
    
    if df is None or features is None:
        with ui.column().style("width:100%; height:100vh; display:flex; align-items:center; justify-content:center;"):
            ui.label("❌ Données manquantes").style("font-size:20px; color:#e74c3c; font-weight:600; margin-bottom:12px;")
            ui.label("Veuillez revenir en arrière et sélectionner les features").style("color:#7f8c8d; margin-bottom:20px;")
            ui.button(" Retour", on_click=lambda: ui.navigate.to("/unsupervised/univariate_analysis")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )
        return

    # Filtrer uniquement les colonnes numériques
    df_selected = df[features].select_dtypes(include=[np.number]).copy()
    
    if df_selected.empty or len(df_selected.columns) < 2:
        with ui.column().style("width:100%; height:100vh; display:flex; align-items:center; justify-content:center;"):
            ui.label("⚠️ Analyse impossible").style("font-size:20px; color:#f39c12; font-weight:600; margin-bottom:12px;")
            ui.label("Il faut au moins 2 variables numériques pour l'analyse multivariée").style("color:#7f8c8d; margin-bottom:20px;")
            ui.button(" Retour", on_click=lambda: ui.navigate.to("/unsupervised/univariate_analysis")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )
        return
    
    # Initialisation des décisions
    if "multivariate_decisions" not in state:
        state["multivariate_decisions"] = {}
    
    multivariate_decisions = state["multivariate_decisions"]
    
    # Calcul de la matrice de corrélation
    corr_matrix = df_selected.corr()
    
    # Identifier les paires fortement corrélées (> 0.9)
    correlated_pairs = []
    for i, j in combinations(corr_matrix.columns, 2):
        corr_val = corr_matrix.loc[i, j]
        if abs(corr_val) > 0.9:
            correlated_pairs.append((i, j, corr_val))

    # Page principale
    with ui.column().style("width:100%; min-height:100vh; padding:40px; background-color:#f5f6fa; font-family:'Inter', sans-serif;"):
        
        # Header simple
        ui.label("Analyse Multivariée").style("font-weight:700; font-size:32px; color:#01335A; margin-bottom:8px;")
        ui.label(f"{len(df_selected.columns)} variables numériques analysées").style("color:#7f8c8d; font-size:16px; margin-bottom:32px;")

        # ==========================================================
        # MATRICE DE CORRÉLATION
        # ==========================================================
        with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:24px;"):
            with ui.card().style("width:100%; max-width:900px; padding:24px; background:white; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.06);"):
                ui.label("Matrice de Corrélation").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:16px;")
                
                # Créer la heatmap
                try:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                                center=0, vmin=-1, vmax=1, square=True, 
                                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
                    plt.xticks(rotation=45, ha='right', fontsize=9)
                    plt.yticks(rotation=0, fontsize=9)
                    plt.tight_layout()
                    
                    # Conversion en base64
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.read()).decode()
                    plt.close(fig)
                    
                    ui.image(f"data:image/png;base64,{img_base64}").style("width:100%; border-radius:6px;")
                except Exception as e:
                    ui.label(f"⚠️ Erreur : {str(e)}").style("color:#e74c3c; font-size:14px;")

        # ==========================================================
        # PAIRES FORTEMENT CORRÉLÉES
        # ==========================================================
        with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:24px;"):
            with ui.card().style("width:100%; max-width:900px; padding:24px; background:white; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.06);"):
                ui.label("Paires fortement corrélées (|r| > 0.9)").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:16px;")
                
                if not correlated_pairs:
                    ui.label("✅ Aucune redondance détectée").style("color:#27ae60; font-size:16px; font-weight:500;")
                else:
                    ui.label(f"{len(correlated_pairs)} paire(s) détectée(s)").style("color:#7f8c8d; font-size:14px; margin-bottom:16px;")
                    
                    for idx, (f1, f2, corr_val) in enumerate(correlated_pairs):
                        pair_key = f"{f1}__{f2}"
                        
                        # Récupérer ou initialiser la décision
                        if pair_key not in multivariate_decisions:
                            multivariate_decisions[pair_key] = "Garder les deux"
                        
                        with ui.card().style("padding:16px; margin-bottom:12px; background:#f8f9fa; border-left:3px solid #e74c3c; border-radius:6px;"):
                            
                            # Variables et corrélation
                            with ui.row().style("display:flex; align-items:center; gap:12px; margin-bottom:12px;"):
                                ui.label(f1).style("font-weight:600; color:#2c3e50; font-size:15px;")
                                ui.label("↔").style("color:#7f8c8d;")
                                ui.label(f2).style("font-weight:600; color:#2c3e50; font-size:15px;")
                                ui.label(f"r = {corr_val:.3f}").style("color:#e74c3c; font-weight:600; margin-left:auto; font-size:15px;")
                            
                            # Sélecteur d'action
                            with ui.row().style("display:flex; align-items:center; gap:12px;"):
                                ui.label("Action:").style("color:#7f8c8d; font-size:14px;")
                                
                                action_select = ui.select(
                                    options=[
                                        "Garder les deux",
                                        f"Garder '{f1}' uniquement",
                                        f"Garder '{f2}' uniquement"
                                    ],
                                    value=multivariate_decisions[pair_key]
                                ).style("width:280px;")
                                
                                # Mise à jour de la décision
                                action_select.on_value_change(lambda e, key=pair_key: multivariate_decisions.update({key: e.value}))

        # ==========================================================
        # FONCTION D'APPLICATION DES DÉCISIONS
        # ==========================================================
        def apply_decisions():
            cols_to_remove = []
            
            for f1, f2, _ in correlated_pairs:
                pair_key = f"{f1}__{f2}"
                decision = multivariate_decisions.get(pair_key, "Garder les deux")
                
                if decision == f"Garder '{f1}' uniquement":
                    cols_to_remove.append(f2)
                elif decision == f"Garder '{f2}' uniquement":
                    cols_to_remove.append(f1)
            
            # Supprimer les doublons
            cols_to_remove = list(set(cols_to_remove))
            
            if cols_to_remove:
                # Mise à jour des features
                updated_features = [f for f in features if f not in cols_to_remove]
                state["selected_features"] = updated_features
                
                ui.notify(
                    f"✅ {len(cols_to_remove)} variable(s) supprimée(s)",
                    type="positive"
                )
            else:
                ui.notify("ℹ️ Aucune modification", type="info")
            
            ui.navigate.to("/unsupervised/missing_values")

        # ==========================================================
        # BOUTONS DE NAVIGATION
        # ==========================================================
        with ui.row().style("display:flex; justify-content:space-between; width:100%; margin-top:32px;"):
            ui.button(" Retour", on_click=lambda: ui.navigate.to("/unsupervised/univariate_analysis")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )
            
            ui.button("Appliquer et Continuer ", on_click=apply_decisions).style(
                "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:250px !important;"
            )




#  ----------------- PAGE /unsupervised/anomalies ----------------- 


@ui.page('/unsupervised/anomalies')
def anomalies_page():
    # Récupérer les données ENCODÉES de l'étape précédente
    df = state.get("encoded_data")
    
    if df is None:
        with ui.column().style("width:100%; height:100vh; display:flex; align-items:center; justify-content:center;"):
            ui.label("❌ Données manquantes").style("font-size:20px; color:#e74c3c; font-weight:600; margin-bottom:12px;")
            ui.button(" Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/encoding'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )
        return
    
    # Ne garder que les colonnes numériques (après encodage, tout devrait être numérique)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        with ui.column().style("width:100%; height:100vh; display:flex; align-items:center; justify-content:center;"):
            ui.label("⚠️ Aucune variable numérique").style("font-size:20px; color:#f39c12; font-weight:600; margin-bottom:12px;")
            ui.button(" Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/encoding'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )
        return
    
    decisions = {}
    
    # Pré-calculer toutes les statistiques (rapide)
    stats_data = {}
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) == 0:
            continue
            
        series = series.reset_index(drop=True)
        q1 = np.percentile(series, 25)
        q3 = np.percentile(series, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        is_outlier_iqr = (series < lower) | (series > upper)
        outliers_iqr_count = is_outlier_iqr.sum()
        outliers_iqr_pct = (outliers_iqr_count / len(series)) * 100
        
        z_scores = np.abs(zscore(series))
        is_outlier_z = z_scores > 3
        outliers_z_count = is_outlier_z.sum()
        outliers_z_pct = (outliers_z_count / len(series)) * 100
        
        stats_data[col] = {
            'series': series,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'lower': lower,
            'upper': upper,
            'is_outlier_iqr': is_outlier_iqr,
            'outliers_iqr_count': outliers_iqr_count,
            'outliers_iqr_pct': outliers_iqr_pct,
            'is_outlier_z': is_outlier_z,
            'outliers_z_count': outliers_z_count,
            'outliers_z_pct': outliers_z_pct
        }
    
    with ui.column().style("width:100%; min-height:100vh; padding:40px; background:#f5f6fa; font-family:'Inter', sans-serif;"):
        
        # Header avec animation hover
        with ui.column().style("margin-bottom:48px; text-align:center;"):
            ui.label("🔍 Détection d'Anomalies").style("font-weight:800; font-size:42px; background:linear-gradient(135deg, #01335A, #09538C); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:12px; letter-spacing:-1px;")
        
        # INFO MÉTHODES
        with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:24px;"):
            with ui.card().style("width:100%; max-width:900px; padding:24px; background:white; display:flex; align-items:center; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                
                # En-tête de section avec ligne décorative
                with ui.column().style("margin-bottom:20px;"):
                    ui.label("📖 Méthodes de Détection").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                    ui.separator().style("width:60px; height:3px; background:#3498db; border-radius:2px; margin:0;")
                
                with ui.column().style("gap:8px; width:100%;"):
                    ui.label("• IQR (Interquartile Range) : Valeurs < Q1-1.5×IQR ou > Q3+1.5×IQR").style("font-size:14px; color:#7f8c8d;")
                    ui.label("• Z-score : Valeurs avec |z-score| > 3 (très éloignées de la moyenne)").style("font-size:14px; color:#7f8c8d;")
        
        # ANALYSE PAR VARIABLE
        with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:24px;"):
            with ui.card().style("width:100%; max-width:900px; padding:24px; background:white; display:flex; align-items:center; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                
                # En-tête de section
                with ui.column().style("margin-bottom:20px;"):
                    ui.label(f"Variables Numériques ({len(stats_data)})").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                    ui.separator().style("width:60px; height:3px; background:#e67e22; border-radius:2px; margin:0;")
                
                for idx, (col, data) in enumerate(stats_data.items(), 1):
                    series = data['series']
                    q1 = data['q1']
                    q3 = data['q3']
                    lower = data['lower']
                    upper = data['upper']
                    is_outlier_iqr = data['is_outlier_iqr']
                    outliers_iqr_count = data['outliers_iqr_count']
                    outliers_iqr_pct = data['outliers_iqr_pct']
                    is_outlier_z = data['is_outlier_z']
                    outliers_z_count = data['outliers_z_count']
                    outliers_z_pct = data['outliers_z_pct']
                    
                    # Déterminer la couleur selon la sévérité
                    if outliers_iqr_pct > 10:
                        severity_color = "#e74c3c"
                        severity_bg = "#fadbd8"
                        severity_label = "Critique"
                    elif outliers_iqr_pct > 5:
                        severity_color = "#e67e22"
                        severity_bg = "#fdebd0"
                        severity_label = "Modéré"
                    else:
                        severity_color = "#27ae60"
                        severity_bg = "#d5f4e6"
                        severity_label = "Faible"
                    
                    # Carte avec expansion
                    with ui.expansion(text=f"{col}", icon="analytics").style(
                        "width:700px; margin-bottom:16px; background:#fafbfc; border-radius:8px; border:1px solid #e1e8ed; transition:all 0.2s;"
                    ):
                        with ui.column().style("width:100%; padding:16px;"):
                            
                            # Statistiques d'anomalies
                            with ui.row().style("display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; width:100%;"):
                                ui.label(f"Outliers détectés").style("font-weight:600; color:#2c3e50; font-size:16px;")
                                with ui.row().style("display:flex; gap:12px; align-items:center;"):
                                    with ui.card().style(f"padding:6px 12px; background:{severity_bg}; border-radius:6px; border:1px solid {severity_color};"):
                                        ui.label(f"IQR: {outliers_iqr_count} ({outliers_iqr_pct:.1f}%)").style(f"color:{severity_color}; font-weight:700; font-size:14px;")
                                    with ui.card().style("padding:6px 12px; background:#f0f0f0; border-radius:6px; border:1px solid #95a5a6;"):
                                        ui.label(f"Z: {outliers_z_count} ({outliers_z_pct:.1f}%)").style("color:#7f8c8d; font-weight:700; font-size:14px;")
                                    ui.label(severity_label).style("color:#7f8c8d; font-size:14px; font-weight:500;")
                            
                            # Statistiques descriptives compactes
                            with ui.row().style("gap:8px; margin-bottom:16px; flex-wrap:wrap;"):
                                stats_items = [
                                    ("Min", series.min()),
                                    ("Q1", q1),
                                    ("Médiane", series.median()),
                                    ("Q3", q3),
                                    ("Max", series.max()),
                                    ("Moyenne", series.mean())
                                ]
                                for label, value in stats_items:
                                    with ui.card().style("padding:8px 12px; background:#f9fafb; border-radius:6px; min-width:80px;"):
                                        ui.label(f"{label}: {value:.2f}").style("font-size:12px; font-weight:600; color:#34495e;")
                            
                            # Bouton pour charger le graphique (lazy loading)
                            graph_container = ui.column().style("width:100%;")
                            
                            def create_graph(container, col_name, series_data, lower_val, upper_val, outliers_mask, outliers_count):
                                def load_graph():
                                    container.clear()
                                    with container:
                                        with ui.pyplot(figsize=(10, 4), close=True):
                                            x_pos = np.arange(len(series_data))
                                            
                                            # Points normaux
                                            normal_mask = ~outliers_mask
                                            if normal_mask.any():
                                                plt.scatter(x_pos[normal_mask], series_data[normal_mask], 
                                                          alpha=0.5, color='#3498db', s=30, label="Valeurs normales")
                                            
                                            # Outliers
                                            if outliers_mask.any():
                                                plt.scatter(x_pos[outliers_mask], series_data[outliers_mask], 
                                                          color="#e74c3c", s=60, zorder=5, 
                                                          edgecolors='#c0392b', linewidth=1.5,
                                                          label=f"Outliers ({outliers_count})")
                                            
                                            # Lignes de seuil
                                            plt.axhline(y=lower_val, color='#e67e22', linestyle='--', linewidth=1.5, 
                                                      alpha=0.7, label=f'Seuil inf. ({lower_val:.2f})')
                                            plt.axhline(y=upper_val, color='#e67e22', linestyle='--', linewidth=1.5, 
                                                      alpha=0.7, label=f'Seuil sup. ({upper_val:.2f})')
                                            
                                            plt.title(f"Distribution de {col_name}", fontsize=12, fontweight='bold')
                                            plt.xlabel("Index", fontsize=9)
                                            plt.ylabel("Valeur", fontsize=9)
                                            plt.legend(fontsize=8, loc='best')
                                            plt.grid(True, alpha=0.2, linestyle=':')
                                            plt.tight_layout()
                                return load_graph
                            
                            ui.button(
                                "📊 Afficher le graphique", 
                                on_click=create_graph(graph_container, col, series, lower, upper, is_outlier_iqr, outliers_iqr_count)
                            ).style("background:#3498db; color:white; padding:8px 16px; border-radius:6px; margin-bottom:12px; cursor:pointer; font-weight:600;")
                            
                            # Dropdown décision utilisateur
                            ui.separator().style("margin:16px 0; background:#ecf0f1;")
                            
                            with ui.row().style("display:flex; align-items:center; justify-content:center; width:100%; gap:16px;"):
                                ui.label("Action :").style("font-weight:600; color:#2c3e50; font-size:14px;")
                                
                                # Recommandation automatique
                                if outliers_iqr_pct > 10:
                                    recommended = "Winsoriser"
                                    recommendation_text = "⚠️ >10% outliers"
                                    recommendation_color = "#e67e22"
                                elif outliers_iqr_pct > 5:
                                    recommended = "Winsoriser"
                                    recommendation_text = "💡 >5% outliers"
                                    recommendation_color = "#3498db"
                                else:
                                    recommended = "Garder"
                                    recommendation_text = "✅ Peu d'outliers"
                                    recommendation_color = "#27ae60"
                                
                                select_widget = ui.select(
                                    options=["Garder", "Supprimer", "Winsoriser"],
                                    value=recommended
                                ).style("width:200px; border:2px solid #e1e8ed; border-radius:6px;")
                                
                                decisions[col] = select_widget
                                
                                ui.label(recommendation_text).style(f"font-size:12px; color:{recommendation_color}; font-weight:600;")
        
        # LÉGENDE DES ACTIONS
        with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:24px;"):
            with ui.card().style("width:100%; max-width:900px; padding:24px; background:white; display:flex; align-items:center; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                
                # En-tête de section
                with ui.column().style("margin-bottom:20px;"):
                    ui.label("📋 Guide des Actions").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                    ui.separator().style("width:60px; height:3px; background:#9b59b6; border-radius:2px; margin:0;")
                
                with ui.column().style("gap:8px; width:100%;"):
                    ui.label("• Garder : Conserver toutes les valeurs sans modification").style("font-size:14px; color:#7f8c8d;")
                    ui.label("• Supprimer : Retirer les lignes contenant des outliers").style("font-size:14px; color:#7f8c8d;")
                    ui.label("• Winsoriser : Remplacer les outliers par les seuils IQR").style("font-size:14px; color:#7f8c8d;")
        
        # BOUTONS DE NAVIGATION
        def save_and_next():
            # Sauvegarder les décisions
            anomaly_decisions = {col: widget.value for col, widget in decisions.items()}
            state["anomaly_decisions"] = anomaly_decisions
            
            # Appliquer les transformations
            df_processed = df.copy()
            removed_rows = 0
            
            for col, decision in anomaly_decisions.items():
                if col not in df_processed.columns:
                    continue
                
                series = pd.to_numeric(df_processed[col], errors="coerce")
                
                if decision == "Supprimer":
                    initial_rows = len(df_processed)
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    
                    mask = (series >= lower) & (series <= upper)
                    df_processed = df_processed[mask]
                    removed_rows += initial_rows - len(df_processed)
                
                elif decision == "Winsoriser":
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    
                    df_processed[col] = series.clip(lower=lower, upper=upper)
            
            state["anomalies_processed_data"] = df_processed
            
            ui.run_javascript("window.location.href='/unsupervised/normalization'")
        
        with ui.row().style("display:flex; justify-content:space-between; width:100%; margin-top:32px;"):
            ui.button(" Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/encoding'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important; transition:all 0.2s !important;"
            )
            ui.button("Appliquer et Continuer ", on_click=save_and_next).style(
                "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:250px !important; transition:all 0.2s !important;"
            )
# 

# ----------------- PAGE /unsupervised/normalization ----------------- 


def apply_normalization(series, method):
    """Applique la normalisation à une série"""
    series_clean = pd.to_numeric(series, errors='coerce')
    values = series_clean.values.reshape(-1, 1)
    
    if method == "Z-Score (StandardScaler)":
        scaler = StandardScaler()
        normalized = scaler.fit_transform(values).flatten()
    elif method == "Min-Max (0-1)":
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(values).flatten()
    else:
        return series_clean
    
    result = series_clean.copy()
    result[series_clean.notna()] = normalized[~np.isnan(values.flatten())]
    return result

@ui.page('/unsupervised/normalization')
def normalization_page():
    # Récupérer les données TRAITÉES de l'étape d'anomalies
    df = state.get("anomalies_processed_data")
    
    if df is None:
        with ui.column().style("width:100%; height:100vh; display:flex; align-items:center; justify-content:center;"):
            ui.label("❌ Données manquantes").style("font-size:20px; color:#e74c3c; font-weight:600; margin-bottom:12px;")
            ui.button(" Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/anomalies'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )
        return
    
    # Identifier les features numériques
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numerical_features:
        with ui.column().style("width:100%; height:100vh; display:flex; align-items:center; justify-content:center;"):
            ui.label("⚠️ Aucune variable numérique").style("font-size:20px; color:#f39c12; font-weight:600; margin-bottom:12px;")
            ui.button(" Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/anomalies'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )
        return
    
    # Calculer statistiques pour chaque feature numérique
    features_stats = []
    for col in numerical_features:
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(series) > 0:
            features_stats.append({
                'col': col,
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'range': series.max() - series.min()
            })
    
    # Stockage des décisions
    normalization_decisions = {}
    
    with ui.column().style("width:100%; min-height:100vh; padding:40px; background:#f5f6fa; font-family:'Inter', sans-serif;"):
        
        # Header avec animation hover
        with ui.column().style("margin-bottom:48px; text-align:center;"):
            ui.label("📐 Normalisation des Features").style("font-weight:800; font-size:42px; background:linear-gradient(135deg, #01335A, #09538C); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:12px; letter-spacing:-1px;")
        
        # INFO BOX - POURQUOI NORMALISER
        with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:24px;"):
            with ui.card().style("width:100%; max-width:900px; padding:24px; background:white; display:flex; align-items:center; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                
                # En-tête de section avec ligne décorative
                with ui.column().style("margin-bottom:20px;"):
                    ui.label("ℹ️ Pourquoi Normaliser ?").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                    ui.separator().style("width:60px; height:3px; background:#3498db; border-radius:2px; margin:0;")
                
                ui.label("La normalisation met toutes les features sur la même échelle, essentiel pour :").style("color:#7f8c8d; margin-bottom:12px; font-size:14px;")
                with ui.column().style("margin-left:20px; gap:6px;"):
                    ui.label("• Algorithmes basés sur les distances (K-Means, DBSCAN, KNN)").style("color:#7f8c8d; font-size:14px;")
                    ui.label("• Algorithmes à gradient (régression, réseaux de neurones)").style("color:#7f8c8d; font-size:14px;")
                    ui.label("• Éviter qu'une feature domine les autres à cause de son échelle").style("color:#7f8c8d; font-size:14px;")
        
        # MÉTHODES DE NORMALISATION
        with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:24px;"):
            with ui.card().style("width:100%; max-width:900px; padding:24px; background:white; display:flex; align-items:center; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                
                # En-tête de section
                with ui.column().style("margin-bottom:20px;"):
                    ui.label("🎯 Méthodes de Normalisation").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                    ui.separator().style("width:60px; height:3px; background:#e67e22; border-radius:2px; margin:0;")
                
                with ui.row().style("gap:16px; margin-bottom:20px; width:100%;"):
                    # Z-Score
                    with ui.card().style("padding:20px; background:#e3f2fd; flex:1; border-radius:8px; border:2px solid #3498db;"):
                        ui.label("📐 Z-Score (StandardScaler)").style("font-weight:700; color:#01335A; margin-bottom:12px; font-size:16px;")
                        ui.label("Formule : (x - μ) / σ").style("font-family:monospace; font-size:13px; color:#7f8c8d; margin-bottom:8px; background:white; padding:8px; border-radius:4px;")
                        ui.label("Résultat : Moyenne = 0, Écart-type = 1").style("font-size:13px; color:#2c3e50; margin-bottom:8px; font-weight:600;")
                        with ui.column().style("gap:4px;"):
                            ui.label("✓ Préserve la forme de la distribution").style("font-size:12px; color:#27ae60;")
                            ui.label("✓ Idéal pour distributions normales").style("font-size:12px; color:#27ae60;")
                            ui.label("✓ Utilisé par la plupart des algorithmes ML").style("font-size:12px; color:#27ae60;")
                    
                    # Min-Max
                    with ui.card().style("padding:20px; background:#f3e5f5; flex:1; border-radius:8px; border:2px solid #9b59b6;"):
                        ui.label("📏 Min-Max (0-1)").style("font-weight:700; color:#8e44ad; margin-bottom:12px; font-size:16px;")
                        ui.label("Formule : (x - min) / (max - min)").style("font-family:monospace; font-size:13px; color:#7f8c8d; margin-bottom:8px; background:white; padding:8px; border-radius:4px;")
                        ui.label("Résultat : Valeurs entre 0 et 1").style("font-size:13px; color:#2c3e50; margin-bottom:8px; font-weight:600;")
                        with ui.column().style("gap:4px;"):
                            ui.label("✓ Préserve les zéros").style("font-size:12px; color:#27ae60;")
                            ui.label("✓ Utile pour réseaux de neurones").style("font-size:12px; color:#27ae60;")
                            ui.label("⚠️ Sensible aux outliers").style("font-size:12px; color:#e67e22;")
                
                # Application globale
                ui.separator().style("margin:20px 0; background:#ecf0f1;")
                
                with ui.row().style("align-items:center; gap:16px; width:100%; justify-content:center;"):
                    ui.label("Appliquer à toutes les features :").style("font-weight:600; font-size:15px; color:#2c3e50;")
                    
                    global_selector = ui.select(
                        options=["Aucune", "Z-Score (StandardScaler)", "Min-Max (0-1)"],
                        value="Aucune"
                    ).style("min-width:250px; border:2px solid #e1e8ed; border-radius:6px;")
                    
                    def apply_global_method():
                        method = global_selector.value
                        for col in numerical_features:
                            if col in normalization_decisions:
                                normalization_decisions[col].value = method
                    
                    ui.button("Appliquer à tout", on_click=apply_global_method).style(
                        "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white; font-weight:600; border-radius:8px; padding:10px 20px; cursor:pointer;"
                    )
        
        # CONFIGURATION PAR FEATURE
        with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:24px;"):
            with ui.card().style("width:100%; max-width:900px; padding:24px; background:white; display:flex; align-items:center; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                
                # En-tête de section
                with ui.column().style("margin-bottom:20px;"):
                    ui.label(f"Configuration par Feature ({len(features_stats)})").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                    ui.separator().style("width:60px; height:3px; background:#9b59b6; border-radius:2px; margin:0;")
                
                for idx, stats in enumerate(features_stats, 1):
                    col = stats['col']
                    
                    # Déterminer si la feature bénéficierait de la normalisation
                    if stats['range'] > 100:
                        recommendation = "Z-Score (StandardScaler)"
                        recommendation_text = "⚠️ Large plage"
                        recommendation_color = "#e67e22"
                        recommendation_bg = "#fdebd0"
                    elif stats['std'] > stats['mean']:
                        recommendation = "Z-Score (StandardScaler)"
                        recommendation_text = "💡 Forte variance"
                        recommendation_color = "#3498db"
                        recommendation_bg = "#d6eaf8"
                    else:
                        recommendation = "Min-Max (0-1)"
                        recommendation_text = "✓ Min-Max adapté"
                        recommendation_color = "#27ae60"
                        recommendation_bg = "#d5f4e6"
                    
                    with ui.expansion(text=f"{col}", icon="tune").style(
                        "width:700px; margin-bottom:16px; background:#fafbfc; border-radius:8px; border:1px solid #e1e8ed; transition:all 0.2s;"
                    ):
                        with ui.column().style("width:100%; padding:16px;"):
                            
                            # Statistiques descriptives
                            with ui.row().style("display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; width:100%;"):
                                ui.label("Statistiques").style("font-weight:600; color:#2c3e50; font-size:16px;")
                                with ui.row().style("display:flex; gap:8px; align-items:center;"):
                                    with ui.card().style(f"padding:6px 12px; background:{recommendation_bg}; border-radius:6px; border:1px solid {recommendation_color};"):
                                        ui.label(recommendation_text).style(f"color:{recommendation_color}; font-weight:700; font-size:14px;")
                            
                            with ui.row().style("gap:8px; margin-bottom:16px; flex-wrap:wrap;"):
                                stats_items = [
                                    ("Min", stats['min']),
                                    ("Max", stats['max']),
                                    ("Moyenne", stats['mean']),
                                    ("Écart-type", stats['std']),
                                    ("Plage", stats['range'])
                                ]
                                for label, value in stats_items:
                                    with ui.card().style("padding:8px 12px; background:#f9fafb; border-radius:6px; min-width:100px;"):
                                        ui.label(f"{label}: {value:.2f}").style("font-size:12px; font-weight:600; color:#34495e;")
                            
                            # Sélecteur de méthode
                            ui.separator().style("margin:16px 0; background:#ecf0f1;")
                            
                            with ui.row().style("display:flex; align-items:center; justify-content:center; width:100%; gap:16px;"):
                                ui.label("Méthode :").style("font-weight:600; color:#2c3e50; font-size:14px;")
                                
                                normalization_decisions[col] = ui.select(
                                    options=["Aucune", "Z-Score (StandardScaler)", "Min-Max (0-1)"],
                                    value=recommendation
                                ).style("width:250px; border:2px solid #e1e8ed; border-radius:6px;")
                            
                            # Aperçu des valeurs avant/après
                            preview_container = ui.column().style("width:100%;")
                            
                            def create_preview_handler(column, container):
                                def show_preview():
                                    container.clear()
                                    with container:
                                        method = normalization_decisions[column].value
                                        
                                        if method == "Aucune":
                                            with ui.card().style("padding:16px; background:#f8f9fa; border-radius:8px; margin-top:12px;"):
                                                ui.label("ℹ️ Sélectionnez une méthode pour voir l'aperçu").style(
                                                    "color:#7f8c8d; font-size:13px; font-style:italic;"
                                                )
                                        else:
                                            original_series = pd.to_numeric(df[column], errors='coerce').dropna()
                                            normalized_series = apply_normalization(df[column], method).dropna()
                                            
                                            # Stats avant/après
                                            with ui.row().style("gap:16px; margin:16px 0;"):
                                                with ui.card().style("padding:16px; background:#e3f2fd; flex:1; border-radius:8px; border:2px solid #3498db;"):
                                                    ui.label("📊 Avant normalisation").style("font-size:13px; color:#01335A; margin-bottom:8px; font-weight:700;")
                                                    ui.label(f"Moyenne : {original_series.mean():.4f}").style("font-size:12px; margin-bottom:4px; color:#7f8c8d;")
                                                    ui.label(f"Écart-type : {original_series.std():.4f}").style("font-size:12px; margin-bottom:4px; color:#7f8c8d;")
                                                    ui.label(f"Min : {original_series.min():.4f}").style("font-size:12px; margin-bottom:4px; color:#7f8c8d;")
                                                    ui.label(f"Max : {original_series.max():.4f}").style("font-size:12px; color:#7f8c8d;")
                                                
                                                with ui.card().style("padding:16px; background:#d5f4e6; flex:1; border-radius:8px; border:2px solid #27ae60;"):
                                                    ui.label("✨ Après normalisation").style("font-size:13px; color:#27ae60; margin-bottom:8px; font-weight:700;")
                                                    ui.label(f"Moyenne : {normalized_series.mean():.4f}").style("font-size:12px; margin-bottom:4px; color:#27ae60;")
                                                    ui.label(f"Écart-type : {normalized_series.std():.4f}").style("font-size:12px; margin-bottom:4px; color:#27ae60;")
                                                    ui.label(f"Min : {normalized_series.min():.4f}").style("font-size:12px; margin-bottom:4px; color:#27ae60;")
                                                    ui.label(f"Max : {normalized_series.max():.4f}").style("font-size:12px; color:#27ae60;")
                                
                                return show_preview
                            
                            preview_handler = create_preview_handler(col, preview_container)
                            ui.button(" Voir les statistiques", on_click=preview_handler).style(
                                "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white; padding:8px 16px; border-radius:6px; margin:12px 0; cursor:pointer; font-size:13px; font-weight:600;"
                            )
                            
                            # Zone de prévisualisation
                            preview_container
        
        # RECOMMANDATIONS
        with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:24px;"):
            with ui.card().style("width:100%; max-width:900px; padding:24px; background:white; display:flex; align-items:center; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                
                # En-tête de section
                with ui.column().style("margin-bottom:20px;"):
                    ui.label("💡 Recommandations").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                    ui.separator().style("width:60px; height:3px; background:#f39c12; border-radius:2px; margin:0;")
                
                with ui.column().style("gap:8px; width:100%;"):
                    ui.label("• Z-Score : Idéal pour features avec distribution normale ou large plage de valeurs").style("color:#7f8c8d; font-size:14px;")
                    ui.label("• Min-Max : Préférable pour features avec distribution uniforme ou besoins d'intervalle fixe").style("color:#7f8c8d; font-size:14px;")
                    ui.label("• Si vous avez traité les outliers, Z-Score est généralement plus robuste").style("color:#7f8c8d; font-size:14px;")
                    ui.label("• Pour des algorithmes comme K-Means, la normalisation est fortement recommandée").style("color:#7f8c8d; font-size:14px;")
        
        # NAVIGATION
        def save_and_next():
            decisions = {col: widget.value for col, widget in normalization_decisions.items()}
            state["normalization_decisions"] = decisions
            
            # Appliquer les normalisations
            df_normalized = df.copy()
            normalized_count = 0
            
            for col, method in decisions.items():
                if method != "Aucune" and col in df_normalized.columns:
                    df_normalized[col] = apply_normalization(df_normalized[col], method)
                    normalized_count += 1
            
            state["normalized_data"] = df_normalized
            
            ui.run_javascript("window.location.href='/unsupervised/pca'")
        
        with ui.row().style("display:flex; justify-content:space-between; width:100%; margin-top:32px;"):
            ui.button(" Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/anomalies'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important; transition:all 0.2s !important;"
            )
            ui.button("Appliquer et Continuer ", on_click=save_and_next).style(
                "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:250px !important; transition:all 0.2s !important;"
            )



# ---------------- PAGE /unsupervised/pca ----------------- 


@ui.page('/unsupervised/pca')
def pca_page():
    # Récupérer les données NORMALISÉES de l'étape précédente
    df = state.get("normalized_data")
    
    if df is None:
        with ui.column().style("width:100%; height:100vh; display:flex; align-items:center; justify-content:center;"):
            ui.label("❌ Données manquantes").style("font-size:20px; color:#e74c3c; font-weight:600; margin-bottom:12px;")
            ui.button(" Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/normalization'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )
        return
    
    # Garder seulement les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        with ui.column().style("width:100%; height:100vh; display:flex; align-items:center; justify-content:center;"):
            ui.label("⚠️ Minimum 2 variables requises").style("font-size:20px; color:#f39c12; font-weight:600; margin-bottom:12px;")
            ui.button(" Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/normalization'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )
        return
    
    # Dataset numérique propre
    numeric_df = df[numeric_cols].copy()
    numeric_df = numeric_df.dropna()
    
    if len(numeric_df) == 0:
        with ui.column().style("width:100%; height:100vh; display:flex; align-items:center; justify-content:center;"):
            ui.label("❌ Données insuffisantes").style("font-size:20px; color:#e74c3c; font-weight:600; margin-bottom:12px;")
            ui.button(" Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/normalization'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )
        return
    
    max_components = min(len(numeric_cols), len(numeric_df))
    
    with ui.column().style("width:100%; min-height:100vh; padding:40px; background:#f5f6fa; font-family:'Inter', sans-serif;"):
        
        # HEADER
        with ui.column().style("margin-bottom:48px; text-align:center;"):
            ui.label("🔬 Réduction de Dimension (PCA)").style("font-weight:800; font-size:42px; background:linear-gradient(135deg, #01335A, #09538C); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:12px; letter-spacing:-1px;")
        
        # INFO BOX - QU'EST-CE QUE LA PCA
        with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:24px;"):
            with ui.card().style("width:100%; max-width:900px; padding:24px; background:white; display:flex; align-items:center; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                
                # En-tête de section
                with ui.column().style("margin-bottom:20px;"):
                    ui.label("ℹ️ Qu'est-ce que la PCA ?").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                    ui.separator().style("width:60px; height:3px; background:#3498db; border-radius:2px; margin:0;")
                
                ui.label("La PCA (Principal Component Analysis) réduit le nombre de variables en créant de nouvelles variables appelées 'composantes principales' qui capturent le maximum d'information.").style("color:#7f8c8d; margin-bottom:12px; font-size:14px;")
                
                with ui.column().style("margin-left:20px; gap:6px;"):
                    ui.label("• Visualiser des données complexes en 2D ou 3D").style("color:#7f8c8d; font-size:14px;")
                    ui.label("• Accélérer les algorithmes en réduisant les dimensions").style("color:#7f8c8d; font-size:14px;")
                    ui.label("• Éliminer les corrélations entre variables").style("color:#7f8c8d; font-size:14px;")
                    ui.label("• Détecter les patterns cachés dans les données").style("color:#7f8c8d; font-size:14px;")
        
        # CONFIGURATION PCA
        with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:24px;"):
            with ui.card().style("width:100%; max-width:900px; padding:24px; background:white; display:flex; align-items:center; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                
                # En-tête de section
                with ui.column().style("margin-bottom:20px;"):
                    ui.label("⚙️ Configuration de la PCA").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                    ui.separator().style("width:60px; height:3px; background:#e67e22; border-radius:2px; margin:0;")
                
                # Switch activation
                with ui.row().style("align-items:center; gap:16px; margin-bottom:20px; padding:16px; background:#f9fafb; border-radius:8px; width:100%;"):
                    apply_switch = ui.switch("Activer la PCA", value=False).style("font-size:16px; font-weight:600;")
                    ui.label("Désactivez si vous souhaitez conserver toutes les variables originales").style("font-size:13px; color:#7f8c8d; font-style:italic;")
                
                # Nombre de composantes
                with ui.column().style("width:100%; margin-bottom:20px;"):
                    with ui.row().style("align-items:center; gap:16px; margin-bottom:12px;"):
                        ui.label("Nombre de composantes :").style("font-weight:600; font-size:15px; color:#2c3e50;")
                        n_comp_input = ui.number(
                            label="",
                            value=2,
                            min=2,
                            max=max_components,
                            step=1
                        ).props('outlined dense').style("max-width:120px;")
                        ui.label(f"(Maximum : {max_components})").style("font-size:13px; color:#7f8c8d;")
                    
                    # Info box
                    with ui.card().style("padding:16px; background:#fff3e0; border-left:3px solid #f39c12; border-radius:8px;"):
                        ui.label("💡 Recommandation").style("font-weight:700; font-size:14px; color:#e67e22; margin-bottom:8px;")
                        with ui.column().style("gap:4px;"):
                            ui.label("• 2 composantes : Idéal pour la visualisation 2D").style("font-size:13px; color:#7f8c8d;")
                            ui.label("• 3 composantes : Pour la visualisation 3D interactive").style("font-size:13px; color:#7f8c8d;")
                            ui.label("• Plus de 3 : Préserve plus d'information mais perd l'aspect visuel").style("font-size:13px; color:#7f8c8d;")
                
                # Bouton aperçu
                preview_area = ui.column().style("width:100%; margin-top:20px;")
                
                def preview_pca():
                    preview_area.clear()
                    
                    with preview_area:
                        if not apply_switch.value:
                            with ui.card().style("padding:20px; background:#f8f9fa; border-radius:8px; text-align:center;"):
                                ui.label("ℹ️ Activez la PCA pour voir l'aperçu").style(
                                    "color:#7f8c8d; font-size:13px; font-style:italic;"
                                )
                            return
                        
                        n_components = int(n_comp_input.value)
                        
                        if n_components > max_components:
                            with ui.card().style("padding:20px; background:#fee2e2; border-left:4px solid #e74c3c; border-radius:8px;"):
                                ui.label(f"❌ Nombre de composantes trop élevé (max: {max_components})").style("font-size:14px; color:#c0392b; font-weight:600;")
                            return
                        
                        try:
                            # Application PCA
                            pca = PCA(n_components=n_components)
                            pca.fit(numeric_df)
                            
                            variance_ratio = pca.explained_variance_ratio_ * 100
                            cumulative_variance = np.cumsum(variance_ratio)
                            
                            # Résumé global
                            with ui.card().style("padding:20px; background:#d5f4e6; border-left:4px solid #27ae60; border-radius:8px; margin-bottom:16px;"):
                                ui.label(f"✅ Aperçu PCA avec {n_components} composante(s)").style("font-size:16px; font-weight:700; color:#27ae60; margin-bottom:12px;")
                                ui.label(f"📊 Variance totale conservée : {cumulative_variance[-1]:.2f}%").style("font-size:15px; color:#27ae60; font-weight:600; margin-bottom:8px;")
                                
                                if cumulative_variance[-1] >= 90:
                                    ui.label("🎯 Excellent ! Plus de 90% de l'information est préservée").style("font-size:13px; color:#27ae60;")
                                elif cumulative_variance[-1] >= 70:
                                    ui.label("👍 Bon ! Une grande partie de l'information est préservée").style("font-size:13px; color:#f39c12;")
                                else:
                                    ui.label("⚠️ Attention : Moins de 70% de l'information est préservée").style("font-size:13px; color:#e74c3c;")
                            
                            # Détail par composante
                            with ui.card().style("padding:20px; background:white; border:1px solid #e1e8ed; border-radius:8px; margin-bottom:16px;"):
                                ui.label("📈 Variance expliquée par composante").style("font-size:15px; font-weight:700; color:#2c3e50; margin-bottom:12px;")
                                
                                for i in range(n_components):
                                    if variance_ratio[i] >= 30:
                                        bg_color = "#d5f4e6"
                                        border_color = "#27ae60"
                                    elif variance_ratio[i] >= 15:
                                        bg_color = "#fdebd0"
                                        border_color = "#f39c12"
                                    else:
                                        bg_color = "#f8f9fa"
                                        border_color = "#95a5a6"
                                    
                                    with ui.row().style(f"padding:12px; margin-bottom:8px; background:{bg_color}; border-left:3px solid {border_color}; border-radius:6px; justify-content:space-between; align-items:center;"):
                                        ui.label(f"PC{i+1}").style("flex:1; font-weight:600; color:#2c3e50;")
                                        ui.label(f"{variance_ratio[i]:.2f}%").style("flex:1; text-align:center; font-weight:600; color:#34495e;")
                                        ui.label(f"{cumulative_variance[i]:.2f}%").style("flex:1; text-align:center; font-weight:600; color:#27ae60;")
                            
                            # Informations
                            with ui.card().style("padding:16px; background:#e3f2fd; border-radius:8px;"):
                                ui.label("📝 Informations").style("font-weight:700; font-size:14px; color:#01335A; margin-bottom:8px;")
                                ui.label(f"• Variables originales : {len(numeric_cols)}").style("font-size:13px; color:#7f8c8d;")
                                ui.label(f"• Variables après PCA : {n_components}").style("font-size:13px; color:#7f8c8d;")
                                ui.label(f"• Réduction : {len(numeric_cols) - n_components} variable(s) en moins").style("font-size:13px; color:#7f8c8d;")
                                ui.label(f"• Information perdue : {100 - cumulative_variance[-1]:.2f}%").style("font-size:13px; color:#7f8c8d;")
                        
                        except Exception as e:
                            with ui.card().style("padding:20px; background:#fee2e2; border-left:4px solid #e74c3c; border-radius:8px;"):
                                ui.label(f"❌ Erreur : {str(e)}").style("font-size:14px; color:#c0392b; font-weight:600;")
                
                ui.button("👁️ Voir l'aperçu", on_click=preview_pca).style(
                    "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white; font-weight:600; border-radius:8px; padding:10px 20px; cursor:pointer; margin-top:12px;"
                )
                
                preview_area
        
        # GUIDE D'INTERPRÉTATION
        with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:24px;"):
            with ui.card().style("width:100%; max-width:900px; padding:24px; background:white; display:flex; align-items:center; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                
                # En-tête de section
                with ui.column().style("margin-bottom:20px;"):
                    ui.label("💡 Comment interpréter les résultats ?").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                    ui.separator().style("width:60px; height:3px; background:#f39c12; border-radius:2px; margin:0;")
                
                with ui.column().style("gap:8px; width:100%;"):
                    ui.label("• Chaque composante (PC1, PC2...) est une combinaison linéaire des variables originales").style("color:#7f8c8d; font-size:14px;")
                    ui.label("• PC1 capture le plus de variance, PC2 la deuxième plus grande variance, etc.").style("color:#7f8c8d; font-size:14px;")
                    ui.label("• Si 2 composantes capturent 80%+ de variance, visualisation 2D possible").style("color:#7f8c8d; font-size:14px;")
                    ui.label("• Les composantes principales sont toujours orthogonales (non corrélées)").style("color:#7f8c8d; font-size:14px;")
        
        # NAVIGATION
        def save_and_next():
            pca_config = {
                "apply": apply_switch.value,
                "n_components": int(n_comp_input.value) if apply_switch.value else None
            }
            
            state["pca_decision"] = pca_config
            
            if apply_switch.value:
                n_components = int(n_comp_input.value)
                pca = PCA(n_components=n_components)
                pca_data = pca.fit_transform(numeric_df)
                
                pca_columns = [f"PC{i+1}" for i in range(n_components)]
                df_pca = pd.DataFrame(pca_data, columns=pca_columns, index=numeric_df.index)
                
                state["pca_transformed_data"] = df_pca
                state["pca_model"] = pca
            else:
                state["pca_transformed_data"] = None
                state["pca_model"] = None
            
            ui.run_javascript("window.location.href='/unsupervised/summary'")
        
        with ui.row().style("display:flex; justify-content:space-between; width:100%; margin-top:32px;"):
            ui.button(" Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/normalization'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important; transition:all 0.2s !important;"
            )
            ui.button("Appliquer et Continuer ", on_click=save_and_next).style(
                "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:250px !important; transition:all 0.2s !important;"
            )


# ----------------- PAGE /unsupervised/summary -----------------


def download_dataset(df):
    """Télécharge le dataset final en CSV"""
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    ui.download(csv_data.encode('utf-8'), filename='dataset_preprocessed.csv')
    ui.notify("✅ Dataset téléchargé avec succès !", type='positive', position='top', timeout=2000)

def validate_and_continue(df):
    """Valide et passe à l'étape de clustering"""
    state["final_preprocessed_data"] = df
    ui.run_javascript("window.location.href='/unsupervised/clustering'")

@ui.page('/unsupervised/summary')
def summary_page():
    # Récupérer toutes les décisions et données
    raw_df = state.get("raw_df")
    selected_features = state.get("selected_features")
    univariate_decisions = state.get("univariate_decisions", {})
    anomaly_decisions = state.get("anomaly_decisions", {})
    missing_decisions = state.get("missing_decisions", {})
    encoding_decisions = state.get("encoding_decisions", {})
    normalization_decisions = state.get("normalization_decisions", {})
    pca_decision = state.get("pca_decision", {'apply': False})
    
    # Récupérer le dataset final traité
    if pca_decision.get('apply'):
        df_final = state.get("pca_transformed_data")
    else:
        df_final = state.get("normalized_data")
    
    if df_final is None or raw_df is None:
        with ui.column().style("width:100%; height:100vh; display:flex; align-items:center; justify-content:center;"):
            ui.label("❌ Données manquantes").style("font-size:20px; color:#e74c3c; font-weight:600; margin-bottom:12px;")
            ui.label("Veuillez compléter toutes les étapes de prétraitement").style("font-size:14px; color:#7f8c8d; margin-bottom:20px;")
            ui.button(" Retour à l'accueil", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )
        return
    
    # Calculer les statistiques
    original_shape = (raw_df.shape[0], len(selected_features)) if selected_features else raw_df.shape
    final_shape = df_final.shape
    
    # Compter les transformations
    cols_removed_univariate = [col for col, dec in univariate_decisions.items() if dec == "Supprimer"]
    cols_winsorized = [col for col, dec in anomaly_decisions.items() if dec == "Winsoriser"]
    cols_removed_anomaly = [col for col, dec in anomaly_decisions.items() if dec == "Supprimer"]
    
    # ============= MISSING VALUES =============
    cols_imputed_dict = {}
    cols_removed_missing = []
    if missing_decisions and 'column_decisions' in missing_decisions:
        for col, action in missing_decisions['column_decisions'].items():
            if action == "Supprimer colonne":
                cols_removed_missing.append(col)
            elif action in ["Imputer Mean", "Imputer Median", "Imputer Mode", "Imputer KNN"]:
                cols_imputed_dict[col] = action
    
    # ============= ENCODING =============
    cols_encoded_dict = {}
    if encoding_decisions:
        categorical = encoding_decisions.get("categorical", {})
        numeric = encoding_decisions.get("numeric", {})
        
        if categorical:
            for col, method in categorical.items():
                cols_encoded_dict[col] = method
        
        if numeric:
            for col, method in numeric.items():
                if method != "Aucun traitement":
                    cols_encoded_dict[col] = method
    
    # ============= NORMALIZATION =============
    cols_normalized = {col: method for col, method in normalization_decisions.items() if method != "Aucune"}
    
    with ui.column().style("width:100%; min-height:100vh; padding:40px; background:#f5f6fa; font-family:'Inter', sans-serif;"):
        
        # HEADER
        with ui.column().style("margin-bottom:48px; text-align:center;"):
            ui.label("📋 Récapitulatif du Prétraitement").style("font-weight:800; font-size:42px; background:linear-gradient(135deg, #01335A, #09538C); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:12px; letter-spacing:-1px;")
            ui.label("Vérifiez vos transformations avant de passer au clustering").style("font-size:16px; color:#7f8c8d;")
        
        # TRANSFORMATION AVANT/APRÈS
        with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:24px;"):
            with ui.card().style("width:100%; max-width:900px; padding:24px; background:white; display:flex; align-items:center; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                
                # En-tête de section
                with ui.column().style("margin-bottom:20px;"):
                    ui.label("📊 Comparaison Avant/Après").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                    ui.separator().style("width:60px; height:3px; background:#3498db; border-radius:2px; margin:0;")
                
                with ui.row().style("gap:20px; margin-top:20px; width:100%; align-items:center;"):
                    # Avant
                    with ui.card().style("padding:24px; background:#fee2e2; flex:1; border-radius:8px; border:2px solid #e74c3c;"):
                        ui.label("📥 Dataset Original").style("font-size:16px; font-weight:700; color:#c0392b; margin-bottom:12px;")
                        ui.label(f"Lignes : {original_shape[0]}").style("font-size:14px; color:#7f8c8d; margin-bottom:6px;")
                        ui.label(f"Colonnes : {original_shape[1]}").style("font-size:14px; color:#7f8c8d;")
                    
                    # Flèche
                    ui.label("").style("font-size:32px; color:#3498db; font-weight:700;")
                    
                    # Après
                    with ui.card().style("padding:24px; background:#d5f4e6; flex:1; border-radius:8px; border:2px solid #27ae60;"):
                        ui.label("📤 Dataset Traité").style("font-size:16px; font-weight:700; color:#27ae60; margin-bottom:12px;")
                        ui.label(f"Lignes : {final_shape[0]}").style("font-size:14px; color:#7f8c8d; margin-bottom:6px;")
                        ui.label(f"Colonnes : {final_shape[1]}").style("font-size:14px; color:#7f8c8d;")
                
                # Statistiques de changement
                rows_diff = original_shape[0] - final_shape[0]
                cols_diff = original_shape[1] - final_shape[1]
                
                if rows_diff > 0 or cols_diff > 0 or pca_decision.get('apply'):
                    with ui.row().style("gap:12px; margin-top:20px; flex-wrap:wrap; width:100%;"):
                        if rows_diff > 0:
                            with ui.card().style("padding:12px 16px; background:#fdebd0; border-left:3px solid #f39c12; border-radius:6px;"):
                                ui.label(f"⚠️ {rows_diff} ligne(s) supprimée(s)").style("font-size:13px; font-weight:600; color:#e67e22;")
                        
                        if cols_diff > 0:
                            with ui.card().style("padding:12px 16px; background:#fdebd0; border-left:3px solid #f39c12; border-radius:6px;"):
                                ui.label(f"⚠️ {cols_diff} colonne(s) supprimée(s)").style("font-size:13px; font-weight:600; color:#e67e22;")
                        
                        if pca_decision.get('apply'):
                            with ui.card().style("padding:12px 16px; background:#d6eaf8; border-left:3px solid #3498db; border-radius:6px;"):
                                ui.label(f"✨ PCA : {pca_decision.get('n_components')} composante(s)").style("font-size:13px; font-weight:600; color:#01335A;")
        
        # TIMELINE DES ÉTAPES
        with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:24px;"):
            with ui.card().style("width:100%; max-width:900px; padding:24px; background:white; display:flex; align-items:center; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                
                # En-tête de section
                with ui.column().style("margin-bottom:20px;"):
                    ui.label("✅ Étapes Complétées").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                    ui.separator().style("width:60px; height:3px; background:#27ae60; border-radius:2px; margin:0;")
                
                timeline_steps = [
                    ("📊 Sélection", f"{len(selected_features)} features", "#3498db", "/unsupervised/user_decisions"),
                    ("📈 Univariée", f"{len(cols_removed_univariate)} supprimées" if cols_removed_univariate else "Aucune suppression", "#9b59b6", "/unsupervised/univariate_analysis"),
                    ("🔍 Anomalies", f"{len(cols_winsorized)} winsorisées" if cols_winsorized else "Aucun traitement", "#e74c3c", "/unsupervised/anomalies"),
                    ("💧 Manquantes", f"{len(cols_imputed_dict)} traitées" if cols_imputed_dict else "Aucune action", "#1abc9c", "/unsupervised/missing_values"),
                    ("🔤 Encodage", f"{len(cols_encoded_dict)} traitées" if cols_encoded_dict else "Aucune action", "#f39c12", "/unsupervised/encoding"),
                    ("📏 Normalisation", f"{len(cols_normalized)} normalisées" if cols_normalized else "Aucune normalisation", "#27ae60", "/unsupervised/normalization"),
                    ("🔬 PCA", f"{pca_decision.get('n_components')} composantes" if pca_decision.get('apply') else "Non appliquée", "#3498db", "/unsupervised/pca")
                ]
                
                for i, (title, desc, color, url) in enumerate(timeline_steps, 1):
                    with ui.row().style("align-items:center; gap:16px; margin-bottom:12px; width:100%;"):
                        # Numéro
                        with ui.element('div').style(
                            f"width:40px; height:40px; border-radius:50%; background:{color}; color:white; "
                            f"display:flex; align-items:center; justify-content:center; font-weight:700; font-size:16px; flex-shrink:0;"
                        ):
                            ui.label(str(i))
                        
                        # Contenu
                        with ui.card().style(f"padding:16px; background:#fafbfc; flex:1; border-radius:8px; border-left:4px solid {color};"):
                            with ui.row().style("width:100%; justify-content:space-between; align-items:center;"):
                                with ui.column().style("gap:4px;"):
                                    ui.label(title).style("font-weight:700; font-size:15px; color:#2c3e50;")
                                    ui.label(desc).style("font-size:13px; color:#7f8c8d;")
                                
                                ui.button("✏️", on_click=lambda u=url: ui.run_javascript(f"window.location.href='{u}'")).props('flat dense').style("color:#7f8c8d; font-size:18px;")
        
        # DÉTAILS DES TRANSFORMATIONS
        if any([cols_removed_univariate, cols_winsorized, cols_removed_anomaly, cols_imputed_dict, cols_encoded_dict, cols_normalized]):
            with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:24px;"):
                with ui.expansion("🔍 Détails des Transformations", icon='info').style(
                    "width:100%; max-width:900px; background:white; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"
                ):
                    with ui.column().style("padding:16px; gap:12px; width:100%;"):
                        
                        if cols_removed_univariate:
                            with ui.card().style("padding:12px; background:#fee2e2; border-left:3px solid #e74c3c; border-radius:6px;"):
                                ui.label(f"❌ Colonnes supprimées (univariée) : {', '.join(cols_removed_univariate)}").style("font-size:13px; color:#c0392b;")
                        
                        if cols_winsorized:
                            with ui.card().style("padding:12px; background:#fdebd0; border-left:3px solid #f39c12; border-radius:6px;"):
                                ui.label(f"🔧 Colonnes winsorisées : {', '.join(cols_winsorized)}").style("font-size:13px; color:#e67e22;")
                        
                        if cols_removed_anomaly:
                            with ui.card().style("padding:12px; background:#fee2e2; border-left:3px solid #e74c3c; border-radius:6px;"):
                                ui.label(f"❌ Colonnes supprimées (anomalies) : {', '.join(cols_removed_anomaly)}").style("font-size:13px; color:#c0392b;")
                        
                        if cols_imputed_dict:
                            with ui.card().style("padding:12px; background:#d6eaf8; border-left:3px solid #3498db; border-radius:6px;"):
                                ui.label("💧 Colonnes imputées :").style("font-size:13px; font-weight:600; color:#01335A; margin-bottom:4px;")
                                for col, method in cols_imputed_dict.items():
                                    ui.label(f"  • {col} : {method}").style("font-size:12px; color:#7f8c8d;")
                        
                        if cols_encoded_dict:
                            with ui.card().style("padding:12px; background:#fdebd0; border-left:3px solid #f39c12; border-radius:6px;"):
                                ui.label("🔤 Colonnes encodées :").style("font-size:13px; font-weight:600; color:#e67e22; margin-bottom:4px;")
                                for col, method in cols_encoded_dict.items():
                                    ui.label(f"  • {col} : {method}").style("font-size:12px; color:#7f8c8d;")
                        
                        if cols_normalized:
                            with ui.card().style("padding:12px; background:#d5f4e6; border-left:3px solid #27ae60; border-radius:6px;"):
                                ui.label("📏 Colonnes normalisées :").style("font-size:13px; font-weight:600; color:#27ae60; margin-bottom:4px;")
                                for col, method in cols_normalized.items():
                                    ui.label(f"  • {col} : {method}").style("font-size:12px; color:#7f8c8d;")
        
        # APERÇU DU DATASET
        with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:24px;"):
            with ui.card().style("width:100%; max-width:900px; padding:24px; background:white; display:flex; align-items:center; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                
                # En-tête de section
                with ui.column().style("margin-bottom:20px;"):
                    ui.label("👀 Aperçu du Dataset Final").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                    ui.separator().style("width:60px; height:3px; background:#9b59b6; border-radius:2px; margin:0;")
                
                # Statistiques par type
                numeric_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df_final.select_dtypes(exclude=[np.number]).columns.tolist()
                
                with ui.row().style("gap:12px; margin-bottom:20px; flex-wrap:wrap; width:100%;"):
                    with ui.card().style("padding:16px 20px; background:#d5f4e6; border-radius:8px; flex:1; min-width:150px;"):
                        ui.label("Variables numériques").style("font-size:12px; color:#27ae60; margin-bottom:4px; font-weight:600;")
                        ui.label(str(len(numeric_cols))).style("font-size:32px; font-weight:700; color:#27ae60;")
                    
                    with ui.card().style("padding:16px 20px; background:#fce7f3; border-radius:8px; flex:1; min-width:150px;"):
                        ui.label("Variables catégorielles").style("font-size:12px; color:#e74c3c; margin-bottom:4px; font-weight:600;")
                        ui.label(str(len(categorical_cols))).style("font-size:32px; font-weight:700; color:#e74c3c;")
                    
                    with ui.card().style("padding:16px 20px; background:#d6eaf8; border-radius:8px; flex:1; min-width:150px;"):
                        ui.label("Total variables").style("font-size:12px; color:#01335A; margin-bottom:4px; font-weight:600;")
                        ui.label(str(final_shape[1])).style("font-size:32px; font-weight:700; color:#3498db;")
                
                # Tableau des premières lignes
                ui.label("Premières lignes du dataset :").style("font-weight:600; font-size:15px; color:#2c3e50; margin-bottom:12px;")
                
                html_table = df_final.head(5).to_html(index=False, classes="preview-table")
                
                ui.html(f"""
                <div style="overflow-x:auto; max-width:100%; border-radius:8px; border:1px solid #e1e8ed;">
                    <style>
                        .preview-table {{
                            border-collapse: collapse;
                            width: 100%;
                            font-size: 13px;
                            font-family: 'Inter', sans-serif;
                        }}
                        .preview-table th {{
                            background: #01335A;
                            color: white;
                            padding: 12px;
                            text-align: left;
                            font-weight: 600;
                            border-bottom: 2px solid #09538C;
                        }}
                        .preview-table td {{
                            border: 1px solid #e1e8ed;
                            padding: 10px 12px;
                            color: #2c3e50;
                        }}
                        .preview-table tr:nth-child(even) {{
                            background: #f8f9fa;
                        }}
                        .preview-table tr:hover {{
                            background: #e8f4f8;
                        }}
                    </style>
                    {html_table}
                </div>
                """, sanitize=False)
        
        # VALIDATION FINALE
        with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:24px;"):
            with ui.card().style("width:100%; max-width:900px; padding:32px; ; display:flex; align-items:center; border-radius:8px; box-shadow:0 8px 20px rgba(1,51,90,0.3);"):
                ui.label(" Prétraitement Terminé !").style("font-size:28px; font-weight:700; color:#01335A !important; text-align:center; margin-bottom:12px;")
                ui.label("Votre dataset est prêt pour le clustering").style("font-size:16px; color:#01335A !important; text-align:center; margin-bottom:24px;")
                
                with ui.row().style("width:100%; justify-content:center; gap:16px;"):
                    ui.button("📥 Télécharger", on_click=lambda: download_dataset(df_final)).style(
                        "background:white !important; color:#01335A !important; font-weight:700 !important; border-radius:8px !important; padding:12px 24px !important; font-size:15px !important; cursor:pointer !important;"
                    )
                    
                    ui.button("🚀 Lancer le Clustering ", on_click=lambda: validate_and_continue(df_final)).style(
                        "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white !important; font-weight:700 !important; border-radius:8px !important; padding:12px 28px !important; font-size:15px !important; cursor:pointer !important; box-shadow:0 4px 12px rgba(39,174,96,0.3) !important;"
                    )
        
        # NAVIGATION
        with ui.row().style("display:flex; justify-content:center; width:100%; margin-top:32px;"):
            ui.button(" Retour à la PCA", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/pca'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important; transition:all 0.2s !important;"
            )
# ==========================================================
# FONCTIONS UTILITAIRES
# ==========================================================
def download_dataset(df):
    """Télécharge le dataset final en CSV"""
    try:
        csv_data = df.to_csv(index=False)
        ui.download(csv_data.encode(), "dataset_preprocessed.csv")
        ui.notify("✅ Dataset téléchargé avec succès", type="positive", position="top")
    except Exception as e:
        ui.notify(f"❌ Erreur lors du téléchargement : {str(e)}", type="negative", position="top")


def validate_and_continue(df):
    """Valide le prétraitement et passe au clustering"""
    try:
        state["final_dataset"] = df
        state["preprocessing_completed"] = True
        ui.notify("✅ Prétraitement validé ! Redirection vers le clustering...", type="positive", position="top", timeout=2000)
        ui.navigate.to("/unsupervised/clustering")
    except Exception as e:
        ui.notify(f"❌ Erreur : {str(e)}", type="negative", position="top")








####################################################################################################################################################################33


# ----------------- PAGE /unsupervised/clustering -----------------
# ----------------- PAGE /unsupervised/clustering -----------------


@ui.page('/unsupervised/clustering')
def algos_page():
    """Page de configuration et lancement des algorithmes de clustering"""
    
    # Récupérer le dataset final depuis le prétraitement
    pca_decision = state.get("pca_decision", {'apply': False})
    
    if pca_decision.get('apply'):
        X = state.get("pca_transformed_data")
    else:
        X = state.get("normalized_data")
    
    # Vérification que le dataset existe
    if X is None:
        with ui.column().style("width:100%; height:100vh; display:flex; align-items:center; justify-content:center;"):
            ui.label("❌ Dataset non disponible").style("font-size:20px; color:#e74c3c; font-weight:600; margin-bottom:12px;")
            ui.label("Veuillez compléter le prétraitement avant de lancer le clustering").style("font-size:14px; color:#7f8c8d; margin-bottom:20px;")
            ui.button(" Retour au résumé", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/summary'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )
        return
    
    # Convertir en numpy array si nécessaire
    if hasattr(X, 'values'):
        X = X.values
    X = np.array(X)
    
    # Stocker dans state pour les autres fonctions
    state['X'] = X

    # Calculer les K optimaux au chargement
    ui.notify("🔍 Calcul des K optimaux en cours...", color='info', timeout=2000)
    
    optimal_k_kmeans = 3
    optimal_k_kmedoids = 3
    optimal_k_agnes = 3
    
    try:
        _, optimal_k_kmeans = plot_elbow_curve(X, max_k=10, algo='kmeans')
        ui.notify(f"✅ K optimal KMeans: {optimal_k_kmeans}", color='positive', timeout=2000)
    except:
        pass
    
    try:
        _, optimal_k_kmedoids = plot_elbow_curve(X, max_k=10, algo='kmedoids')
        ui.notify(f"✅ K optimal KMedoids: {optimal_k_kmedoids}", color='positive', timeout=2000)
    except:
        pass
    
    optimal_k_agnes = optimal_k_kmeans

    with ui.column().style("width:100%; min-height:100vh; padding:40px; background:#f5f6fa; font-family:'Inter', sans-serif;"):
        
        # HEADER
        with ui.column().style("margin-bottom:48px; text-align:center;"):
            ui.label("🤖 Algorithmes de Clustering").style("font-weight:800; font-size:42px; background:linear-gradient(135deg, #01335A, #09538C); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:12px; letter-spacing:-1px;")
            ui.label("Configurez les paramètres et lancez l'analyse").style("font-size:16px; color:#7f8c8d;")
        
        # INFO BOX - DATASET
        with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:32px;"):
            with ui.card().style("width:100%; max-width:900px; padding:20px; background:white; display:flex; align-items:center; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                with ui.row().style("width:100%; align-items:center; gap:16px; justify-content:center;"):
                    ui.label("📊 Dataset :").style("font-weight:600; font-size:16px; color:#2c3e50;")
                    ui.label(f"{X.shape[0]} lignes × {X.shape[1]} variables").style("font-size:16px; color:#3498db; font-weight:700;")

        # ALGORITHMES
        with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:32px;"):
            with ui.row().style("width:100%; max-width:1200px; gap:20px; flex-wrap:wrap; justify-content:center;"):
                
                # -------- KMEANS --------
                with ui.card().style("width:320px; padding:24px; background:white; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                    ui.label("KMeans").style("font-size:20px; font-weight:700; color:#2c3e50; margin-bottom:8px;")
                    ui.label("Scikit-learn").style("font-size:12px; color:#7f8c8d; margin-bottom:16px;")
                    
                    with ui.card().style("padding:12px; background:#d5f4e6; border-left:3px solid #27ae60; border-radius:6px; margin-bottom:16px;"):
                        ui.label(f"✨ K optimal suggéré: {optimal_k_kmeans}").style("font-size:13px; color:#27ae60; font-weight:700;")
                    
                    k_kmeans = ui.number("Nombre de clusters", value=optimal_k_kmeans, min=2).props('outlined dense').style("width:100%; margin-bottom:12px;")
                    ui.label("💡 Vous pouvez modifier le K ou garder le K optimal").style("font-size:11px; color:#7f8c8d; margin-bottom:16px; font-style:italic;")
                    
                    kmeans_chk = ui.switch("Activer", value=True).style("font-weight:600;")

                # -------- KMEDOIDS --------
                with ui.card().style("width:320px; padding:24px; background:white; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                    ui.label("KMedoids").style("font-size:20px; font-weight:700; color:#2c3e50; margin-bottom:8px;")
                    ui.label("Scikit-learn-extra").style("font-size:12px; color:#7f8c8d; margin-bottom:16px;")
                    
                    with ui.card().style("padding:12px; background:#d5f4e6; border-left:3px solid #27ae60; border-radius:6px; margin-bottom:16px;"):
                        ui.label(f"✨ K optimal suggéré: {optimal_k_kmedoids}").style("font-size:13px; color:#27ae60; font-weight:700;")
                    
                    k_kmed = ui.number("Nombre de clusters", value=optimal_k_kmedoids, min=2).props('outlined dense').style("width:100%; margin-bottom:12px;")
                    ui.label("💡 Vous pouvez modifier le K ou garder le K optimal").style("font-size:11px; color:#7f8c8d; margin-bottom:16px; font-style:italic;")
                    
                    kmed_chk = ui.switch("Activer", value=True).style("font-weight:600;")

                # -------- DBSCAN --------
                diag = diagnose_dbscan(X)
                with ui.card().style("width:320px; padding:24px; background:white; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                    ui.label("DBSCAN").style("font-size:20px; font-weight:700; color:#2c3e50; margin-bottom:8px;")
                    ui.label("Scikit-learn").style("font-size:12px; color:#7f8c8d; margin-bottom:16px;")
                    
                    with ui.card().style("padding:12px; background:#d6eaf8; border-left:3px solid #3498db; border-radius:6px; margin-bottom:16px;"):
                        ui.label(f"💡 eps suggéré: {diag['suggested_eps']:.2f}").style("font-size:13px; color:#01335A; font-weight:700;")

                    eps_label = ui.label(f"Epsilon (eps): {max(0.5, diag['suggested_eps']):.2f}").style("font-size:13px; color:#2c3e50; margin-bottom:8px; font-weight:600;")
                    eps_val = ui.slider(
                        min=0.1, 
                        max=5, 
                        step=0.1,
                        value=max(0.5, diag['suggested_eps'])
                    ).props('label-always').style("margin-bottom:12px;")
                    eps_val.on_value_change(lambda e: eps_label.set_text(f"Epsilon (eps): {e.value:.2f}"))

                    min_samples = ui.number("min_samples", value=3, min=2).props('outlined dense').style("width:100%; margin-bottom:16px;")
                    dbscan_chk = ui.switch("Activer", value=True).style("font-weight:600;")
                
                # -------- AGNES --------
                with ui.card().style("width:320px; padding:24px; background:white; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                    ui.label("AgglomerativeClustering").style("font-size:20px; font-weight:700; color:#2c3e50; margin-bottom:8px;")
                    ui.label("Scikit-learn (AGNES)").style("font-size:12px; color:#7f8c8d; margin-bottom:16px;")
                    
                    with ui.card().style("padding:12px; background:#d5f4e6; border-left:3px solid #27ae60; border-radius:6px; margin-bottom:16px;"):
                        ui.label(f"✨ K optimal suggéré: {optimal_k_agnes}").style("font-size:13px; color:#27ae60; font-weight:700;")
                    
                    agnes_k = ui.number("Nombre de clusters", value=optimal_k_agnes, min=2).props('outlined dense').style("width:100%; margin-bottom:12px;")
                    ui.label("💡 Vous pouvez modifier le K ou garder le K optimal").style("font-size:11px; color:#7f8c8d; margin-bottom:12px; font-style:italic;")
                    
                    agnes_link = ui.select(
                        ['ward', 'complete', 'average', 'single'],
                        value='ward',
                        label="Linkage"
                    ).props('outlined dense').style("width:100%; margin-bottom:16px;")
                    
                    agnes_chk = ui.switch("Activer", value=True).style("font-weight:600;")

                # -------- DIANA --------
                with ui.card().style("width:320px; padding:24px; background:white; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                    ui.label("DIANA").style("font-size:20px; font-weight:700; color:#2c3e50; margin-bottom:8px;")
                    ui.label("Implémentation Custom").style("font-size:12px; color:#7f8c8d; margin-bottom:16px;")
                    
                    diana_k = ui.number("Nombre de clusters", value=3, min=2).props('outlined dense').style("width:100%; margin-bottom:16px;")
                    diana_chk = ui.switch("Activer", value=False).style("font-weight:600;")

        # INFO BOX - GUIDE
        with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:32px;"):
            with ui.card().style("width:100%; max-width:900px; padding:24px; background:white; display:flex; align-items:center; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                
                # En-tête de section
                with ui.column().style("margin-bottom:20px;"):
                    ui.label("💡 Guide de Configuration").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                    ui.separator().style("width:60px; height:3px; background:#f39c12; border-radius:2px; margin:0;")
                
                with ui.column().style("gap:8px; width:100%;"):
                    ui.label("• KMeans & KMedoids : Utilisez le K optimal suggéré ou ajustez selon vos besoins").style("color:#7f8c8d; font-size:14px;")
                    ui.label("• DBSCAN : Ajustez eps selon la densité de vos données (valeur suggérée fournie)").style("color:#7f8c8d; font-size:14px;")
                    ui.label("• AGNES : Choisissez la méthode de linkage (ward recommandé pour données normalisées)").style("color:#7f8c8d; font-size:14px;")
                    ui.label("• DIANA : Approche divisive, utile pour structures hiérarchiques descendantes").style("color:#7f8c8d; font-size:14px;")

        # NAVIGATION
        def run_all():
            # Initialiser le dictionnaire results dans state s'il n'existe pas
            if 'results' not in state:
                state['results'] = {}
            
            results = {}
            optimal_ks = {
                'kmeans': optimal_k_kmeans,
                'kmedoids': optimal_k_kmedoids,
                'agnes': optimal_k_agnes
            }

            try:
                state['X_pca'] = PCA(n_components=2).fit_transform(X)
            except:
                state['X_pca'] = None

            # -------- KMEANS --------
            if kmeans_chk.value:
                k_to_use = int(k_kmeans.value)
                
                ui.notify("Génération Elbow Method pour KMeans...", color='info')
                elbow_img, _ = plot_elbow_curve(X, max_k=10, algo='kmeans')
                state['results']['kmeans_elbow'] = elbow_img
                
                km = KMeans(n_clusters=k_to_use, random_state=0, n_init='auto')
                labels = km.fit_predict(X)
                results['kmeans'] = {
                    'labels': labels,
                    'silhouette': silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'davies_bouldin': davies_bouldin_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'calinski_harabasz': calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'n_clusters': len(np.unique(labels)),
                    'k_used': k_to_use,
                    'k_optimal': optimal_k_kmeans,
                    'used_optimal': (k_to_use == optimal_k_kmeans)
                }

            # -------- KMEDOIDS --------
            if kmed_chk.value:
                k_to_use = int(k_kmed.value)
                
                ui.notify("Génération Elbow Method pour KMedoids...", color='info')
                elbow_img, _ = plot_elbow_curve(X, max_k=10, algo='kmedoids')
                state['results']['kmedoids_elbow'] = elbow_img
                
                kmed = KMedoids(n_clusters=k_to_use, random_state=0, method='pam')
                labels = kmed.fit_predict(X)
                results['kmedoids'] = {
                    'labels': labels,
                    'silhouette': silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'davies_bouldin': davies_bouldin_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'calinski_harabasz': calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'n_clusters': len(np.unique(labels)),
                    'k_used': k_to_use,
                    'k_optimal': optimal_k_kmedoids,
                    'used_optimal': (k_to_use == optimal_k_kmedoids)
                }

            # -------- DBSCAN --------
            if dbscan_chk.value:
                dbs = DBSCAN(eps=float(eps_val.value), min_samples=int(min_samples.value))
                labels = dbs.fit(X).labels_
                valid_clusters = [l for l in np.unique(labels) if l != -1]
                results['dbscan'] = {
                    'labels': labels,
                    'n_clusters': len(valid_clusters),
                    'n_noise': np.sum(labels == -1),
                    'eps_used': float(eps_val.value),
                    'min_samples_used': int(min_samples.value)
                }

            # -------- AGNES --------
            if agnes_chk.value:
                k_to_use = int(agnes_k.value)
                linkage_used = agnes_link.value
                
                if linkage_used == 'ward':
                    ag = AgglomerativeClustering(n_clusters=k_to_use, linkage=linkage_used, metric='euclidean')
                else:
                    ag = AgglomerativeClustering(n_clusters=k_to_use, linkage=linkage_used)
                
                labels = ag.fit_predict(X)
                results['agnes'] = {
                    'labels': labels,
                    'silhouette': silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'davies_bouldin': davies_bouldin_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'calinski_harabasz': calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'n_clusters': len(np.unique(labels)),
                    'k_used': k_to_use,
                    'k_optimal': optimal_k_agnes,
                    'used_optimal': (k_to_use == optimal_k_agnes)
                }
            
            # -------- DIANA --------
            if diana_chk.value:
                k_to_use = int(diana_k.value)
                
                from scipy.spatial.distance import pdist
                from scipy.cluster.hierarchy import fcluster
                
                distances = pdist(X, metric='euclidean')
                Z = linkage(distances, method='complete')
                labels = fcluster(Z, k_to_use, criterion='maxclust') - 1
                
                results['diana'] = {
                    'labels': labels,
                    'silhouette': silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'davies_bouldin': davies_bouldin_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'calinski_harabasz': calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'n_clusters': len(np.unique(labels)),
                    'k_used': k_to_use
                }

            state['results'].update(results)
            state['optimal_k'] = optimal_ks
            
            ui.notify("✅ Clustering terminé", color='positive')
            ui.run_javascript("window.location.href='/unsupervised/results'")

        with ui.row().style("display:flex; justify-content:space-between; width:100%; max-width:900px; margin:0 auto; margin-top:32px;"):
            ui.button(" Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/summary'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important; transition:all 0.2s !important;"
            )
            ui.button("🚀 Lancer les algorithmes", on_click=run_all).style(
                "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:250px !important; transition:all 0.2s !important;"
            )


# ----------------- PAGE /unsupervised/results -----------------
@ui.page('/unsupervised/results')
def results_page():
    """Page d'affichage des résultats de clustering"""
    
    # Vérifier que les résultats existent
    if not state.get('results'):
        with ui.column().style("width:100%; height:100vh; display:flex; align-items:center; justify-content:center;"):
            ui.label("❌ Aucun résultat disponible").style("font-size:20px; color:#e74c3c; font-weight:600; margin-bottom:12px;")
            ui.label("Veuillez d'abord lancer le clustering").style("font-size:14px; color:#7f8c8d; margin-bottom:20px;")
            ui.button(" Retour au clustering", on_click=lambda: ui.navigate.to("/unsupervised/clustering")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )
        return

    results = state['results']
    X = state.get('X')
    X_pca = state.get('X_pca')

    # Calcul des métriques
    metrics_dict = {}
    for algo, res in results.items():
        if algo.endswith('_elbow'):
            continue
            
        labels = res.get('labels')
        m = {}

        if labels is not None and X is not None:
            mask = labels != -1
            m["n_clusters"] = len(set(labels[mask])) if mask.sum() > 0 else 0
            m["n_noise"] = list(labels).count(-1) if -1 in labels else 0

            if m["n_clusters"] > 1:
                X_masked = X[mask]
                labels_masked = labels[mask]
                
                m["silhouette"] = round(float(silhouette_score(X_masked, labels_masked)), 3)
                m["davies_bouldin"] = round(float(davies_bouldin_score(X_masked, labels_masked)), 3)
                m["calinski_harabasz"] = round(float(calinski_harabasz_score(X_masked, labels_masked)), 2)
            else:
                m["silhouette"] = "N/A"
                m["davies_bouldin"] = "N/A"
                m["calinski_harabasz"] = "N/A"

        metrics_dict[algo] = m
        res.update(m)

    # Déterminer le meilleur algorithme
    best_algo = None
    best_score = -1
    for algo, m in metrics_dict.items():
        if m.get("silhouette") != "N/A" and m.get("silhouette") is not None:
            if m["silhouette"] > best_score:
                best_score = m["silhouette"]
                best_algo = algo

    with ui.column().style("width:100%; min-height:100vh; padding:40px; background:#f5f6fa; font-family:'Inter', sans-serif;"):
        
        # HEADER
        with ui.column().style("margin-bottom:48px; text-align:center;"):
            ui.label("📊 Résultats du Clustering").style("font-weight:800; font-size:42px; background:linear-gradient(135deg, #01335A, #09538C); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:12px; letter-spacing:-1px;")
            ui.label("Analysez et comparez les performances des algorithmes").style("font-size:16px; color:#7f8c8d;")
            if best_algo:
                with ui.card().style("display:inline-block; padding:12px 24px; background:linear-gradient(135deg, #fbbf24, #f59e0b); border-radius:8px; margin-top:16px; box-shadow:0 4px 12px rgba(251,191,36,0.3);"):
                    ui.label(f"🏆 Meilleur algorithme: {best_algo.upper()} (Silhouette: {best_score:.3f})").style("font-size:16px; color:white; font-weight:700;")

        # TABLEAU COMPARATIF
        with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:32px;"):
            with ui.card().style("width:100%; max-width:900px; padding:24px; background:white; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                
                with ui.column().style("margin-bottom:20px;"):
                    ui.label("📈 Tableau Comparatif des Métriques").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                    ui.separator().style("width:60px; height:3px; background:#3498db; border-radius:2px; margin:0;")
                
                rows = [{**{'algo': algo.upper()}, **m} for algo, m in metrics_dict.items()]
                ui.table(
                    rows=rows,
                    columns=[
                        {"name": "algo", "label": "Algorithme", "field": "algo", "align": "left"},
                        {"name": "n_clusters", "label": "Clusters", "field": "n_clusters", "align": "center"},
                        {"name": "n_noise", "label": "Bruit", "field": "n_noise", "align": "center"},
                        {"name": "silhouette", "label": "Silhouette ↑", "field": "silhouette", "align": "center"},
                        {"name": "davies_bouldin", "label": "Davies-Bouldin ↓", "field": "davies_bouldin", "align": "center"},
                        {"name": "calinski_harabasz", "label": "Calinski-Harabasz ↑", "field": "calinski_harabasz", "align": "center"},
                    ],
                ).style("width:100%; font-size: 14px;").props("flat bordered")

        # HISTOGRAMME COMPARATIF
        with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:32px;"):
            with ui.card().style("width:100%; max-width:900px; padding:24px; background:white; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                
                with ui.column().style("margin-bottom:20px;"):
                    ui.label("📊 Comparaison Visuelle des Métriques").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                    ui.separator().style("width:60px; height:3px; background:#9b59b6; border-radius:2px; margin:0;")
                
                hist_img = plot_grouped_histogram(metrics_dict, "Comparaison des métriques")
                ui.image(hist_img).style("width:100%; max-width:800px; height:auto; display:block; margin:0 auto; border-radius:8px;")

        # DÉTAIL PAR ALGORITHME
        for idx, (algo, res) in enumerate(results.items()):
            if algo.endswith('_elbow'):
                continue

            colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981']
            color = colors[idx % len(colors)]
            is_best = (algo == best_algo)
            
            with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:32px;"):
                with ui.card().style("width:100%; max-width:900px; padding:24px; background:white; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                    
                    # EN-TÊTE DE L'ALGORITHME
                    with ui.element('div').style(f"width:100%; padding:20px; background:linear-gradient(135deg, {color}22 0%, {color}11 100%); border-left:4px solid {color}; border-radius:8px; margin-bottom:20px;"):
                        with ui.row().style("align-items:center; gap:12px; flex-wrap:wrap;"):
                            ui.label(f"{algo.upper()}").style(f"font-size:28px; font-weight:700; color:{color};")
                            if is_best:
                                with ui.card().style("padding:8px 16px; background:linear-gradient(135deg, #fbbf24, #f59e0b); border-radius:6px;"):
                                    ui.label("🏆 MEILLEUR").style("font-size:14px; font-weight:700; color:white;")
                        
                        # Afficher K utilisé vs K optimal
                        k_used = res.get('k_used')
                        k_optimal = res.get('k_optimal')
                        
                        if k_used is not None and k_optimal is not None:
                            if res.get('used_optimal'):
                                with ui.card().style("padding:10px 16px; background:#d5f4e6; border-left:3px solid #27ae60; border-radius:6px; margin-top:12px; display:inline-block;"):
                                    ui.label(f"✅ K utilisé: {k_used} (K optimal)").style("font-size:14px; color:#27ae60; font-weight:600;")
                            else:
                                with ui.card().style("padding:10px 16px; background:#fdebd0; border-left:3px solid #f39c12; border-radius:6px; margin-top:12px; display:inline-block;"):
                                    ui.label(f"⚙️ K utilisé: {k_used} | K optimal suggéré: {k_optimal}").style("font-size:14px; color:#e67e22; font-weight:600;")
                        elif algo == 'dbscan':
                            eps_used = res.get('eps_used')
                            min_samples_used = res.get('min_samples_used')
                            with ui.card().style("padding:10px 16px; background:#d6eaf8; border-left:3px solid #3498db; border-radius:6px; margin-top:12px; display:inline-block;"):
                                ui.label(f"⚙️ eps: {eps_used:.2f} | min_samples: {min_samples_used}").style("font-size:14px; color:#01335A; font-weight:600;")

                    m = metrics_dict[algo]

                    # RÉSUMÉ DES MÉTRIQUES
                    ui.label("📌 Résumé des Métriques").style("font-weight:600; font-size:18px; color:#2c3e50; margin-bottom:16px;")
                    
                    with ui.row().style("gap:12px; width:100%; flex-wrap:wrap; margin-bottom:24px;"):
                        with ui.card().style(f"padding:16px; background:{color}11; border-top:3px solid {color}; border-radius:8px; flex:1; min-width:140px; text-align:center;"):
                            ui.label("Clusters").style("font-size:12px; color:#7f8c8d; font-weight:600; margin-bottom:8px;")
                            ui.label(str(m["n_clusters"])).style(f"font-size:32px; font-weight:700; color:{color};")
                        
                        with ui.card().style("padding:16px; background:#fdebd022; border-top:3px solid #f39c12; border-radius:8px; flex:1; min-width:140px; text-align:center;"):
                            ui.label("Points de bruit").style("font-size:12px; color:#7f8c8d; font-weight:600; margin-bottom:8px;")
                            ui.label(str(m["n_noise"])).style("font-size:32px; font-weight:700; color:#f39c12;")
                        
                        with ui.card().style("padding:16px; background:#d5f4e622; border-top:3px solid #27ae60; border-radius:8px; flex:1; min-width:140px; text-align:center;"):
                            ui.label("Silhouette ↑").style("font-size:12px; color:#7f8c8d; font-weight:600; margin-bottom:8px;")
                            ui.label(str(m["silhouette"])).style("font-size:24px; font-weight:700; color:#27ae60;")
                        
                        with ui.card().style("padding:16px; background:#d6eaf822; border-top:3px solid #3498db; border-radius:8px; flex:1; min-width:140px; text-align:center;"):
                            ui.label("Davies-Bouldin ↓").style("font-size:12px; color:#7f8c8d; font-weight:600; margin-bottom:8px;")
                            ui.label(str(m["davies_bouldin"])).style("font-size:24px; font-weight:700; color:#3498db;")
                        
                        with ui.card().style("padding:16px; background:#e9d5ff22; border-top:3px solid #9b59b6; border-radius:8px; flex:1; min-width:140px; text-align:center;"):
                            ui.label("Calinski-Harabasz ↑").style("font-size:12px; color:#7f8c8d; font-weight:600; margin-bottom:8px;")
                            ui.label(str(m["calinski_harabasz"])).style("font-size:24px; font-weight:700; color:#9b59b6;")

                    # VISUALISATIONS
                    
                    # Graphique Elbow si disponible
                    elbow_key = f'{algo}_elbow'
                    has_elbow = elbow_key in state['results']
                    
                    if has_elbow:
                        ui.separator().style("margin:24px 0;")
                        ui.label("📉 Méthode du Coude (Elbow Method)").style("font-weight:600; font-size:18px; color:#2c3e50; margin-bottom:12px;")
                        
                        k_optimal_val = res.get('k_optimal', 'N/A')
                        k_used_val = res.get('k_used', 'N/A')
                        
                        if k_used_val == k_optimal_val:
                            with ui.card().style("padding:10px 16px; background:#d5f4e6; border-left:3px solid #27ae60; border-radius:6px; margin-bottom:16px; display:inline-block;"):
                                ui.label(f"✅ K utilisé: {k_used_val} (identique au K optimal)").style("font-size:14px; color:#27ae60; font-weight:600;")
                        else:
                            with ui.card().style("padding:10px 16px; background:#fdebd0; border-left:3px solid #f39c12; border-radius:6px; margin-bottom:16px; display:inline-block;"):
                                ui.label(f"📊 K optimal trouvé: {k_optimal_val} | K utilisé: {k_used_val}").style("font-size:14px; color:#e67e22; font-weight:600;")
                        
                        elbow_img = state['results'][elbow_key]
                        ui.image(elbow_img).style("width:100%; max-width:700px; height:auto; display:block; margin:0 auto; border-radius:8px;")
                    
                    # PCA et Dendrogramme
                    ui.separator().style("margin:24px 0;")
                    ui.label("🎨 Visualisations").style("font-weight:600; font-size:18px; color:#2c3e50; margin-bottom:16px;")
                    
                    num_cols = 1
                    if algo.lower() == "agnes":
                        num_cols = 2
                        
                    with ui.row().style("gap:20px; width:100%; flex-wrap:wrap;"):
                        # PCA
                        with ui.card().style("padding:20px; background:#fafbfc; border-radius:8px; flex:1; min-width:300px;"):
                            ui.label("🎨 Visualisation PCA").style("font-weight:600; font-size:16px; color:#2c3e50; margin-bottom:12px;")
                            if X_pca is None:
                                ui.label("PCA non disponible").style("font-size:14px; color:#7f8c8d; text-align:center; padding:40px 0;")
                            else:
                                img64 = scatter_plot_2d(X_pca, res['labels'], f"{algo.upper()} - PCA")
                                ui.image(img64).style("width:100%; max-width:500px; height:auto; display:block; margin:0 auto; border-radius:8px;")

                        # Dendrogramme (seulement pour AGNES)
                        if algo.lower() == "agnes":
                            with ui.card().style("padding:20px; background:#fafbfc; border-radius:8px; flex:1; min-width:300px;"):
                                ui.label("🌳 Dendrogramme").style("font-weight:600; font-size:16px; color:#2c3e50; margin-bottom:12px;")
                                try:
                                    dendro64 = generate_dendrogram(X, algo)
                                    if dendro64:
                                        ui.image(dendro64).style("width:100%; max-width:500px; height:auto; display:block; margin:0 auto; border-radius:8px;")
                                    else:
                                        ui.label("⚠️ Impossible de générer le dendrogramme").style("font-size:14px; color:#f39c12; text-align:center; padding:40px 0;")
                                except Exception as e:
                                    ui.label(f"❌ Erreur : {e}").style("font-size:14px; color:#e74c3c; text-align:center; padding:40px 0;")
        
        # BOUTONS DE NAVIGATION
        with ui.row().style("display:flex; justify-content:center; width:100%; margin-top:32px;"):
            ui.button(" Retour au Clustering", on_click=lambda: ui.navigate.to('/unsupervised/clustering')).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:220px !important;"
            )
# ----------------- FONCTIONS UTILITAIRES -----------------

def fig_to_base64(fig):
    """Convertit une figure matplotlib en base64"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def plot_grouped_histogram(metrics_dict, title):
    """Crée un histogramme groupé comparant les métriques de différents algorithmes"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Préparer les données
    algos = list(metrics_dict.keys())
    metrics_names = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
    
    # Extraire les valeurs
    data = {metric: [] for metric in metrics_names}
    for algo in algos:
        for metric in metrics_names:
            val = metrics_dict[algo].get(metric, 0)
            # Convertir en float si possible, sinon mettre 0
            if val == "N/A" or val is None:
                data[metric].append(0)
            else:
                data[metric].append(float(val))
    
    # Normaliser Davies-Bouldin (inverser car plus bas = mieux)
    if data['davies_bouldin']:
        max_db = max([v for v in data['davies_bouldin'] if v > 0], default=1)
        data['davies_bouldin'] = [max_db - v if v > 0 else 0 for v in data['davies_bouldin']]
    
    # Normaliser Calinski-Harabasz (diviser par 1000 pour une meilleure échelle)
    data['calinski_harabasz'] = [v / 1000 if v > 0 else 0 for v in data['calinski_harabasz']]
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(algos))
    width = 0.25
    
    colors = ['#3b82f6', '#10b981', '#8b5cf6']
    labels_display = ['Silhouette', 'Davies-Bouldin (inv)', 'Calinski-Harabasz (÷1000)']
    
    for i, (metric, color, label) in enumerate(zip(metrics_names, colors, labels_display)):
        offset = (i - 1) * width
        ax.bar(x + offset, data[metric], width, label=label, color=color, alpha=0.8)
    
    ax.set_xlabel('Algorithmes', fontweight='bold', fontsize=12)
    ax.set_ylabel('Valeurs normalisées', fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in algos], rotation=15, ha='right')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    
    img_base64 = fig_to_base64(fig)
    return f"data:image/png;base64,{img_base64}"

def scatter_plot_2d(X_pca, labels, title):
    """Génère un scatter plot 2D pour la visualisation PCA"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = 'black'
            marker = 'x'
            label_name = 'Bruit'
        else:
            marker = 'o'
            label_name = f'Cluster {label}'
        
        mask = labels == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                  c=[color], marker=marker, label=label_name, s=80, alpha=0.7, 
                  edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('Composante Principale 1', fontweight='bold', fontsize=12)
    ax.set_ylabel('Composante Principale 2', fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14, pad=15)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    img_base64 = fig_to_base64(fig)
    return f"data:image/png;base64,{img_base64}"

def diagnose_dbscan(X):
    """Diagnostic pour suggérer les paramètres DBSCAN"""
    from sklearn.neighbors import NearestNeighbors
    
    k = min(4, X.shape[0] - 1)
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = np.sort(distances[:, -1])
    
    suggested_eps = np.percentile(distances, 90)
    
    return {
        'suggested_eps': suggested_eps,
        'distances': distances
    }

def find_elbow_point(K_range, inertias):
    """
    Trouve le point du coude en utilisant la méthode de la distance maximale
    à la ligne reliant le premier et le dernier point
    """
    # Convertir en numpy arrays
    K_range = np.array(list(K_range))
    inertias = np.array(inertias)
    
    # Normaliser les valeurs entre 0 et 1 pour un meilleur calcul
    K_norm = (K_range - K_range.min()) / (K_range.max() - K_range.min())
    inertias_norm = (inertias - inertias.min()) / (inertias.max() - inertias.min())
    
    # Points de départ et de fin
    p1 = np.array([K_norm[0], inertias_norm[0]])
    p2 = np.array([K_norm[-1], inertias_norm[-1]])
    
    # Calculer la distance de chaque point à la ligne p1-p2
    distances = []
    for i in range(len(K_norm)):
        p = np.array([K_norm[i], inertias_norm[i]])
        # Distance d'un point à une ligne
        d = np.abs(np.cross(p2-p1, p1-p)) / np.linalg.norm(p2-p1)
        distances.append(d)
    
    # Le coude est le point avec la distance maximale
    elbow_idx = np.argmax(distances)
    return K_range[elbow_idx]

def plot_elbow_curve(X, max_k=10, algo='kmeans'):
    """Génère la courbe du coude et retourne l'image + K optimal"""
    from sklearn.metrics import silhouette_score
    
    K_range = range(2, min(max_k + 1, X.shape[0]))
    inertias = []
    silhouettes = []
    
    for k in K_range:
        if algo == 'kmeans':
            model = KMeans(n_clusters=k, random_state=0, n_init='auto')
        else:  # kmedoids
            model = KMedoids(n_clusters=k, random_state=0, method='pam')
        
        labels = model.fit_predict(X)
        
        if hasattr(model, 'inertia_'):
            inertias.append(model.inertia_)
        else:
            # Pour KMedoids, calculer l'inertie manuellement
            centers = X[model.medoid_indices_]
            inertia = sum(np.min([np.sum((X[i] - c)**2) for c in centers]) for i in range(len(X)))
            inertias.append(inertia)
        
        silhouettes.append(silhouette_score(X, labels))
    
    # Trouver le K optimal avec la méthode du coude (distance maximale)
    optimal_k = find_elbow_point(K_range, inertias)
    
    # Créer le graphique
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Graphique 1: Inertie avec le coude marqué
    ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(optimal_k, color='red', linestyle='--', linewidth=2, label=f'Coude détecté: K={optimal_k}')
    ax1.scatter([optimal_k], [inertias[list(K_range).index(optimal_k)]], 
                color='red', s=200, zorder=5, marker='*', edgecolors='darkred', linewidth=2)
    ax1.set_xlabel('Nombre de clusters (K)', fontweight='bold')
    ax1.set_ylabel('Inertie', fontweight='bold')
    ax1.set_title('Méthode du Coude - Inertie', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Graphique 2: Score Silhouette
    ax2.plot(K_range, silhouettes, 'go-', linewidth=2, markersize=8)
    ax2.axvline(optimal_k, color='red', linestyle='--', linewidth=2, label=f'K du coude = {optimal_k}')
    ax2.scatter([optimal_k], [silhouettes[list(K_range).index(optimal_k)]], 
                color='red', s=200, zorder=5, marker='*', edgecolors='darkred', linewidth=2)
    ax2.set_xlabel('Nombre de clusters (K)', fontweight='bold')
    ax2.set_ylabel('Score Silhouette', fontweight='bold')
    ax2.set_title('Score Silhouette par K', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    img_base64 = fig_to_base64(fig)
    return f"data:image/png;base64,{img_base64}", optimal_k

def generate_dendrogram(X, algo):
    """Génère un dendrogramme pour les algorithmes hiérarchiques"""
    try:
        # Limiter le nombre de points pour la lisibilité
        if X.shape[0] > 100:
            indices = np.random.choice(X.shape[0], 100, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Choisir la méthode de liaison
        method = 'ward' if algo.lower() == 'agnes' else 'average'
        Z = linkage(X_sample, method=method)
        
        # Créer le dendrogramme
        fig, ax = plt.subplots(figsize=(12, 6))
        dendrogram(Z, ax=ax, leaf_font_size=10)
        ax.set_title(f'Dendrogramme - {algo.upper()}', fontweight='bold', fontsize=14, pad=15)
        ax.set_xlabel('Index des échantillons', fontweight='bold', fontsize=12)
        ax.set_ylabel('Distance', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        
        img_base64 = fig_to_base64(fig)
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"Erreur lors de la génération du dendrogramme: {e}")
        return None








# LANCEMENT
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Clustering Data Mining", port=8080, reload=True)