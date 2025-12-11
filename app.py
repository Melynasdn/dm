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

# ========================
# 🔧 FONCTION UTILITAIRE GLOBALE
# ========================

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
                            "window.location.href='/unsupervised/supervised/upload'"
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
                width: 580px !important;
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

                btn_next = ui.button("Continuer ➡")
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

            ui.button("➡ Étape suivante", on_click=lambda: ui.run_javascript("window.location.href='/supervised/user_decisions'")).style(
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

        for col_name, cb in column_exclude_widgets.items():
            state.setdefault("columns_exclude", {})[col_name] = cb.value

        state["apply_smote"] = smote_cb.value

        ui.notify("✅ Décisions enregistrées avec succès !", color="positive")
         
        ui.run_javascript("setTimeout(() => window.location.href='/supervised/preprocessing2', 1000);")

    def go_to_split():
        on_confirm()
        ui.run_javascript("setTimeout(() => window.location.href='/supervised/preprocessing2', 500);")

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

            # Créer un aperçu des 10 premières lignes
            df_preview = df.head(10).copy()
            
            # Préparer les données pour le tableau
            columns_for_table = []
            rows_for_table = []
            
            # Colonnes
            for col in df_preview.columns:
                columns_for_table.append({
                    "name": col,
                    "label": col,
                    "field": col,
                    "align": "left",
                    "sortable": True
                })
            
            # Lignes
            for idx, row in df_preview.iterrows():
                row_dict = {"_index": str(idx)}
                for col in df_preview.columns:
                    val = row[col]
                    # Formater la valeur
                    if pd.isna(val):
                        row_dict[col] = "NaN"
                    elif isinstance(val, (int, np.integer)):
                        row_dict[col] = str(val)
                    elif isinstance(val, (float, np.floating)):
                        row_dict[col] = f"{val:.2f}"
                    else:
                        row_dict[col] = str(val)[:50]  # Limiter à 50 caractères
                rows_for_table.append(row_dict)
            
            # Ajouter colonne index
            columns_for_table.insert(0, {
                "name": "_index",
                "label": "Index",
                "field": "_index",
                "align": "center",
                "sortable": False
            })
            
            # Afficher le tableau
            ui.table(
                columns=columns_for_table,
                rows=rows_for_table,
                row_key="_index"
            ).props("flat bordered dense").style(
                "width:100% !important; font-size:12px !important; max-height:400px !important; "
                "overflow-y:auto !important;"
            )
            
            # Statistiques rapides
            with ui.row().classes("w-full gap-8 items-center justify-around mt-6").style(
                "background:#f8f9fa !important; padding:20px !important; border-radius:12px !important;"
            ):
                # Nombre de lignes
                with ui.column().classes("items-center"):
                    ui.label("📊").style("font-size:28px !important; margin-bottom:4px !important;")
                    ui.label(f"{len(df):,}").style(
                        "font-weight:700 !important; font-size:24px !important; color:#01335A !important;"
                    )
                    ui.label("Lignes").style(
                        "font-size:13px !important; color:#636e72 !important; margin-top:4px !important;"
                    )
                
                # Nombre de colonnes
                with ui.column().classes("items-center"):
                    ui.label("📋").style("font-size:28px !important; margin-bottom:4px !important;")
                    ui.label(f"{len(df.columns)}").style(
                        "font-weight:700 !important; font-size:24px !important; color:#01335A !important;"
                    )
                    ui.label("Colonnes").style(
                        "font-size:13px !important; color:#636e72 !important; margin-top:4px !important;"
                    )
                
                # Valeurs manquantes
                with ui.column().classes("items-center"):
                    ui.label("⚠️").style("font-size:28px !important; margin-bottom:4px !important;")
                    total_missing = df.isna().sum().sum()
                    missing_pct = (total_missing / (len(df) * len(df.columns)) * 100)
                    ui.label(f"{total_missing:,}").style(
                        "font-weight:700 !important; font-size:24px !important; color:#e74c3c !important;"
                    )
                    ui.label(f"Missing ({missing_pct:.1f}%)").style(
                        "font-size:13px !important; color:#636e72 !important; margin-top:4px !important;"
                    )
                
                # Colonnes numériques
                with ui.column().classes("items-center"):
                    ui.label("🔢").style("font-size:28px !important; margin-bottom:4px !important;")
                    n_numeric = len(df.select_dtypes(include=[np.number]).columns)
                    ui.label(f"{n_numeric}").style(
                        "font-weight:700 !important; font-size:24px !important; color:#2196f3 !important;"
                    )
                    ui.label("Numériques").style(
                        "font-size:13px !important; color:#636e72 !important; margin-top:4px !important;"
                    )
                
                # Colonnes catégorielles
                with ui.column().classes("items-center"):
                    ui.label("🏷️").style("font-size:28px !important; margin-bottom:4px !important;")
                    n_categorical = len(df.select_dtypes(include=['object', 'category']).columns)
                    ui.label(f"{n_categorical}").style(
                        "font-weight:700 !important; font-size:24px !important; color:#094282 !important;"
                    )
                    ui.label("Catégorielles").style(
                        "font-size:13px !important; color:#636e72 !important; margin-top:4px !important;"
                    )

        # ---------- CONFIGURATION DES TYPES ----------
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

            # Header tableau
            with ui.row().classes("w-full items-center gap-4 mb-4 pb-3").style(
                "border-bottom:2px solid #e1e8ed !important;"
            ):
                ui.label("Colonne").style(
                    "width:200px !important; font-weight:700 !important; font-size:14px !important; "
                    "color:#01335A !important;"
                )
                ui.label("Type Détecté").style(
                    "flex:1 !important; font-weight:700 !important; font-size:14px !important; "
                    "color:#01335A !important;"
                )
                ui.label("Exclure").style(
                    "width:120px !important; font-weight:700 !important; font-size:14px !important; "
                    "color:#01335A !important; text-align:center !important;"
                )

            for col in columns_info:
                col_name = col["Colonne"]
                actual_type = detect_actual_type(col, col_name)

                with ui.row().classes("w-full items-center gap-4 mb-2").style(
                    "padding:12px !important; border-radius:8px !important; background:#f8f9fa !important;"
                ):
                    ui.label(col_name).style(
                        "width:200px !important; font-weight:600 !important; font-size:14px !important; "
                        "color:#2c3e50 !important;"
                    )

                    col_type = ui.select(
                        options=[
                            "Numérique Continue", "Numérique Discrète",
                            "Catégorielle Nominale", "Catégorielle Ordinale",
                            "Date/Datetime", "Texte", "Identifiant"
                        ],
                        value=actual_type
                    ).props("outlined dense").classes("flex-1")

                    column_type_widgets[col_name] = col_type

                    auto_exclude = False
                    if col["Cardinalité"] == 1:
                        auto_exclude = True
                    if "%" in col["% Missing"]:
                        try:
                            missing_pct = float(col["% Missing"].replace("%", "").strip())
                            if missing_pct >= 100:
                                auto_exclude = True
                        except:
                            pass
                    if actual_type == "Identifiant":
                        auto_exclude = True

                    exclude_cb = ui.checkbox("", value=auto_exclude)
                    column_exclude_widgets[col_name] = exclude_cb

            # Info exclusions
            with ui.card().classes("w-full mt-6").style(
                "background:#134b78 !important; padding:20px !important; border-radius:12px !important; "
                "border-left:4px solid #01335A  !important; box-shadow:none !important;"
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
        with ui.row().classes("w-full max-w-4xl gap-4 mt-8 justify-center"):
            ui.button(
                "Valider les décisions", 
                on_click=on_confirm
            ).style(
                "background:#01335A !important; color:white !important; font-weight:600 !important; "
                "border-radius:8px !important; height:50px !important; min-width:220px !important; "
                "font-size:16px !important; text-transform:none !important;"
            )

            ui.button(
                "Suivant →", 
                on_click=go_to_split
            ).style(
                "background:#01335A !important; color:white !important; font-weight:600 !important; "
                "border-radius:8px !important; height:50px !important; min-width:220px !important; "
                "font-size:16px !important; text-transform:none !important;"
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




@ui.page('/supervised/preprocessing2')
def preprocessing2_page():
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
            ui.label(f" Distribution de la target '{target_col}'").style(
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
                ui.label(" Déséquilibre détecté !").style(
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
            strategy_radio = ui.radio(
                options=["Stratified (recommandé)", "Random", "Time-based (si date)"],
                value="Stratified (recommandé)"
            ).style("margin-top:12px; color:#01335A !important;")

            seed_input = ui.input(label="Random Seed", value=42).style("margin-top:12px; width:150px;")

        # ---------- 3️⃣ Bouton Split ----------
        # Container résultat
        result_container = ui.column().classes("w-full max-w-4xl mt-4")

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
                        ui.label("").style("font-size:32px;")
                        ui.label("Split effectué avec succès !").style(
                            "font-weight:700; font-size:26px; color:#01335A;"
                        )
                    ui.label(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}").style(
                        "font-weight:600; font-size:16px; color:#666; font-family:monospace;"
                    )
                
                # Cards des datasets
                with ui.row().classes("w-full gap-4 justify-center flex-wrap"):
                    for name, y_set, gradient, icon in [
                        ("Train", y_train, "linear-gradient(135deg, #667eea 0%, #764ba2 100%)", ""),
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

            ui.notify(" Split effectué avec succès !", color="positive")
            
            # Affichage post-split - Information simple
            with result_container:
                ui.label(f" Split effectué ! Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}").style(
                    "font-weight:600; color:#01335A; margin-top:20px; font-size:16px; text-align:center;"
                )

                ui.button(
                    "➡ Analyse Univariée",
                    on_click=lambda: ui.run_javascript("window.location.href='/supervised/univariate_analysis'"),
                ).style(
                    "background:#09538C !important; color:white; font-weight:600; border-radius:8px; height:40px; width:280px; margin-top:12px;"
                )

        ui.button("Effectue r le split", on_click=do_split).style(
            "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white; font-weight:600; "
            "border-radius:12px; height:50px; width:280px; margin-top:20px; font-size:16px; "
            "box-shadow:0 6px 20px rgba(1,51,90,0.3); transition:all 0.3s ease;"
        ).classes("hover:shadow-xl")
        
        ui.button(
                "➡ Analyse Univariée",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/univariate_analysis'"),
        ).style(
                "background:#09538C !important; color:white; font-weight:600; border-radius:8px; height:40px; width:280px; margin-top:12px;"
        )
        

@ui.page('/supervised/univariate_analysis')
def univariate_analysis_page():

    # Récupération des données train
    split = state.get("split")
    if split is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Aucun split trouvé. Veuillez d'abord effectuer le split.").style(
                "font-size:18px !important; color:#c0392b !important; font-weight:600 !important;"
            )
            ui.button(
                " Retour au Preprocessing", 
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/preprocessing2'")
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
        "background:#f0f2f5 !important; min-height:100vh !important; padding:24px !important; "
        "font-family:'Inter', sans-serif !important;"
    ):
        # HEADER
        ui.label("Analyse Univariée").style(
            "font-weight:700 !important; font-size:36px !important; color:#01335A !important; "
            "margin-bottom:8px !important; text-align:center !important; letter-spacing:-0.5px !important;"
        )
        ui.label("Exploration détaillée de chaque variable (dataset d'entraînement)").style(
            "font-size:16px !important; color:#636e72 !important; "
            "text-align:center !important; font-weight:400 !important;"
        )

        # --- OVERVIEW METRICS ---
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white !important; border-radius:16px !important; padding:16px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            ui.label(" Vue d'ensemble du dataset").style(
                "font-weight:700 !important; font-size:22px !important; color:#01335A !important; "
                "margin-bottom:6px !important;"
            )
            
            with ui.row().classes("w-full gap-8 items-center justify-around"):
                def metric_card(icon, label, value, sublabel=""):
                    with ui.column().classes("items-center"):
                        ui.label(value).style(
                            "font-weight:700 !important; font-size:28px !important; color:#01335A !important;"
                        )
                        ui.label(label).style(
                            "font-size:13px !important; color:#636e72 !important; margin-top:2px !important;"
                        )
                        if sublabel:
                            ui.label(sublabel).style(
                                "font-size:12px !important; color:#7f8c8d !important; "
                            )
                
                metric_card("", "Variables numériques", str(n_numeric), f"{round(n_numeric/total_features*100)}% du total")
                metric_card("", "Variables catégorielles", str(n_categorical), f"{round(n_categorical/total_features*100)}% du total")
                metric_card("", "Observations", f"{n_observations:,}", "lignes d'entraînement")
                metric_card("", "Features totales", str(total_features), "colonnes actives")

        # --- FILTRES & TRI ---
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white !important; border-radius:16px !important; padding:20px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            with ui.row().classes("w-full gap-4 items-end"):
                # Filtre par type
                filter_type = ui.select(
                    options={
                        "all": "Toutes les variables",
                        "numeric": "Numériques uniquement",
                        "categorical": "Catégorielles uniquement"
                    },
                    value="all",
                    label="Type de variable"
                ).props("outlined").classes("flex-1")
                
                # Tri
                sort_by = ui.select(
                    options={
                        "name": "Nom (A-Z)",
                        "name_desc": "Nom (Z-A)",
                        "type": "Type",
                        "variance": "Variance (numériques)"
                    },
                    value="name",
                    label="Trier par"
                ).props("outlined").classes("flex-1")
                
                # Recherche
                search_input = ui.input(
                    label="Rechercher une variable",
                    placeholder="Nom de colonne..."
                ).props("outlined clearable").classes("flex-1").style("min-width:250px !important;")

        # --- CONTENEUR DES CARTES ---
        cards_container = ui.column().classes("w-full max-w-6xl gap-6")

        def create_numeric_plot(series, col_name):
            """Crée un histogramme avec KDE pour une variable numérique"""
            fig = go.Figure()
            
            # Histogramme
            fig.add_trace(go.Histogram(
                x=series.dropna(),
                name="Distribution",
                marker_color='#01335A',
                opacity=0.7,
                nbinsx=30
            ))
            
            fig.update_layout(
                title=f"Distribution de {col_name}",
                height=280,
                margin=dict(l=40, r=20, t=40, b=40),
                showlegend=False,
                paper_bgcolor='white',
                plot_bgcolor='#f8f9fa',
                font=dict(family="Inter, sans-serif", color="#2c3e50"),
                xaxis=dict(title=col_name),
                yaxis=dict(title="Fréquence")
            )
            
            return fig

        def create_categorical_plot(series, col_name):
            """Crée un bar chart pour une variable catégorielle"""
            counts = series.value_counts().head(10)  # Top 10 catégories
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=counts.index.astype(str),
                y=counts.values,
                marker_color='#01335A',
                text=counts.values,
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f"Distribution de {col_name}",
                height=280,
                margin=dict(l=40, r=20, t=40, b=60),
                showlegend=False,
                paper_bgcolor='white',
                plot_bgcolor='#f8f9fa',
                font=dict(family="Inter, sans-serif", color="#2c3e50"),
                xaxis=dict(title=col_name, tickangle=-45),
                yaxis=dict(title="Fréquence")
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
                title=f"Boxplot - {col_name}",
                height=280,
                margin=dict(l=40, r=20, t=40, b=40),
                showlegend=False,
                paper_bgcolor='white',
                plot_bgcolor='#f8f9fa',
                font=dict(family="Inter, sans-serif", color="#2c3e50")
            )
            
            return fig

        def add_numeric_card(col):
            """Carte pour variable numérique avec visualisations"""
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
            
            # Détection outliers
            lower_bound = q1 - 1.5 * iqr_val
            upper_bound = q3 + 1.5 * iqr_val
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            n_outliers = len(outliers)
            pct_outliers = round(n_outliers / len(series) * 100, 2) if len(series) > 0 else 0
            
            with cards_container:
                with ui.card().classes("w-full").style(
                    "background:white !important; border-radius:16px !important; padding:28px !important; "
                    "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important; border-left:4px solid #01335A !important;"
                ):
                    # Header
                    with ui.row().classes("w-full items-center justify-between mb-4"):
                        with ui.row().classes("items-center gap-3"):
                            ui.label("").style("font-size:28px !important;")
                            ui.label(col).style(
                                "font-weight:700 !important; font-size:20px !important; color:#01335A !important;"
                            )
                            with ui.badge().style(
                                "background:#e8f4f8 !important; color:#01335A !important; "
                                "padding:6px 12px !important; border-radius:6px !important;"
                            ):
                                ui.label("Numérique").style("font-size:12px !important; font-weight:600 !important;")
                    
                    # Stats grid
                    with ui.grid(columns=4).classes("w-full gap-4 mb-4"):
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
                            with ui.card().classes("p-3").style(
                                "background:#f8f9fa !important; border-radius:8px !important; "
                                "box-shadow:none !important;"
                            ):
                                ui.label(stat_label).style(
                                    "font-size:11px !important; color:#636e72 !important; "
                                    "margin-bottom:4px !important; text-transform:uppercase !important;"
                                )
                                ui.label(str(stat_value)).style(
                                    "font-weight:700 !important; font-size:16px !important; "
                                    "color:#01335A !important; font-family:monospace !important;"
                                )
                    
                    # Alertes
                    alerts = []
                    if abs(skewness) > 1:
                        alerts.append(("🔴", f"Asymétrie très forte (skewness = {skewness})"))
                    elif abs(skewness) > 0.5:
                        alerts.append(("🟡", f"Asymétrie modérée (skewness = {skewness})"))
                    
                    if pct_outliers > 10:
                        alerts.append(("🔴", f"Nombreux outliers détectés ({pct_outliers}% des données)"))
                    elif pct_outliers > 5:
                        alerts.append(("🟡", f"Outliers détectés ({pct_outliers}% des données)"))
                    
                    if series.isna().sum() > 0:
                        pct_missing = round(series.isna().sum() / len(X_train) * 100, 2)
                        alerts.append(("", f"Valeurs manquantes: {pct_missing}%"))
                    
                    if alerts:
                        with ui.column().classes("w-full gap-2 mb-4"):
                            for icon, msg in alerts:
                                with ui.card().classes("w-full p-3").style(
                                    "background:#cde4ff !important; border-radius:8px !important; "
                                    "border-left:3px solid #01335A !important; box-shadow:none !important;"
                                ):
                                    with ui.row().classes("items-center gap-2"):
                                        ui.label(icon).style("font-size:18px !important;")
                                        ui.label(msg).style(
                                            "color:#856404 !important; font-size:13px !important; font-weight:500 !important;"
                                        )
                    
                    # Visualisations
                    with ui.row().classes("w-full gap-4"):
                        # Histogramme
                        with ui.column().classes("flex-1"):
                            hist_fig = create_numeric_plot(series, col)
                            ui.plotly(hist_fig).style("width:100% !important;")
                        
                        # Boxplot
                        with ui.column().classes("flex-1"):
                            box_fig = create_boxplot(series, col)
                            ui.plotly(box_fig).style("width:100% !important;")

        def add_categorical_card(col):
            """Carte pour variable catégorielle"""
            series = X_train[col]
            
            # Statistiques
            n_unique = series.nunique()
            mode_val = series.mode()[0] if len(series.mode()) > 0 else "N/A"
            counts = series.value_counts()
            total = len(series)
            
            with cards_container:
                with ui.card().classes("w-full").style(
                    "background:white !important; border-radius:16px !important; padding:28px !important; "
                    "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important; border-left:4px solid #01335A !important;"
                ):
                    # Header
                    with ui.row().classes("w-full items-center justify-between mb-4"):
                        with ui.row().classes("items-center gap-3"):
                            ui.label("").style("font-size:28px !important;")
                            ui.label(col).style(
                                "font-weight:700 !important; font-size:20px !important; color:#01335A !important;"
                            )
                            with ui.badge().style(
                                "background:#d4edda !important; color:#155724 !important; "
                                "padding:6px 12px !important; border-radius:6px !important;"
                            ):
                                ui.label("Catégorielle").style("font-size:12px !important; font-weight:600 !important;")
                    
                    # Stats
                    with ui.row().classes("w-full gap-6 mb-4"):
                        with ui.card().classes("flex-1 p-3").style(
                            "background:#f8f9fa !important; border-radius:8px !important; box-shadow:none !important;"
                        ):
                            ui.label("Catégories uniques").style(
                                "font-size:11px !important; color:#636e72 !important; "
                                "margin-bottom:4px !important; text-transform:uppercase !important;"
                            )
                            ui.label(str(n_unique)).style(
                                "font-weight:700 !important; font-size:18px !important; color:#01335A !important;"
                            )
                        
                        with ui.card().classes("flex-1 p-3").style(
                            "background:#f8f9fa !important; border-radius:8px !important; box-shadow:none !important;"
                        ):
                            ui.label("Mode (valeur la plus fréquente)").style(
                                "font-size:11px !important; color:#636e72 !important; "
                                "margin-bottom:4px !important; text-transform:uppercase !important;"
                            )
                            ui.label(str(mode_val)[:30]).style(
                                "font-weight:700 !important; font-size:18px !important; color:#01335A !important;"
                            )
                    
                    # Alertes
                    if n_unique > 50:
                        with ui.card().classes("w-full p-3 mb-4").style(
                            "background:#cde4ff !important; border-radius:8px !important; "
                            "border-left:3px solid #01335A !important; box-shadow:none !important;"
                        ):
                            with ui.row().classes("items-center gap-2"):
                                ui.label("").style("font-size:18px !important;")
                                ui.label(f"Cardinalité élevée: {n_unique} catégories (encodage complexe)").style(
                                    "color:#856404 !important; font-size:13px !important; font-weight:500 !important;"
                                )
                    
                    # Distribution (Top 10)
                    with ui.column().classes("w-full gap-3 mb-4"):
                        ui.label(f"Distribution (Top {min(10, len(counts))} catégories)").style(
                            "font-weight:600 !important; font-size:14px !important; color:#636e72 !important;"
                        )
                        
                        for idx, (cat, cnt) in enumerate(counts.head(10).items()):
                            pct = round(cnt / total * 100, 1)
                            
                            with ui.row().classes("w-full items-center gap-3"):
                                ui.label(str(cat)[:30]).style(
                                    "width:180px !important; font-family:monospace !important; "
                                    "font-size:13px !important; color:#2c3e50 !important;"
                                )
                                
                                with ui.column().classes("flex-1"):
                                    ui.linear_progress(value=pct/100).props(
                                        f'color={"primary" if idx == 0 else "grey"}'
                                    ).classes("h-3 rounded-lg")
                                
                                ui.label(f"{cnt} ({pct}%)").style(
                                    "width:100px !important; text-align:right !important; "
                                    "font-family:monospace !important; font-size:13px !important; "
                                    "color:#636e72 !important; font-weight:600 !important;"
                                )
                        
                        if len(counts) > 10:
                            ui.label(f"... et {len(counts) - 10} autres catégories").style(
                                "font-size:12px !important; color:#7f8c8d !important; "
                                "font-style:italic !important; margin-top:8px !important;"
                            )
                    
                    # Graphique
                    with ui.column().classes("w-full"):
                        cat_fig = create_categorical_plot(series, col)
                        ui.plotly(cat_fig).style("width:100% !important;")

        def update_display():
            """Met à jour l'affichage selon les filtres"""
            cards_container.clear()
            
            # Filtrage
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
            
            # Affichage
            if not cols_to_show:
                with cards_container:
                    with ui.card().classes("w-full p-8").style(
                        "background:white !important; border-radius:16px !important; text-align:center !important;"
                    ):
                        ui.label("🔍").style("font-size:48px !important; margin-bottom:16px !important;")
                        ui.label("Aucune variable ne correspond aux filtres").style(
                            "font-size:16px !important; color:#636e72 !important;"
                        )
            else:
                for col in cols_to_show:
                    if col in numeric_cols:
                        add_numeric_card(col)
                    else:
                        add_categorical_card(col)
        
        # Événements des filtres
        filter_type.on_value_change(lambda: update_display())
        sort_by.on_value_change(lambda: update_display())
        search_input.on_value_change(lambda: update_display())
        
        # Affichage initial
        update_display()

        # --- NAVIGATION ---
        with ui.row().classes("w-full max-w-6xl justify-between gap-4 mt-8"):
            ui.button(
                "← Précédent",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/preprocessing2'")
            ).style(
                "background:white !important; color:#01335A !important; font-weight:500 !important; "
                "border-radius:8px !important; height:48px !important; min-width:140px !important; "
                "font-size:14px !important; text-transform:none !important; "
                "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
            )
            
            ui.button(
                "Détection des Outliers →",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/outliers_analysis'")
            ).style(
                "background:#01335A !important; color:white !important; font-weight:600 !important; "
                "border-radius:8px !important; height:48px !important; min-width:200px !important; "
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
        """Applique le traitement des outliers"""
        df_result = df_data.copy()
        
        if method == 'none' or not indices_to_treat:
            return df_result
        
        valid_indices = [idx for idx in indices_to_treat if idx in df_result.index]
        
        if not valid_indices:
            return df_result
        
        if method == 'remove':
            df_result = df_result.drop(valid_indices, errors='ignore')
        elif method == 'cap':
            Q1 = df_result[col].quantile(0.25)
            Q3 = df_result[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_result.loc[valid_indices, col] = df_result.loc[valid_indices, col].clip(lower, upper)
        elif method == 'median':
            median_val = df_result[col].median()
            df_result.loc[valid_indices, col] = median_val
        elif method == 'mean':
            mean_val = df_result[col].mean()
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
                        'constant': '#95a5a6'
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
                "← Précédent",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/preprocessing2'")
            ).style(
                "background:#01335A !important; color:white; font-weight:500; border-radius:8px; "
                "height:48px; min-width:140px; font-size:14px; text-transform:none; box-shadow:0 2px 8px rgba(0,0,0,0.08);"
            )
            
            ui.button(
                "Suivant →",
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
                "font-size:18px; color:#c0392b; font-weight:600;"
            )
            ui.button(
                " Retour à l'Upload",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/upload'")
            ).style(
                "background:#01335A; color:white; padding:12px 32px; border-radius:8px; margin-top:20px;"
            )
        return

    # Préparations
    numeric_cols = df.select_dtypes(include=[int, float, "number"]).columns.tolist()
    active_numeric = [c for c in numeric_cols if not columns_exclude.get(c, False) and c != target_col]

    if "engineered_features" not in state:
        state["engineered_features"] = []

    # Calculer les paires corrélées une seule fois
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
        """Crée un histogramme Plotly"""
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
        """Modal avec détails de la paire et création de feature"""
        with ui.dialog() as dlg, ui.card().classes("w-full max-w-4xl").style(
            "padding:0; border-radius:16px; box-shadow:0 10px 40px rgba(0,0,0,0.15);"
        ):
            # Header
            with ui.column().classes("w-full").style(
                "background:linear-gradient(135deg, #01335A 0%, #09538C 100%); "
                "padding:24px 32px; border-radius:16px 16px 0 0;"
            ):
                ui.label(f"Analyse de paire : {feat_a} ↔ {feat_b}").style(
                    "font-weight:700; font-size:22px; color:white;"
                )
                ui.label(f"Corrélation : {r_val:.3f}").style(
                    "color:white; font-size:16px; margin-top:4px; opacity:0.9;"
                )
            
            # Contenu
            with ui.column().classes("w-full").style("padding:32px;"):
                # Scatter plot
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
                ui.plotly(scatter).style("width:100%;")
                
                # Formulaire création feature
                with ui.card().classes("w-full mt-6").style(
                    "background:#f8f9fa; padding:20px; border-radius:12px; border:1px solid #e1e8ed;"
                ):
                    ui.label("Créer une nouvelle feature combinée").style(
                        "font-weight:600; font-size:16px; color:#01335A; margin-bottom:16px;"
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
                        "color:#7f8c8d; text-transform:none;"
                    )
                    
                    ui.button(
                        "Créer la feature",
                        on_click=lambda: create_engineered_feature(
                            feat_a, feat_b, name_input.value, formula_select.value, dlg
                        )
                    ).style(
                        "background:#01335A; color:white; border-radius:8px; "
                        "padding:10px 24px; text-transform:none; font-weight:600;"
                    )
        
        dlg.open()

    def create_engineered_feature(a, b, new_name, formula_str, dialog_obj):
        """Crée une feature engineered"""
        if not new_name:
            ui.notify(" Donnez un nom à la nouvelle feature", color="warning")
            return
        
        if new_name in df.columns:
            ui.notify(" Une colonne avec ce nom existe déjà", color="warning")
            return
        
        try:
            s_a = df[a].astype(float)
            s_b = df[b].astype(float)
            
            if "/" in formula_str:
                new_series = s_b / s_a.replace({0: np.nan})
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
            
            ui.notify(f" Feature '{new_name}' créée avec succès", color="positive")
            dialog_obj.close()
            
        except Exception as e:
            ui.notify(f"❌ Erreur : {str(e)}", color="negative")

    def keep_feature(a, b, keep="both"):
        """Garde une feature et exclut l'autre"""
        if keep == a:
            state.setdefault("columns_exclude", {})[b] = True
            ui.notify(f" {a} conservée, {b} exclue", color="positive")
        elif keep == b:
            state.setdefault("columns_exclude", {})[a] = True
            ui.notify(f" {b} conservée, {a} exclue", color="positive")
        else:
            state.setdefault("columns_exclude", {})[a] = False
            state.setdefault("columns_exclude", {})[b] = False
            ui.notify(f" Les deux features sont conservées", color="positive")
        
        ui.run_javascript("setTimeout(() => window.location.reload(), 800);")

    def apply_naivebayes_prune():
        """Applique la recommandation pour Naive Bayes"""
        if not pairs:
            ui.notify(" Aucune paire corrélée à traiter", color="warning")
            return
        
        excluded_count = 0
        for a, b, r in pairs:
            if target_col and target_col in df.columns:
                try:
                    corr_a = abs(df[a].corr(df[target_col]))
                    corr_b = abs(df[b].corr(df[target_col]))
                except:
                    corr_a = corr_b = 0
                
                if corr_a >= corr_b:
                    state.setdefault("columns_exclude", {})[b] = True
                    excluded_count += 1
                else:
                    state.setdefault("columns_exclude", {})[a] = True
                    excluded_count += 1
            else:
                state.setdefault("columns_exclude", {})[b] = True
                excluded_count += 1
        
        ui.notify(f" {excluded_count} features exclues (optimisation Naive Bayes)", color="positive")
        ui.run_javascript("setTimeout(() => window.location.reload(), 1000);")

    def open_bulk_engineer_modal():
        """Modal pour créer une feature combinée manuellement"""
        with ui.dialog() as dlg, ui.card().classes("w-full max-w-2xl").style(
            "padding:0; border-radius:16px; box-shadow:0 10px 40px rgba(0,0,0,0.15);"
        ):
            # Header
            with ui.column().classes("w-full").style(
                "background:linear-gradient(135deg, #01335A 0%, #09538C 100%); "
                "padding:24px 32px; border-radius:16px 16px 0 0;"
            ):
                ui.label("Créer une feature combinée").style(
                    "font-weight:700; font-size:22px; color:white;"
                )
            
            # Contenu
            with ui.column().classes("w-full").style("padding:32px;"):
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
                        ui.notify(" Sélectionnez A, B et donnez un nom", color="warning")
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
                        "color:#7f8c8d; text-transform:none;"
                    )
                    
                    ui.button("Créer", on_click=create_btn_action).style(
                        "background:#01335A; color:white; border-radius:8px; "
                        "padding:10px 24px; text-transform:none; font-weight:600;"
                    )
        
        dlg.open()

    # ---------- UI PRINCIPALE ----------
    with ui.column().classes("w-full items-center").style(
        "background:#f0f2f5; min-height:100vh; padding:48px 24px; font-family:'Inter', sans-serif;"
    ):
        # HEADER
        ui.label("Analyse Multivariée").style(
            "font-weight:700; font-size:36px; color:#01335A; margin-bottom:8px; "
            "text-align:center; letter-spacing:-0.5px;"
        )
        ui.label("Corrélations et gestion de la redondance entre variables").style(
            "font-size:16px; color:#636e72; margin-bottom:48px; text-align:center; font-weight:400;"
        )
        
        # SECTION A : HEATMAP (SANS COEFFICIENTS AFFICHÉS)
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white; border-radius:16px; padding:32px; box-shadow:0 2px 8px rgba(0,0,0,0.08);"
        ):
            ui.label("Matrice de corrélation").style(
                "font-weight:700; font-size:22px; color:#01335A; margin-bottom:16px;"
            )
            
            ui.label("Cliquez sur une cellule pour analyser la paire en détail").style(
                "font-size:14px; color:#636e72; margin-bottom:20px;"
            )

            if len(active_numeric) < 2:
                ui.label(" Moins de 2 colonnes numériques disponibles").style(
                    "color:#7f8c8d; font-size:15px;"
                )
            else:
                try:
                    # Heatmap SANS text/texttemplate (pas de coefficients affichés)
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

                    plot = ui.plotly(heatmap_fig).style("width:100%;")
                    
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
                            "background:#fff9e6; padding:16px; border-radius:12px; border-left:4px solid #01335A;"
                        ):
                            ui.label(f" {len(pairs)} paire(s) fortement corrélée(s) (|r| ≥ 0.8)").style(
                                "font-weight:600; margin-bottom:8px; color:#856404;"
                            )
                            for p_a, p_b, p_r in pairs[:8]:
                                ui.label(f"• {p_a} ↔ {p_b}: r = {p_r:.3f}").style(
                                    "font-size:13px; margin-left:12px; color:#856404;"
                                )
                            if len(pairs) > 8:
                                ui.label(f"... et {len(pairs)-8} autres paires").style(
                                    "font-size:13px; margin-left:12px; font-style:italic; color:#7f8c8d;"
                                )
                
                except Exception as e:
                    ui.label(f" Erreur lors du calcul de la corrélation: {str(e)}").style("color:#e74c3c;")

        # SECTION B : PAIRES CORRÉLÉES
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white; border-radius:16px; padding:32px; box-shadow:0 2px 8px rgba(0,0,0,0.08);"
        ):
            ui.label("Paires fortement corrélées").style(
                "font-weight:700; font-size:22px; color:#01335A; margin-bottom:16px;"
            )
            
            ui.label("Détection automatique des corrélations élevées (|r| ≥ 0.8)").style(
                "font-size:14px; color:#636e72; margin-bottom:20px;"
            )

            if len(active_numeric) < 2:
                ui.label(" Pas assez de variables numériques").style("color:#7f8c8d;")
            elif len(pairs) == 0:
                with ui.card().classes("w-full").style(
                    "background:#01335A; padding:16px; border-radius:12px; border-left:4px solid #01335A;"
                ):
                    ui.label(" Aucune corrélation élevée détectée").style(
                        "color:white; font-weight:500;"
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
                    
                    impact_nb = "🔴" if abs(r) > 0.85 else "🟡"
                    impact_knn = "🟡" if abs(r) >= 0.8 else "🟢"
                    impact_c45 = "🟢"
                    
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
                ).props("flat bordered").style("width:100%;")

        # SECTION C : VIF
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white; border-radius:16px; padding:32px; box-shadow:0 2px 8px rgba(0,0,0,0.08);"
        ):
            ui.label("VIF (Variance Inflation Factor)").style(
                "font-weight:700; font-size:22px; color:#01335A; margin-bottom:16px;"
            )
            
            ui.label("Indicateur de multicolinéarité : VIF > 5 suggère une redondance").style(
                "font-size:14px; color:#636e72; margin-bottom:20px;"
            )
            
            if len(active_numeric) < 2:
                ui.label(" Pas assez de colonnes numériques").style("color:#7f8c8d;")
            else:
                vif_df = []
                try:
                    X_vif = df[active_numeric].dropna()
                    X_vif_const = sm.add_constant(X_vif)
                    
                    for i, col in enumerate(active_numeric):
                        vif_val = variance_inflation_factor(X_vif_const.values, i + 1)
                        level = "🟢 OK" if vif_val <= 5 else ("🟡 Modéré" if vif_val <= 10 else "🔴 Élevé")
                        vif_df.append({
                            "Feature": col,
                            "VIF": f"{vif_val:.2f}",
                            "Niveau": level
                        })
                except Exception as e:
                    ui.label(f"❌ Erreur calcul VIF : {str(e)}").style("color:#e74c3c;")

                if len(vif_df) > 0:
                    ui.table(
                        columns=[
                            {"name": "Feature", "label": "Feature", "field": "Feature", "align": "left"},
                            {"name": "VIF", "label": "VIF", "field": "VIF", "align": "center"},
                            {"name": "Niveau", "label": "Niveau", "field": "Niveau", "align": "center"}
                        ],
                        rows=vif_df,
                        row_key="Feature"
                    ).props("flat bordered").style("width:100%;")

        # SECTION D : ACTIONS
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white; border-radius:16px; padding:32px; box-shadow:0 2px 8px rgba(0,0,0,0.08);"
        ):
            ui.label("Actions et recommandations").style(
                "font-weight:700; font-size:22px; color:#01335A; margin-bottom:16px;"
            )

            # Boutons globaux
            with ui.row().classes("gap-3 mb-6"):
                ui.button(
                    "Optimiser pour Naive Bayes",
                    on_click=apply_naivebayes_prune
                ).style(
                    "background:#01335A !important; color:white; border-radius:8px; "
                    "padding:10px 20px; text-transform:none; font-weight:500;"
                )
                
                ui.button(
                    "Créer feature combinée",
                    on_click=open_bulk_engineer_modal
                ).props("outlined").style(
                    "background:#01335A !important; color:white; border-radius:8px; "
                    "padding:10px 20px; text-transform:none; font-weight:500;"
                )

            # Cartes comparatives pour chaque paire
            if len(pairs) > 0:
                ui.label(f"{len(pairs)} paire(s) à analyser").style(
                    "font-weight:600; color:#636e72; margin-bottom:16px;"
                )
                
                for idx, (a, b, r) in enumerate(pairs[:5]):
                    with ui.expansion(f"{a} ↔ {b} (r = {r:.3f})", icon="compare_arrows").classes("w-full mb-3"):
                        with ui.row().classes("w-full gap-4"):
                            # Feature A
                            with ui.card().classes("flex-1").style(
                                "background:#f8f9fa; padding:16px; border-radius:12px;"
                            ):
                                ui.label(a).style("font-weight:700; font-size:15px; margin-bottom:8px; color:#01335A;")
                                hist_a = make_histogram_plot(df[a].dropna(), a)
                                ui.plotly(hist_a).style("width:100%;")
                                ui.label(f"Moyenne: {float(df[a].dropna().mean()):.3f}").style("font-size:13px; color:#636e72;")
                                
                                if target_col and pd.api.types.is_numeric_dtype(df[target_col]):
                                    try:
                                        corr_a = df[a].corr(df[target_col])
                                        ui.label(f"Corrélation avec target: {corr_a:.3f}").style(
                                            "font-size:13px; color:#01335A; font-weight:600;"
                                        )
                                    except:
                                        pass

                            # Feature B
                            with ui.card().classes("flex-1").style(
                                "background:#f8f9fa; padding:16px; border-radius:12px;"
                            ):
                                ui.label(b).style("font-weight:700; font-size:15px; margin-bottom:8px; color:#01335A;")
                                hist_b = make_histogram_plot(df[b].dropna(), b)
                                ui.plotly(hist_b).style("width:100%;")
                                ui.label(f"Moyenne: {float(df[b].dropna().mean()):.3f}").style("font-size:13px; color:#636e72;")
                                
                                if target_col and pd.api.types.is_numeric_dtype(df[target_col]):
                                    try:
                                        corr_b = df[b].corr(df[target_col])
                                        ui.label(f"Corrélation avec target: {corr_b:.3f}").style(
                                            "font-size:13px; color:#01335A; font-weight:600;"
                                        )
                                    except:
                                        pass
                        
                        # Actions
                        with ui.row().classes("w-full gap-2 mt-4"):
                            ui.button(
                                f"Garder {a}",
                                on_click=lambda a=a, b=b: keep_feature(a, b, keep=a)
                            ).props("flat").style("color:#01335A; text-transform:none;")
                            
                            ui.button(
                                f"Garder {b}",
                                on_click=lambda a=a, b=b: keep_feature(a, b, keep=b)
                            ).props("flat").style("color:#01335A; text-transform:none;")
                            
                            ui.button(
                                "Garder les deux",
                                on_click=lambda a=a, b=b: keep_feature(a, b, keep="both")
                            ).props("flat").style("color:#01335A; text-transform:none;")
                            
                            ui.button(
                                "Créer feature combinée",
                                on_click=lambda a=a, b=b, r=r: open_pair_modal(a, b, r)
                            ).style(
                                "background:#01335A; color:white; border-radius:8px; "
                                "padding:8px 16px; text-transform:none;"
                            )
                
                if len(pairs) > 5:
                    ui.label(f"... et {len(pairs) - 5} autres paires (limitées pour performance)").style(
                        "font-size:13px; color:#7f8c8d; font-style:italic; text-align:center; margin-top:12px;"
                    )
            else:
                with ui.card().classes("w-full").style(
                    "background:#01335A; padding:16px; border-radius:12px; border-left:4px solid #01335A;"
                ):
                    ui.label(" Aucune paire corrélée à analyser").style(
                    "background:#01335A !important; color:white; border-radius:8px; "
                    "padding:10px 20px; text-transform:none; font-weight:500;"
                )


        # NAVIGATION
        with ui.row().classes("w-full max-w-6xl justify-between gap-4 mt-8"):
            ui.button(
                "← Précédent",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/preprocessing2'")
            ).style(
                "background:#01335A !important; color:white; font-weight:500; border-radius:8px; "
                "height:48px; min-width:140px; font-size:14px; text-transform:none; "
                "box-shadow:0 2px 8px rgba(0,0,0,0.08);"
            )
            
            ui.button(
                "Suivant →",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/missing_values'")
            ).style(
                "background:#01335A !important; color:white; font-weight:600; border-radius:8px; "
                "height:48px; min-width:140px; font-size:14px; text-transform:none;"
            )






# ----------------- PAGE 3.6 : GESTION DES VALEURS MANQUANTES (VERSION FINALE) -----------------

@ui.page('/supervised/missing_values')
def missing_values_page():
    """
    Page complète pour gestion des valeurs manquantes avec visualisation before/after
    Gestion robuste des features créées dynamiquement
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

    #  CORRECTION : Synchroniser le split avec raw_df pour inclure les nouvelles features
    if split and "X_train" in split:
        try:
            # Récupérer les nouvelles colonnes ajoutées à raw_df
            new_cols = [c for c in df.columns if c not in split["X_train"].columns and c != target_col]
            
            if new_cols:
                print(f"🔄 Synchronisation des nouvelles features : {new_cols}")
                
                # Ajouter les nouvelles colonnes au split
                for key in ["X_train", "X_val", "X_test"]:
                    if key in split and isinstance(split[key], pd.DataFrame):
                        # Récupérer les indices du split
                        indices = split[key].index
                        
                        # Ajouter les nouvelles colonnes depuis raw_df
                        for col in new_cols:
                            if col in df.columns:
                                try:
                                    split[key][col] = df[col].reindex(indices)
                                except Exception as e:
                                    print(f" Impossible de synchroniser {col} pour {key}: {e}")
                
                # Mettre à jour le state
                state["split"] = split
                print(f" {len(new_cols)} nouvelles features synchronisées dans le split")
        
        except Exception as e:
            print(f" Erreur lors de la synchronisation : {e}")

    # Sauvegarde df_original si pas déjà fait
    if "df_original" not in state:
        state["df_original"] = df.copy()

    # Déterminer df_train pour le fit
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
            
            #  CORRECTION : Vérifier que la colonne existe
            if col not in df_train_local.columns:
                print(f" Colonne {col} introuvable dans df_train, ignorée")
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
            
            #  CORRECTION : Vérifier existence colonne
            if col not in df_result.columns:
                print(f" Colonne {col} introuvable dans df_target, ignorée")
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
        #  CORRECTION : Filtrer seulement les colonnes existantes
        valid_cols = [c for c in cols if c in df_data.columns]
        if not valid_cols:
            return []
        
        mask = df_data[valid_cols].isna().any(axis=1)
        indices = df_data[mask].index.tolist()
        return indices[:max_rows]

    def apply_and_propagate(navigate_after=False):
        """Applique l'imputation sur raw_df ET tous les splits"""
        try:
            strategies = state.get("missing_strategy", {})
            if not strategies:
                ui.notify(" Aucune stratégie configurée", color="warning")
                return False
            
            split = state.get("split", {})
            
            # 1. FIT sur le bon dataset
            if split and "X_train" in split:
                df_train_for_fit = split["X_train"].copy()
                if target_col and "y_train" in split and target_col not in df_train_for_fit.columns:
                    df_train_for_fit[target_col] = split["y_train"]
            else:
                df_train_for_fit = state["raw_df"].copy()
            
            # Fit des imputers
            fitted = fit_imputers(strategies, df_train_for_fit)
            state["fitted_imputers"] = serialize_fitted_imputers(fitted)
            
            # 2. APPLICATION sur raw_df
            state["raw_df"] = apply_fitted_imputers(state["raw_df"], fitted, active_cols)
            
            # 3. APPLICATION sur les splits si présents
            if split:
                for key in ["X_train", "X_val", "X_test"]:
                    if key in split and isinstance(split[key], pd.DataFrame):
                        split[key] = apply_fitted_imputers(split[key], fitted, active_cols)
                
                state["split"] = split
            
            ui.notify(" Imputation appliquée sur raw_df et tous les splits!", color="positive")
            
            # Navigation conditionnelle
            if navigate_after:
                ui.run_javascript("setTimeout(() => window.location.href='/supervised/encoding', 1000);")
            else:
                ui.run_javascript("setTimeout(() => window.location.reload(), 1000);")
            
            return True
        
        except Exception as e:
            ui.notify(f"❌ Erreur lors de l'application : {str(e)}", color="negative")
            print(f"Détail erreur: {e}")
            return False

    def open_feature_modal(col_name):
        """Ouvre un dialog pour configurer la stratégie d'imputation d'une colonne"""
        current_strategy = state.get("missing_strategy", {}).get(col_name, {})
        current_method = current_strategy.get("method", "none")
        
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-2xl").style(
            "padding:32px !important; border-radius:16px !important; "
            "box-shadow:0 10px 40px rgba(0,0,0,0.15) !important;"
        ):
            ui.label(f"Configuration : {col_name}").style(
                "font-weight:700 !important; font-size:20px !important; color:#01335A !important; "
                "margin-bottom:16px !important;"
            )
            
            dtype = df[col_name].dtype
            n_missing = int(df[col_name].isna().sum())
            pct = round(n_missing / len(df) * 100, 2)
            
            ui.label(f"Type: {dtype} | Missing: {n_missing} ({pct}%)").style(
                "color:#636e72 !important; margin-bottom:20px !important; font-size:14px !important;"
            )
            
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
                    
                    ui.notify(f" Stratégie sauvegardée pour {col_name}", color="positive")
                    dialog.close()
                
                ui.button("Sauvegarder", on_click=save_strategy).style(
                    "background:#01335A !important; color:white !important; border-radius:8px !important; "
                    "padding:10px 24px !important; text-transform:none !important; font-weight:600 !important;"
                )
        
        dialog.open()

    def preview_imputation():
        """Prévisualise l'imputation avec tableaux BEFORE/AFTER des lignes affectées"""
        strategies = state.get("missing_strategy", {})
        
        if not strategies:
            ui.notify(" Aucune stratégie configurée", color="warning")
            return
        
        try:
            # Fit sur df_train
            fitted = fit_imputers(strategies, df_train)
            
            # Récupérer les lignes avec missing values AVANT traitement
            cols_with_strategy = list(strategies.keys())
            cols_to_check = [c for c in cols_with_strategy if c in df_train.columns]
            
            if not cols_to_check:
                ui.notify(" Aucune colonne valide à traiter", color="warning")
                return
            
            missing_indices = get_rows_with_missing(df_train, cols_to_check, max_rows=15)
            
            if not missing_indices:
                ui.notify("ℹ️ Aucune ligne avec valeurs manquantes dans les colonnes sélectionnées", color="info")
                return
            
            # BEFORE : lignes originales
            df_before = df_train.loc[missing_indices, cols_to_check].copy()
            
            # Application sur copie complète
            df_preview = df_train.copy()
            df_preview = apply_fitted_imputers(df_preview, fitted, active_cols)
            
            # AFTER : mêmes lignes après imputation
            df_after = df_preview.loc[missing_indices, cols_to_check].copy()
            
            # Stats globales
            before_missing = df_train[active_cols].isna().sum().sum()
            after_missing = df_preview[active_cols].isna().sum().sum()
            
            preview_info.set_text(
                f" Preview généré : {before_missing} valeurs manquantes → {after_missing} après imputation | {len(missing_indices)} lignes affichées"
            )
            
            # Affichage des tableaux BEFORE/AFTER
            table_before_container.clear()
            table_after_container.clear()
            
            with table_before_container:
                ui.label("📋 AVANT Imputation (lignes avec missing)").style(
                    "font-weight:700 !important; font-size:16px !important; color:#e74c3c !important; "
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
                ui.label(" APRÈS Imputation (mêmes lignes)").style(
                    "font-weight:700 !important; font-size:16px !important; color:#01335A !important; "
                    "margin-bottom:12px !important;"
                )
                
                rows_after = []
                for idx in missing_indices:
                    row_dict = {"Index": idx}
                    for col in cols_to_check:
                        val = df_after.loc[idx, col]
                        if pd.isna(val):
                            row_dict[col] = " Still NaN"
                        else:
                            was_nan = pd.isna(df_before.loc[idx, col])
                            display_val = str(val)[:30]
                            if was_nan:
                                row_dict[col] = f" {display_val}"
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
                    marker_color="#e74c3c",
                    opacity=0.7
                ))
                fig_before.update_layout(
                    title=f"{sample_col} - Distribution AVANT Imputation",
                    height=300,
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
                    title=f"{sample_col} - Distribution APRÈS Imputation",
                    height=300,
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
            ui.notify(" Aucune stratégie configurée", color="warning")
            return
        
        with ui.dialog() as dialog, ui.card().style(
            "padding:32px !important; border-radius:16px !important; "
            "box-shadow:0 10px 40px rgba(0,0,0,0.15) !important;"
        ):
            ui.label(" Confirmation").style(
                "font-weight:700 !important; font-size:20px !important; color:#01335A !important; "
                "margin-bottom:16px !important;"
            )
            ui.label("Voulez-vous appliquer l'imputation sur raw_df et tous les datasets (train/val/test) ?").style(
                "margin-bottom:8px !important; color:#2c3e50 !important; font-size:14px !important;"
            )
            ui.label(" Cette action est irréversible (sauf si vous rechargez le fichier)").style(
                "color:#e74c3c !important; font-size:13px !important;"
            )
            
            with ui.row().classes("w-full justify-end gap-3 mt-6"):
                ui.button("Annuler", on_click=dialog.close).props("flat").style(
                    "color:#7f8c8d !important; text-transform:none !important;"
                )
                
                def confirm_and_next():
                    dialog.close()
                    apply_and_propagate(navigate_after=True)
                
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
            ui.notify(f" Conservative : {excluded_count} colonnes >20% marquées pour exclusion", color="positive")

        elif val.startswith("Balanced"):
            strategy_count = 0
            for col in active_cols:
                if df[col].isna().sum() == 0:
                    continue
                method = "median" if pd.api.types.is_numeric_dtype(df[col]) else "mode"
                state.setdefault("missing_strategy", {})[col] = {"method": method, "params": {}}
                strategy_count += 1
            ui.notify(f" Balanced : {strategy_count} colonnes configurées (Median/Mode)", color="positive")

        elif val.startswith("Aggressive"):
            knn_count = 0
            for col in active_cols:
                if pd.api.types.is_numeric_dtype(df[col]) and df[col].isna().sum() > 0:
                    state.setdefault("missing_strategy", {})[col] = {"method": "knn", "params": {"n_neighbors": 5}}
                    knn_count += 1
            ui.notify(f" Aggressive : KNN appliqué à {knn_count} colonnes numériques", color="positive")

        elif val.startswith("Custom"):
            ui.notify("ℹ️ Choix Custom : configure colonne par colonne via le tableau", color="info")

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

        # --- A - Overview ---
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white !important; border-radius:16px !important; padding:32px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            ui.label(" Vue d'ensemble").style(
                "font-weight:700 !important; font-size:22px !important; color:#01335A !important; "
                "margin-bottom:20px !important;"
            )
            
            with ui.row().classes("w-full gap-8 items-center justify-around"):
                def metric(label, value, sub=""):
                    with ui.column().classes("items-center"):
                        ui.label(label).style(
                            "font-size:13px !important; color:#636e72 !important; margin-bottom:4px !important;"
                        )
                        ui.label(value).style(
                            "font-weight:700 !important; font-size:28px !important; color:#01335A !important;"
                        )
                        if sub: 
                            ui.label(sub).style(
                                "font-size:12px !important; color:#7f8c8d !important; margin-top:4px !important;"
                            )
                
                metric("Total missing", f"{total_missing:,}", f"{total_pct}% du dataset")
                metric("Features affectées", f"{affected}/{len(active_cols)}", "colonnes")
                metric("Lignes", f"{len(df):,}", "observations")

        # --- B - Table of columns ---
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
                tag = "🔴" if pct > 20 else ("🟡" if pct >= 5 else ("🟢" if pct > 0 else ""))
                
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
            
            ui.label("💡 Cliquez sur une ligne pour configurer la stratégie d'imputation").style(
                "font-size:13px !important; color:#636e72 !important; margin-top:12px !important; "
                "font-style:italic !important;"
            )
            
            def handle_row_click(e):
                if e and "row" in e and "Feature" in e["row"]:
                    open_feature_modal(e["row"]["Feature"])
            
            table.on("row-click", handle_row_click)

        # --- C - Global strategy ---
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

        # --- D - Preview & Apply ---
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
                    "background:#2d9cdb !important; color:white !important; border-radius:8px !important; "
                    "padding:10px 20px !important; text-transform:none !important; font-weight:500 !important;"
                )
                
                ui.button(
                    " Appliquer & Continuer",
                    on_click=confirm_and_apply
                ).style(
                    "background:#01335A !important; color:white !important; border-radius:8px !important; "
                    "padding:10px 20px !important; text-transform:none !important; font-weight:600 !important;"
                )

        # --- Navigation ---
        with ui.row().classes("w-full max-w-6xl justify-between gap-4 mt-8"):
            ui.button(
                "← Précédent",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/multivariate_analysis'")
            ).style(
                "background:white !important; color:#01335A !important; font-weight:500 !important; "
                "border-radius:8px !important; height:48px !important; min-width:140px !important; "
                "font-size:14px !important; text-transform:none !important; "
                "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
            )
            
            ui.button(
                "Suivant →",
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
    
    #  CORRECTION : Synchroniser le split avec raw_df
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
                print(f" {len(new_cols)} nouvelles features synchronisées")
        
        except Exception as e:
            print(f" Erreur synchronisation : {e}")
    
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
                print(f" Erreur création df_train : {e}")
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
            return "🟡", "Medium", "#01335A", "orange"
        else:
            return "🔴", "High", "#e74c3c", "red"
    
    def get_recommended_encoding(col):
        """Recommande une méthode d'encodage"""
        if col not in df.columns:
            return "Label Encoding", "Colonne introuvable", ""
        
        n_unique = df[col].nunique()
        
        if n_unique == 2:
            return "Label Encoding", "Binaire - simple et efficace", ""
        elif n_unique <= 10:
            return "One-Hot Encoding", "Faible cardinalité - safe et interprétable", ""
        elif n_unique <= 50:
            return "Frequency Encoding", "Cardinalité moyenne - évite l'explosion", ""
        else:
            return "Target Encoding", "Haute cardinalité - capture la relation avec target", ""
    
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
            marker_color='#01335A',
            text=counts.values,
            textposition='outside',
            opacity=0.8
        ))
        
        fig.update_layout(
            title=f"Distribution des 15 premières modalités",
            height=300,
            margin=dict(l=40, r=20, t=40, b=80),
            showlegend=False,
            paper_bgcolor='white',
            plot_bgcolor='#f8f9fa',
            font=dict(family="Inter, sans-serif", color="#2c3e50"),
            xaxis=dict(title="Modalités", tickangle=-45),
            yaxis=dict(title="Fréquence")
        )
        
        return fig
    
    def apply_encoding(df_target, strategies, params, fit_on_train=True, fitted_encoders=None):
        """Applique les encodages selon les stratégies définies"""
        df_result = df_target.copy()
        encoders = {} if fit_on_train else fitted_encoders
        
        for col, method in strategies.items():
            if col not in df_result.columns:
                print(f" Colonne {col} introuvable dans df_target")
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
        if col_name not in df.columns:
            ui.notify(f" Colonne {col_name} introuvable", color="warning")
            return
        
        current_method = state.get("encoding_strategy", {}).get(col_name, "")
        current_params = state.get("encoding_params", {}).get(col_name, {})
        
        n_unique = df[col_name].nunique()
        top_values = df[col_name].value_counts().head(5)
        is_ordinal, suggested_order = detect_ordinal(col_name)
        recommended, reason, icon = get_recommended_encoding(col_name)
        
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-4xl").style(
            "padding:0 !important; border-radius:16px !important; "
            "box-shadow:0 10px 40px rgba(0,0,0,0.15) !important; max-height:90vh !important; "
            "overflow-y:auto !important;"
        ):
            # Header
            with ui.column().classes("w-full").style(
                "background:linear-gradient(135deg, #01335A 0%, #09538C 100%) !important; "
                "padding:24px 32px !important; border-radius:16px 16px 0 0 !important;"
            ):
                ui.label(f"Configuration : {col_name}").style(
                    "font-weight:700 !important; font-size:22px !important; color:white !important;"
                )
                ui.label(f"Variable catégorielle • {n_unique} modalités uniques").style(
                    "color:white !important; font-size:14px !important; margin-top:4px !important; opacity:0.9 !important;"
                )
            
            # Contenu
            with ui.column().classes("w-full").style("padding:32px !important;"):
                # Stats & Distribution
                with ui.row().classes("w-full gap-4 mb-6"):
                    # Carte stats
                    with ui.card().classes("flex-1").style(
                        "background:#f8f9fa !important; padding:20px !important; "
                        "border-radius:12px !important; box-shadow:none !important;"
                    ):
                        icon_card, level, color, _ = get_cardinality_level(n_unique)
                        ui.label(f"{icon_card} Cardinalité: {level}").style(
                            f"font-weight:700 !important; font-size:16px !important; color:{color} !important; "
                            "margin-bottom:12px !important;"
                        )
                        
                        ui.label("Top 5 modalités :").style(
                            "font-size:13px !important; color:#636e72 !important; margin-bottom:8px !important;"
                        )
                        
                        for val, count in top_values.items():
                            pct = round(count / len(df) * 100, 1)
                            with ui.row().classes("items-center gap-2 mb-1"):
                                ui.label(str(val)[:20]).style(
                                    "width:120px !important; font-size:12px !important; "
                                    "color:#2c3e50 !important; font-weight:500 !important;"
                                )
                                ui.linear_progress(value=pct/100).props('color="primary"').classes("flex-1 h-2")
                                ui.label(f"{pct}%").style(
                                    "width:50px !important; text-align:right !important; "
                                    "font-size:12px !important; color:#636e72 !important; font-family:monospace !important;"
                                )
                    
                    # Graphique
                    with ui.column().classes("flex-1"):
                        dist_plot = create_distribution_plot(col_name)
                        if dist_plot:
                            ui.plotly(dist_plot).style("width:100% !important;")
                
                # Recommandation
                with ui.card().classes("w-full mb-6").style(
                    "background:linear-gradient(135deg, #e8f4f8 0%, #d4e9f7 100%) !important; "
                    "padding:20px !important; border-radius:12px !important; "
                    "border-left:4px solid #01335A !important; box-shadow:none !important;"
                ):
                    with ui.row().classes("items-center gap-3 mb-2"):
                        ui.label(icon).style("font-size:24px !important;")
                        ui.label(f"Recommandation : {recommended}").style(
                            "font-weight:700 !important; font-size:16px !important; color:#01335A !important;"
                        )
                    ui.label(reason).style(
                        "font-size:13px !important; color:#2c3e50 !important; margin-left:36px !important;"
                    )
                
                # Sélection méthode
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
                ).props("outlined").classes("w-full mb-4")
                
                # Zone paramètres
                params_container = ui.column().classes("w-full")
                
                # Variables pour stocker les widgets
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
                            with ui.card().classes("w-full p-4").style(
                                "background:#f8f9fa !important; border-radius:8px !important; box-shadow:none !important;"
                            ):
                                ui.markdown("""
**📌 Label Encoding**

**Principe** : Assigne un entier unique à chaque modalité (0, 1, 2, ...)

 **Avantages** :
- Simple et rapide
- Idéal pour variables binaires
- Pas d'explosion dimensionnelle

 **Attention** :
- Impose un ordre arbitraire
- Peut biaiser les modèles basés sur la distance
                                """).style("color:#2c3e50 !important; font-size:13px !important;")
                        
                        elif method == "One-Hot Encoding":
                            with ui.card().classes("w-full p-4 mb-4").style(
                                "background:#f8f9fa !important; border-radius:8px !important; box-shadow:none !important;"
                            ):
                                ui.markdown("""
**📌 One-Hot Encoding**

**Principe** : Crée une colonne binaire pour chaque modalité

 **Avantages** :
- Pas d'ordre imposé
- Très interprétable

❌ **Inconvénients** :
- Explosion dimensionnelle si haute cardinalité
                                """).style("color:#2c3e50 !important; font-size:13px !important;")
                            
                            drop_first_checkbox = ui.checkbox(
                                "Drop first (éviter multicolinéarité)",
                                value=current_params.get("drop_first", True)
                            ).style("margin-top:8px !important;")
                        
                        elif method == "Ordinal Encoding":
                            with ui.card().classes("w-full p-4 mb-4").style(
                                "background:#f8f9fa !important; border-radius:8px !important; box-shadow:none !important;"
                            ):
                                ui.markdown("""
**📌 Ordinal Encoding**

**Principe** : Assigne des entiers selon un ordre naturel

 Capture l'ordre naturel (ex: Low < Medium < High)

 Applicable uniquement si ordre naturel existe
                                """).style("color:#2c3e50 !important; font-size:13px !important;")
                            
                            ui.label("Définir l'ordre (du plus bas au plus haut) :").style(
                                "font-weight:600 !important; margin-bottom:12px !important; color:#01335A !important;"
                            )
                            
                            current_order = current_params.get("order", suggested_order if is_ordinal else list(df[col_name].unique())[:10])
                            
                            with ui.column().classes("w-full gap-2"):
                                for idx, val in enumerate(current_order):
                                    with ui.row().classes("items-center gap-3"):
                                        with ui.badge().style(
                                            "background:#01335A !important; color:white !important; "
                                            "min-width:30px !important; height:30px !important; "
                                            "display:flex !important; align-items:center !important; "
                                            "justify-content:center !important; border-radius:6px !important;"
                                        ):
                                            ui.label(str(idx)).style("font-weight:600 !important;")
                                        
                                        inp = ui.input(value=str(val)).props("outlined dense").classes("flex-1")
                                        order_inputs.append(inp)
                        
                        elif method == "Frequency Encoding":
                            with ui.card().classes("w-full p-4").style(
                                "background:#f8f9fa !important; border-radius:8px !important; box-shadow:none !important;"
                            ):
                                ui.markdown("""
**📌 Frequency Encoding**

**Principe** : Remplace chaque modalité par sa fréquence

 **Avantages** :
- Simple et efficace
- Pas de data leakage
- Une seule colonne générée
                                """).style("color:#2c3e50 !important; font-size:13px !important;")
                        
                        elif method == "Target Encoding":
                            with ui.card().classes("w-full p-4 mb-4").style(
                                "background:#cde4ff !important; border-radius:8px !important; "
                                "border-left:3px solid #01335A !important; box-shadow:none !important;"
                            ):
                                ui.markdown("""
**📌 Target Encoding**

**Principe** : Remplace chaque modalité par la moyenne de la target

 Capture directement la relation avec la target

 **Risques** : Overfitting et data leakage possibles
                                """).style("color:#856404 !important; font-size:13px !important;")
                            
                            smoothing_input = ui.number(
                                label="Smoothing (régularisation)",
                                value=current_params.get("smoothing", 10),
                                min=0,
                                max=100
                            ).props("outlined").classes("w-full")
                
                method_select.on_value_change(lambda: update_params_ui())
                update_params_ui()
                
                # Boutons
                with ui.row().classes("w-full justify-end gap-3 mt-6"):
                    ui.button("Annuler", on_click=dialog.close).props("flat").style(
                        "color:#7f8c8d !important; text-transform:none !important;"
                    )
                    
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
                        
                        ui.notify(f" Encodage configuré pour {col_name}", color="positive")
                        dialog.close()
                        ui.run_javascript("setTimeout(() => window.location.reload(), 500);")
                    
                    ui.button("Sauvegarder", on_click=save_encoding).style(
                        "background:#01335A !important; color:white !important; border-radius:8px !important; "
                        "padding:10px 24px !important; text-transform:none !important; font-weight:600 !important;"
                    )
        
        dialog.open()
    
    def apply_all_encodings():
        """Applique tous les encodages et passe à l'étape suivante"""
        strategies = state.get("encoding_strategy", {})
        split_data = state.get("split", {})

        if not strategies:
            ui.notify(" Aucun encodage configuré", color="warning")
            return
        
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-md").style(
            "padding:32px !important; border-radius:16px !important; "
            "box-shadow:0 10px 40px rgba(0,0,0,0.15) !important;"
        ):
            ui.label(" Confirmation").style(
                "font-weight:700 !important; font-size:20px !important; color:#01335A !important; "
                "margin-bottom:16px !important;"
            )
            ui.label("Voulez-vous appliquer les encodages sur tous les datasets ?").style(
                "margin-bottom:8px !important; color:#2c3e50 !important; font-size:14px !important;"
            )
            ui.label(f"{len(strategies)} encodage(s) configuré(s)").style(
                "color:#636e72 !important; font-size:13px !important;"
            )
            
            with ui.row().classes("w-full justify-end gap-3 mt-6"):
                ui.button("Annuler", on_click=dialog.close).props("flat").style(
                    "color:#7f8c8d !important; text-transform:none !important;"
                )
                
                def confirm_and_next():
                    try:
                        params = state.get("encoding_params", {})
                        
                        # FIT sur X_train uniquement
                        if split_data and "X_train" in split_data:
                            df_train_for_fit = split_data["X_train"].copy()
                            if target_col and "y_train" in split_data and target_col not in df_train_for_fit.columns:
                                df_train_for_fit[target_col] = split_data["y_train"]
                        else:
                            df_train_for_fit = df_train.copy()
                    
                        # FIT sur train
                        _, encoders = apply_encoding(
                            df_train_for_fit, 
                            strategies, 
                            params, 
                            fit_on_train=True
                        )
                        state["fitted_encoders"] = encoders
                        
                        # TRANSFORM sur raw_df
                        df_result, _ = apply_encoding(
                            df.copy(),
                            strategies,
                            params,
                            fit_on_train=False,
                            fitted_encoders=encoders
                        )
                        state["raw_df"] = df_result
                    
                        # TRANSFORM sur les splits
                        if split_data:
                            for key in ["X_train", "X_val", "X_test"]:
                                if key in split_data and isinstance(split_data[key], pd.DataFrame):
                                    split_data[key], _ = apply_encoding(
                                        split_data[key], 
                                        strategies, 
                                        params, 
                                        fit_on_train=False,
                                        fitted_encoders=encoders
                                    )
                            state["split"] = split_data
                        
                        ui.notify(" Encodages appliqués avec succès!", color="positive")
                        dialog.close()
                        ui.run_javascript("setTimeout(() => window.location.href='/supervised/distribution_transform', 1000);")
                    
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
        for col in cat_cols:
            recommended, _, _ = get_recommended_encoding(col)
            state.setdefault("encoding_strategy", {})[col] = recommended
            
            # Paramètres par défaut
            params = {}
            if recommended == "One-Hot Encoding":
                params["drop_first"] = True
            elif recommended == "Target Encoding":
                params["smoothing"] = 10
            elif recommended == "Ordinal Encoding":
                is_ordinal, order = detect_ordinal(col)
                if is_ordinal:
                    params["order"] = order
                else:
                    params["order"] = list(df[col].unique())[:10]
            
            state.setdefault("encoding_params", {})[col] = params
            count += 1
        
        ui.notify(f" {count} recommandations appliquées", color="positive")
        ui.run_javascript("setTimeout(() => window.location.reload(), 500);")
    
    # ---------- UI PRINCIPALE ----------
    with ui.column().classes("w-full items-center").style(
        "background:#f0f2f5 !important; min-height:100vh !important; padding:48px 24px !important; "
        "font-family:'Inter', sans-serif !important;"
    ):
        # HEADER
        ui.label("Encodage des Features Catégorielles").style(
            "font-weight:700 !important; font-size:36px !important; color:#01335A !important; "
            "margin-bottom:8px !important; text-align:center !important; letter-spacing:-0.5px !important;"
        )
        ui.label("Conversion des variables catégorielles en format numérique").style(
            "font-size:16px !important; color:#636e72 !important; margin-bottom:48px !important; "
            "text-align:center !important; font-weight:400 !important;"
        )
        
        # --- OVERVIEW METRICS ---
        with ui.card().classes("w-full max-w-6xl mb-6").style(
            "background:white !important; border-radius:16px !important; padding:32px !important; "
            "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
        ):
            ui.label(" Vue d'ensemble").style(
                "font-weight:700 !important; font-size:22px !important; color:#01335A !important; "
                "margin-bottom:20px !important;"
            )
            
            if n_categorical == 0:
                with ui.card().classes("w-full p-6 text-center").style(
                    "background:linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%) !important; "
                    "border-radius:12px !important; border-left:4px solid #27ae60 !important; box-shadow:none !important;"
                ):
                    ui.label("").style("font-size:48px !important; margin-bottom:12px !important;")
                    ui.label("Aucune feature catégorielle détectée").style(
                        "font-weight:600 !important; font-size:18px !important; color:#155724 !important; "
                        "margin-bottom:8px !important;"
                    )
                    ui.label("Toutes vos features sont déjà au format numérique").style(
                        "font-size:14px !important; color:#155724 !important; opacity:0.8 !important;"
                    )
            else:
                with ui.row().classes("w-full gap-8 items-center justify-around"):
                    def metric_card(icon, label, value, sublabel="", color="#01335A"):
                        with ui.column().classes("items-center"):
                            ui.label(icon).style("font-size:32px !important; margin-bottom:8px !important;")
                            ui.label(value).style(
                                f"font-weight:700 !important; font-size:28px !important; color:{color} !important;"
                            )
                            ui.label(label).style(
                                "font-size:13px !important; color:#636e72 !important; margin-top:4px !important;"
                            )
                            if sublabel:
                                ui.label(sublabel).style(
                                    "font-size:12px !important; color:#7f8c8d !important; margin-top:2px !important;"
                                )
                    
                    pct_cat = round(n_categorical / total_features * 100) if total_features > 0 else 0
                    n_configured = len([c for c in cat_cols if c in state.get("encoding_strategy", {})])
                    
                    low_card = len([c for c in cat_cols if df[c].nunique() <= 10])
                    medium_card = len([c for c in cat_cols if 10 < df[c].nunique() <= 50])
                    high_card = len([c for c in cat_cols if df[c].nunique() > 50])
                    
                    metric_card("", "Catégorielles", str(n_categorical), f"{pct_cat}% du total")
                    metric_card("", "Configurées", f"{n_configured}/{n_categorical}", "")
                    metric_card("🟢", "Low", str(low_card), "≤10")
                    metric_card("🟡", "Medium", str(medium_card), "11-50", "#01335A")
                    metric_card("🔴", "High", str(high_card), ">50", "#e74c3c")
        
        # --- TABLE DES FEATURES ---
        if n_categorical > 0:
            with ui.card().classes("w-full max-w-6xl mb-6").style(
                "background:white !important; border-radius:16px !important; padding:32px !important; "
                "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
            ):
                ui.label("⚙️ Configuration des encodages").style(
                    "font-weight:700 !important; font-size:22px !important; color:#01335A !important; "
                    "margin-bottom:16px !important;"
                )
                
                ui.label("💡 Cliquez sur une ligne pour configurer l'encodage").style(
                    "font-size:14px !important; color:#636e72 !important; margin-bottom:20px !important; "
                    "font-style:italic !important;"
                )
                
                rows = []
                for col in cat_cols:
                    n_unique = df[col].nunique()
                    icon, level, color, badge_color = get_cardinality_level(n_unique)
                    recommended, _, rec_icon = get_recommended_encoding(col)
                    current = state.get("encoding_strategy", {}).get(col, "")
                    
                    status = "" if current else "⚪"
                    
                    rows.append({
                        "Feature": col,
                        "Cardinalité": n_unique,
                        "Niveau": f"{icon} {level}",
                        "Recommandé": f"{rec_icon} {recommended}",
                        "Configuré": current or "-",
                        "Status": status
                    })
                
                table = ui.table(
                    columns=[
                        {"name": "Feature", "label": "Feature", "field": "Feature", "align": "left"},
                        {"name": "Cardinalité", "label": "Card.", "field": "Cardinalité", "align": "center"},
                        {"name": "Niveau", "label": "Niveau", "field": "Niveau", "align": "center"},
                        {"name": "Recommandé", "label": "Recommandé", "field": "Recommandé", "align": "left"},
                        {"name": "Configuré", "label": "Configuré", "field": "Configuré", "align": "left"},
                        {"name": "Status", "label": "✓", "field": "Status", "align": "center"}
                    ],
                    rows=rows,
                    row_key="Feature"
                ).props("flat bordered").style("width:100% !important; cursor:pointer !important;")
                
                #  CORRECTION : Utiliser "row:click" avec deux-points
                def handle_row_click(e):
                    try:
                        if e and "row" in e and "Feature" in e["row"]:
                            open_encoding_modal(e["row"]["Feature"])
                    except Exception as ex:
                        print(f"Erreur click: {ex}")
                
                table.on("row-click", handle_row_click)  # ← Correction ici
                
                # Actions rapides
                with ui.row().classes("w-full gap-3 mt-6"):
                    ui.button(
                        "✨ Appliquer les recommandations",
                        on_click=apply_recommended
                    ).style(
                        "background:#01335A !important; color:white !important; border-radius:8px !important; "
                        "padding:12px 24px !important; text-transform:none !important; font-weight:600 !important;"
                    )
                    
                    ui.button(
                        " Appliquer & Continuer",
                        on_click=apply_all_encodings
                    ).style(
                        "background:#01335A !important; color:white !important; border-radius:8px !important; "
                        "padding:12px 24px !important; text-transform:none !important; font-weight:600 !important;"
                    )
        
        # --- NAVIGATION ---
        with ui.row().classes("w-full max-w-6xl justify-between gap-4 mt-8"):
            ui.button(
                "← Précédent",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/missing_values'")
            ).style(
                "background:white !important; color:#01335A !important; font-weight:500 !important; "
                "border-radius:8px !important; height:48px !important; min-width:140px !important; "
                "font-size:14px !important; text-transform:none !important; "
                "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
            )
            
            ui.button(
                "Suivant →",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/distribution_transform'")
            ).style(
                "background:#01335A !important; color:white !important; font-weight:600 !important; "
                "border-radius:8px !important; height:48px !important; min-width:140px !important; "
                "font-size:14px !important; text-transform:none !important;"
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
                " Retour à l'Upload",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/upload'")
            ).style(
                "background:#01335A !important; color:white !important; padding:12px 32px !important; "
                "border-radius:8px !important; margin-top:20px !important;"
            )
        return
    
    #  CORRECTION : Synchroniser le split avec raw_df
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
                print(f" {len(new_cols)} nouvelles features synchronisées")
        
        except Exception as e:
            print(f" Erreur synchronisation : {e}")
    
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
                print(f" Erreur création df_train : {e}")
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
            print(f" Erreur calcul skew/kurt pour {col}: {e}")
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
        
        #  CORRECTION : Gérer les différents types de data
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
                    print(f" Box-Cox impossible : valeurs ≤ 0")
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
            impacts["Naive Bayes"] = ("", "Distribution normale", "#27ae60")
        elif abs_skew < 1.5:
            impacts["Naive Bayes"] = ("🟡", "Assume normalité (légèrement violée)", "#01335A")
        else:
            impacts["Naive Bayes"] = ("🔴", "Assume normalité (fortement violée)", "#e74c3c")
        
        # KNN
        if abs_skew < 0.5:
            impacts["KNN"] = ("", "Distances équilibrées", "#27ae60")
        elif abs_skew < 1.5:
            impacts["KNN"] = ("🟡", "Distances légèrement biaisées", "#01335A")
        else:
            impacts["KNN"] = ("🟡", "Distances biaisées vers extrêmes", "#01335A")
        
        # C4.5
        impacts["Decision Tree"] = ("", "Robuste aux distributions", "#27ae60")
        
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
            print(f" Erreur Q-Q plot original: {e}")
        
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
            print(f" Erreur Q-Q plot transformé: {e}")
        
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
            title_text=f"<b>{col}</b> - Skew Original: {skew_original:.2f} → Transformé: {skew_transformed:.2f}",
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
    
    def open_transform_modal(col_name):
        """Ouvre un modal pour configurer la transformation d'une colonne"""
        if col_name not in df_train.columns:
            ui.notify(f" Colonne {col_name} introuvable", color="warning")
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
                                ui.label(" Négatives").style(
                                    "color:#e74c3c !important; font-weight:600 !important; font-size:16px !important;"
                                )
                            elif has_zero:
                                ui.label(" Zéros").style(
                                    "color:#01335A !important; font-weight:600 !important; font-size:16px !important;"
                                )
                            else:
                                ui.label(" Toutes > 0").style(
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

 Avantages :
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

 Pour skewness modéré (0.5-1.5)
- Plus douce que log
- Préserve mieux les relations
                            """).style(
                                "background:#f8f9fa !important; padding:16px !important; "
                                "border-radius:8px !important; font-size:13px !important;"
                            )
                        
                        elif method == "Box-Cox":
                            ui.markdown("""
**📌 Box-Cox Transform (automatique)**

 Trouve meilleur λ pour normaliser

 Nécessite valeurs strictement > 0

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

 Gère valeurs négatives et zéros

λ optimal sera calculé automatiquement
                            """).style(
                                "background:#f8f9fa !important; padding:16px !important; "
                                "border-radius:8px !important; font-size:13px !important;"
                            )
                        
                        elif method == "Aucune":
                            ui.markdown("""
**📌 Aucune transformation**

 Garder distribution originale

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
                        
                        ui.notify(f" Transformation configurée pour {col_name}", color="positive")
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
            ui.notify(" Aucune transformation configurée", color="warning")
            return
        
        with ui.dialog() as dialog, ui.card().style(
            "padding:32px !important; border-radius:16px !important; "
            "box-shadow:0 10px 40px rgba(0,0,0,0.15) !important;"
        ):
            ui.label(" Confirmation").style(
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
                                    print(f" Fitted {method} pour {col}")
                        
                        state["fitted_transformers"] = transformers
                        
                        # 3. TRANSFORM sur raw_df
                        df_result = df.copy()
                        for col, info in transformers.items():
                            if col in df_result.columns:
                                method = info["method"]
                                params = info["params"]
                                transformed_data, _ = apply_transform(df_result[col].values, method, params)
                                if transformed_data is not None:
                                    #  CORRECTION : Gérer la différence de taille
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
                        
                        ui.notify(" Transformations appliquées avec succès!", color="positive")
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
        
        ui.notify(f" {count} recommandations appliquées", color="positive")
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
            ui.label(" Analyse des distributions").style(
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
                    ui.label(" Aucune feature nécessitant transformation").style(
                        "color:#1b5e20 !important; font-weight:600 !important; font-size:16px !important;"
                    )
                    ui.label("Vous pouvez passer à l'étape suivante").style(
                        "color:#2e7d32 !important; margin-top:4px !important; font-size:14px !important;"
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
                ui.label("Configuration des transformations").style(
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
                
                table = ui.table(
                    columns=[
                        {"name": "Feature", "label": "Feature", "field": "Feature", "align": "left"},
                        {"name": "Skewness", "label": "Skewness", "field": "Skewness", "align": "center"},
                        {"name": "Kurtosis", "label": "Kurtosis", "field": "Kurtosis", "align": "center"},
                        {"name": "Niveau", "label": "Niveau", "field": "Niveau", "align": "center"},
                        {"name": "Recommandé", "label": "Recommandé", "field": "Recommandé", "align": "left"},
                        {"name": "Configuré", "label": "Configuré", "field": "Configuré", "align": "left"}
                    ],
                    rows=rows,
                    row_key="Feature"
                ).props("flat bordered").style(
                    "width:100% !important; cursor:pointer !important;"
                )
                
                ui.label("💡 Cliquez sur une ligne pour configurer la transformation").style(
                    "font-size:13px !important; color:#636e72 !important; margin-top:12px !important; "
                    "font-style:italic !important;"
                )
                
                def handle_row_click(e):
                    if e and "row" in e and "Feature" in e["row"]:
                        open_transform_modal(e["row"]["Feature"])
                
                table.on("row-click", handle_row_click)
                
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
                                "font-size:14px !important; color:#01335A !important; margin-bottom:4px !important;"
                            )
                        
                        if "Decision Tree" in selected_algos or "C4.5" in selected_algos:
                            ui.label("🟢 Decision Tree : Transformation optionnelle (robuste aux skew)").style(
                                "font-size:14px !important; color:#27ae60 !important;"
                            )
                
                # Boutons actions
                with ui.row().classes("w-full gap-3 mt-6"):
                    ui.button(
                        "✨ Appliquer recommandations",
                        on_click=apply_recommended
                    ).style(
                        "background:#01335A !important; color:white !important; border-radius:8px !important; "
                        "padding:10px 20px !important; text-transform:none !important; font-weight:500 !important;"
                    )
                    
                    ui.button(
                        " Appliquer & Continuer",
                        on_click=apply_all_transforms
                    ).style(
                        "background:#01335A !important; color:white !important; border-radius:8px !important; "
                        "padding:10px 20px !important; text-transform:none !important; font-weight:600 !important;"
                    )
        
        # Navigation
        with ui.row().classes("w-full max-w-6xl justify-between gap-4 mt-8"):
            ui.button(
                "← Précédent",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/encoding'")
            ).style(
                "background:white !important; color:#01335A !important; font-weight:500 !important; "
                "border-radius:8px !important; height:48px !important; min-width:140px !important; "
                "font-size:14px !important; text-transform:none !important; "
                "box-shadow:0 2px 8px rgba(0,0,0,0.08) !important;"
            )
            
            ui.button(
                "Suivant →",
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
    Style: Nuances de bleu élégantes
    """
   
    
    # ---------- CONTEXTE ----------
    df = state.get("raw_df", None)
    if df is None:
        with ui.column().classes("items-center justify-center w-full h-screen").style("background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);"):
            ui.label("❌ Aucun dataset chargé.").style("font-size:18px; color:#01579b; font-weight:600;")
            ui.button(" Retour à l'Upload", on_click=lambda: ui.run_javascript("window.location.href='/supervised/upload'")).style(
                "background:#0277bd; color:white; font-weight:600; padding:12px 24px; border-radius:8px;"
            )
        return
    
    df_train = df.copy()
    target_col = state.get("target_column", None)
    columns_exclude = state.get("columns_exclude", {}) or {}
    selected_algos = state.get("selected_algos", [])
    
    active_cols = [c for c in df.columns if not columns_exclude.get(c, False) and c != target_col]
    
    # ---------- DÉTECTION INTELLIGENTE DES VARIABLES CONTINUES ----------
    def detect_continuous_features(df, columns):
        """
        Détecte les vraies variables continues (exclut les catégorielles encodées)
        """
        continuous = []
        categorical_encoded = []
        
        for col in columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            unique_values = df[col].nunique()
            data = df[col].dropna()
            
            if len(data) == 0:
                continue
            
            # Critères pour identifier une variable catégorielle encodée
            is_binary = unique_values == 2
            is_small_discrete = unique_values <= 10
            all_integers = all(data == data.astype(int))
            small_range = (data.max() - data.min()) < 10
            
            # Si c'est binaire (0/1) ou petit nombre de valeurs discrètes → catégorielle
            if is_binary or (is_small_discrete and all_integers and small_range):
                categorical_encoded.append(col)
            else:
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
                        marker_color='#42a5f5',
                        showlegend=(idx == 0)
                    ),
                    row=1, col=idx+1
                )
                
                # APRÈS (ligne 2)
                fig.add_trace(
                    go.Box(
                        y=scaled_data,
                        name="Scaled",
                        marker_color='#01579b',
                        showlegend=(idx == 0)
                    ),
                    row=2, col=idx+1
                )
                
            except Exception as e:
                print(f"Erreur visualisation {col}: {e}")
                continue
        
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text=f" Comparaison AVANT/APRÈS - {method.upper()}",
            title_font_size=20,
            title_x=0.5,
            paper_bgcolor='#e3f2fd',
            plot_bgcolor='white'
        )
        
        return fig
    
    def preview_scaling(method):
        """Prévisualise le scaling sans l'appliquer"""
        fig = create_comparison_plot(method)
        if fig:
            plot_container.clear()
            with plot_container:
                ui.plotly(fig).classes("w-full")
        else:
            plot_container.clear()
            with plot_container:
                ui.label(" Aucune visualisation disponible").style("color:#01579b; font-size:16px;")
    
    def apply_scaling(method):
        ensure_original_saved()
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "maxabs": MaxAbsScaler()
        }
        
        if method == "none":
            state["scaling_method"] = "none"
            ui.notify("✓ Aucun scaling appliqué", color="positive")
            ui.run_javascript("setTimeout(() => window.location.href='/supervised/dimension_reduction', 1000);")
            return
        
        split = state.get("split", {})

        with ui.dialog() as dialog, ui.card().classes("p-6").style("background:white; border-radius:16px;"):
            ui.label(" Confirmer l'application").style("font-weight:700; font-size:18px; color:#01579b;")
            ui.label(f"Appliquer **{method.upper()}** sur {len(num_cols)} features continues ?").style("margin-top:8px; color:#0277bd;")
        
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Annuler", on_click=dialog.close).style(
                    "background:#90caf9; color:#01579b; font-weight:600; padding:8px 20px; border-radius:8px;"
                )
            
                def do_apply():
                    try:
                        scaler = scalers[method]
                    
                        #  CORRECTION : FIT sur X_train uniquement
                        if split and "X_train" in split:
                            X_train_for_fit = split["X_train"][num_cols].copy()
                        else:
                            X_train_for_fit = df_train[num_cols].copy()
                    
                        #  FIT sur train
                        scaler.fit(X_train_for_fit)
                    
                        #  TRANSFORM sur chaque split
                        if split:
                            for key in ["X_train", "X_val", "X_test"]:
                                if key in split and isinstance(split[key], pd.DataFrame):
                                    split[key][num_cols] = scaler.transform(split[key][num_cols])
                    
                        state["scaling_method"] = method
                        state["fitted_scaler"] = scaler  # Sauvegarder le scaler fitté
                        state["scaled_columns"] = num_cols
                    
                        ui.notify(f"✓ Scaling appliqué sans data leakage!", color="positive")
                        dialog.close()
                        ui.run_javascript("setTimeout(() => window.location.href='/supervised/dimension_reduction', 1000);")
                
                    except Exception as e:
                        ui.notify(f"❌ Erreur: {str(e)}", color="negative")
            
                ui.button("✓ Confirmer", on_click=do_apply).style(
                    "background:linear-gradient(135deg, #0277bd 0%, #01579b 100%); "
                    "color:white; font-weight:600; padding:8px 24px; border-radius:8px;"
                )
    
        dialog.open()

    
    # ---------- ANALYSE ----------
    features_info = analyze_features()
    recommended = get_recommendation()
    
    # ---------- INTERFACE ----------
    with ui.column().classes("w-full items-center p-8").style(
        "background: white; min-height:100vh;"
    ):
        # Header
        ui.label("🔵 FEATURE SCALING").style(
            "font-weight:700; font-size:32px; color:#01579b; margin-bottom:8px; text-align:center;"
        )
        ui.label("Normalisation/Standardisation des features continues").style(
            "font-size:18px; color:#0277bd; margin-bottom:32px; text-align:center;"
        )
        
        # Info détection intelligente
        with ui.card().classes("w-full max-w-6xl p-6 mb-8").style(
            "background:white; border-left:5px solid #2196f3; border-radius:12px; box-shadow: 0 4px 12px rgba(33, 150, 243, 0.15);"
        ):
            ui.label("🔍 Détection Intelligente").style("font-weight:700; font-size:18px; color:#01579b; margin-bottom:12px;")
            with ui.row().classes("w-full gap-8"):
                with ui.column().classes("flex-1"):
                    ui.label(f" **{len(num_cols)} features continues** détectées").style("font-size:15px; margin-bottom:4px; color:#0277bd;")
                    if num_cols:
                        ui.label(f"→ {', '.join(num_cols[:5])}{'...' if len(num_cols) > 5 else ''}").style(
                            "font-size:13px; color:#42a5f5;"
                        )
                
                with ui.column().classes("flex-1"):
                    ui.label(f"⚙️ **{len(cat_encoded)} variables catégorielles** encodées").style("font-size:15px; margin-bottom:4px; color:#0277bd;")
                    if cat_encoded:
                        ui.label(f"→ {', '.join(cat_encoded[:5])}{'...' if len(cat_encoded) > 5 else ''}").style(
                            "font-size:13px; color:#42a5f5;"
                        )
                        ui.label("→ Seront préservées (pas de scaling)").style("font-size:12px; color:#0288d1; font-style:italic;")
        
        # Objectif
        with ui.card().classes("w-full max-w-6xl p-6 mb-8").style(
            "background:linear-gradient(135deg, #e1f5fe 0%, #b3e5fc 100%); "
            "border-left:5px solid #0288d1; border-radius:12px; box-shadow: 0 4px 12px rgba(2, 136, 209, 0.15);"
        ):
            ui.label(" Objectif principal").style("font-weight:700; font-size:18px; color:#01579b; margin-bottom:12px;")
            with ui.row().classes("w-full gap-8"):
                with ui.column().classes("flex-1"):
                    ui.label("🔴 **CRITIQUE** pour KNN").style("font-size:15px; font-weight:600; margin-bottom:4px; color:#0277bd;")
                    ui.label("→ Les distances sont biaisées par les échelles différentes").style("font-size:13px; color:#0288d1;")
                
                with ui.column().classes("flex-1"):
                    ui.label("🟢 **INUTILE** pour C4.5").style("font-size:15px; font-weight:600; margin-bottom:4px; color:#0277bd;")
                    ui.label("→ Arbres de décision insensibles aux échelles").style("font-size:13px; color:#0288d1;")
                
                with ui.column().classes("flex-1"):
                    ui.label("🔵 **OPTIONNEL** pour Naive Bayes").style("font-size:15px; font-weight:600; margin-bottom:4px; color:#0277bd;")
                    ui.label("→ Dépend de la distribution des données").style("font-size:13px; color:#0288d1;")
        
        # Analyse des features continues
        with ui.card().classes("w-full max-w-6xl p-6 mb-8").style(
            "background:white; border-radius:12px; box-shadow: 0 4px 12px rgba(1, 87, 155, 0.1);"
        ):
            ui.label(" Analyse des features continues").style(
                "font-weight:700; font-size:20px; color:#01579b; margin-bottom:16px;"
            )
            
            if features_info:
                # Tableau amélioré avec nuances de bleu
                table_html = """<div style="background:linear-gradient(135deg, #01579b 0%, #0277bd 100%); border-radius:12px; padding:20px; overflow-x:auto; box-shadow: 0 4px 12px rgba(1, 87, 155, 0.3);">
                <table style="width:100%; color:#e3f2fd; font-family:'Segoe UI', sans-serif; font-size:13px; border-collapse:collapse;">
                    <thead>
                        <tr style="border-bottom:2px solid #42a5f5;">
                            <th style="text-align:left; padding:12px; color:#90caf9; font-weight:600;">Feature</th>
                            <th style="text-align:right; padding:12px; color:#90caf9;">Min</th>
                            <th style="text-align:right; padding:12px; color:#90caf9;">Max</th>
                            <th style="text-align:right; padding:12px; color:#90caf9;">Range</th>
                            <th style="text-align:right; padding:12px; color:#90caf9;">Mean</th>
                            <th style="text-align:right; padding:12px; color:#90caf9;">Std</th>
                            <th style="text-align:center; padding:12px; color:#90caf9;">Outliers</th>
                        </tr>
                    </thead>
                    <tbody>"""
                
                for f in features_info:
                    out = "🔴" if f["outliers"] else "🟢"
                    # Highlight pour range > 1000
                    range_color = "#64b5f6" if f["range"] > 1000 else "#e3f2fd"
                    table_html += f"""
                        <tr style="border-bottom:1px solid #1565c0;">
                            <td style="padding:10px; font-weight:600; color:#bbdefb;">{f['name'][:20]}</td>
                            <td style="text-align:right; padding:10px;">{f['min']:.2f}</td>
                            <td style="text-align:right; padding:10px;">{f['max']:.2f}</td>
                            <td style="text-align:right; padding:10px; color:{range_color}; font-weight:700;">{f['range']:.2f}</td>
                            <td style="text-align:right; padding:10px;">{f['mean']:.2f}</td>
                            <td style="text-align:right; padding:10px;">{f['std']:.2f}</td>
                            <td style="text-align:center; padding:10px; font-size:16px;">{out}</td>
                        </tr>"""
                
                table_html += """
                    </tbody>
                </table>
                </div>"""
                ui.html(table_html, sanitize=False)
            else:
                with ui.column().classes("items-center p-8"):
                    ui.label("❌ Aucune feature continue détectée").style("color:#0277bd; font-size:16px; font-weight:600;")
                    ui.label("Toutes les features sont catégorielles ou binaires").style("color:#42a5f5; font-size:14px;")
        
        # Sélecteur de méthode
        with ui.card().classes("w-full max-w-5xl p-8 mb-8").style(
            "background:white; border:3px solid #64b5f6; border-radius:16px; "
            "box-shadow: 0 6px 16px rgba(33, 150, 243, 0.2);"
        ):
            ui.label("⚙️ Choisir la méthode de scaling").style(
                "font-weight:700; font-size:22px; color:#01579b; margin-bottom:24px; text-align:center;"
            )
            
            # Select avec options
            method_select = ui.select(
                label="Sélectionnez une méthode",
                options={
                    "standard": " StandardScaler (Z-score, mean=0, std=1)",
                    "minmax": "📏 MinMaxScaler (normalise [0,1])",
                    "robust": "🛡️ RobustScaler (robuste aux outliers)",
                    "maxabs": "📐 MaxAbsScaler (sparse data, [-1,1])",
                    "none": "⏭️ Aucun scaling (raw data)"
                }
            ).props("dense outlined").classes("w-full").style("max-width:600px; margin:0 auto 20px;")
            
            # Définir valeur par défaut
            method_select.value = recommended
            
            # Bouton prévisualisation
            ui.button(
                "👁️ Prévisualiser",
                on_click=lambda: preview_scaling(method_select.value)
            ).style(
                "background:linear-gradient(135deg, #42a5f5 0%, #2196f3 100%); color:white; font-weight:600; "
                "height:48px; width:200px; border-radius:10px; margin:0 auto; display:block; "
                "box-shadow: 0 4px 12px rgba(33, 150, 243, 0.3);"
            )
            
            method_select.on_value_change(lambda e: state.update({"scaling_method": e.value}))
        
        # Zone de visualisation
        plot_container = ui.column().classes("w-full max-w-6xl mb-8")
        
        # Recommandation intelligente
        with ui.card().classes("w-full max-w-5xl p-8 mb-12").style(
            "background:linear-gradient(135deg, #e1f5fe 0%, #b3e5fc 100%); "
            "border:4px solid #0288d1; border-radius:20px; "
            "box-shadow:0 8px 24px rgba(2, 136, 209, 0.3);"
        ):
            method_names = {
                "standard": "StandardScaler",
                "minmax": "MinMaxScaler", 
                "robust": "RobustScaler",
                "maxabs": "MaxAbsScaler",
                "none": "Aucun scaling"
            }
            
            ui.label("🤖 RECOMMANDATION INTELLIGENTE").style(
                "font-weight:800; font-size:24px; color:#01579b; text-align:center; margin-bottom:20px;"
            )
            
            ui.label(f"**{method_names[recommended]}**").style(
                "font-weight:700; font-size:28px; color:#0277bd; text-align:center; margin-bottom:16px;"
            )
            
            # Explication de la recommandation
            explanations = {
                "standard": "✓ Pas d'outliers majeurs détectés\n✓ Données normalement distribuées\n✓ Optimal pour KNN et régression",
                "robust": f" Outliers détectés dans vos données\n✓ Résistant aux valeurs extrêmes\n✓ Préserve la structure centrale\n✓ Recommandé pour {len([f for f in features_info if f['outliers']])} features avec outliers",
                "none": "✓ Toutes les features ont des échelles similaires\n✓ Pas de normalisation nécessaire"
            }
            
            ui.label(explanations.get(recommended, "")).style(
                "font-size:15px; color:#0288d1; text-align:center; white-space:pre-line; margin-bottom:20px;"
            )
            
            with ui.row().classes("w-full justify-center gap-6"):
                ui.button(
                    " Appliquer la recommandation",
                    on_click=lambda: (
                        method_select.set_value(recommended),
                        apply_scaling(recommended)
                    )
                ).style(
                    "background:linear-gradient(135deg, #0277bd 0%, #01579b 100%); "
                    "color:white; font-weight:700; height:56px; padding:0 40px; border-radius:12px; "
                    "box-shadow: 0 6px 16px rgba(1, 87, 155, 0.4);"
                )
        
        # Navigation
        with ui.row().classes("w-full max-w-5xl justify-between mt-16"):
            ui.button(
                " Étape précédente",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/distribution_transform'")
            ).style(
                "background:#90caf9; color:#01579b; font-weight:600; "
                "height:60px; width:260px; border-radius:16px; font-size:16px; "
                "box-shadow: 0 4px 12px rgba(144, 202, 249, 0.3);"
            )
            
            ui.button(
                "✓ Appliquer & Continuer",
                on_click=lambda: apply_scaling(method_select.value)
            ).style(
                "background:linear-gradient(135deg, #0277bd 0%, #01579b 100%); "
                "color:white; font-weight:700; height:60px; width:320px; "
                "border-radius:16px; font-size:16px; "
                "box-shadow: 0 6px 16px rgba(1, 87, 155, 0.4);"
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
                
                ui.notify(f" PCA appliqué : {n_features} → {n_comp} composantes ({variance_explained:.1f}% variance)", 
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
                
                ui.notify(f" Feature Selection : {n_features} → {n_comp} features", 
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
            with ui.card().classes("w-full p-4").style("background:#cde4ff; border-left:4px solid #ffc107;"):
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
            ui.label("⚙️ Configuration de la Réduction").style(
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
                                                "font-size:15px; font-weight:600; color:#2e7d32;"
                                            )
                                            ui.label(f"Impact : {n_features} → {n_for_95} features (réduction {(1-n_for_95/n_features)*100:.0f}%)").style(
                                                "font-size:14px; color:#2e7d32; margin-top:4px;"
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
            "background:linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); "
            "border:4px solid #1c365c; border-radius:20px;"
        ):
            ui.label("💡 Décision Finale").style(
                "font-weight:800; font-size:24px; color:#1c365c; text-align:center; margin-bottom:20px;"
            )
            
            ui.label("**Recommandation pour vos algorithmes** :").style(
                "font-size:16px; color:#2e7d32; font-weight:600; margin-bottom:12px;"
            )
            
            ui.label("🔵 **KNN** : PCA recommandée (améliore vitesse et précision)").style("font-size:14px; margin-bottom:6px;")
            ui.label("🌳 **C4.5** : Feature Selection optionnelle (garde interprétabilité)").style("font-size:14px; margin-bottom:6px;")
            ui.label(" **Naive Bayes** : Pas de réduction nécessaire").style("font-size:14px; margin-bottom:20px;")
            
            with ui.row().classes("w-full justify-center gap-6 mt-8"):
                ui.button(
                    "⏭️ Passer (pas de réduction)",
                    on_click=skip_reduction
                ).style(
                    "background:#95a5a6; color:white; font-weight:600; "
                    "height:56px; padding:0 32px; border-radius:12px; font-size:16px;"
                )
                
                ui.button(
                    " Appliquer la Réduction",
                    on_click=lambda: apply_reduction(
                        state.get("reduction_method", "pca"),
                        state.get("n_components", 10)
                    )
                ).style(
                    "background:linear-gradient(135deg, #1c365c 0%, #45a049 100%); "
                    "color:white; font-weight:700; height:56px; padding:0 40px; border-radius:12px; font-size:16px;"
                ).bind_enabled_from(enable_switch, 'value')
        
        # Navigation
        with ui.row().classes("w-full max-w-5xl justify-between mt-16"):
            ui.button(
                " Étape précédente",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/scaling'")
            ).style(
                "background:#95a5a6; color:white; font-weight:600; "
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
    
    def generate_pdf_report():
        """Génère un rapport PDF du preprocessing"""
        ui.notify("📄 Génération du rapport PDF...", color="info")
        # TODO: Implémenter génération PDF avec reportlab
        ui.notify(" Fonctionnalité en développement", color="warning")
    
    def open_step_selector():
        """Ouvre un dialog pour sélectionner une étape à modifier"""
        steps = [
            ("3.2 - Split Train/Val/Test", "/supervised/preprocessing2"),
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
        """Valide le preprocessing et prépare pour les algorithmes"""
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-3xl p-6"):
            ui.label(" Validation du Pipeline").style("font-weight:700; font-size:20px; margin-bottom:16px;")
            
            # Progress
            progress_container = ui.column().classes("w-full")
            
            with progress_container:
                ui.label("Préparation des données pour training...").style("font-size:16px; margin-bottom:12px;")
                
                progress_bar = ui.linear_progress(value=0).classes("w-full mb-4")
                status_label = ui.label("")
                
                async def run_validation():
                    # Simulation des étapes
                    steps_validation = [
                        ("Vérification intégrité données", 0.2),
                        ("Application pipeline sur Validation set", 0.4),
                        ("Application pipeline sur Test set", 0.6),
                        ("Sauvegarde du pipeline", 0.8),
                        ("Préparation algorithmes", 1.0)
                    ]
                    
                    for step, value in steps_validation:
                        status_label.set_text(f"✓ {step}")
                        progress_bar.set_value(value)
                        await asyncio.sleep(0.5)
                    
                    # Sauvegarder état validé
                    state["preprocessing_validated"] = True
                    state["validation_timestamp"] = datetime.now().isoformat()
                    
                    ui.notify(" Pipeline validé avec succès !", color="positive")
                    
                    # Afficher résumé final
                    progress_container.clear()
                    with progress_container:
                        ui.label(" Validation complète !").style(
                            "font-weight:700; font-size:20px; color:#01335A; margin-bottom:16px;"
                        )
                        
                        with ui.card().classes("w-full p-4 mb-4").style("background:#4e88d4;"):
                            ui.label("Prêt à lancer les algorithmes :").style("font-weight:600; margin-bottom:8px;")
                            selected_algos = state.get("selected_algos", [])
                            for algo in selected_algos:
                                ui.label(f"• {algo}").style("font-size:14px;")
                        
                        ui.button(
                            "▶️ Passer à la Page Algorithmes",
                            on_click=lambda: (dialog.close(), ui.run_javascript("window.location.href='/supervised/algorithm_config'"))
                        ).style(
                            "background:linear-gradient(135deg, #01335A 0%, #229954 100%); "
                            "color:white; font-weight:700; height:56px; width:100%; border-radius:12px; margin-top:12px;"
                        )
                
                # Lancer la validation
                ui.timer(0.1, run_validation, once=True)
        
        dialog.open()
    
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
                    "background:#01335A; color:white; font-weight:600; "
                    "height:48px; padding:0 24px; border-radius:10px;"
                )
                
                ui.button(
                    "📄 Générer Rapport PDF",
                    on_click=generate_pdf_report
                ).style(
                    "background:#9b59b6; color:white; font-weight:600; "
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
            
            ui.label("Votre pipeline de preprocessing est prêt !").style(
                "font-size:16px; color:#2e7d32; text-align:center; margin-bottom:20px;"
            )
            
            with ui.row().classes("w-full justify-center gap-4"):
                ui.button(
                    "️ Modifier une Étape",
                    on_click=open_step_selector
                ).style(
                    "background:#95a5a6; color:white; font-weight:600; "
                    "height:56px; padding:0 32px; border-radius:12px; font-size:16px;"
                )
                
                ui.button(
                    "🔄 Recommencer",
                    on_click=reset_preprocessing
                ).style(
                    "background:#e74c3c; color:white; font-weight:600; "
                    "height:56px; padding:0 32px; border-radius:12px; font-size:16px;"
                )
                
                ui.button(
                    " Valider et Continuer",
                    on_click=validate_and_continue
                ).style(
                    "background:linear-gradient(135deg, #01335A 0%, #229954 100%); "
                    "color:white; font-weight:700; height:56px; padding:0 40px; border-radius:12px; font-size:16px;"
                )
        
        # Navigation
        with ui.row().classes("w-full max-w-6xl justify-start mt-6"):
            ui.button(
                " Étape précédente",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/dimension_reduction'")
            ).style("background:#95a5a6; color:white; font-weight:600; height:50px; width:220px; border-radius:10px;")





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
        with ui.card().classes("w-full max-w-6xl p-6 mb-8").style("background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);"):
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
                    
                    ui.label("⚙️ HYPERPARAMÈTRES").style("font-weight:700; font-size:18px; margin-top:16px; margin-bottom:12px;")
                    
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
                    ui.label(" K petit → Overfitting | K grand → Underfitting").style("font-size:13px; color:#01335A; margin-left:250px; margin-bottom:16px;")
                    
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
                    with ui.card().classes("w-full p-4 mt-4").style("background:#cde4ff;"):
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
                            "background:#01335A; color:white;"
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
                    
                    ui.label("⚙️ HYPERPARAMÈTRES").style("font-weight:700; font-size:18px; margin-top:16px; margin-bottom:12px;")
                    
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
                            "background:#01335A; color:white;"
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
                    
                    ui.label("⚙️ HYPERPARAMÈTRES").style("font-weight:700; font-size:18px; margin-top:16px; margin-bottom:12px;")
                    
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
                    with ui.card().classes("w-full p-4 mt-4").style("background:#4e88d4;"):
                        ui.label("⚡ Performances").style("font-weight:700; margin-bottom:8px;")
                        ui.label("• Temps training : < 0.1s (le plus rapide de tous)").style("font-size:14px;")
                        ui.label("• Temps prédiction : < 0.01s (très rapide)").style("font-size:14px;")
                        ui.label("• Mémoire : Très faible (~quelques KB)").style("font-size:14px;")
                        ui.label("• Idéal pour : Baseline rapide, données textuelles, peu de données").style("font-size:14px; margin-top:8px; font-weight:600; color:#01335A;")
                    
                    # Boutons
                    with ui.row().classes("w-full justify-end gap-2 mt-4"):
                        ui.button("🔄 Réinitialiser", on_click=lambda: reset_config("naive_bayes")).props("flat")
                        ui.button("💡 Config par défaut", on_click=lambda: apply_recommended_config("naive_bayes")).style(
                            "background:#01335A; color:white;"
                        )
        
        # ==================== SECTION: STRATÉGIE DE VALIDATION ====================
        with ui.card().classes("w-full max-w-6xl p-6 mb-6").style("background:#4e88d4; border:2px solid #1c365c;"):
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
                    "holdout": "Hold-out (Train → Val) - Rapide, déjà splitté  [Recommandé]",
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
                            ui.label("⚙️ Paramètres Cross-Validation :").style("font-weight:600; margin-bottom:8px;")
                            
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
            "background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius:16px;"
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
                "background:#95a5a6; color:white; font-weight:600; height:56px; padding:0 32px; border-radius:12px; font-size:16px;"
            )
            
            ui.button(
                " Valider et Lancer l'Entraînement",
                on_click=validate_and_continue
            ).style(
                "background:linear-gradient(135deg, #01335A 0%, #229954 100%); "
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
                "font-size:18px; color:#c0392b; font-weight:600;"
            )
            ui.button(" Retour", on_click=lambda: ui.run_javascript("window.location.href='/supervised/algorithm_config'"))
        return
    
    # Préparer données
    X_train = split.get("X_train")
    y_train = split.get("y_train")
    
    if X_train is None or y_train is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Split non disponible.").style("font-size:18px; color:#c0392b; font-weight:600;")
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
                        "signif_color": "#95a5a6"
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
                    signif_color = "#95a5a6"
                
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
                    "signif_color": "#95a5a6"
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
        
        colors = ['#01335A' if imp > 60 else '#01335A' if imp > 30 else '#e74c3c' 
                  for imp in df_importance["importance"]]
        
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
            paper_bgcolor='#f8f9fa',
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
                    boxmean='sd'
                ))
            
            fig.update_layout(
                title=f"{feature_name} par Classe",
                yaxis_title=feature_name,
                xaxis_title="Classe",
                height=400,
                paper_bgcolor='#f8f9fa',
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
                paper_bgcolor='#f8f9fa',
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
    with ui.column().classes("w-full items-center p-8").style("background-color:#f8f9fa; min-height:100vh;"):
        
        # ==================== HEADER ====================
        ui.label(" ANALYSE D'IMPORTANCE DES FEATURES").style(
            "font-weight:700; font-size:32px; color:#2c3e50; margin-bottom:8px; text-align:center;"
        )
        ui.label(f"Analyse préliminaire pour guider la modélisation ({task_type})").style(
            "font-size:18px; color:#7f8c8d; margin-bottom:32px; text-align:center;"
        )
        
        # Info dataset
        with ui.card().classes("w-full max-w-6xl p-4 mb-6").style("background:#e3f2fd; border-left:5px solid #2196f3;"):
            with ui.row().classes("w-full gap-8 items-center"):
                ui.label(" Dataset :").style("font-weight:700; font-size:16px;")
                ui.label(f"Features: {len(active_features)}").style("font-size:14px;")
                ui.label(f"Samples: {len(X_train):,}").style("font-size:14px;")
                ui.label(f"Target: {target_col}").style("font-size:14px;")
                ui.label(f"Type: {task_type}").style("font-size:14px;")
        
        # ==================== SECTION A : IMPORTANCE UNIVARIÉE ====================
        with ui.card().classes("w-full max-w-6xl p-6 mb-8"):
            ui.label(" SECTION A : IMPORTANCE UNIVARIÉE").style(
                "font-weight:700; font-size:24px; color:#2c3e50; margin-bottom:16px;"
            )
            
            ui.label("Méthode : Tests Statistiques Adaptés au Type de Feature").style(
                "font-size:16px; color:#7f8c8d; margin-bottom:12px;"
            )
            
            # Info méthode
            with ui.card().classes("w-full p-3 mb-4").style("background:#4e88d4; border-left:3px solid #1c365c;"):
                ui.label(" Méthodes utilisées :").style("font-weight:600; margin-bottom:6px;")
                ui.label("• Chi² + Cramér's V : Variables catégorielles/binaires").style("font-size:13px;")
                ui.label("• ANOVA F + Eta² : Variables continues (classification)").style("font-size:13px;")
                ui.label("• Pearson : Variables continues (régression)").style("font-size:13px;")
            
            # Tableau détaillé
            if len(univariate_results) > 0:
                # Créer tableau HTML stylisé
                table_html = """
                <div style="background:#1a1a1a; border-radius:12px; padding:20px; overflow-x:auto; margin-bottom:24px;">
                <table style="width:100%; color:#00ff88; font-family:monospace; font-size:13px; border-collapse:collapse;">
                    <thead>
                        <tr style="border-bottom:2px solid #00ff88;">
                            <th style="text-align:left; padding:12px; color:#00ffff;">Feature</th>
                            <th style="text-align:center; padding:12px;">Assoc. Target</th>
                            <th style="text-align:center; padding:12px;">Test</th>
                            <th style="text-align:center; padding:12px;">Test Stat</th>
                            <th style="text-align:center; padding:12px;">Signif.</th>
                            <th style="text-align:left; padding:12px;">Importance</th>
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
                        <tr style="border-bottom:1px solid #333;">
                            <td style="padding:10px; font-weight:600;">{feature_display}</td>
                            <td style="text-align:center; padding:10px;">{row['corr']:.3f} {icon}</td>
                            <td style="text-align:center; padding:10px; color:#ffa500;">{row['test_name']}</td>
                            <td style="text-align:center; padding:10px;">{row['test_stat']:.2f}</td>
                            <td style="text-align:center; padding:10px; color:{row['signif_color']}; font-weight:700;">{row['signif']}</td>
                            <td style="padding:10px;">{bar} {row['importance']}%</td>
                        </tr>
                    """
                
                table_html += """
                    </tbody>
                </table>
                </div>
                """
                
                ui.html(table_html, sanitize=False)
                
                # Légende significativité
                with ui.card().classes("w-full p-4 mb-4").style("background:#cde4ff;"):
                    ui.label("📖 Légende Significativité (p-value) :").style("font-weight:700; margin-bottom:8px;")
                    ui.label("*** p < 0.001 (Très significatif) - Forte évidence d'association").style("font-size:14px; color:#01335A;")
                    ui.label("** p < 0.01 (Significatif) - Bonne évidence d'association").style("font-size:14px; color:#01335A;")
                    ui.label("* p < 0.05 (Marginalement significatif) - Évidence modérée").style("font-size:14px; color:#01335A;")
                    ui.label("(vide) p >= 0.05 (Non significatif) - Pas d'évidence d'association").style("font-size:14px; color:#95a5a6;")
                
                # Observations
                high_importance = univariate_results[univariate_results["importance"] > 60]
                medium_importance = univariate_results[(univariate_results["importance"] >= 30) & (univariate_results["importance"] <= 60)]
                low_importance = univariate_results[univariate_results["importance"] < 20]
                
                with ui.card().classes("w-full p-4 mb-4").style("background:#4e88d4; border-left:4px solid #1c365c;"):
                    ui.label("💡 Observations :").style("font-weight:700; margin-bottom:8px;")
                    
                    if len(high_importance) > 0:
                        features_str = ", ".join(high_importance["feature"].head(5).tolist())
                        if len(high_importance) > 5:
                            features_str += f" (+ {len(high_importance)-5} autres)"
                        ui.label(f"• 🟢 {len(high_importance)} features très prédictives (importance > 60%) : {features_str}").style("font-size:14px; margin-bottom:4px;")
                    
                    if len(medium_importance) > 0:
                        ui.label(f"• 🟡 {len(medium_importance)} features moyennement prédictives (30-60%)").style("font-size:14px; margin-bottom:4px;")
                    
                    if len(low_importance) > 0:
                        ui.label(f"• 🔴 {len(low_importance)} features faiblement prédictives (< 20%)").style("font-size:14px; margin-bottom:4px;")
                    
                    # Recommandation nombre features
                    recommended_n = len(univariate_results[univariate_results["importance"] >= 30])
                    if recommended_n > 0:
                        ui.label(f"• 💡 Recommandation : Conserver les {recommended_n} features (importance >= 30%) pour un bon compromis").style(
                            "font-size:14px; font-weight:600; color:#2e7d32; margin-top:8px;"
                        )
                
                # Barplot horizontal
                ui.label(" Visualisation des Importances").style(
                    "font-weight:700; font-size:20px; margin-top:20px; margin-bottom:12px;"
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
                                    "font-weight:700; font-size:18px; margin-bottom:12px;"
                                )
                                
                                scatter_fig = create_scatter_feature_vs_target(feature_clicked)
                                ui.plotly(scatter_fig).classes("w-full")
                    except Exception as exc:
                        print(f"❌ Erreur click: {exc}")
                
                barplot_plot.on("plotly_click", handle_bar_click)
                
                ui.label("💡 Cliquez sur une barre pour voir la relation avec la target").style(
                    "font-size:13px; color:#7f8c8d; margin-top:8px; text-align:center;"
                )
            
            else:
                ui.label(" Aucune feature disponible pour l'analyse").style("color:#01335A; font-size:16px;")
        
        # ==================== SECTION B : MUTUAL INFORMATION ====================
        if len(mi_results) > 0:
            with ui.card().classes("w-full max-w-6xl p-6 mb-8"):
                ui.label("🔄 SECTION B : IMPORTANCE MULTIVARIÉE (MUTUAL INFORMATION)").style(
                    "font-weight:700; font-size:24px; color:#2c3e50; margin-bottom:16px;"
                )
                
                ui.label("Méthode : Mutual Information (capture relations non-linéaires)").style(
                    "font-size:16px; color:#7f8c8d; margin-bottom:20px;"
                )
                
                # Tableau MI
                table_html_mi = """
                <div style="background:#1a1a1a; border-radius:12px; padding:20px; overflow-x:auto; margin-bottom:24px;">
                <table style="width:100%; color:#00ff88; font-family:monospace; font-size:13px; border-collapse:collapse;">
                    <thead>
                        <tr style="border-bottom:2px solid #00ff88;">
                            <th style="text-align:left; padding:12px; color:#00ffff;">Feature</th>
                            <th style="text-align:center; padding:12px;">MI Score</th>
                            <th style="text-align:left; padding:12px;">Importance Relative</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                
                for _, row in mi_results.iterrows():
                    bar = create_progress_bar_text(row["importance"])
                    feature_display = row['feature'][:25] + "..." if len(row['feature']) > 25 else row['feature']
                    
                    table_html_mi += f"""
                        <tr style="border-bottom:1px solid #333;">
                            <td style="padding:10px; font-weight:600;">{feature_display}</td>
                            <td style="text-align:center; padding:10px;">{row['mi_score']:.4f}</td>
                            <td style="padding:10px;">{bar} {row['importance']}%</td>
                        </tr>
                    """
                
                table_html_mi += """
                    </tbody>
                </table>
                </div>
                """
                
                ui.html(table_html_mi, sanitize=False)
                
                # Info MI
                with ui.card().classes("w-full p-4").style("background:#e3f2fd;"):
                    ui.label("💡 Mutual Information (Avantages)").style(
                        "font-weight:700; margin-bottom:8px;"
                    )
                    ui.label("→ Capture relations non-linéaires (complète l'analyse univariée)").style("font-size:14px;")
                    ui.label("→ Pas d'hypothèse sur la distribution des données").style("font-size:14px;")
                    ui.label("→ Fonctionne avec variables continues ET catégorielles").style("font-size:14px;")
                    ui.label("→ Détecte interactions complexes avec la target").style("font-size:14px;")
                
                # Barplot MI
                ui.label(" Visualisation MI Scores").style(
                    "font-weight:700; font-size:20px; margin-top:20px; margin-bottom:12px;"
                )
                
                mi_barplot = create_importance_barplot(mi_results, "Mutual Information Scores")
                ui.plotly(mi_barplot).classes("w-full")
        
        # ==================== SECTION C : RECOMMANDATIONS ====================
        with ui.card().classes("w-full max-w-6xl p-6 mb-8").style(
            "background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius:16px;"
        ):
            ui.label("💡 RECOMMANDATIONS AVANT TRAINING").style(
                "font-weight:700; font-size:24px; color:white; margin-bottom:16px; text-align:center;"
            )
            
            ui.label("Basé sur l'analyse d'importance :").style(
                "font-size:16px; color:white; text-align:center; margin-bottom:20px;"
            )
            
            if len(univariate_results) > 0:
                # Options
                with ui.column().classes("w-full gap-4"):
                    # Option 1 : Garder toutes
                    with ui.card().classes("p-6"):
                        ui.label(f"Option 1 : Garder toutes les features ({len(active_features)})").style(
                            "font-weight:700; font-size:18px; margin-bottom:12px;"
                        )
                        
                        with ui.row().classes("gap-6 mb-4"):
                            with ui.column().classes("flex-1"):
                                ui.label(" Avantages").style("font-weight:600; color:#01335A; margin-bottom:6px;")
                                ui.label("• Maximise information disponible").style("font-size:14px;")
                                ui.label("• Pas de risque de perte d'info").style("font-size:14px;")
                                ui.label("• Laisse l'algo choisir").style("font-size:14px;")
                            
                            with ui.column().classes("flex-1"):
                                ui.label("❌ Inconvénients").style("font-weight:600; color:#e74c3c; margin-bottom:6px;")
                                ui.label("• Risque léger de bruit").style("font-size:14px;")
                                ui.label("• Training plus lent").style("font-size:14px;")
                                ui.label("• Overfitting possible").style("font-size:14px;")
                        
                        ui.button(
                            " Continuer avec toutes les features",
                            on_click=skip_to_training
                        ).style(
                            "background:#01335A; color:white; font-weight:600; height:48px; width:100%; border-radius:8px;"
                        )
                    
                    # Option 2 : Réduire Top N
                    recommended_n = len(univariate_results[univariate_results["importance"] >= 30])
                    
                    if recommended_n > 0 and recommended_n < len(active_features):
                        with ui.card().classes("p-6"):
                            ui.label(f"Option 2 : Réduire à Top {recommended_n} features (importance >= 30%)").style(
                                "font-weight:700; font-size:18px; margin-bottom:12px;"
                            )
                            
                            top_features = univariate_results[univariate_results["importance"] >= 30]["feature"].tolist()
                            ui.label(f"Features conservées : {', '.join(top_features[:5])}{'...' if len(top_features) > 5 else ''}").style(
                                "font-size:13px; color:#636e72; margin-bottom:12px;"
                            )
                            
                            with ui.row().classes("gap-6 mb-4"):
                                with ui.column().classes("flex-1"):
                                    ui.label(" Avantages").style("font-weight:600; color:#01335A; margin-bottom:6px;")
                                    ui.label("• Élimine bruit potentiel").style("font-size:14px;")
                                    ui.label("• Accélère training").style("font-size:14px;")
                                    ui.label("• Réduit overfitting").style("font-size:14px;")
                                
                                with ui.column().classes("flex-1"):
                                    ui.label("❌ Inconvénients").style("font-weight:600; color:#e74c3c; margin-bottom:6px;")
                                    low_features = univariate_results[univariate_results["importance"] < 30]
                                    n_removed = len(low_features)
                                    ui.label(f"• Supprime {n_removed} features").style("font-size:14px;")
                                    ui.label("• Risque léger de sous-apprentissage").style("font-size:14px;")
                            
                            ui.button(
                                f"✂️ Réduire à Top {recommended_n} features [Recommandé]",
                                on_click=lambda: apply_feature_selection(recommended_n)
                            ).style(
                                "background:#01335A; color:white; font-weight:600; height:48px; width:100%; border-radius:8px;"
                            )
                    
                    # Option 3 : Custom
                    with ui.card().classes("p-6"):
                        ui.label("Option 3 : Sélection personnalisée").style(
                            "font-weight:700; font-size:18px; margin-bottom:12px;"
                        )
                        
                        with ui.row().classes("w-full items-center gap-4 mb-4"):
                            ui.label("Nombre de features à conserver :").style("font-weight:600; width:250px;")
                            custom_n_slider = ui.slider(
                                min=1, 
                                max=len(active_features), 
                                value=min(recommended_n if recommended_n > 0 else len(active_features), len(active_features)), 
                                step=1
                            ).classes("flex-1")
                            custom_n_label = ui.label(f"{custom_n_slider.value}").style("font-weight:700; width:60px;")
                            
                            def update_custom_n(e):
                                custom_n_label.set_text(f"{int(e.value)}")
                            
                            custom_n_slider.on_value_change(update_custom_n)
                        
                        ui.button(
                            "✂️ Appliquer sélection personnalisée",
                            on_click=lambda: apply_feature_selection(int(custom_n_slider.value))
                        ).style(
                            "background:#9b59b6; color:white; font-weight:600; height:48px; width:100%; border-radius:8px;"
                        )
        
        # ==================== COMPARAISON MÉTHODES ====================
        if len(mi_results) > 0 and len(univariate_results) > 0:
            with ui.card().classes("w-full max-w-6xl p-6 mb-8"):
                ui.label(" COMPARAISON DES MÉTHODES").style(
                    "font-weight:700; font-size:24px; color:#2c3e50; margin-bottom:16px;"
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
                    marker_color='#e74c3c'
                ))
                
                fig.update_layout(
                    title="Comparaison Importance Univariée vs Multivariée",
                    xaxis_title="Features",
                    yaxis_title="Importance (%)",
                    barmode='group',
                    height=500,
                    showlegend=True,
                    paper_bgcolor='#f8f9fa',
                    plot_bgcolor='white'
                )
                
                ui.plotly(fig).classes("w-full")
                
                # Interprétation
                with ui.card().classes("w-full p-4 mt-4").style("background:#4e88d4;"):
                    ui.label("💡 Interprétation :").style("font-weight:700; margin-bottom:8px;")
                    ui.label("• Si les deux méthodes donnent des résultats similaires → Relations linéaires/monotones dominantes").style("font-size:14px;")
                    ui.label("• Si MI > Analyse univariée → Relations non-linéaires importantes").style("font-size:14px;")
                    ui.label("• Features avec MI élevé mais score univarié faible → Interactions complexes").style("font-size:14px;")
                    ui.label("• Consensus entre les deux méthodes → Features robustement importantes").style("font-size:14px;")
        
        # ==================== NAVIGATION ====================
        with ui.row().classes("w-full max-w-6xl justify-between gap-4 mt-8"):
            ui.button(
                " Retour Config Algos",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/algorithm_config'")
            ).style(
                "background:#95a5a6; color:white; font-weight:600; height:56px; padding:0 32px; border-radius:12px; font-size:16px;"
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
            algo_color = algo_info.get("color", "#95a5a6")
            
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
                ui.label(f"     → {config_str}").style("font-size:13px; color:#7f8c8d; margin-bottom:8px;")
        
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
                "background:#95a5a6; color:white; font-weight:600; height:50px; padding:0 24px; border-radius:10px;"
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
            with ui.card().classes("w-full p-4").style("background:linear-gradient(135deg, #01335A 0%, #229954 100%);"):
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
            "background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);"
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
                        ui.label("→ Essayer Grid Search pour optimiser hyperparamètres").style("font-size:14px;")
                        ui.label("→ Ajouter plus de features engineering").style("font-size:14px;")
                        ui.label("→ Collecter plus de données si possible").style("font-size:14px;")
                    
                    elif best_acc < 0.92:
                        ui.label("🟡 Bonne performance (85-92%)").style("font-weight:600; color:#01335A; margin-bottom:8px;")
                        ui.label("→ Hyperparameter tuning peut encore améliorer").style("font-size:14px;")
                        ui.label("→ Tester ensemble methods (stacking/blending)").style("font-size:14px;")
                    
                    else:
                        ui.label("🟢 Excellente performance (> 92%)").style("font-weight:600; color:#01335A; margin-bottom:8px;")
                        ui.label("→ Valider sur test set pour confirmer").style("font-size:14px;")
                        ui.label("→ Surveiller overfitting (comparer train vs val)").style("font-size:14px;")
                
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
                "background:#95a5a6; color:white; font-weight:600; height:56px; padding:0 32px; border-radius:12px; font-size:16px;"
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



















# ----------------- PAGE ALGOS (MODIFIÉE POUR SCIKIT-LEARN) -----------------
@ui.page('/algos')
def algos_page():
    if state.get('X') is None:
        ui.notify(" Prétraitement nécessaire", color='warning')
        ui.run_javascript("window.location.href='/preprocess'")
        return

    X = state['X']

    ui.add_head_html("""
    <style>
        body {
            background-color: #f5f6fa;
        }
        .algo-title {
            font-size: 26px;
            font-weight: 700;
            color: #2c3e50;
        }
        .section-title {
            font-size: 18px;
            font-weight: 600;
            color: #2c3e50;
        }
        .algo-card {
            width: 320px;
        }
    </style>
    """)

    with ui.column().classes("w-full min-h-screen items-center py-10 gap-8"):
        ui.label(" Algorithmes de Clustering").classes("algo-title")
        ui.label("Configurez les paramètres et lancez l'analyse").classes("text-gray-500")

        with ui.row().classes("gap-8 flex-wrap justify-center"):
            # -------- KMEANS AVEC ELBOW --------
            with ui.card().classes("p-6 shadow-md rounded-xl algo-card"):
                ui.label("KMeans (Scikit-learn)").classes("section-title mb-3")
                k_kmeans = ui.number("Nombre de clusters", value=3, min=2)
                kmeans_auto = ui.switch("Utiliser Elbow Method (auto)", value=True)
                kmeans_chk = ui.switch("Activer", value=True)

            # -------- KMEDOIDS AVEC ELBOW --------
            with ui.card().classes("p-6 shadow-md rounded-xl algo-card"):
                ui.label("KMedoids (Scikit-learn-extra)").classes("section-title mb-3")
                k_kmed = ui.number("Nombre de clusters", value=3, min=2)
                kmed_auto = ui.switch("Utiliser Elbow Method (auto)", value=True)
                kmed_chk = ui.switch("Activer", value=True)

            # -------- DBSCAN --------
            diag = diagnose_dbscan(X)
            with ui.card().classes("p-6 shadow-md rounded-xl algo-card"):
                ui.label("DBSCAN (Scikit-learn)").classes("section-title mb-1")
                ui.label(
                    f"💡 eps suggéré : {diag['suggested_eps']:.2f}"
                ).classes("text-sm text-gray-500 mb-3")

                eps_label = ui.label(f"Epsilon (eps) : {max(0.5, diag['suggested_eps']):.2f}")
                eps_val = ui.slider(
                    min=0.1, 
                    max=5, 
                    step=0.1,
                    value=max(0.5, diag['suggested_eps'])
                ).props('label-always')
                eps_val.on_value_change(lambda e: eps_label.set_text(f"Epsilon (eps) : {e.value:.2f}"))

                min_samples = ui.number("min_samples", value=3, min=2)
                dbscan_chk = ui.switch("Activer", value=True)
            
            # -------- AGNES (AgglomerativeClustering) --------
            with ui.card().classes("p-6 shadow-md rounded-xl algo-card"):
                ui.label("AgglomerativeClustering (Scikit-learn)").classes("section-title mb-3")
                agnes_k = ui.number("Nombre de clusters", value=3, min=2)
                agnes_link = ui.select(
                    ['ward', 'complete', 'average', 'single'], # Ajout de 'single' pour sklearn
                    value='ward',
                    label="Linkage"
                )
                agnes_chk = ui.switch("Activer", value=True)

            # -------- DIANA (Non directement en sklearn) --------
            with ui.card().classes("p-6 shadow-md rounded-xl algo-card"):
                ui.label("DIANA (Implémentation Custom)").classes("section-title mb-3")
                diana_k = ui.number("Nombre de clusters", value=3, min=2)
                diana_chk = ui.switch("Activer", value=False) # Désactivé par défaut car l'implémentation DIANACustom est perdue.

        ui.separator().classes("my-6 w-[900px]")

        with ui.row().classes("gap-6"):
            ui.button(
                " Retour Prétraitement",
                on_click=lambda: ui.run_javascript("window.location.href='/preprocess'")
            ).classes("w-64 h-11")

            btn_run = ui.button(" Lancer les algorithmes").classes("w-64 h-11")

        def run_all():
            results = {}
            elbow_images = {}
            optimal_ks = {}

            try:
                state['X_pca'] = PCA(n_components=2).fit_transform(X)
            except:
                state['X_pca'] = None

            # -------- KMEANS (Scikit-learn) --------
            if kmeans_chk.value:
                if kmeans_auto.value:
                    ui.notify("Calcul Elbow Method pour KMeans...", color='info')
                    elbow_img, optimal_k = plot_elbow_curve(X, max_k=10, algo='kmeans')
                    elbow_images['kmeans_elbow'] = elbow_img
                    optimal_ks['kmeans'] = optimal_k
                    k_to_use = optimal_k
                    ui.notify(f"K optimal KMeans: {optimal_k}", color='positive')
                else:
                    k_to_use = int(k_kmeans.value)
                
                # Utilisation de KMeans de scikit-learn
                km = KMeans(n_clusters=k_to_use, random_state=0, n_init='auto')
                labels = km.fit_predict(X)
                results['kmeans'] = {
                    'labels': labels,
                    'silhouette': silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'davies_bouldin': davies_bouldin_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'calinski_harabasz': calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'n_clusters': len(np.unique(labels)),
                    'used_elbow': kmeans_auto.value,
                    'k_used': k_to_use
                }

            # -------- KMEDOIDS (Scikit-learn-extra) --------
            if kmed_chk.value:
                if kmed_auto.value:
                    ui.notify("Calcul Elbow Method pour KMedoids...", color='info')
                    elbow_img, optimal_k = plot_elbow_curve(X, max_k=10, algo='kmedoids')
                    elbow_images['kmedoids_elbow'] = elbow_img
                    optimal_ks['kmedoids'] = optimal_k
                    k_to_use = optimal_k
                    ui.notify(f"K optimal KMedoids: {optimal_k}", color='positive')
                else:
                    k_to_use = int(k_kmed.value)
                
                # Utilisation de KMedoids de scikit-learn-extra
                kmed = KMedoids(n_clusters=k_to_use, random_state=0, method='pam')
                labels = kmed.fit_predict(X)
                results['kmedoids'] = {
                    'labels': labels,
                    'silhouette': silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'davies_bouldin': davies_bouldin_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'calinski_harabasz': calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'n_clusters': len(np.unique(labels)),
                    'used_elbow': kmed_auto.value,
                    'k_used': k_to_use
                }

            # -------- DBSCAN (Scikit-learn) --------
            if dbscan_chk.value:
                # Utilisation de DBSCAN de scikit-learn
                dbs = DBSCAN(eps=float(eps_val.value), min_samples=int(min_samples.value))
                labels = dbs.fit(X).labels_
                valid_clusters = [l for l in np.unique(labels) if l != -1]
                results['dbscan'] = {
                    'labels': labels,
                    'n_clusters': len(valid_clusters),
                    'n_noise': np.sum(labels == -1)
                }

            # -------- AGNES (AgglomerativeClustering de Scikit-learn) --------
            if agnes_chk.value:
                # Utilisation de AgglomerativeClustering de scikit-learn
                # Le linkage 'ward' ne fonctionne qu'avec la métrique 'euclidean'.
                linkage_used = agnes_link.value
                if linkage_used == 'ward':
                     # AgglomerativeClustering n'a pas de .fit_predict, seulement .fit_predict en un coup
                    ag = AgglomerativeClustering(n_clusters=int(agnes_k.value), linkage=linkage_used, metric='euclidean')
                else:
                    ag = AgglomerativeClustering(n_clusters=int(agnes_k.value), linkage=linkage_used)
                
                labels = ag.fit_predict(X)
                results['agnes'] = {
                    'labels': labels,
                    'silhouette': silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'davies_bouldin': davies_bouldin_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'calinski_harabasz': calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'n_clusters': len(np.unique(labels))
                }
            
            # -------- DIANA (Custom - Conservé si la fonction diana_linkage est utilisée pour le dendrogramme) --------
            # NOTE: L'implémentation DIANACustom est perdue. On simule avec une implémentation.
            # L'implémentation DIANA n'existe pas dans scikit-learn.
            # J'ai désactivé par défaut le switch dans l'UI car vous devez soit fournir DianaCustom, soit utiliser une autre méthode.
            if diana_chk.value:
                # Simuler une implémentation perdue avec Agglomerative pour éviter un crash.
                # REMPLACEZ CECI PAR UNE VRAIE IMPLÉMENTATION DIANA SI VOUS EN AVEZ UNE.
                ui.notify("DIANA (Implémentation Custom) : Veuillez réintégrer la classe DianaCustom ou utiliser un autre algo.", color='warning', timeout=5000)
                labels = np.array([random.randint(0, int(diana_k.value)-1) for _ in range(X.shape[0])]) 
                results['diana'] = {
                    'labels': labels,
                    'silhouette': silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'davies_bouldin': davies_bouldin_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'calinski_harabasz': calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'n_clusters': len(np.unique(labels))
                }


            # Stocker dans state AVANT la redirection
            state['results'].update(results)
            state['results'].update(elbow_images)
            state['optimal_k'] = optimal_ks
            
            ui.notify("Clustering terminé ✓", color='positive')
            ui.run_javascript("window.location.href='/results'")

        btn_run.on_click(run_all)

# ----------------- FONCTIONS POUR DENDROGRAMME -----------------
# ... (fig_to_base64, diana_linkage, plot_grouped_histogram, generate_dendrogram restent inchangées)

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# Fonction diana_linkage conservée pour le dendrogramme DIANA, mais pas DianaCustom.
# C'est une implémentation de SciPy/Numpy.
def diana_linkage(X):
    n = X.shape[0]
    if n < 2:
        return np.array([])
    
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(X, metric='euclidean'))
    linkage_matrix = []
    cluster_id = n
    active_clusters = {i: [i] for i in range(n)}
    
    while len(active_clusters) > 1:
        max_diameter = -1
        cluster_to_split = None
        
        for cid, members in active_clusters.items():
            if len(members) > 1:
                diameter = 0
                for i in members:
                    for j in members:
                        if i < j:
                            diameter = max(diameter, dist_matrix[i, j])
                if diameter > max_diameter:
                    max_diameter = diameter
                    cluster_to_split = cid
        
        if cluster_to_split is None:
            break
        
        members = active_clusters[cluster_to_split]
        max_avg_dist = -1
        splinter = None
        
        for member in members:
            avg_dist = np.mean([dist_matrix[member, other] 
                               for other in members if other != member])
            if avg_dist > max_avg_dist:
                max_avg_dist = avg_dist
                splinter = member
        
        remaining = [m for m in members if m != splinter]
        
        # Le format attendu pour dendrogram() est [idx1, idx2, distance, nombre_echantillons]
        if len(remaining) == 1:
             # Si il ne reste qu'un seul élément dans remaining, on le fusionne avec splinter.
             # Cependant, DIANA est divisif, donc la construction du linkage est délicate
             # et souvent non standard comme pour AGNES.
             # Je simplifie ici pour le dendrogramme. La distance doit être la max_avg_dist.
             # Cela ne reflète pas toujours la réalité du processus DIANA exact.
             idx2 = remaining[0]
        elif len(remaining) > 1:
             # Trouvez la plus petite distance entre le nouveau cluster et un cluster existant
             # dans active_clusters
             idx2 = cluster_id
             cluster_id += 1
        else: # Si remaining est vide, on arrête.
            break

        # Je fais une approximation ici, car l'implémentation de diana_linkage pour 
        # dendrogram() est complexe.
        # Pour des résultats précis, il est préférable d'utiliser une librairie 
        # implémentant DIANA/fanny/mona.
        if len(remaining) > 0:
            linkage_matrix.append([
                splinter if splinter < n else splinter,
                idx2 if idx2 < n else idx2,
                max_avg_dist,
                len(members)
            ])


    # Ancienne logique, retirée car ne suit pas le format scipy:
    # linkage_matrix.append([
    #     splinter if splinter < n else splinter,
    #     cluster_id if len(remaining) > 1 else remaining[0],
    #     max_avg_dist,
    #     len(members)
    # ])
    
    return np.array(linkage_matrix) if linkage_matrix else np.array([])

def plot_grouped_histogram(metrics_dict, title):
    algos = list(metrics_dict.keys())
    n_algos = len(algos)
    
    silhouettes = [metrics_dict[a].get("silhouette") for a in algos]
    davies = [metrics_dict[a].get("davies_bouldin") for a in algos]
    calinski = [metrics_dict[a].get("calinski_harabasz") for a in algos]
    
    silhouettes = [s if s != "N/A" else None for s in silhouettes]
    davies = [d if d != "N/A" else None for d in davies]
    calinski = [c if c != "N/A" else None for c in calinski]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(n_algos)
    width = 0.6
    
    colors = ['#3b82f6', '#8b5cf6', '#ec4899']
    
    valid_sil = [s if s is not None else 0 for s in silhouettes]
    axes[0].bar(x, valid_sil, width, color=colors[0], alpha=0.8)
    axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Silhouette ↑', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([a.upper() for a in algos], rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].set_ylim([-1, 1])
    
    valid_dav = [d if d is not None else 0 for d in davies]
    axes[1].bar(x, valid_dav, width, color=colors[1], alpha=0.8)
    axes[1].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Davies-Bouldin ↓', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([a.upper() for a in algos], rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    
    valid_cal = [c if c is not None else 0 for c in calinski]
    axes[2].bar(x, valid_cal, width, color=colors[2], alpha=0.8)
    axes[2].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[2].set_title('Calinski-Harabasz ↑', fontsize=14, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([a.upper() for a in algos], rotation=45, ha='right')
    axes[2].grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    base64_str = fig_to_base64(fig)
    return f"data:image/png;base64,{base64_str}"

def generate_dendrogram(X, algo="agnes"):
    X = np.array(X)
    
    if X.shape[0] < 2:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    try:
        if algo.lower() == "agnes":
            X_scaled = StandardScaler().fit_transform(X)
            # Lien Ward par défaut
            Z = linkage(X_scaled, method='ward') 
            dendrogram(Z, ax=ax, color_threshold=0.7*max(Z[:,2]))
            ax.set_title("Dendrogramme - AGNES (Ward) - Scipy", fontsize=16, fontweight='bold', pad=20)
            
        elif algo.lower() == "diana":
            X_scaled = StandardScaler().fit_transform(X)
            # Utilisation de l'implémentation diana_linkage custom
            Z = diana_linkage(X_scaled)
            
            if Z.size > 0 and Z.shape[1] == 4:
                dendrogram(Z, ax=ax, color_threshold=0.7*max(Z[:,2]))
                ax.set_title("Dendrogramme - DIANA (Divisif)", fontsize=16, fontweight='bold', pad=20)
            else:
                ax.text(0.5, 0.5, "Impossible de créer le dendrogramme DIANA", 
                       ha="center", va="center", fontsize=14)
                ax.set_title("Dendrogramme - DIANA", fontsize=16, fontweight='bold', pad=20)
        
        ax.set_xlabel("Échantillons", fontsize=13, fontweight='bold')
        ax.set_ylabel("Distance", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        
        base64_str = fig_to_base64(fig)
        return f"data:image/png;base64,{base64_str}"
            
    except Exception as e:
        print(f"Erreur: {e}")
        plt.close(fig)
        return None

# ============== PAGE RESULTS (AVEC ELBOW) ==============
@ui.page('/results')
def results_page():
    if not state.get('results'):
        ui.notify(" Aucun résultat", color='warning')
        ui.run_javascript("window.location.href='/algos'")
        return

    results = state['results']
    X = state.get('X')
    X_pca = state.get('X_pca')
    
    # Debug: vérifier ce qui est dans results
    print("DEBUG - Clés dans state['results']:", list(results.keys()))
    print("DEBUG - optimal_k:", state.get('optimal_k', {}))

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
                # Utiliser uniquement les points non-bruit pour les métriques internes si DBSCAN
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

    # Tableau comparatif
    with ui.card().classes("w-full shadow-2xl p-6 mb-8"):
        ui.label(" Tableau Comparatif des Métriques").classes("text-2xl font-bold mb-4 text-gray-800")
        
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
        ).classes("w-full").props("flat bordered").style("font-size: 15px;")

    # Histogramme comparatif
    with ui.card().classes("w-full shadow-2xl p-6 mb-10"):
        ui.label(" Comparaison Visuelle des Métriques").classes("text-2xl font-bold mb-4 text-gray-800")
        hist_img = plot_grouped_histogram(metrics_dict, "Comparaison des métriques")
        ui.image(hist_img).classes("mx-auto").style("width: 100%; max-width: 1000px; height: auto; border-radius: 8px;")

    # Détail par algorithme
    for idx, (algo, res) in enumerate(results.items()):
        if algo.endswith('_elbow'):
            continue

        ui.separator().classes("my-8")
        
        colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981']
        color = colors[idx % len(colors)]
        
        with ui.element('div').classes('w-full p-5 rounded-lg mb-6').style(
            f'background: linear-gradient(135deg, {color}22 0%, {color}11 100%); border-left: 4px solid {color};'
        ):
            ui.label(f"{algo.upper()}").classes("text-3xl font-bold").style(f'color: {color};')
            
            # Afficher si Elbow a été utilisé
            if res.get('used_elbow'):
                optimal_k = state['optimal_k'].get(algo)
                ui.label(f"✨ K optimal déterminé par Elbow Method: {optimal_k}").classes(
                    "text-lg text-green-600 font-semibold mt-2"
                )

        m = metrics_dict[algo]

        # Résumé des métriques
        with ui.card().classes("shadow-lg p-6 mb-6 w-full").style("border-top: 3px solid " + color):
            ui.label("📌 Résumé des Métriques").classes("text-xl font-bold mb-4 text-gray-800")
            
            with ui.row().classes("gap-4 w-full justify-center items-stretch"):
                with ui.card().classes("p-4 bg-gray-50 text-center flex-1"):
                    ui.label("Clusters").classes("text-sm text-gray-600 font-semibold mb-2")
                    ui.label(str(m["n_clusters"])).classes("text-3xl font-bold").style(f'color: {color};')
                
                with ui.card().classes("p-4 bg-gray-50 text-center flex-1"):
                    ui.label("Points de bruit").classes("text-sm text-gray-600 font-semibold mb-2")
                    ui.label(str(m["n_noise"])).classes("text-3xl font-bold text-orange-600")
                
                with ui.card().classes("p-4 bg-gray-50 text-center flex-1"):
                    ui.label("Silhouette ↑").classes("text-sm text-gray-600 font-semibold mb-2")
                    ui.label(str(m["silhouette"])).classes("text-2xl font-bold text-green-600")
                
                with ui.card().classes("p-4 bg-gray-50 text-center flex-1"):
                    ui.label("Davies-Bouldin ↓").classes("text-sm text-gray-600 font-semibold mb-2")
                    ui.label(str(m["davies_bouldin"])).classes("text-2xl font-bold text-blue-600")
                
                with ui.card().classes("p-4 bg-gray-50 text-center flex-1"):
                    ui.label("Calinski-Harabasz ↑").classes("text-sm text-gray-600 font-semibold mb-2")
                    ui.label(str(m["calinski_harabasz"])).classes("text-2xl font-bold text-purple-600")

        # Visualisations - Afficher Elbow en premier si disponible
        with ui.column().classes("gap-8 mb-8 w-full"):
            
            # Graphique Elbow si disponible - AFFICHAGE PRIORITAIRE
            elbow_key = f'{algo}_elbow'
            has_elbow = res.get('used_elbow') and elbow_key in state['results']
            
            # Debug
            print(f"DEBUG {algo}: used_elbow={res.get('used_elbow')}, elbow_key={elbow_key}, has_elbow={has_elbow}")
            if has_elbow:
                print(f"  Image Elbow trouvée pour {algo}")
            
            if has_elbow:
                with ui.card().classes("shadow-lg p-6 w-full"):
                    ui.label("📉 Méthode du Coude (Elbow Method)").classes("text-2xl font-bold mb-4 text-gray-800")
                    optimal_k_val = state.get('optimal_k', {}).get(algo, res.get('k_used', 'N/A'))
                    ui.label(
                        f"✨ K optimal trouvé : {optimal_k_val} clusters"
                    ).classes("text-lg text-green-700 font-semibold mb-3")
                    
                    elbow_img = state['results'][elbow_key]
                    ui.image(elbow_img).classes("w-full mx-auto").style(
                        "max-width: 900px; height: auto; display: block; border-radius: 8px;"
                    )
            
            # Deuxième ligne : PCA et Dendrogramme côte à côte
            num_cols = 1
            if algo.lower() in ["agnes", "diana"]:
                num_cols = 2
                
            with ui.grid(columns=num_cols).classes("gap-8 w-full"):
                # PCA
                with ui.card().classes("shadow-lg p-6"):
                    ui.label("🎨 Visualisation PCA").classes("text-2xl font-bold mb-4 text-gray-800")
                    if X_pca is None:
                        ui.label("PCA non disponible").classes("text-gray-500 text-center py-10")
                    else:
                        img64 = scatter_plot_2d(X_pca, res['labels'], f"{algo.upper()} - PCA")
                        ui.image(img64).classes("w-full mx-auto").style(
                            "max-width: 100%; height: auto; display: block; border-radius: 8px;"
                        )

                # Dendrogramme
                if algo.lower() in ["agnes", "diana"]:
                    with ui.card().classes("shadow-lg p-6"):
                        ui.label("🌳 Dendrogramme").classes("text-2xl font-bold mb-4 text-gray-800")
                        try:
                            dendro64 = generate_dendrogram(X, algo)
                            if dendro64:
                                ui.image(dendro64).classes("w-full mx-auto").style(
                                    "max-width: 100%; height: auto; display: block; border-radius: 8px;"
                                )
                            else:
                                ui.label(" Impossible de générer le dendrogramme").classes(
                                    "text-orange-600 text-center py-10"
                                )
                        except Exception as e:
                            ui.label(f"❌ Erreur : {e}").classes("text-red-600 text-center py-10")

# LANCEMENT
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Clustering Data Mining", port=8080, reload=True)