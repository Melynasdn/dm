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

# ----------------- PAGE INDEX -----------------
@ui.page('/')
def home_page():
    with ui.column().classes("w-full h-screen items-center justify-center gap-8 p-10").style(
        "background-color: #f5f6fa !important; font-family: 'Inter', sans-serif !important;"
    ):
        # Titres
        ui.label("Plateforme d'Analyse de Données").style(
            "font-weight:700 !important; font-size:36px !important; color:#01335A !important; margin-bottom: -12px !important; text-align:center !important;"
        )
        ui.label("Veuillez choisir le mode d'apprentissage").style(
            "color:#09538C !important; font-size:16px !important; text-align:center !important;"
        )

        with ui.row().classes("gap-12 flex-wrap justify-center"):
            # Carte Non Supervisé
            with ui.card().classes("p-8 w-120 shadow-md rounded-xl transition transform hover:scale-105 flex flex-col items-center justify-center").style(
                "transition: box-shadow 0.3s ease !important; text-align:center !important;"
            ):
                ui.label("📊").style("font-size:50px !important; color:#09538C !important;")
                ui.label("Non Supervisé").style(
                    "font-size:22px !important; font-weight:600 !important; color:#01335A !important; margin-top:8px !important;"
                )
                ui.label(
                    "Découvrir les patterns et clusters dans vos données sans labels."
                ).style("font-size:15px !important; color:#636e72 !important; margin-top:6px !important; text-align:center !important;")
                ui.label(
                    "Exemples : segmentation clients, détection d'anomalies"
                ).style("font-size:15px !important; color:#636e72 !important; margin-top:4px !important; text-align:center !important;")
                with ui.row().classes("gap-4 mt-6 w-full justify-center"):
                    ui.button(
                        "Commencer",
                        on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/upload'")
                    ).style(
                        "background: linear-gradient(135deg, #01335A, #09538C) !important; "
                        "color:white !important; font-weight:600 !important; height:48px !important; width:100% !important; border-radius:8px !important;"
                    )
                    ui.button(
                        "En savoir plus",
                        on_click=lambda: ui.notify(
                            "L'apprentissage non supervisé permet de découvrir des structures cachées dans les données.",
                            color="info"
                        )
                    ).style(
                        "background-color:#dfe6e9 !important; color:#2d3436 !important; font-weight:500 !important; height:48px !important; width:100% !important; border-radius:8px !important;"
                    )

            # Carte Supervisé
            with ui.card().classes("p-8 w-120 shadow-md rounded-xl transition transform hover:scale-105 flex flex-col items-center justify-center").style(
                "transition: box-shadow 0.3s ease !important; text-align:center !important;"
            ):
                ui.label("🧠").style("font-size:50px !important; color:#09538C !important;")
                ui.label("Supervisé").style(
                    "font-size:22px !important; font-weight:600 !important; color:#01335A !important; margin-top:8px !important;"
                )
                ui.label(
                    "Prédire des valeurs ou classes connues à partir de vos données labellisées."
                ).style("font-size:15px !important; color:#636e72 !important; margin-top:6px !important; text-align:center !important;")
                ui.label(
                    "Exemples : classification d'email, prédiction de prix"
                ).style("font-size:15px !important; color:#636e72 !important; margin-top:4px !important; text-align:center !important;")
                with ui.row().classes("gap-4 mt-6 w-full justify-center"):
                    ui.button(
                        "Commencer",
                        on_click=lambda: ui.run_javascript("window.location.href='/supervised/upload'")
                    ).style(
                        "background: linear-gradient(135deg, #01335A, #09538C) !important; "
                        "color:white !important; font-weight:600 !important; height:48px !important; width:100% !important; border-radius:8px !important;"
                    )
                    ui.button(
                        "En savoir plus",
                        on_click=lambda: ui.notify(
                            "L'apprentissage supervisé utilise des labels pour apprendre à prédire de nouvelles données.",
                            color="info"
                        )
                    ).style(
                        "background-color:#dfe6e9 !important; color:#2d3436 !important; font-weight:500 !important; height:48px !important; width:100% !important; border-radius:8px !important;"
                    )
    with ui.column().classes("w-full h-screen items-center justify-center gap-8 p-10").style(
        "background-color: #f5f6fa !important; font-family: 'Inter', sans-serif !important;"
    ):
        # Titres
        ui.label("Plateforme d'Analyse de Données").style(
            "font-weight:700 !important; font-size:36px !important; color:#01335A !important; margin-bottom: -12px !important; text-align:center !important;"
        )
        ui.label("Veuillez choisir le mode d'apprentissage").style(
            "color:#09538C !important; font-size:16px !important; text-align:center !important;"
        )

        with ui.row().classes("gap-12 flex-wrap justify-center"):
            # Carte Non Supervisé
            with ui.card().classes("p-8 w-120 shadow-md rounded-xl transition transform hover:scale-105 flex flex-col items-center justify-center").style(
                "transition: box-shadow 0.3s ease !important; text-align:center !important;"
            ):
                ui.label("📊").style("font-size:50px !important; color:#09538C !important;")
                ui.label("Non Supervisé").style(
                    "font-size:22px !important; font-weight:600 !important; color:#01335A !important; margin-top:8px !important;"
                )
                ui.label(
                    "Découvrir les patterns et clusters dans vos données sans labels."
                ).style("font-size:15px !important; color:#636e72 !important; margin-top:6px !important; text-align:center !important;")
                ui.label(
                    "Exemples : segmentation clients, détection d'anomalies"
                ).style("font-size:15px !important; color:#636e72 !important; margin-top:4px !important; text-align:center !important;")
                with ui.row().classes("gap-4 mt-6 w-full justify-center"):
                    ui.button(
                        "Commencer",
                        on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/upload'")
                    ).style(
                        "background: linear-gradient(135deg, #01335A, #09538C) !important; "
                        "color:white !important; font-weight:600 !important; height:48px !important; width:100% !important; border-radius:8px !important;"
                    )
                    ui.button(
                        "En savoir plus",
                        on_click=lambda: ui.notify(
                            "L'apprentissage non supervisé permet de découvrir des structures cachées dans les données.",
                            color="info"
                        )
                    ).style(
                        "background-color:#dfe6e9 !important; color:#2d3436 !important; font-weight:500 !important; height:48px !important; width:100% !important; border-radius:8px !important;"
                    )

            # Carte Supervisé
            with ui.card().classes("p-8 w-120 shadow-md rounded-xl transition transform hover:scale-105 flex flex-col items-center justify-center").style(
                "transition: box-shadow 0.3s ease !important; text-align:center !important;"
            ):
                ui.label("🧠").style("font-size:50px !important; color:#09538C !important;")
                ui.label("Supervisé").style(
                    "font-size:22px !important; font-weight:600 !important; color:#01335A !important; margin-top:8px !important;"
                )
                ui.label(
                    "Prédire des valeurs ou classes connues à partir de vos données labellisées."
                ).style("font-size:15px !important; color:#636e72 !important; margin-top:6px !important; text-align:center !important;")
                ui.label(
                    "Exemples : classification d'email, prédiction de prix"
                ).style("font-size:15px !important; color:#636e72 !important; margin-top:4px !important; text-align:center !important;")
                with ui.row().classes("gap-4 mt-6 w-full justify-center"):
                    ui.button(
                        "Commencer",
                        on_click=lambda: ui.run_javascript("window.location.href='/supervised/upload'")
                    ).style(
                        "background: linear-gradient(135deg, #01335A, #09538C) !important; "
                        "color:white !important; font-weight:600 !important; height:48px !important; width:100% !important; border-radius:8px !important;"
                    )
                    ui.button(
                        "En savoir plus",
                        on_click=lambda: ui.notify(
                            "L'apprentissage supervisé utilise des labels pour apprendre à prédire de nouvelles données.",
                            color="info"
                        )
                    ).style(
                        "background-color:#dfe6e9 !important; color:#2d3436 !important; font-weight:500 !important; height:48px !important; width:100% !important; border-radius:8px !important;"
                    )




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

            btn_next = ui.button("Continuer ➡")
            btn_next.disable()
            btn_next.style(
                """
                width: 100% !important;
                height: 48px !important;
                margin-top: 14px !important;
                border-radius: 8px !important;
                background: linear-gradient(135deg, #01335A, #09538C) !important;
                color: white !important;
                font-weight: 600 !important;
                font-size: 15px !important;
                border: none !important;
                cursor: pointer !important;
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
                        color: #27ae60 !important;
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

                ui.button(
                    "⬅ Retour",
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
            ui.button("⬅ Retour à l'Upload",
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
            ui.button("⬅ Retour",
                      on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/upload'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )

            ui.button("➡ Étape suivante",
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
    

    df = state.get("raw_df", None)
    columns_info = state.get("columns_info", None)

    if df is None or columns_info is None:
        with ui.column().classes("w-full h-screen").style("display:flex; align-items:center; justify-content:center;"):
            ui.label("❌ Aucun dataset chargé ou informations de colonnes manquantes.").style(
                "font-size:18px; color:#c0392b; font-weight:600;"
            )
            ui.button("⬅ Retour à l'Upload",
                      on_click=lambda: ui.run_javascript("window.location.href='/upload'")).style(
                "margin-top:20px; background:#01335A; color:white; font-weight:600;"
            )
        return

    # ----------- STYLES ----------- 
    with ui.column().style(
        "width:100%; min-height:100vh; padding:40px; background-color:#f5f6fa; font-family:'Inter', sans-serif; display:flex; align-items:center;"
    ):
        ui.label("Phase 3.2 : Décisions Utilisateur (Clustering)").style(
            "font-weight:700; font-size:32px; color:#01335A; margin-bottom:32px; text-align:center;"
        )

        # ---------- 1️⃣ Sélection des Features ----------
        with ui.card().style(
            "width:100%; max-width:1200px; padding:24px; margin-bottom:24px; background:white; border-radius:12px; box-shadow:0 4px 15px rgba(0,0,0,0.08);"
        ):
            with ui.row().style("display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;"):
                ui.label("🧩 Sélection des colonnes pour le clustering").style(
                    "font-weight:700; font-size:20px; color:#01335A;"
                )
                
                # Boutons de sélection rapide
                with ui.row().style("display:flex; gap:8px;"):
                    ui.button("Tout sélectionner", 
                             on_click=lambda: feature_dropdown.set_value(all_columns)).props("flat dense").style(
                        " color:#01335A; font-weight:500; font-size:12px;"
                    )
                    ui.button("Désélectionner", 
                             on_click=lambda: feature_dropdown.set_value([])).props("flat dense").style(
                        " color:#6c757d; font-weight:500; font-size:12px;"
                    )

            all_columns = [col["Colonne"] for col in columns_info]
            
            feature_dropdown = ui.select(
                options=all_columns,
                multiple=True,
                label="Sélectionnez les colonnes à utiliser pour le clustering"
            ).props("dense")

            feature_warning = ui.label("").style("color:#e67e22; font-weight:600; margin-top:6px;")

        # ---------- 2️⃣ Correction des Types ----------
        with ui.card().style(
            "width:100%; max-width:1200px; padding:24px; margin-bottom:24px; background:white; border-radius:12px; box-shadow:0 4px 15px rgba(0,0,0,0.08);"
        ):
            ui.label("🛠 Correction des types de colonnes").style(
                "font-weight:700; font-size:20px; color:#01335A; margin-bottom:12px;"
            )

            column_type_widgets = {}
            column_exclude_widgets = {}

            for col in columns_info:
                with ui.row().style("display:flex; align-items:center; gap:12px; margin-bottom:8px;"):
                    ui.label(col["Colonne"]).style("width:180px; font-weight:600;")
                    col_type = ui.select(
                        options=[
                            "Numérique Continue",
                            "Numérique Discrète",
                            "Catégorielle Nominale",
                            "Catégorielle Ordinale",
                            "Date/Datetime",
                            "Texte",
                            "Identifiant"
                        ],
                        value=map_detected_type(col["Type Détecté"])
                    )
                    column_type_widgets[col["Colonne"]] = col_type

                    # Exclusion automatique (comme supervised)
                    auto_exclude = False
                    if col["Cardinalité"] == 1:
                        auto_exclude = True
                    if "%" in col["% Missing"]:
                        if float(col["% Missing"].replace("%","")) >= 100:
                            auto_exclude = True
                    if col["Colonne"].lower().startswith("id") or col["Cardinalité"] / len(df) > 0.95:
                        auto_exclude = True

                    exclude_cb = ui.checkbox("Exclure", value=auto_exclude)
                    column_exclude_widgets[col["Colonne"]] = exclude_cb

        # ---------- BOUTONS (SÉPARÉS GAUCHE/DROITE) ----------
        with ui.row().style("display:flex; justify-content:space-between; margin-top:20px; width:100%; max-width:1200px;"):
           ui.button("⬅ Retour",
                      on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/preprocessing'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )
           ui.button("➡ Étape suivante", on_click=lambda: save_and_go()).style(
                "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )

    # ----------------- FONCTIONS -----------------
    def on_confirm():
        selected_features = feature_dropdown.value
        if not selected_features or len(selected_features) < 2:
            feature_warning.text = "⚠️ Sélectionnez au moins 2 colonnes pour le clustering"
            return

        state["selected_features"] = selected_features

        for col_name, widget in column_type_widgets.items():
            state.setdefault("columns_types", {})[col_name] = widget.value

        for col_name, cb in column_exclude_widgets.items():
            state.setdefault("columns_exclude", {})[col_name] = cb.value

        ui.notify("✅ Décisions enregistrées avec succès !", color="positive")

    def save_and_go():
        on_confirm()
        ui.run_javascript("window.location.href='/unsupervised/univariate_analysis'")
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
            ui.button("← Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/multivariate_analysis'")).style(
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
                            
                            ui.label("→").style("font-size:24px; color:#01335A;")
                            
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
            ui.button("← Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/multivariate_analysis'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important; transition:all 0.2s !important;"
            )
            ui.button("Sauvegarder et Continuer →", on_click=save_and_next).style(
                "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:250px !important; transition:all 0.2s !important;"
            )
#----------------- PAGE /unsupervised/encoding -----------------



@ui.page('/unsupervised/encoding')
def encoding_page():

    df = state.get("cleaned_data")

    if df is None:
        with ui.column().style("width:100%; height:100vh; display:flex; align-items:center; justify-content:center;"):
            ui.label("❌ Données manquantes").style("font-size:20px; color:#e74c3c; font-weight:600; margin-bottom:12px;")
            ui.button("← Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/missing_values'")).style(
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
                            
                            ui.label("→").style("font-size:24px; color:#01335A;")
                            
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
            ui.button("← Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/missing_values'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important; transition:all 0.2s !important;"
            )
            ui.button("Sauvegarder et Continuer →", on_click=save_and_next).style(
                "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:250px !important; transition:all 0.2s !important;"
            ) 
# 
#   ----------------- PAGE /unsupervised/univariate_analysis -----------------




@ui.page('/unsupervised/univariate_analysis')
def unsupervised_univariate_page():

    df = state.get("raw_df")
    features = state.get("selected_features")

    if df is None or features is None:
        with ui.column().style("width:100%; height:100vh; display:flex; align-items:center; justify-content:center;"):
            ui.label("❌ Données manquantes").style("font-size:20px; color:#e74c3c; font-weight:600; margin-bottom:12px;")
            ui.label("Veuillez revenir en arrière et sélectionner les features").style("color:#7f8c8d; margin-bottom:20px;")
            ui.button("← Retour", on_click=lambda: ui.navigate.to("/unsupervised/user_decisions")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )
        return

    # Page principale
    with ui.column().style("width:100%; min-height:100vh; padding:40px; background:#f5f6fa; font-family:'Inter', sans-serif;"):
        
        # Header moderne
        with ui.column().style("margin-bottom:48px; text-align:center;"):
            ui.label("📊 Analyse Univariée").style("font-weight:800; font-size:42px; background:linear-gradient(135deg, #01335A, #09538C); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:12px; letter-spacing:-1px;")
            ui.label(f"Explorez et transformez vos {len(features)} variables").style("font-size:16px; color:#7f8c8d;")

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
            with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:32px;"):
                with ui.card().style("width:100%; max-width:900px; padding:24px; background:white; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                    
                    # En-tête de section
                    with ui.column().style("margin-bottom:20px;"):
                        ui.label("Variables Numériques").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                        ui.separator().style("width:60px; height:3px; background:#3498db; border-radius:2px; margin:0;")
                    
                    for col in numeric_features:
                        series = pd.to_numeric(df[col], errors="coerce").dropna()

                        if len(series) == 0:
                            continue

                        # Calculs statistiques essentiels
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

                        # Déterminer la couleur selon l'asymétrie
                        if abs(sk) > 1.0:
                            skew_color = "#e74c3c"
                            skew_bg = "#fadbd8"
                            skew_label = "Forte asymétrie"
                        elif abs(sk) > 0.5:
                            skew_color = "#e67e22"
                            skew_bg = "#fdebd0"
                            skew_label = "Asymétrie modérée"
                        else:
                            skew_color = "#27ae60"
                            skew_bg = "#d5f4e6"
                            skew_label = "Distribution normale"
                        
                        # Carte variable
                        with ui.card().style("padding:20px; margin-bottom:16px; background:#fafbfc; border-radius:8px; width:100%; border:1px solid #e1e8ed; transition:all 0.2s;"):
                            
                            # En-tête avec nom et asymétrie
                            with ui.row().style("display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; width:100%;"):
                                ui.label(col).style("font-weight:600; color:#2c3e50; font-size:18px;")
                                with ui.row().style("display:flex; gap:12px; align-items:center;"):
                                    with ui.card().style(f"padding:6px 12px; background:{skew_bg}; border-radius:6px; border:1px solid {skew_color};"):
                                        ui.label(f"Skew: {sk}").style(f"color:{skew_color}; font-weight:700; font-size:14px;")
                                    ui.label(skew_label).style("color:#7f8c8d; font-size:14px; font-weight:500;")
                            
                            # Statistiques en cartes
                            with ui.row().style("display:flex; gap:12px; margin-bottom:16px; flex-wrap:wrap;"):
                                with ui.card().style("padding:10px 14px; background:#f8f9fa; border-radius:6px; flex:1; min-width:120px;"):
                                    ui.label("Moyenne").style("font-size:11px; color:#7f8c8d; margin-bottom:4px;")
                                    ui.label(str(mean_val)).style("font-size:16px; font-weight:700; color:#2c3e50;")
                                
                                with ui.card().style("padding:10px 14px; background:#f8f9fa; border-radius:6px; flex:1; min-width:120px;"):
                                    ui.label("Écart-type").style("font-size:11px; color:#7f8c8d; margin-bottom:4px;")
                                    ui.label(str(std_val)).style("font-size:16px; font-weight:700; color:#2c3e50;")
                                
                                with ui.card().style("padding:10px 14px; background:#f8f9fa; border-radius:6px; flex:1; min-width:100px;"):
                                    ui.label("Min").style("font-size:11px; color:#7f8c8d; margin-bottom:4px;")
                                    ui.label(str(min_val)).style("font-size:16px; font-weight:700; color:#2c3e50;")
                                
                                with ui.card().style("padding:10px 14px; background:#f8f9fa; border-radius:6px; flex:1; min-width:100px;"):
                                    ui.label("Max").style("font-size:11px; color:#7f8c8d; margin-bottom:4px;")
                                    ui.label(str(max_val)).style("font-size:16px; font-weight:700; color:#2c3e50;")
                                
                                if outliers > 0:
                                    with ui.card().style("padding:10px 14px; background:#fadbd8; border-radius:6px; border:1px solid #e74c3c; flex:1; min-width:120px;"):
                                        ui.label("Outliers").style("font-size:11px; color:#c0392b; margin-bottom:4px; font-weight:600;")
                                        ui.label(str(outliers)).style("font-size:16px; font-weight:700; color:#e74c3c;")
                            
                            # Sélecteur d'action
                            with ui.row().style("display:flex; align-items:center; gap:12px; margin-top:16px;"):
                                ui.label("Action:").style("color:#7f8c8d; font-size:14px; font-weight:600;")
                                action_select = ui.select(
                                    options=["Garder tel quel", "Transformer (log)", "Transformer (sqrt)", "Supprimer"],
                                    value="Garder tel quel"
                                ).style("width:250px; border:2px solid #e1e8ed; border-radius:6px;")
                                
                                decisions[col] = action_select

        # ==========================================================
        # VARIABLES CATÉGORIELLES
        # ==========================================================
        if categorical_features:
            with ui.column().style("width:100%; display:flex; align-items:center; margin-bottom:32px;"):
                with ui.card().style("width:100%; max-width:900px; padding:24px; background:white; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.08);"):
                    
                    # En-tête de section
                    with ui.column().style("margin-bottom:20px;"):
                        ui.label("Variables Catégorielles").style("font-weight:600; font-size:20px; color:#2c3e50; margin-bottom:8px;")
                        ui.separator().style("width:60px; height:3px; background:#9b59b6; border-radius:2px; margin:0;")
                    
                    for col in categorical_features:
                        series = df[col].dropna()
                        
                        if len(series) == 0:
                            continue

                        # Calculs statistiques essentiels
                        n_unique = series.nunique()
                        most_common = series.value_counts().head(3)
                        missing = df[col].isna().sum()
                        missing_pct = round((missing / len(df)) * 100, 1)

                        # Carte variable
                        with ui.card().style("padding:20px; margin-bottom:16px; background:#fafbfc; border-radius:8px; width:100%; border:1px solid #e1e8ed; transition:all 0.2s;"):
                            
                            # En-tête avec nom et modalités
                            with ui.row().style("display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; width:100%;"):
                                ui.label(col).style("font-weight:600; color:#2c3e50; font-size:18px;")
                                with ui.row().style("display:flex; gap:8px; align-items:center;"):
                                    ui.label(f"{n_unique}").style("font-size:20px; font-weight:700; color:#9b59b6;")
                                    ui.label("modalités").style("color:#7f8c8d; font-weight:500; font-size:14px;")
                            
                            # Statistiques
                            with ui.row().style("display:flex; gap:12px; margin-bottom:16px; flex-wrap:wrap;"):
                                with ui.card().style("padding:10px 14px; background:#f8f9fa; border-radius:6px;"):
                                    ui.label("Observations").style("font-size:11px; color:#7f8c8d; margin-bottom:4px;")
                                    ui.label(str(len(series))).style("font-size:16px; font-weight:700; color:#2c3e50;")
                                
                                if missing > 0:
                                    with ui.card().style("padding:10px 14px; background:#fadbd8; border-radius:6px; border:1px solid #e74c3c;"):
                                        ui.label("Manquantes").style("font-size:11px; color:#c0392b; margin-bottom:4px; font-weight:600;")
                                        ui.label(f"{missing} ({missing_pct}%)").style("font-size:16px; font-weight:700; color:#e74c3c;")
                            
                            # Top 3 modalités
                            if len(most_common) > 0:
                                with ui.card().style("padding:12px; background:#f8f9fa; border-radius:6px; margin-bottom:16px;"):
                                    ui.label("Top 3 modalités:").style("color:#7f8c8d; font-size:13px; font-weight:600; margin-bottom:8px;")
                                    for category, count in most_common.items():
                                        pct = round((count / len(series)) * 100, 1)
                                        with ui.row().style("display:flex; align-items:center; gap:8px; margin-bottom:4px;"):
                                            ui.label("•").style("color:#9b59b6; font-weight:700;")
                                            ui.label(str(category)[:40]).style("color:#2c3e50; font-size:13px; flex:1;")
                                            with ui.card().style("padding:4px 8px; background:white; border-radius:4px;"):
                                                ui.label(f"{count} ({pct}%)").style("color:#9b59b6; font-size:12px; font-weight:600;")
                            
                            # Sélecteur d'action
                            with ui.row().style("display:flex; align-items:center; gap:12px; margin-top:16px;"):
                                ui.label("Action:").style("color:#7f8c8d; font-size:14px; font-weight:600;")
                                action_select = ui.select(
                                    options=["Garder tel quel", "Encoder (One-Hot)", "Encoder (Ordinal)", "Supprimer"],
                                    value="Garder tel quel"
                                ).style("width:250px; border:2px solid #e1e8ed; border-radius:6px;")
                                
                                decisions[col] = action_select

        # ==========================================================
        # BOUTONS DE NAVIGATION
        # ==========================================================
        def save_and_next():
            state["univariate_decisions"] = {
                col: widget.value for col, widget in decisions.items()
            }
            ui.notify("✅ Décisions sauvegardées", type="positive")
            ui.navigate.to("/unsupervised/multivariate_analysis")

        with ui.row().style("display:flex; justify-content:space-between; width:100%; margin-top:32px;"):
            ui.button("← Retour", on_click=lambda: ui.navigate.to("/unsupervised/user_decisions")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important; transition:all 0.2s !important;"
            )
            
            ui.button("Sauvegarder et Continuer →", on_click=save_and_next).style(
                "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:250px !important; transition:all 0.2s !important;"
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
            ui.button("← Retour", on_click=lambda: ui.navigate.to("/unsupervised/univariate_analysis")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )
        return

    # Filtrer uniquement les colonnes numériques
    df_selected = df[features].select_dtypes(include=[np.number]).copy()
    
    if df_selected.empty or len(df_selected.columns) < 2:
        with ui.column().style("width:100%; height:100vh; display:flex; align-items:center; justify-content:center;"):
            ui.label("⚠️ Analyse impossible").style("font-size:20px; color:#f39c12; font-weight:600; margin-bottom:12px;")
            ui.label("Il faut au moins 2 variables numériques pour l'analyse multivariée").style("color:#7f8c8d; margin-bottom:20px;")
            ui.button("← Retour", on_click=lambda: ui.navigate.to("/unsupervised/univariate_analysis")).style(
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
            ui.button("← Retour", on_click=lambda: ui.navigate.to("/unsupervised/univariate_analysis")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )
            
            ui.button("Appliquer et Continuer →", on_click=apply_decisions).style(
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
            ui.button("← Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/encoding'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )
        return
    
    # Ne garder que les colonnes numériques (après encodage, tout devrait être numérique)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        with ui.column().style("width:100%; height:100vh; display:flex; align-items:center; justify-content:center;"):
            ui.label("⚠️ Aucune variable numérique").style("font-size:20px; color:#f39c12; font-weight:600; margin-bottom:12px;")
            ui.button("← Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/encoding'")).style(
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
            ui.button("← Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/encoding'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important; transition:all 0.2s !important;"
            )
            ui.button("Appliquer et Continuer →", on_click=save_and_next).style(
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
            ui.button("← Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/anomalies'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )
        return
    
    # Identifier les features numériques
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numerical_features:
        with ui.column().style("width:100%; height:100vh; display:flex; align-items:center; justify-content:center;"):
            ui.label("⚠️ Aucune variable numérique").style("font-size:20px; color:#f39c12; font-weight:600; margin-bottom:12px;")
            ui.button("← Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/anomalies'")).style(
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
            ui.button("← Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/anomalies'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important; transition:all 0.2s !important;"
            )
            ui.button("Appliquer et Continuer →", on_click=save_and_next).style(
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
            ui.button("← Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/normalization'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )
        return
    
    # Garder seulement les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        with ui.column().style("width:100%; height:100vh; display:flex; align-items:center; justify-content:center;"):
            ui.label("⚠️ Minimum 2 variables requises").style("font-size:20px; color:#f39c12; font-weight:600; margin-bottom:12px;")
            ui.button("← Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/normalization'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )
        return
    
    # Dataset numérique propre
    numeric_df = df[numeric_cols].copy()
    numeric_df = numeric_df.dropna()
    
    if len(numeric_df) == 0:
        with ui.column().style("width:100%; height:100vh; display:flex; align-items:center; justify-content:center;"):
            ui.label("❌ Données insuffisantes").style("font-size:20px; color:#e74c3c; font-weight:600; margin-bottom:12px;")
            ui.button("← Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/normalization'")).style(
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
            ui.button("← Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/normalization'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important; transition:all 0.2s !important;"
            )
            ui.button("Appliquer et Continuer →", on_click=save_and_next).style(
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
            ui.button("← Retour à l'accueil", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised'")).style(
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
                    ui.label("→").style("font-size:32px; color:#3498db; font-weight:700;")
                    
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
                    
                    ui.button("🚀 Lancer le Clustering →", on_click=lambda: validate_and_continue(df_final)).style(
                        "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white !important; font-weight:700 !important; border-radius:8px !important; padding:12px 28px !important; font-size:15px !important; cursor:pointer !important; box-shadow:0 4px 12px rgba(39,174,96,0.3) !important;"
                    )
        
        # NAVIGATION
        with ui.row().style("display:flex; justify-content:center; width:100%; margin-top:32px;"):
            ui.button("← Retour à la PCA", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/pca'")).style(
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
            ui.button("← Retour au résumé", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/summary'")).style(
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
            ui.button("← Retour", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/summary'")).style(
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
            ui.button("← Retour au clustering", on_click=lambda: ui.navigate.to("/unsupervised/clustering")).style(
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
            ui.button("← Retour au Clustering", on_click=lambda: ui.navigate.to('/unsupervised/clustering')).style(
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
    ui.run(title="Clustering Data Mining", port=5000, reload=True)
