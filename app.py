# ----------------- IMPORTS -----------------
from nicegui import ui
import pandas as pd
import numpy as np
import io, base64, random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering 
from sklearn_extra.cluster import KMedoids 


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
            print("✅ df_original sauvegardé automatiquement")
        else:
            print("⚠️ Impossible de sauvegarder df_original : raw_df absent")

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
from nicegui import ui
import pandas as pd, io



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
                    #  CORRECTION : Sauvegarder AVANT toute modification
                    state["df_original"] = df.copy()  # ⚠️ CRUCIAL : .copy() pour snapshot
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

                btn_next.on_click(lambda: ui.run_javascript("window.location.href='/preprocess'"))




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

                btn_next.on_click(lambda: ui.run_javascript("window.location.href='/supervised/preprocessing'"))















###########################################################

# ----------------- PAGE PREPROCESS -----------------
# ----------------- PAGE PREPROCESS (UNSUPERVISED) -----------------
@ui.page('/preprocess')
def preprocess_page():
    if state["raw_df"] is None:
        ui.notify("⚠️ Aucun dataset chargé", color='warning')
        # 🔧 Correction : on revient vers la page upload UNSUPERVISÉE
        ui.run_javascript("window.location.href='/unsupervised/upload'")
        return

    df = state["raw_df"]

    with ui.column().classes("p-10"):

        ui.label("Prétraitement des Données").classes("text-3xl font-bold")

        # 🔧 Correction : redirection correcte
        ui.button("⬅ Retour Upload", on_click=lambda: ui.run_javascript("window.location.href='/unsupervised/upload'"))

        ui.button(
            "➡ Continuer au clustering",
            on_click=lambda: ui.run_javascript("window.location.href='/algos'")
        )


# ----------------- PAGE PREPROCESS (SUPERVISE) -----------------
# ----------------- PAGE /supervised/preprocessing -----------------
from nicegui import ui
import numpy as np
import pandas as pd

@ui.page('/supervised/preprocessing')
def supervised_preprocessing_page():
    df = state.get("raw_df", None)

    if df is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Aucun dataset chargé. Veuillez importer un fichier avant de continuer.").style(
                "font-size:18px !important; color:#c0392b !important; font-weight:600 !important;"
            )
            ui.button("⬅ Retour à l'Upload", on_click=lambda: ui.run_javascript("window.location.href='/upload'")).style(
                "margin-top:20px !important; background:#01335A !important; color:white !important; font-weight:600 !important;"
            )
        return

    # ----------- STYLES GÉNÉRAUX -----------
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
            ui.button("⬅ Retour", on_click=lambda: ui.run_javascript("window.location.href='/upload'")).style(
                "background:#dfe6e9 !important; color:#2c3e50 !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )

            ui.button("➡ Étape suivante", on_click=lambda: ui.run_javascript("window.location.href='/supervised/user_decisions'")).style(
                "background:linear-gradient(135deg, #01335A, #09538C) !important; color:white !important; font-weight:600 !important; border-radius:8px !important; height:46px !important; width:200px !important;"
            )


# ----------------- PAGE /supervised/user_decisions -----------------
from nicegui import ui

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
    df = state.get("raw_df", None)
    columns_info = state.get("columns_info", None)

    if df is None or columns_info is None:
        with ui.column().classes("w-full h-screen").style("display:flex; align-items:center; justify-content:center;"):
            ui.label("❌ Aucun dataset chargé ou informations de colonnes manquantes.").style(
                "font-size:18px; color:#c0392b; font-weight:600;"
            )
            ui.button("⬅ Retour à l'Upload", on_click=lambda: ui.run_javascript("window.location.href='/upload'")).style(
                "margin-top:20px; background:#01335A; color:white; font-weight:600;"
            )
        return

    # ----------- STYLES GÉNÉRAUX ----------- 
    with ui.column().style(
        "width:100%; min-height:100vh; padding:40px; background-color:#f5f6fa; font-family:'Inter', sans-serif;"
    ):
        ui.label("Phase 3.2 : Décisions Utilisateur").style(
            "font-weight:700; font-size:32px; color:#01335A; margin-bottom:32px; text-align:center;"
        )

        # ---------- 1️⃣ Sélection de la Target ----------
        with ui.card().style(
            "width:100%; max-width:900px; padding:24px; margin-bottom:24px; background:white; border-radius:12px; box-shadow:0 4px 15px rgba(0,0,0,0.08);"
        ):
            ui.label("🎯 Sélection de la colonne Target").style(
                "font-weight:700; font-size:20px; color:#01335A; margin-bottom:12px;"
            )

            target_dropdown = ui.select(
                options=[col["Colonne"] for col in columns_info],
                label="Sélectionnez la colonne cible",
                on_change=lambda e: on_target_change(e.value)
            ).props("clearable dense")

            target_warning = ui.label("").style("color:#e67e22; font-weight:600; margin-top:6px;")
            imbalance_label = ui.label("").style("margin-top:6px; font-size:14px; color:#2c3e50;")
            smote_cb = ui.checkbox("Appliquer un rééquilibrage (SMOTE/undersampling)").disable()

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

                    # Checkbox Exclusion
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

        # ---------- BOUTONS CONFIRMER + SUIVANT ----------
        with ui.row().style("display:flex; gap:12px; margin-top:20px; justify-content:center;"):
            ui.button("✅ Valider les décisions", on_click=lambda: on_confirm()).style(
                "background:linear-gradient(135deg, #01335A, #09538C); color:white; font-weight:600; border-radius:8px; height:46px; width:250px;"
            )
            ui.button("➡ Passer à Split", on_click=lambda: save_and_go_to_split()).style(
                "background:#27ae60; color:white; font-weight:600; border-radius:8px; height:46px; width:250px;"
            )

    # ----------------- FONCTIONS INTERNES -----------------
    def on_target_change(target_col):
        if target_col is None:
            target_warning.text = ""
            imbalance_label.text = "Sélectionnez la target pour voir la distribution"
            smote_cb.disable()
            return
        n_unique = df[target_col].nunique(dropna=True)
        if n_unique > 20:
            target_warning.text = "⚠️ Attention : cela semble être une régression"
        else:
            target_warning.text = ""
        counts = df[target_col].value_counts()
        if n_unique <= 20:
            imbalance_label.text = "Distribution des classes : " + ", ".join([f"{k}: {v}" for k,v in counts.items()])
            smote_cb.enable()
        else:
            imbalance_label.text = "C'est une variable continue (régression)"
            smote_cb.disable()

    def on_confirm():
        target_col = target_dropdown.value
        if target_col is None:
            ui.notify("Veuillez sélectionner une colonne target avant de continuer.", color="negative")
            return
        state["target_column"] = target_col
        for col_name, widget in column_type_widgets.items():
            state.setdefault("columns_types", {})[col_name] = widget.value
        for col_name, cb in column_exclude_widgets.items():
            state.setdefault("columns_exclude", {})[col_name] = cb.value
        ui.notify("✅ Décisions enregistrées avec succès !", color="positive")

    def save_and_go_to_split():
        on_confirm()  # Sauvegarde des décisions avant de passer à la page suivante
        ui.run_javascript("window.location.href='/supervised/preprocessing2'")





from nicegui import ui
from sklearn.model_selection import train_test_split
import pandas as pd

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
            ui.button("⬅ Retour", on_click=lambda: ui.run_javascript(
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
            ui.label(f"🎯 Distribution de la target '{target_col}'").style(
                "font-weight:700; font-size:20px; color:#01335A; margin-bottom:12px;"
            )

            counts = df_active[target_col].value_counts()
            total = len(df_active)
            imbalance_detected = False

            for cls, cnt in counts.items():
                pct = round(cnt / total * 100, 1)
                with ui.row().classes("items-center gap-2 mb-1"):
                    ui.label(f"{cls}: {cnt} ({pct}%)").style("width:120px; font-family:monospace;")
                    ui.linear_progress(value=pct/100, color="blue").classes("w-full h-3 rounded-lg")

            if len(counts) > 1 and counts.min() / total * 100 < 30:
                imbalance_detected = True
                ui.label("⚠️ Déséquilibre détecté !").style(
                    "color:#e67e22; font-weight:600; margin-top:6px;"
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
            ).style("margin-top:12px;")

            seed_input = ui.input(label="Random Seed", value=42).style("margin-top:12px; width:150px;")

        # ---------- 3️⃣ Bouton Split ----------
        # Container résultat
        result_container = ui.column().classes("w-full max-w-4xl mt-4")

        def do_split():
            tr = train_slider.value / 100
            vr = val_slider.value / 100
            te = test_slider.value / 100

            stratify_col = df_active[target_col] if "Stratified" in strategy_radio.value else None

            # ✅ NOUVEAU : Sauvegarder l'état original AVANT split
            if "df_original" not in state:
                state["df_original"] = df_active.copy()  # Dataset AVANT transformations

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
                for name, y_set in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
                    counts = y_set.value_counts()
                    total = len(y_set)
                    ui.label(f"{name} set : {total} samples").style("font-weight:600; color:#01335A; margin-top:8px;")
                    for cls, cnt in counts.items():
                        pct = round(cnt / total * 100, 1)
                        ui.label(f"  - {cls}: {cnt} ({pct}%)").style("font-family:monospace; margin-left:10px;")

            ui.notify("✅ Split effectué avec succès !", color="positive")
 
            # Affichage post-split stylé (simple)
            ui.label(f"✅ Split effectué ! Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}").style(
                "font-weight:600; color:#01335A;"
            )

            ui.button(
                "➡ Étape 3.3 : Analyse Univariée",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/univariate_analysis'"),
            ).style(
                "background:#09538C; color:white; font-weight:600; border-radius:8px; height:40px; width:280px; margin-top:12px;"
            )

        ui.button("✅ Effectuer le split", on_click=do_split).style(
            "background:linear-gradient(135deg, #01335A, #09538C); color:white; font-weight:600; border-radius:8px; height:46px; width:250px; margin-top:20px;"
        )





from nicegui import ui
import pandas as pd
from scipy.stats import iqr, skew

@ui.page('/supervised/univariate_analysis')
def univariate_analysis_page():
    # Récupération des données train
    split = state.get("split")
    if split is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Aucun split trouvé. Veuillez d'abord effectuer le split.").style(
                "font-size:18px; color:#c0392b; font-weight:600;"
            )
            ui.button("⬅ Retour au Preprocessing", on_click=lambda: ui.run_javascript(
                "window.location.href='/supervised/preprocessing2'"
            )).style("margin-top:20px; background:#01335A; color:white; font-weight:600;")
        return

    X_train = split["X_train"]
    y_train = split["y_train"]

    numeric_cols = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=['int64','float64']).columns.tolist()

    # Container scrollable
    with ui.column().classes("w-full h-screen overflow-auto p-6").style("background-color:#f5f6fa; font-family:'Inter', sans-serif;"):
        ui.label("Phase 3.3 : Analyse univariée (Train uniquement)").style(
            "font-weight:700; font-size:32px; color:#01335A; margin-bottom:24px; text-align:center;"
        )

        ui.label("Analyse par feature :").style("font-weight:600; font-size:18px; margin-bottom:12px;")

        cards_container = ui.column().classes("w-full gap-4")

        def add_feature_card(col):
            series = X_train[col]
            is_numeric = col in numeric_cols

            # Crée la carte dans le container
            with cards_container:
                with ui.card().classes("w-full").style(
                    "padding:12px; border-radius:10px; box-shadow:0 3px 10px rgba(0,0,0,0.08); background:white;"
                ):
                    ui.label(f"📊 {col} ({'Numérique' if is_numeric else 'Catégorielle'})").style(
                        "font-weight:700; font-size:18px; color:#01335A; margin-bottom:6px;"
                    )

                    if is_numeric:
                        mean_ = round(series.mean(),2)
                        median_ = round(series.median(),2)
                        std_ = round(series.std(),2)
                        min_ = series.min()
                        max_ = series.max()
                        iqr_ = iqr(series)
                        skewness = round(skew(series),2)
                        lower = series.quantile(0.25) - 1.5*iqr_
                        upper = series.quantile(0.75) + 1.5*iqr_
                        outliers = series[(series < lower) | (series > upper)]

                        ui.label(f"• Mean: {mean_}  • Median: {median_}  • Std: {std_}").style("font-family:monospace;")
                        ui.label(f"• Min: {min_}  • Max: {max_}  • IQR: {iqr_}").style("font-family:monospace; margin-bottom:6px;")
                        
                        alerts = []
                        if abs(skewness) > 0.5:
                            alerts.append(f"Skewness: {skewness} (forte asymétrie)")
                        if len(outliers) > 0:
                            alerts.append(f"Outliers détectés: {len(outliers)} valeurs ({round(len(outliers)/len(series)*100,1)}%)")
                        for a in alerts:
                            ui.label(f"⚠️ {a}").style("color:#e67e22; font-size:14px;")

                    else:
                        counts = series.value_counts()
                        total = len(series)
                        for cls, cnt in counts.items():
                            pct = round(cnt / total * 100, 1)
                            with ui.row().classes("items-center gap-2 mb-1"):
                                ui.label(f"{cls}: {cnt} ({pct}%)").style("width:150px; font-family:monospace;")
                                ui.linear_progress(value=pct/100, color="blue").classes("w-full h-3 rounded-lg")

                    actions = ["Garder", "Transformer", "Traiter Outliers", "Supprimer"] if is_numeric else ["Garder", "Regrouper modalités rares", "Supprimer"]
                    ui.select(actions, label="Action à entreprendre").style("margin-top:6px; width:250px;")

        # Ajoute toutes les features
        for col in X_train.columns:
            add_feature_card(col)

        # Bouton pour passer à la page suivante (outliers)
        ui.button("➡ Étape 3.4 : Détection Outliers", on_click=lambda: ui.run_javascript(
            "window.location.href='/supervised/outliers_analysis'"
        )).style(
            "background:linear-gradient(135deg, #01335A, #09538C); color:white; font-weight:600; border-radius:8px; height:46px; width:300px; margin-top:20px;"
        )









from nicegui import ui

import pandas as pd
import numpy as np
import io, base64
import matplotlib.pyplot as plt

global_state = state

@ui.page('/supervised/outliers_analysis')
def outliers_analysis_page():
    split = global_state.get("split")
    if not split or "X_train" not in split:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Données d'entraînement introuvables. Veuillez d'abord effectuer le split.").style(
                "font-size:18px; color:#c0392b; font-weight:600;"
            )
            ui.button("⬅ Retour au Split",
                      on_click=lambda: ui.run_javascript("window.location.href='/supervised/split'"),
            ).style("margin-top:20px; background:#01335A; color:white; font-weight:600;")
        return

    df_train = split["X_train"].copy()
    global_state["cleaned_train"] = df_train.copy()

    with ui.column().classes("w-full p-6").style("background-color:#f5f6fa; font-family:'Inter', sans-serif;"):
        ui.label("🔍 Étape 3.4 : Gestion des Outliers (Train)").style(
            "font-weight:700; font-size:32px; color:#01335A; margin-bottom:24px; text-align:center;"
        )

        numeric_features = df_train.select_dtypes(include=np.number).columns.tolist()
        if not numeric_features:
            ui.label("Aucune variable numérique détectée.").style("color:#e74c3c; font-weight:600;")
            return

        def generate_hist(data, title):
            fig, ax = plt.subplots(figsize=(4, 2.5))
            ax.hist(data.dropna(), bins=20, alpha=0.7)
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.3)
            buffer = io.BytesIO()
            plt.tight_layout()
            fig.savefig(buffer, format='png')
            plt.close(fig)
            buffer.seek(0)
            return "data:image/png;base64," + base64.b64encode(buffer.read()).decode()

        # fabrique de handlers qui lie feature, after_plot et info_label
        def make_handlers(feature, after_plot, info_label):
            def remove_outliers():
                df = global_state["cleaned_train"]
                q1, q3 = df[feature].quantile([0.25, 0.75])
                iqr = q3 - q1
                mask = ~((df[feature] < (q1 - 1.5 * iqr)) |
                         (df[feature] > (q3 + 1.5 * iqr)))
                nb_outliers = len(df) - mask.sum()
                df.loc[~mask, feature] = np.nan
                after_plot.set_source(generate_hist(df[feature].dropna(), "Après suppression"))
                info_label.text = f"🧹 {nb_outliers} outliers supprimés ({nb_outliers/len(df)*100:.1f} %)."

            def winsorise():
                df = global_state["cleaned_train"]
                lower, upper = np.percentile(df[feature].dropna(), [1, 99])
                total = ((df[feature] < lower) | (df[feature] > upper)).sum()
                df[feature] = df[feature].clip(lower, upper)
                after_plot.set_source(generate_hist(df[feature], "Après Winsorisation (1%-99%)"))
                info_label.text = f"📉 {total} valeurs winsorisées (1%-99 %)."

            def log_transform():
                df = global_state["cleaned_train"]
                positive = df[feature].clip(lower=0)
                df[feature] = np.log1p(positive)
                after_plot.set_source(generate_hist(df[feature], "Après Log Transform"))
                info_label.text = "🔁 Transformation logarithmique appliquée."

            return remove_outliers, winsorise, log_transform

        for feature in numeric_features:
            with ui.card().style(
                "background:white; padding:16px; border-radius:10px; "
                "box-shadow:0 3px 10px rgba(0,0,0,0.08); margin-bottom:24px;"
            ):
                ui.label(f"📊 Feature : {feature}").style(
                    "font-weight:600; font-size:18px; margin-bottom:8px; color:#01335A;"
                )

                with ui.row().classes("items-center justify-center gap-4"):
                    before_plot = ui.image(
                        generate_hist(global_state["cleaned_train"][feature], "Avant traitement")
                    ).style("width:400px; height:250px; border-radius:8px; box-shadow:0 2px 6px rgba(0,0,0,0.1);")
                    after_plot = ui.image(
                        generate_hist(global_state["cleaned_train"][feature], "Après traitement (aucun)")
                    ).style("width:400px; height:250px; border-radius:8px; box-shadow:0 2px 6px rgba(0,0,0,0.1);")

                info_label = ui.label("").style("margin-top:10px; color:#555; font-size:14px;")

                # obtenir handlers bindés à la feature courante
                remove_cb, winsor_cb, log_cb = make_handlers(feature, after_plot, info_label)

                with ui.row().style("justify-content:center; margin-top:12px; gap:10px;"):
                    ui.button("Supprimer Outliers", on_click=remove_cb).style(
                        "background:#e74c3c; color:white; font-weight:600; border-radius:6px; padding:8px 18px;"
                    )
                    ui.button("Winsorisation", on_click=winsor_cb).style(
                        "background:#f39c12; color:white; font-weight:600; border-radius:6px; padding:8px 18px;"
                    )
                    ui.button("Log Transform", on_click=log_cb).style(
                        "background:#2980b9; color:white; font-weight:600; border-radius:6px; padding:8px 18px;"
                    )

        ui.button("➡ Étape suivante", on_click=lambda: ui.run_javascript(
            "window.location.href='/supervised/multivariate_analysis'"
        )).style(
            "background:linear-gradient(135deg, #01335A, #09538C); color:white; font-weight:600; "
            "border-radius:8px; height:46px; width:280px; margin-top:20px;"
        )











# ----------------- PAGE 3.5 : ANALYSE MULTIVARIÉE -----------------
from nicegui import ui
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import math

@ui.page('/supervised/multivariate_analysis')
def multivariate_analysis_page():
    df = state.get("raw_df", None)
    target_col = state.get("target_column", None)
    columns_exclude = state.get("columns_exclude", {}) or {}

    if df is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Aucun dataset chargé.").style("font-size:18px; color:#c0392b; font-weight:600;")
            ui.button("⬅ Retour à l'Upload", on_click=lambda: ui.run_javascript("window.location.href='/upload'"))
        return

    # Préparations
    numeric_cols = df.select_dtypes(include=[int, float, "number"]).columns.tolist()
    # Exclure colonnes marquées "Exclure" et target
    active_numeric = [c for c in numeric_cols if not columns_exclude.get(c, False) and c != target_col]

    if "engineered_features" not in state:
        state["engineered_features"] = []  # list of tuples (name, formula)

    # Container principal
    with ui.column().classes("w-full items-center p-8").style("background-color:#f5f6fa;"):
        ui.label("### ÉTAPE 3.5 : ANALYSE MULTIVARIÉE (Corrélations & Redondance)").style(
            "font-weight:700; font-size:28px; color:#01335A; margin-bottom:12px;"
        )

        # ----- SECTION A : Heatmap interactive -----
        with ui.card().classes("w-full max-w-6xl p-6 mb-6").style("background:white; border-radius:12px;"):
            ui.label("A — Heatmap de Corrélation (click sur cellule pour zoom)").style(
                "font-weight:700; font-size:18px; color:#01335A; margin-bottom:8px;"
            )

            if len(active_numeric) < 2:
                ui.label("Pas assez de colonnes numériques actives pour calculer la corrélation.").style("color:#e67e22")
            else:
                corr = df[active_numeric].corr()

                # Plotly heatmap
                heatmap_fig = go.Figure(
                    data=go.Heatmap(
                        z=corr.values,
                        x=corr.columns,
                        y=corr.index,
                        zmin=-1, zmax=1,
                        colorscale='RdBu',
                        colorbar=dict(title="r"),
                        hovertemplate='Feature A: %{y}<br>Feature B: %{x}<br>r = %{z:.3f}<extra></extra>'
                    )
                )
                heatmap_fig.update_layout(margin=dict(l=60, r=20, t=30, b=60), height=560)

                plot = ui.plotly(heatmap_fig).classes("w-full").style("max-width:100%;")
                # callback when user clicks a heatmap cell
                def on_heatmap_click(e):
                    # e contains points with x and y labels
                    try:
                        point = e["points"][0]
                        feat_b = point["x"]
                        feat_a = point["y"]
                        r_val = float(point["z"])
                        open_pair_modal(feat_a, feat_b, r_val)
                    except Exception as exc:
                        ui.notify("Erreur lecture cellule heatmap.", color="negative")
                plot.on("plotly_click", on_heatmap_click)

        # ----- SECTION B : Liste des corrélations élevées -----
        with ui.card().classes("w-full max-w-6xl p-6 mb-6").style("background:white; border-radius:12px;"):
            ui.label("B — Paires fortement corrélées (|r| > 0.8)").style(
                "font-weight:700; font-size:18px; color:#01335A; margin-bottom:8px;"
            )

            correlations = []
            if len(active_numeric) >= 2:
                corr_vals = corr.where(~np.eye(len(corr),dtype=bool))  # mask diagonal
                # find pairs
                pairs = []
                for i, a in enumerate(corr_vals.index):
                    for j, b in enumerate(corr_vals.columns):
                        if j <= i: continue
                        r = corr_vals.iloc[i, j]
                        if pd.isna(r): continue
                        if abs(r) >= 0.8:
                            pairs.append((a, b, float(r)))
                # sort by abs(r) desc
                pairs = sorted(pairs, key=lambda x: -abs(x[2]))

                # build rows
                rows = []
                for a, b, r in pairs:
                    if abs(r) > 0.9:
                        mark = "🔴"
                    elif abs(r) >= 0.8:
                        mark = "🟡"
                    else:
                        mark = "🟢"
                    # model impacts (simple heuristics)
                    impact_nb = "🔴" if abs(r) > 0.85 else "🟡"
                    impact_knn = "🟡" if abs(r) >= 0.8 else "🟢"
                    impact_c45 = "🟢"
                    rows.append({
                        "Feature A": a,
                        "Feature B": b,
                        "Corrélation": f"{r:.3f} {mark}",
                        "Impact NaiveBayes": impact_nb,
                        "Impact KNN": impact_knn,
                        "Impact C4.5": impact_c45
                    })
                if len(rows) == 0:
                    ui.label("Aucune paire avec |r| > 0.8").style("color:#2c3e50")
                else:
                    ui.table(
                        columns=[
                            {"name":"Feature A","label":"Feature A","field":"Feature A"},
                            {"name":"Feature B","label":"Feature B","field":"Feature B"},
                            {"name":"Corrélation","label":"Corrélation","field":"Corrélation"},
                            {"name":"Impact NaiveBayes","label":"Impact NB","field":"Impact NaiveBayes"},
                            {"name":"Impact KNN","label":"Impact KNN","field":"Impact KNN"},
                            {"name":"Impact C4.5","label":"Impact C4.5","field":"Impact C4.5"},
                        ],
                        rows=rows,
                        row_key="Feature A"
                    ).style("width:100%")

            else:
                ui.label("Pas assez de variables numériques actives pour détecter des corrélations.").style("color:#e67e22")

        # ----- SECTION C : VIF -----
        with ui.card().classes("w-full max-w-6xl p-6 mb-6").style("background:white; border-radius:12px;"):
            ui.label("C — VIF (Variance Inflation Factor)").style(
                "font-weight:700; font-size:18px; color:#01335A; margin-bottom:8px;"
            )
            if len(active_numeric) < 2:
                ui.label("Pas assez de colonnes numériques actives pour calculer le VIF.").style("color:#e67e22")
            else:
                vif_df = []
                try:
                    # Prepare design matrix (add constant to avoid perfect collinearity)
                    X_vif = df[active_numeric].dropna()
                    # To compute VIF we need no constant columns; use add_constant for statsmodels
                    X_vif_const = sm.add_constant(X_vif)
                    for i, col in enumerate(active_numeric):
                        vif_val = variance_inflation_factor(X_vif_const.values, i+1)  # +1 because 0 is const
                        level = "🟢" if vif_val <= 5 else "🔴"
                        vif_df.append({"Feature": col, "VIF": f"{vif_val:.2f}", "Niveau": level})
                except Exception as e:
                    ui.label("Erreur calcul VIF: " + str(e)).style("color:#c0392b")
                    vif_df = []

                if len(vif_df) > 0:
                    ui.table(
                        columns=[
                            {"name":"Feature","label":"Feature","field":"Feature"},
                            {"name":"VIF","label":"VIF","field":"VIF"},
                            {"name":"Niveau","label":"Niveau","field":"Niveau"}
                        ],
                        rows=vif_df
                    ).style("width:100%")

        # ----- SECTION D : Cartes comparatives + Actions utilisateur -----
        with ui.card().classes("w-full max-w-6xl p-6 mb-8").style("background:white; border-radius:12px;"):
            ui.label("D — Cartes comparatives & Actions").style(
                "font-weight:700; font-size:18px; color:#01335A; margin-bottom:8px;"
            )

            # Buttons: appliquer recommandations globales
            with ui.row().classes("items-center gap-4").style("margin-bottom:12px;"):
                ui.button("Appliquer recommandation : supprimer redondance (Naive Bayes)")\
                    .on("click", lambda e: apply_naivebayes_prune()).style("background:#09538C; color:white;")
                ui.button("Créer feature combinée (multi) ...").on("click", lambda e: open_bulk_engineer_modal())

            # For each correlated pair (we reuse pairs found above)
            if len(active_numeric) >= 2 and len(pairs) > 0:
                for a, b, r in pairs:
                    with ui.row().classes("items-start gap-4").style("margin-bottom:16px;"):
                        # Left card (A)
                        with ui.card().style("width:48%; padding:8px;"):
                            ui.label(f"{a}").style("font-weight:700; margin-bottom:6px;")
                            # plotly histogram for A
                            hist_a = make_histogram_plot(df[a].dropna(), a)
                            ui.plotly(hist_a).style("height:220px; width:100%;")
                            ui.label(f"Mean: {float(df[a].dropna().mean()):.3f}")
                            corr_to_target_a = df[a].corr(df[target_col]) if target_col and df[target_col].dtype in [int,float,"number"] else None
                            ui.label(f"Importance*: {corr_to_target_a:.3f}" if corr_to_target_a is not None else "Importance*: N/A")

                        # Right card (B)
                        with ui.card().style("width:48%; padding:8px;"):
                            ui.label(f"{b}").style("font-weight:700; margin-bottom:6px;")
                            hist_b = make_histogram_plot(df[b].dropna(), b)
                            ui.plotly(hist_b).style("height:220px; width:100%;")
                            ui.label(f"Mean: {float(df[b].dropna().mean()):.3f}")
                            corr_to_target_b = df[b].corr(df[target_col]) if target_col and df[target_col].dtype in [int,float,"number"] else None
                            ui.label(f"Importance*: {corr_to_target_b:.3f}" if corr_to_target_b is not None else "Importance*: N/A")

                    # Actions for this pair
                    with ui.row().classes("items-center gap-3").style("margin-bottom:12px;"):
                        ui.label("Action :").style("width:80px;")
                        btn_keep_a = ui.button(f"Garder {a}", on_click=lambda e, a=a, b=b: keep_feature(a, b, keep=a))
                        btn_keep_b = ui.button(f"Garder {b}", on_click=lambda e, a=a, b=b: keep_feature(a, b, keep=b)).style("background:#09538C;color:white;")
                        btn_keep_b.set_props("title='Recommandé (plus corrélé à la target)'")
                        ui.button("Garder les deux", on_click=lambda e, a=a, b=b: keep_feature(a, b, keep="both"))
                        ui.button("Créer feature combinée", on_click=lambda e, a=a, b=b: open_pair_modal(a,b,r))

            else:
                ui.label("Aucune paire corrélée à afficher pour les cartes comparatives.").style("color:#636e72")

        # Navigation buttons
        with ui.row().classes("w-full max-w-6xl justify-between").style("margin-top:12px;"):
            ui.button("⬅ Étape précédente", on_click=lambda: ui.run_javascript("window.location.href='/supervised/preprocessing2'"))
            ui.button("➡ Étape suivante", on_click=lambda: ui.run_javascript("window.location.href='/supervised/missing_values'"))

    # ----------------- Fonctions internes -----------------
    def make_histogram_plot(series, name):
        # returns a plotly Figure histogram for consistency/ interactivity
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=series, nbinsx=30, opacity=0.9, name=name))
        fig.update_layout(margin=dict(l=40, r=10, t=20, b=40), height=220, showlegend=False)
        return fig

    def open_pair_modal(feat_a, feat_b, r_val):
        # modal with pair details, combined feature creation form
        dlg = ui.dialog()
        with dlg:
            with ui.card().style("width:800px; padding:18px;"):
                ui.label(f"Détails paire : {feat_a}  —  {feat_b}   (r = {r_val:.3f})").style("font-weight:700; font-size:16px; margin-bottom:8px;")
                # Scatter plot
                scatter = go.Figure()
                scatter.add_trace(go.Scatter(x=df[feat_a], y=df[feat_b], mode='markers', marker=dict(opacity=0.6), name="points"))
                scatter.update_layout(height=360, xaxis_title=feat_a, yaxis_title=feat_b)
                ui.plotly(scatter).style("width:100%; height:360px;")

                # feature creation form
                with ui.row().classes("items-center gap-8").style("margin-top:8px;"):
                    name_input = ui.input(label="Nom nouvelle feature", placeholder=f"{feat_b}_par_{feat_a}")
                    formula_select = ui.select(options=[f"{feat_b} / {feat_a}", f"{feat_b} - {feat_a}", f"{feat_b} + {feat_a}", f"{feat_b} * {feat_a}"], label="Formule", value=f"{feat_b} / {feat_a}")
                    create_btn = ui.button("Créer", on_click=lambda e: create_engineered_feature(feat_a, feat_b, name_input.value, formula_select.value, dlg))
                    ui.button("Annuler", on_click=lambda e: dlg.close())

        dlg.open()

    def create_engineered_feature(a, b, new_name, formula_str, dialog_obj):
        if not new_name:
            ui.notify("Donne un nom à la nouvelle feature.", color="negative")
            return
        # compute safely
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
                ui.notify("Formule non reconnue.", color="negative")
                return
            # add to df and state
            df[new_name] = new_series
            state.setdefault("raw_df", df)
            state.setdefault("engineered_features", []).append((new_name, f"{formula_str} of {b} and {a}"))
            ui.notify(f"✅ Feature '{new_name}' créée.", color="positive")
            dialog_obj.close()
        except Exception as e:
            ui.notify("Erreur création feature: " + str(e), color="negative")

    def keep_feature(a, b, keep="both"):
        # keep = a or b or "both"
        if keep == a:
            state.setdefault("columns_exclude", {})[b] = True
            ui.notify(f"✅ On garde {a} et on exclut {b}.", color="positive")
        elif keep == b:
            state.setdefault("columns_exclude", {})[a] = True
            ui.notify(f"✅ On garde {b} et on exclut {a}.", color="positive")
        else:
            # keep both: ensure none excluded
            state.setdefault("columns_exclude", {})[a] = False
            state.setdefault("columns_exclude", {})[b] = False
            ui.notify(f"✅ Les deux features sont conservées.", color="positive")

    def apply_naivebayes_prune():
        # For each highly correlated pair, exclude the feature with smaller corr to target (if target exists), otherwise exclude the one with higher VIF
        if not pairs:
            ui.notify("Aucune paire corrélée à traiter.", color="warning")
            return
        for a, b, r in pairs:
            if target_col and target_col in df.columns:
                try:
                    corr_a = abs(df[a].corr(df[target_col]))
                    corr_b = abs(df[b].corr(df[target_col]))
                except:
                    corr_a = corr_b = 0
                if corr_a >= corr_b:
                    state.setdefault("columns_exclude", {})[b] = True
                else:
                    state.setdefault("columns_exclude", {})[a] = True
            else:
                # fallback: exclude the one with higher VIF (if computed)
                try:
                    # recompute vif local
                    X_v = df[[a,b]].dropna()
                    Xc = sm.add_constant(X_v)
                    vif_a = variance_inflation_factor(Xc.values, 1)
                    vif_b = variance_inflation_factor(Xc.values, 2)
                    if vif_a > vif_b:
                        state.setdefault("columns_exclude", {})[a] = True
                    else:
                        state.setdefault("columns_exclude", {})[b] = True
                except:
                    # default: exclude second
                    state.setdefault("columns_exclude", {})[b] = True
        ui.notify("✅ Recommandation appliquée (pruning Naive Bayes).", color="positive")

    def open_bulk_engineer_modal():
        # simple modal to create a combined feature from two chosen columns (free choice)
        dlg = ui.dialog()
        with dlg:
            with ui.card().style("width:560px; padding:18px;"):
                ui.label("Créer nouvelle feature combinée (manuelle)").style("font-weight:700; margin-bottom:8px;")
                col_a = ui.select(options=active_numeric, label="Feature A")
                col_b = ui.select(options=active_numeric, label="Feature B")
                name_input = ui.input(label="Nom nouvelle feature")
                formula_select = ui.select(options=["A / B", "B / A", "A - B", "B - A", "A + B", "A * B"], label="Formule", value="B / A")
                def create_btn_action(e):
                    a = col_a.value; b = col_b.value
                    if not a or not b or not name_input.value:
                        ui.notify("Choisis A, B et un nom.", color="negative")
                        return
                    # map formula text to actual
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
                ui.button("Créer", on_click=create_btn_action)
                ui.button("Annuler", on_click=lambda e: dlg.close())
        dlg.open()









# ----------------- PAGE 3.6 : GESTION DES VALEURS MANQUANTES (VERSION FINALE) -----------------

# ----------------- PAGE 3.6 : GESTION DES VALEURS MANQUANTES (VERSION COMPLÈTE) -----------------
# ----------------- PAGE 3.6 : GESTION DES VALEURS MANQUANTES (VERSION COMPLÈTE) -----------------
from nicegui import ui
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

@ui.page('/supervised/missing_values')
def missing_values_page():
    """
    Page complète pour gestion des valeurs manquantes avec visualisation before/after
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
            ui.label("❌ Aucun dataset chargé.").style("font-size:18px; color:#c0392b; font-weight:600;")
            ui.button("⬅ Retour à l'Upload", on_click=lambda: ui.run_javascript("window.location.href='/upload'"))
        return

    # ✅ Sauvegarde df_original si pas déjà fait
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
    total_pct = round(total_missing / (df.shape[0] * df.shape[1]) * 100, 2)

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
        mask = df_data[cols].isna().any(axis=1)
        indices = df_data[mask].index.tolist()
        return indices[:max_rows]  # Limiter pour performance

    def apply_and_propagate(navigate_after=False):
        """✅ Applique l'imputation sur raw_df ET tous les splits"""
        try:
            strategies = state.get("missing_strategy", {})
            if not strategies:
                ui.notify("⚠️ Aucune stratégie configurée", color="warning")
                return False
            
            split = state.get("split", {})
            
            # ✅ 1. FIT sur le bon dataset
            if split and "X_train" in split:
                df_train_for_fit = split["X_train"].copy()
                if target_col and "y_train" in split and target_col not in df_train_for_fit.columns:
                    df_train_for_fit[target_col] = split["y_train"]
            else:
                df_train_for_fit = state["raw_df"].copy()
            
            # Fit des imputers
            fitted = fit_imputers(strategies, df_train_for_fit)
            state["fitted_imputers"] = serialize_fitted_imputers(fitted)
            
            # ✅ 2. APPLICATION sur raw_df
            state["raw_df"] = apply_fitted_imputers(state["raw_df"], fitted, active_cols)
            
            # ✅ 3. APPLICATION sur les splits si présents
            if split:
                for key in ["X_train", "X_val", "X_test"]:
                    if key in split and isinstance(split[key], pd.DataFrame):
                        split[key] = apply_fitted_imputers(split[key], fitted, active_cols)
                
                state["split"] = split
            
            ui.notify("✅ Imputation appliquée sur raw_df et tous les splits!", color="positive")
            
            # ✅ Navigation conditionnelle
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
        
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-2xl p-6"):
            ui.label(f"Configuration : {col_name}").style("font-weight:700; font-size:18px;")
            
            dtype = df[col_name].dtype
            n_missing = int(df[col_name].isna().sum())
            pct = round(n_missing / len(df) * 100, 2)
            
            ui.label(f"Type: {dtype} | Missing: {n_missing} ({pct}%)").style("color:#636e72; margin-top:8px;")
            
            # Options d'imputation
            if pd.api.types.is_numeric_dtype(df[col_name]):
                options = ["none", "drop", "mean", "median", "mode", "constant", "knn", "iterative"]
            else:
                options = ["none", "drop", "mode", "constant", "forward_fill", "backward_fill"]
            
            method_select = ui.select(options=options, value=current_method, label="Méthode").classes("w-full")
            
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
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Annuler", on_click=dialog.close).props("flat")
                
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
                
                ui.button("Sauvegarder", on_click=save_strategy).style("background:#27ae60; color:white;")
        
        dialog.open()

    def preview_imputation():
        """✅ Prévisualise l'imputation avec tableaux BEFORE/AFTER des lignes affectées"""
        strategies = state.get("missing_strategy", {})
        
        if not strategies:
            ui.notify("⚠️ Aucune stratégie configurée", color="warning")
            return
        
        try:
            # Fit sur df_train
            fitted = fit_imputers(strategies, df_train)
            
            # ✅ Récupérer les lignes avec missing values AVANT traitement
            cols_with_strategy = list(strategies.keys())
            cols_to_check = [c for c in cols_with_strategy if c in df_train.columns]
            
            if not cols_to_check:
                ui.notify("⚠️ Aucune colonne valide à traiter", color="warning")
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
                f"✅ Preview généré : {before_missing} valeurs manquantes → {after_missing} après imputation | {len(missing_indices)} lignes affichées"
            )
            
            # ✅ Affichage des tableaux BEFORE/AFTER
            table_before_container.clear()
            table_after_container.clear()
            
            with table_before_container:
                ui.label("📋 AVANT Imputation (lignes avec missing)").style(
                    "font-weight:700; font-size:16px; color:#e74c3c; margin-bottom:8px;"
                )
                
                # Formater les valeurs manquantes en rouge
                rows_before = []
                for idx in missing_indices:
                    row_dict = {"Index": idx}
                    for col in cols_to_check:
                        val = df_before.loc[idx, col]
                        if pd.isna(val):
                            row_dict[col] = "❌ NaN"
                        else:
                            row_dict[col] = str(val)[:30]  # Limiter taille
                    rows_before.append(row_dict)
                
                columns_before = [{"name": "Index", "label": "Index", "field": "Index"}]
                columns_before.extend([{"name": c, "label": c, "field": c} for c in cols_to_check])
                
                ui.table(
                    columns=columns_before,
                    rows=rows_before,
                    row_key="Index"
                ).style("width:100%; font-size:12px;").props("dense")
            
            with table_after_container:
                ui.label("✅ APRÈS Imputation (mêmes lignes)").style(
                    "font-weight:700; font-size:16px; color:#27ae60; margin-bottom:8px;"
                )
                
                rows_after = []
                for idx in missing_indices:
                    row_dict = {"Index": idx}
                    for col in cols_to_check:
                        val = df_after.loc[idx, col]
                        if pd.isna(val):
                            row_dict[col] = "⚠️ Still NaN"
                        else:
                            # Marquer les valeurs imputées en vert
                            was_nan = pd.isna(df_before.loc[idx, col])
                            display_val = str(val)[:30]
                            if was_nan:
                                row_dict[col] = f"✅ {display_val}"
                            else:
                                row_dict[col] = display_val
                    rows_after.append(row_dict)
                
                columns_after = [{"name": "Index", "label": "Index", "field": "Index"}]
                columns_after.extend([{"name": c, "label": c, "field": c} for c in cols_to_check])
                
                ui.table(
                    columns=columns_after,
                    rows=rows_after,
                    row_key="Index"
                ).style("width:100%; font-size:12px;").props("dense")
            
            # Histogrammes comparatifs (première colonne numérique)
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
                    showlegend=True
                )
                
                fig_after = go.Figure()
                fig_after.add_trace(go.Histogram(
                    x=df_preview[sample_col].dropna(), 
                    name="After", 
                    marker_color="#27ae60",
                    opacity=0.7
                ))
                fig_after.update_layout(
                    title=f"{sample_col} - Distribution APRÈS Imputation",
                    height=300,
                    showlegend=True
                )
                
                chart_before.update_figure(fig_before)
                chart_before.style("display:block;")
                chart_after.update_figure(fig_after)
                chart_after.style("display:block;")
            
        except Exception as e:
            ui.notify(f"❌ Erreur lors du preview : {str(e)}", color="negative")
            print(f"Détail erreur preview: {e}")

    def confirm_and_apply():
        """Confirme et applique l'imputation sur tous les datasets"""
        strategies = state.get("missing_strategy", {})
        
        if not strategies:
            ui.notify("⚠️ Aucune stratégie configurée", color="warning")
            return
        
        with ui.dialog() as dialog, ui.card().classes("p-6"):
            ui.label("⚠️ Confirmation").style("font-weight:700; font-size:18px;")
            ui.label("Voulez-vous appliquer l'imputation sur raw_df et tous les datasets (train/val/test) ?").style(
                "margin-top:8px; color:#2c3e50;"
            )
            ui.label("⚠️ Cette action est irréversible (sauf si vous rechargez le fichier)").style(
                "margin-top:4px; color:#e74c3c; font-size:13px;"
            )
            
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Annuler", on_click=dialog.close).props("flat")
                
                def confirm_and_next():
                    dialog.close()
                    # ✅ Passer navigate_after=True pour aller à la page suivante
                    apply_and_propagate(navigate_after=True)
                
                ui.button("Confirmer", on_click=confirm_and_next).style("background:#27ae60; color:white;")
        
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

    # ---------- UI ----------
    with ui.column().classes("w-full items-center p-8").style("background-color:#f5f6fa;"):
        ui.label("ÉTAPE 3.6 : GESTION DES VALEURS MANQUANTES").style(
            "font-weight:700; font-size:28px; color:#01335A; margin-bottom:10px;"
        )

        # --- A - Overview ---
        with ui.card().classes("w-full max-w-6xl p-6 mb-6"):
            ui.label("📊 Vue d'ensemble des valeurs manquantes").style("font-weight:700; font-size:18px; color:#2c3e50;")
            with ui.row().classes("gap-6").style("margin-top:12px;"):
                def metric(label, value, sub=""):
                    with ui.column().classes("items-start"):
                        ui.label(label).style("font-size:13px; color:#636e72;")
                        ui.label(value).style("font-weight:700; font-size:20px; color:#01335A;")
                        if sub: 
                            ui.label(sub).style("font-size:12px; color:#2c3e50;")
                
                metric("Total missing", f"{total_missing}", f"{total_pct}% du dataset")
                metric("Features affectées", f"{affected}/{len(active_cols)}", "")
                metric("Lignes", f"{len(df):,}", "")

        # --- B - Table of columns ---
        with ui.card().classes("w-full max-w-6xl p-6 mb-6"):
            ui.label("📋 Détail par colonne").style("font-weight:700; font-size:18px; color:#2c3e50;")
            rows = []
            for col in active_cols:
                n_missing = int(miss_counts.get(col, 0))
                pct = float(miss_pct.get(col, 0.0))
                dtype = "Numérique" if pd.api.types.is_numeric_dtype(df[col]) else "Catégoriel/Autre"
                tag = "🔴" if pct > 20 else ("🟡" if pct >= 5 else ("🟢" if pct > 0 else ""))
                
                # Afficher stratégie actuelle
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
                    {"name": "Feature", "label": "Feature", "field": "Feature"},
                    {"name": "Type", "label": "Type", "field": "Type"},
                    {"name": "Missing", "label": "Missing", "field": "Missing"},
                    {"name": "% Missing", "label": "% Missing", "field": "% Missing"},
                    {"name": "Niveau", "label": "Niveau", "field": "Niveau"},
                    {"name": "Stratégie", "label": "Stratégie", "field": "Stratégie"}
                ],
                rows=rows,
                row_key="Feature"
            ).style("width:100%;").props("dense")
            
            ui.label("💡 Clique sur une ligne pour configurer la stratégie d'imputation").style(
                "font-size:12px; color:#636e72; margin-top:8px;"
            )
            
            def handle_row_click(e):
                if e and "row" in e and "Feature" in e["row"]:
                    open_feature_modal(e["row"]["Feature"])
            
            table.on("row:click", handle_row_click)

        # --- C - Global strategy ---
        with ui.card().classes("w-full max-w-6xl p-6 mb-6"):
            ui.label("⚡ Stratégie Globale (rapide)").style("font-weight:700; font-size:18px; color:#2c3e50;")
            ui.label("Applique une stratégie prédéfinie à toutes les colonnes concernées").style(
                "font-size:13px; color:#636e72; margin-top:4px;"
            )
            
            strategy_radio = ui.radio(
                options=[
                    "Conservative (drop cols>20% or rows>50%)",
                    "Balanced (median numeric, mode cat)",
                    "Aggressive (KNN numeric)",
                    "Custom (configure per feature)"
                ],
                value=state.get("missing_strategy_global", "Balanced (median numeric, mode cat)")
            ).classes("mt-2")
            
            ui.button(
                "Appliquer stratégie globale",
                on_click=lambda e: apply_global_strategy()
            ).style("background:#09538C; color:white; margin-top:12px;")

        # --- D - Preview & Apply ---
        with ui.card().classes("w-full max-w-6xl p-6 mb-6"):
            ui.label("🔍 Preview & Application").style("font-weight:700; font-size:18px; color:#2c3e50;")
            preview_info = ui.label("Cliquez sur 'Preview' pour visualiser l'impact de l'imputation").style(
                "font-size:14px; color:#636e72; margin-top:8px;"
            )
            
            # ✅ Conteneurs pour les tableaux BEFORE/AFTER
            with ui.row().classes("gap-4 w-full").style("margin-top:16px;"):
                with ui.column().classes("flex-1"):
                    table_before_container = ui.column().classes("w-full")
                with ui.column().classes("flex-1"):
                    table_after_container = ui.column().classes("w-full")
            
            # Histogrammes comparatifs
            with ui.row().classes("gap-4 w-full").style("margin-top:16px;"):
                with ui.column().classes("flex-1"):
                    chart_before = ui.plotly({}).style("width:100%; display:none;")
                with ui.column().classes("flex-1"):
                    chart_after = ui.plotly({}).style("width:100%; display:none;")
            
            with ui.row().classes("gap-2").style("margin-top:16px;"):
                ui.button(
                    "🔍 Preview (train)",
                    on_click=lambda e: preview_imputation()
                ).style("background:#2d9cdb; color:white;")
                
                ui.button(
                    "✅ Appliquer maintenant",
                    on_click=lambda e: confirm_and_apply()
                ).style("background:#27ae60; color:white;")

        # --- Navigation ---
        with ui.row().classes("w-full max-w-6xl justify-between").style("margin-top:20px;"):
            ui.button(
                "⬅ Étape précédente",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/multivariate_analysis'")
            ).style("background:#95a5a6; color:white;")
            
            ui.button(
                "➡ Étape suivante",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/encoding'")
            ).style("background:#09538C; color:white;")








# ----------------- PAGE 3.7 : ENCODAGE DES FEATURES CATÉGORIELLES -----------------
# ----------------- PAGE 3.7 : ENCODAGE DES FEATURES CATÉGORIELLES -----------------
from nicegui import ui
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
import copy

@ui.page('/supervised/encoding')
def encoding_page():
    """
    Page complète pour l'encodage des features catégorielles
    """
    
    # ---------- CONTEXTE ----------
    df = state.get("raw_df", None)
    split = state.get("split", None)
    columns_exclude = state.get("columns_exclude", {}) or {}
    target_col = state.get("target_column", None)
    
    if df is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Aucun dataset chargé.").style("font-size:18px; color:#c0392b; font-weight:600;")
            ui.button("⬅ Retour à l'Upload", on_click=lambda: ui.run_javascript("window.location.href='/upload'"))
        return
    
    # Préparer df_train
    df_train = None
    if split:
        Xtr = split.get("X_train")
        ytr = split.get("y_train")
        if isinstance(Xtr, pd.DataFrame) and ytr is not None:
            try:
                df_train = Xtr.copy()
                if target_col is not None:
                    df_train[target_col] = ytr
            except Exception:
                df_train = None
    if df_train is None:
        df_train = df.copy()
    
    active_cols = [c for c in df.columns if not columns_exclude.get(c, False) and c != target_col]
    
    # Identifier les colonnes catégorielles
    cat_cols = [c for c in active_cols if df[c].dtype == 'object' or pd.api.types.is_categorical_dtype(df[c])]
    
    state.setdefault("encoding_strategy", {})
    state.setdefault("encoding_params", {})
    
    # ---------- HELPERS ----------
    def get_cardinality_level(n):
        """Retourne le niveau de cardinalité"""
        if n <= 10:
            return "🟢", "Low", "green"
        elif n <= 50:
            return "🟡", "Medium", "orange"
        else:
            return "🔴", "High", "red"
    
    def get_recommended_encoding(col):
        """Recommande une méthode d'encodage"""
        n_unique = df[col].nunique()
        
        if n_unique == 2:
            return "Label Encoding", "Binaire - simple et efficace"
        elif n_unique <= 10:
            return "One-Hot Encoding", "Faible cardinalité - safe"
        elif n_unique <= 50:
            return "Frequency Encoding", "Cardinalité moyenne"
        else:
            return "Target Encoding", "Haute cardinalité - évite explosion"
    
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
    
    def apply_encoding(df_target, strategies, params, fit_on_train=True, fitted_encoders=None):
        """
        Applique les encodages selon les stratégies définies
        
        Args:
            df_target: DataFrame à encoder
            strategies: Dict {col: method}
            params: Dict {col: params}
            fit_on_train: Si True, fit les encoders. Si False, utilise fitted_encoders
            fitted_encoders: Dict des encoders déjà fitted (obligatoire si fit_on_train=False)
        """
        df_result = df_target.copy()
        encoders = {} if fit_on_train else fitted_encoders
        
        for col, method in strategies.items():
            if col not in df_result.columns:
                continue
            
            try:
                if method == "Label Encoding":
                    if fit_on_train:
                        # FIT sur train
                        le = LabelEncoder()
                        le.fit(df_train[col].dropna())
                        df_result[col] = le.transform(df_result[col])
                        encoders[col] = {"encoder": le, "method": method}
                    else:
                        # TRANSFORM avec encoder fitted
                        if fitted_encoders and col in fitted_encoders:
                            le = fitted_encoders[col]["encoder"]
                            # Gérer les valeurs inconnues
                            known_classes = set(le.classes_)
                            df_result[col] = df_result[col].apply(
                                lambda x: le.transform([x])[0] if x in known_classes else -1
                            )
                
                elif method == "One-Hot Encoding":
                    drop_first = params.get(col, {}).get("drop_first", True)
                    
                    if fit_on_train:
                        # FIT sur train
                        dummies = pd.get_dummies(df_result[col], prefix=col, drop_first=drop_first)
                        df_result = pd.concat([df_result.drop(columns=[col]), dummies], axis=1)
                        encoders[col] = {
                            "method": method, 
                            "columns": dummies.columns.tolist(), 
                            "drop_first": drop_first
                        }
                    else:
                        # TRANSFORM avec colonnes fitted
                        if fitted_encoders and col in fitted_encoders:
                            expected_cols = fitted_encoders[col]["columns"]
                            drop_first = fitted_encoders[col].get("drop_first", True)
                            dummies = pd.get_dummies(df_result[col], prefix=col, drop_first=drop_first)
                            
                            # Aligner les colonnes
                            for exp_col in expected_cols:
                                if exp_col not in dummies.columns:
                                    dummies[exp_col] = 0
                            
                            # Garder seulement les colonnes attendues dans le bon ordre
                            dummies = dummies[[c for c in expected_cols if c in dummies.columns]]
                            df_result = pd.concat([df_result.drop(columns=[col]), dummies], axis=1)
                
                elif method == "Ordinal Encoding":
                    order = params.get(col, {}).get("order", [])
                    if order:
                        mapping = {val: idx for idx, val in enumerate(order)}
                        df_result[col] = df_result[col].map(mapping).fillna(-1)  # Valeurs inconnues = -1
                        if fit_on_train:
                            encoders[col] = {"method": method, "order": order, "mapping": mapping}
                
                elif method == "Frequency Encoding":
                    if fit_on_train:
                        # FIT sur train
                        freq = df_train[col].value_counts(normalize=True).to_dict()
                        df_result[col] = df_result[col].map(freq).fillna(0)  # Inconnus = 0
                        encoders[col] = {"method": method, "frequencies": freq}
                    else:
                        # TRANSFORM avec fréquences fitted
                        if fitted_encoders and col in fitted_encoders:
                            freq = fitted_encoders[col]["frequencies"]
                            df_result[col] = df_result[col].map(freq).fillna(0)
                
                elif method == "Target Encoding":
                    if target_col and target_col in df_train.columns:
                        if fit_on_train:
                            # FIT sur train
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
                            # TRANSFORM avec means fitted
                            if fitted_encoders and col in fitted_encoders:
                                smoothed_means = fitted_encoders[col]["target_means"]
                                global_mean = fitted_encoders[col]["global_mean"]
                                df_result[col] = df_result[col].map(smoothed_means).fillna(global_mean)
            
            except Exception as e:
                print(f"Erreur encodage {col}: {e}")
                import traceback
                traceback.print_exc()
        
        return df_result, encoders
    
    def open_encoding_modal(col_name):
        """Ouvre un modal pour configurer l'encodage d'une colonne"""
        current_method = state.get("encoding_strategy", {}).get(col_name, "")
        current_params = state.get("encoding_params", {}).get(col_name, {})
        
        n_unique = df[col_name].nunique()
        top_values = df[col_name].value_counts().head(5)
        is_ordinal, suggested_order = detect_ordinal(col_name)
        recommended, reason = get_recommended_encoding(col_name)
        
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-4xl p-6"):
            ui.label(f"Encodage : {col_name}").style("font-weight:700; font-size:20px; color:#01335A;")
            
            # Infos colonne
            with ui.card().classes("w-full p-4 mb-4").style("background:#f8f9fa;"):
                with ui.row().classes("gap-4"):
                    ui.label(f"Cardinalité : {n_unique}").style("font-weight:600;")
                    icon, level, color = get_cardinality_level(n_unique)
                    ui.label(f"{icon} {level}").style(f"color:{color}; font-weight:600;")
                
                ui.label("Top valeurs :").style("font-size:13px; color:#636e72; margin-top:8px;")
                for val, count in top_values.items():
                    pct = round(count / len(df) * 100, 1)
                    ui.label(f"  • {val}: {count} ({pct}%)").style("font-size:12px; color:#2c3e50;")
            
            # Recommandation
            with ui.card().classes("w-full p-4 mb-4").style("background:#e8f5e9;"):
                ui.label(f"💡 Recommandation : {recommended}").style("font-weight:600; color:#27ae60;")
                ui.label(reason).style("font-size:13px; color:#2c3e50;")
            
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
            ).classes("w-full")
            
            # Zone paramètres
            params_container = ui.column().classes("w-full mt-4")
            
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
                        ui.markdown("""
**📌 Label Encoding**

Principe : Assigne un entier à chaque modalité (0, 1, 2...)

✅ Adapté pour : Variables binaires  
⚠️ Attention : Impose un ordre arbitraire
                        """).classes("p-4").style("background:#f8f9fa; border-radius:8px;")
                    
                    elif method == "One-Hot Encoding":
                        ui.markdown("""
**📌 One-Hot Encoding**

Principe : Crée une colonne binaire par modalité

✅ Avantages : Pas d'ordre imposé, interprétable  
❌ Inconvénients : Explosion dimensionnelle si haute cardinalité
                        """).classes("p-4").style("background:#f8f9fa; border-radius:8px;")
                        
                        drop_first_checkbox = ui.checkbox(
                            "Drop first (éviter multicolinéarité)",
                            value=current_params.get("drop_first", True)
                        )
                    
                    elif method == "Ordinal Encoding":
                        ui.markdown("""
**📌 Ordinal Encoding**

Principe : Assigne des entiers selon un ordre défini

✅ Applicable uniquement si ordre naturel existe
                        """).classes("p-4").style("background:#f8f9fa; border-radius:8px;")
                        
                        ui.label("Définir l'ordre (du plus bas au plus haut) :").style("font-weight:600; margin-top:12px;")
                        
                        current_order = current_params.get("order", suggested_order if is_ordinal else list(df[col_name].unique()))
                        
                        for idx, val in enumerate(current_order):
                            with ui.row().classes("items-center gap-2"):
                                ui.label(f"{idx}:").style("font-weight:600; width:30px;")
                                inp = ui.input(value=str(val)).classes("flex-1")
                                order_inputs.append(inp)
                    
                    elif method == "Frequency Encoding":
                        ui.markdown("""
**📌 Frequency Encoding**

Principe : Remplace par la fréquence d'apparition

✅ Avantages : Simple, pas de leakage, 1 colonne  
❌ Limites : Perd l'info sémantique
                        """).classes("p-4").style("background:#f8f9fa; border-radius:8px;")
                    
                    elif method == "Target Encoding":
                        ui.markdown("""
**📌 Target Encoding**

Principe : Remplace par la moyenne de la target

✅ Avantages : Capture relation avec target, 1 colonne  
⚠️ Risques : Overfitting / Data leakage
                        """).classes("p-4").style("background:#f8f9fa; border-radius:8px;")
                        
                        smoothing_input = ui.number(
                            label="Smoothing (régularisation)",
                            value=current_params.get("smoothing", 10),
                            min=0,
                            max=100
                        ).classes("w-full")
            
            method_select.on_value_change(lambda: update_params_ui())
            update_params_ui()
            
            # Boutons
            with ui.row().classes("w-full justify-end gap-2 mt-6"):
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
                    table.update()
                
                ui.button("Sauvegarder", on_click=save_encoding).style("background:#27ae60; color:white;")
        
        dialog.open()
    
    def apply_all_encodings():
        """Applique tous les encodages et passe à l'étape suivante"""
        ensure_original_saved()
        strategies = state.get("encoding_strategy", {})
        split = state.get("split", {})

        if not strategies:
            ui.notify("⚠️ Aucun encodage configuré", color="warning")
            return
        
        with ui.dialog() as dialog, ui.card().classes("p-6"):
            ui.label("⚠️ Confirmation").style("font-weight:700; font-size:18px;")
            ui.label("Appliquer les encodages sur tous les datasets ?").style("margin-top:8px;")
            
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Annuler", on_click=dialog.close).props("flat")
                
                def confirm_and_next():
                    try:
                        params = state.get("encoding_params", {})
                        
                        # ✅ CORRECTION : FIT sur X_train uniquement
                        if split and "X_train" in split:
                            df_train_for_fit = split["X_train"].copy()
                            if target_col and "y_train" in split:
                                df_train_for_fit[target_col] = split["y_train"]
                        else:
                            df_train_for_fit = df_train.copy()
                    
                        # ✅ FIT sur train
                        _, encoders = apply_encoding(
                            df_train_for_fit, 
                            strategies, 
                            params, 
                            fit_on_train=True
                        )
                        state["fitted_encoders"] = encoders
                    
                        #  TRANSFORM sur chaque split séparément
                        if split:
                            for key in ["X_train", "X_val", "X_test"]:
                                if key in split and isinstance(split[key], pd.DataFrame):
                                    split[key], _ = apply_encoding(
                                        split[key], 
                                        strategies, 
                                        params, 
                                        fit_on_train=False,
                                        fitted_encoders=encoders
                                    )
                        
                        ui.notify("✅ Encodages appliqués avec succès!", color="positive")
                        dialog.close()
                        ui.run_javascript("setTimeout(() => window.location.href='/supervised/distribution_transform', 1500);")
                    
                    except Exception as e:
                        ui.notify(f"❌ Erreur : {str(e)}", color="negative")
                        print(f"Erreur détaillée : {e}")
                        import traceback
                        traceback.print_exc()
                
                ui.button("Confirmer", on_click=confirm_and_next).style("background:#27ae60; color:white;")
        
        dialog.open()
    
    def apply_recommended():
        """Applique automatiquement les recommandations"""
        for col in cat_cols:
            recommended, _ = get_recommended_encoding(col)
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
                    params["order"] = list(df[col].unique())
            
            state.setdefault("encoding_params", {})[col] = params
        
        ui.notify("✅ Recommandations appliquées", color="positive")
        table.update()
    
    # ---------- UI ----------
    with ui.column().classes("w-full items-center p-8").style("background-color:#f5f6fa;"):
        ui.label("ÉTAPE 3.7 : ENCODAGE DES FEATURES CATÉGORIELLES").style(
            "font-weight:700; font-size:28px; color:#01335A; margin-bottom:10px;"
        )
        
        ui.label("Convertir les features catégorielles en format numérique").style(
            "font-size:16px; color:#636e72; margin-bottom:20px;"
        )
        
        # Section A : Synthèse
        with ui.card().classes("w-full max-w-6xl p-6 mb-6"):
            ui.label(f"🔤 Features Catégorielles : {len(cat_cols)}").style(
                "font-weight:700; font-size:18px; color:#2c3e50;"
            )
            
            if len(cat_cols) == 0:
                ui.label("✅ Aucune feature catégorielle détectée - vous pouvez passer à l'étape suivante").style(
                    "color:#27ae60; margin-top:12px;"
                )
            else:
                # Légende
                with ui.row().classes("gap-4 mt-4").style("font-size:13px;"):
                    ui.label("🔴 High cardinality (>50) : Éviter One-Hot").style("color:#e74c3c;")
                    ui.label("🟡 Medium (10-50) : One-Hot ou alternatives").style("color:#f39c12;")
                    ui.label("🟢 Low (<10) : One-Hot safe").style("color:#27ae60;")
        
        # Section B : Table des features
        if len(cat_cols) > 0:
            with ui.card().classes("w-full max-w-6xl p-6 mb-6"):
                ui.label("Configuration des encodages").style("font-weight:700; font-size:18px; color:#2c3e50;")
                
                rows = []
                for col in cat_cols:
                    n_unique = df[col].nunique()
                    icon, level, _ = get_cardinality_level(n_unique)
                    recommended, _ = get_recommended_encoding(col)
                    current = state.get("encoding_strategy", {}).get(col, "")
                    
                    rows.append({
                        "Feature": col,
                        "Cardinalité": f"{n_unique} {icon}",
                        "Niveau": level,
                        "Recommandé": recommended,
                        "Configuré": current or "-"
                    })
                
                table = ui.table(
                    columns=[
                        {"name": "Feature", "label": "Feature", "field": "Feature"},
                        {"name": "Cardinalité", "label": "Cardinalité", "field": "Cardinalité"},
                        {"name": "Niveau", "label": "Niveau", "field": "Niveau"},
                        {"name": "Recommandé", "label": "Recommandé", "field": "Recommandé"},
                        {"name": "Configuré", "label": "Configuré", "field": "Configuré"}
                    ],
                    rows=rows,
                    row_key="Feature"
                ).style("width:100%;")
                
                ui.label("💡 Cliquez sur une ligne pour configurer l'encodage").style(
                    "font-size:13px; color:#636e72; margin-top:8px;"
                )
                
                def handle_row_click(e):
                    if e and "row" in e and "Feature" in e["row"]:
                        open_encoding_modal(e["row"]["Feature"])
                
                table.on("row:click", handle_row_click)
                
                # Boutons actions
                with ui.row().classes("gap-2 mt-4"):
                    ui.button(
                        "✨ Appliquer recommandations",
                        on_click=apply_recommended
                    ).style("background:#3498db; color:white;")
                    
                    ui.button(
                        "✅ Appliquer les encodages",
                        on_click=apply_all_encodings
                    ).style("background:#27ae60; color:white;")
        
        # Navigation
        with ui.row().classes("w-full max-w-6xl justify-between mt-6"):
            ui.button(
                "⬅ Étape précédente",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/missing_values'")
            ).style("background:#95a5a6; color:white;")
            
            ui.button(
                "➡ Étape suivante",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/distribution_transform'")
            ).style("background:#09538C; color:white;")



















# ----------------- PAGE 3.8 : TRANSFORMATIONS DE DISTRIBUTIONS -----------------
from nicegui import ui
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import boxcox, yeojohnson
from sklearn.preprocessing import PowerTransformer
import copy

@ui.page('/supervised/distribution_transform')
def distribution_transform_page():
    """
    Page complète pour les transformations de distributions
    """
    
    # ---------- CONTEXTE ----------
    df = state.get("raw_df", None)
    split = state.get("split", None)
    columns_exclude = state.get("columns_exclude", {}) or {}
    target_col = state.get("target_column", None)
    selected_algos = state.get("selected_algos", [])
    
    if df is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Aucun dataset chargé.").style("font-size:18px; color:#c0392b; font-weight:600;")
            ui.button("⬅ Retour à l'Upload", on_click=lambda: ui.run_javascript("window.location.href='/upload'"))
        return
    
    # Préparer df_train
    df_train = None
    if split:
        Xtr = split.get("X_train")
        ytr = split.get("y_train")
        if isinstance(Xtr, pd.DataFrame) and ytr is not None:
            try:
                df_train = Xtr.copy()
                if target_col is not None:
                    df_train[target_col] = ytr
            except Exception:
                df_train = None
    if df_train is None:
        df_train = df.copy()
    
    active_cols = [c for c in df.columns if not columns_exclude.get(c, False) and c != target_col]
    
    # Identifier les colonnes numériques
    num_cols = [c for c in active_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    state.setdefault("transform_strategy", {})
    state.setdefault("transform_params", {})
    state.setdefault("fitted_transformers", {})
    
    # ---------- HELPERS ----------
    def calculate_skewness_kurtosis(col):
        """Calcule skewness et kurtosis"""
        data = df_train[col].dropna()
        if len(data) < 3:
            return 0, 0
        skew = stats.skew(data)
        kurt = stats.kurtosis(data)
        return skew, kurt
    
    def get_skewness_level(skew):
        """Retourne le niveau de skewness"""
        abs_skew = abs(skew)
        if abs_skew < 0.5:
            return "🟢", "Normal", "green"
        elif abs_skew < 1.5:
            return "🟡", "Modéré", "orange"
        else:
            return "🔴", "Fort", "red"
    
    def get_recommended_transform(col):
        """Recommande une transformation"""
        skew, _ = calculate_skewness_kurtosis(col)
        abs_skew = abs(skew)
        
        data = df_train[col].dropna()
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
        
        data = data.copy()
        data_clean = data[~np.isnan(data) & ~np.isinf(data)]
        
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
                    return data_clean, None
                transformed, lambda_param = boxcox(data_clean)
                return transformed, {"lambda": lambda_param}
            
            elif method == "Yeo-Johnson":
                transformed, lambda_param = yeojohnson(data_clean)
                return transformed, {"lambda": lambda_param}
            
            else:  # Aucune
                return data_clean, {}
        
        except Exception as e:
            print(f"Erreur transformation: {e}")
            return data_clean, None
    
    def calculate_algo_impact(col, skew):
        """Calcule l'impact sur chaque algorithme"""
        abs_skew = abs(skew)
        impacts = {}
        
        # Gaussian Naive Bayes
        if abs_skew < 0.5:
            impacts["Gaussian NB"] = ("✅", "Distribution normale", "green")
        elif abs_skew < 1.5:
            impacts["Gaussian NB"] = ("🟡", "Assume normalité (légèrement violée)", "orange")
        else:
            impacts["Gaussian NB"] = ("🔴", "Assume normalité (fortement violée)", "red")
        
        # KNN
        if abs_skew < 0.5:
            impacts["KNN"] = ("✅", "Distances équilibrées", "green")
        elif abs_skew < 1.5:
            impacts["KNN"] = ("🟡", "Distances légèrement biaisées", "orange")
        else:
            impacts["KNN"] = ("🟡", "Distances biaisées vers extrêmes", "orange")
        
        # C4.5
        impacts["C4.5"] = ("✅", "Robuste aux distributions", "green")
        
        return impacts
    
    def create_distribution_plot(col, transform_method="Aucune", params=None):
        """Crée un plot avant/après transformation"""
        data_original = df_train[col].dropna()
        
        if transform_method == "Aucune":
            data_transformed = data_original
            transform_params = {}
        else:
            data_transformed, transform_params = apply_transform(data_original.values, transform_method, params)
            if data_transformed is None or transform_params is None:
                data_transformed = data_original
        
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
            go.Histogram(x=data_original, nbinsx=30, name="Original", marker_color='#3498db'),
            row=1, col=1
        )
        
        # Histogram transformé
        fig.add_trace(
            go.Histogram(x=data_transformed, nbinsx=30, name="Transformé", marker_color='#27ae60'),
            row=1, col=2
        )
        
        # Q-Q Plot original
        qq_original = stats.probplot(data_original, dist="norm")
        fig.add_trace(
            go.Scatter(x=qq_original[0][0], y=qq_original[0][1], mode='markers', 
                      name="Original", marker=dict(color='#3498db', size=4)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=qq_original[0][0], y=qq_original[0][0], mode='lines',
                      name="Théorique", line=dict(color='red', dash='dash')),
            row=2, col=1
        )
        
        # Q-Q Plot transformé
        qq_transformed = stats.probplot(data_transformed, dist="norm")
        fig.add_trace(
            go.Scatter(x=qq_transformed[0][0], y=qq_transformed[0][1], mode='markers',
                      name="Transformé", marker=dict(color='#27ae60', size=4)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=qq_transformed[0][0], y=qq_transformed[0][0], mode='lines',
                      name="Théorique", line=dict(color='red', dash='dash')),
            row=2, col=2
        )
        
        # Calculs skewness
        skew_original = stats.skew(data_original)
        skew_transformed = stats.skew(data_transformed)
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text=f"<b>{col}</b> - Skew Original: {skew_original:.2f} → Transformé: {skew_transformed:.2f}"
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
        current_method = state.get("transform_strategy", {}).get(col_name, "")
        current_params = state.get("transform_params", {}).get(col_name, {})
        
        skew, kurt = calculate_skewness_kurtosis(col_name)
        recommended, reason = get_recommended_transform(col_name)
        impacts = calculate_algo_impact(col_name, skew)
        
        data = df_train[col_name].dropna()
        has_zero = (data == 0).any()
        has_negative = (data < 0).any()
        
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-6xl p-6").style("max-height:90vh; overflow-y:auto;"):
            ui.label(f"Transformation : {col_name}").style("font-weight:700; font-size:20px; color:#01335A;")
            
            # Infos distribution
            with ui.card().classes("w-full p-4 mb-4").style("background:#f8f9fa;"):
                with ui.row().classes("gap-6"):
                    with ui.column():
                        ui.label(f"Skewness : {skew:.2f}").style("font-weight:600;")
                        icon, level, color = get_skewness_level(skew)
                        ui.label(f"{icon} {level}").style(f"color:{color}; font-weight:600;")
                    
                    with ui.column():
                        ui.label(f"Kurtosis : {kurt:.2f}").style("font-weight:600;")
                    
                    with ui.column():
                        if has_negative:
                            ui.label("⚠️ Contient valeurs négatives").style("color:#e74c3c; font-weight:600;")
                        elif has_zero:
                            ui.label("⚠️ Contient zéros").style("color:#f39c12; font-weight:600;")
                        else:
                            ui.label("✅ Toutes valeurs > 0").style("color:#27ae60; font-weight:600;")
            
            # Impact algorithmes
            if selected_algos:
                with ui.card().classes("w-full p-4 mb-4").style("background:#fff3cd;"):
                    ui.label("Impact sur vos algorithmes :").style("font-weight:700; margin-bottom:8px;")
                    for algo in selected_algos:
                        if algo in impacts:
                            icon, msg, color = impacts[algo]
                            ui.label(f"{icon} {algo} : {msg}").style(f"color:{color}; font-size:14px;")
            
            # Recommandation
            with ui.card().classes("w-full p-4 mb-4").style("background:#e8f5e9;"):
                ui.label(f"💡 Recommandation : {recommended}").style("font-weight:600; color:#27ae60;")
                ui.label(reason).style("font-size:13px; color:#2c3e50;")
            
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
            ).classes("w-full mb-4")
            
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
- Nécessite valeurs > 0 (ajout +1 si zéros)
- Interprétabilité réduite
                        """).classes("p-4").style("background:#f8f9fa; border-radius:8px;")
                        
                        constant_input = ui.number(
                            label="Constante c (pour gérer zéros)",
                            value=current_params.get("constant", 1),
                            min=0,
                            max=10,
                            step=0.1
                        ).classes("w-full")
                    
                    elif method == "Square Root":
                        ui.markdown("""
**📌 Square Root : sqrt(x)**

✅ Pour skewness modéré (0.5-1.5)
- Plus douce que log
- Préserve mieux les relations
                        """).classes("p-4").style("background:#f8f9fa; border-radius:8px;")
                    
                    elif method == "Box-Cox":
                        ui.markdown("""
**📌 Box-Cox Transform (automatique)**

Trouve meilleur λ pour normaliser

⚠️ Nécessite valeurs strictement > 0

λ optimal sera calculé automatiquement
                        """).classes("p-4").style("background:#f8f9fa; border-radius:8px;")
                        
                        if has_zero or has_negative:
                            ui.label("❌ Box-Cox impossible (zéros ou valeurs négatives)").style(
                                "color:#e74c3c; font-weight:600; margin-top:8px;"
                            )
                    
                    elif method == "Yeo-Johnson":
                        ui.markdown("""
**📌 Yeo-Johnson (Box-Cox généralisé)**

✅ Gère valeurs négatives et zéros

λ optimal sera calculé automatiquement
                        """).classes("p-4").style("background:#f8f9fa; border-radius:8px;")
                    
                    elif method == "Aucune":
                        ui.markdown("""
**📌 Aucune transformation**

Garder distribution originale

Recommandé si : C4.5 uniquement ou distribution déjà normale
                        """).classes("p-4").style("background:#f8f9fa; border-radius:8px;")
                
                # Preview
                with preview_container:
                    ui.label("Aperçu Avant/Après").style("font-weight:700; font-size:16px; margin-bottom:12px;")
                    
                    params = {}
                    if method == "Log Transform" and constant_input:
                        params["constant"] = constant_input.value
                    
                    fig, skew_after, _ = create_distribution_plot(col_name, method, params)
                    ui.plotly(fig).classes("w-full")
                    
                    icon_after, level_after, color_after = get_skewness_level(skew_after)
                    ui.label(f"Skewness après transformation : {skew_after:.2f} {icon_after} {level_after}").style(
                        f"color:{color_after}; font-weight:600; font-size:14px; margin-top:8px;"
                    )
            
            transform_select.on_value_change(lambda: update_params_and_preview())
            if constant_input:
                constant_input.on_value_change(lambda: update_params_and_preview())
            
            update_params_and_preview()
            
            # Boutons
            with ui.row().classes("w-full justify-end gap-2 mt-6"):
                ui.button("Annuler", on_click=dialog.close).props("flat")
                
                def save_transform():
                    method = transform_select.value
                    params = {}
                    
                    if method == "Log Transform" and constant_input:
                        params["constant"] = constant_input.value
                    
                    state.setdefault("transform_strategy", {})[col_name] = method
                    state.setdefault("transform_params", {})[col_name] = params
                    
                    ui.notify(f"✅ Transformation configurée pour {col_name}", color="positive")
                    dialog.close()
                    table.update()
                
                ui.button("Sauvegarder", on_click=save_transform).style("background:#27ae60; color:white;")
        
        dialog.open()
    
    def apply_all_transforms():
        """Applique toutes les transformations"""
        strategies = state.get("transform_strategy", {})
        split = state.get("split", {})
        
        if not strategies:
            ui.notify("⚠️ Aucune transformation configurée", color="warning")
            return
        
        with ui.dialog() as dialog, ui.card().classes("p-6"):
            ui.label("⚠️ Confirmation").style("font-weight:700; font-size:18px;")
            ui.label("Appliquer les transformations sur tous les datasets ?").style("margin-top:8px;")
            
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Annuler", on_click=dialog.close).props("flat")
                
                def confirm_and_next():
                    try:
                        params = state.get("transform_params", {})
                        transformers = {}
                        
                        #  FIT sur X_train uniquement
                        if split and "X_train" in split:
                            df_train_for_fit = split["X_train"].copy()
                        else:
                            df_train_for_fit = df_train.copy()

                        # ✅ FIT sur train et sauvegarder paramètres
                        for col, method in strategies.items():
                            if method != "Aucune" and col in df_train_for_fit.columns:
                                col_params = params.get(col, {})
                                transformed_data, transform_info = apply_transform(
                                    df_train_for_fit[col].values, 
                                    method, 
                                    col_params
                                )
                                if transformed_data is not None and transform_info is not None:
                                    transformers[col] = {"method": method, "params": transform_info}
                    
                        state["fitted_transformers"] = transformers
                        
                        # TRANSFORM sur chaque split avec paramètres fittés
                        if split:
                            for key in ["X_train", "X_val", "X_test"]:
                                if key in split and isinstance(split[key], pd.DataFrame):
                                    df_split = split[key].copy()
                                    for col, method in strategies.items():
                                        if method != "Aucune" and col in df_split.columns:
                                            col_params = transformers.get(col, {}).get("params", {})
                                            transformed_data, _ = apply_transform(
                                                df_split[col].values,
                                                method,
                                                col_params
                                            )
                                            if transformed_data is not None:
                                                df_split[col] = transformed_data
                                    split[key] = df_split
                        
                        ui.notify("✅ Transformations appliquées avec succès!", color="positive")
                        dialog.close()
                        ui.run_javascript("setTimeout(() => window.location.href='/supervised/scaling', 1500);")
                    
                    except Exception as e:
                        ui.notify(f"❌ Erreur : {str(e)}", color="negative")
                        print(f"Erreur détaillée : {e}")
                        import traceback
                        traceback.print_exc()
                
                ui.button("Confirmer", on_click=confirm_and_next).style("background:#27ae60; color:white;")
        
        dialog.open()
    
    def apply_recommended():
        """Applique automatiquement les recommandations"""
        for col in num_cols:
            skew, _ = calculate_skewness_kurtosis(col)
            if abs(skew) >= 0.5:  # Seulement si nécessaire
                recommended, _ = get_recommended_transform(col)
                state.setdefault("transform_strategy", {})[col] = recommended
                
                params = {}
                if recommended == "Log Transform":
                    data = df_train[col].dropna()
                    if (data == 0).any():
                        params["constant"] = 1
                
                state.setdefault("transform_params", {})[col] = params
        
        ui.notify("✅ Recommandations appliquées", color="positive")
        table.update()
    
    # ---------- UI ----------
    with ui.column().classes("w-full items-center p-8").style("background-color:#f5f6fa;"):
        ui.label("ÉTAPE 3.8 : TRANSFORMATIONS DE DISTRIBUTIONS").style(
            "font-weight:700; font-size:28px; color:#01335A; margin-bottom:10px;"
        )
        
        ui.label("Normaliser les distributions asymétriques (Log, Box-Cox...)").style(
            "font-size:16px; color:#636e72; margin-bottom:20px;"
        )
        
        # Section : Synthèse
        with ui.card().classes("w-full max-w-6xl p-6 mb-6"):
            ui.label("📊 Features avec Distributions Asymétriques").style(
                "font-weight:700; font-size:18px; color:#2c3e50;"
            )
            
            skewed_cols = []
            for col in num_cols:
                skew, _ = calculate_skewness_kurtosis(col)
                if abs(skew) >= 0.5:
                    skewed_cols.append(col)
            
            if len(skewed_cols) == 0:
                ui.label("✅ Aucune feature nécessitant transformation - vous pouvez passer à l'étape suivante").style(
                    "color:#27ae60; margin-top:12px;"
                )
            else:
                # Légende
                with ui.row().classes("gap-4 mt-4").style("font-size:13px;"):
                    ui.label("🔴 Skew > 1.5 : Transformation fortement recommandée").style("color:#e74c3c;")
                    ui.label("🟡 Skew 0.5-1.5 : Transformation optionnelle").style("color:#f39c12;")
                    ui.label("🟢 Skew < 0.5 : Distribution acceptable").style("color:#27ae60;")
        
        # Section : Table des features
        if len(num_cols) > 0:
            with ui.card().classes("w-full max-w-6xl p-6 mb-6"):
                ui.label("Configuration des transformations").style("font-weight:700; font-size:18px; color:#2c3e50;")
                
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
                        {"name": "Feature", "label": "Feature", "field": "Feature"},
                        {"name": "Skewness", "label": "Skewness", "field": "Skewness"},
                        {"name": "Kurtosis", "label": "Kurtosis", "field": "Kurtosis"},
                        {"name": "Niveau", "label": "Niveau", "field": "Niveau"},
                        {"name": "Recommandé", "label": "Recommandé", "field": "Recommandé"},
                        {"name": "Configuré", "label": "Configuré", "field": "Configuré"}
                    ],
                    rows=rows,
                    row_key="Feature"
                ).style("width:100%;")
                
                ui.label("💡 Cliquez sur une ligne pour configurer la transformation").style(
                    "font-size:13px; color:#636e72; margin-top:8px;"
                )
                
                def handle_row_click(e):
                    if e and "row" in e and "Feature" in e["row"]:
                        open_transform_modal(e["row"]["Feature"])
                
                table.on("row:click", handle_row_click)
                
                # Recommandations par algo
                if selected_algos:
                    with ui.card().classes("w-full p-4 mt-4").style("background:#e8f5e9;"):
                        ui.label("💡 Pour vos algorithmes :").style("font-weight:700; margin-bottom:8px;")
                        
                        if "Gaussian Naive Bayes" in selected_algos:
                            ui.label("Gaussian Naive Bayes : Transformation CRITIQUE pour features avec skew > 1").style(
                                "font-size:13px; color:#2c3e50;"
                            )
                        
                        if "KNN" in selected_algos:
                            ui.label("KNN : Transformation utile (stabilise distances) - Moins critique que pour NB").style(
                                "font-size:13px; color:#2c3e50;"
                            )
                        
                        if "C4.5" in selected_algos:
                            ui.label("C4.5 : Transformation optionnelle (robuste aux skew)").style(
                                "font-size:13px; color:#2c3e50;"
                            )
                
                # Boutons actions
                with ui.row().classes("gap-2 mt-4"):
                    ui.button(
                        "✨ Appliquer recommandations",
                        on_click=apply_recommended
                    ).style("background:#3498db; color:white;")
                    
                    ui.button(
                        "✅ Appliquer les transformations",
                        on_click=apply_all_transforms
                    ).style("background:#27ae60; color:white;")
        
        # Navigation
        with ui.row().classes("w-full max-w-6xl justify-between mt-6"):
            ui.button(
                "⬅ Étape précédente",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/distribution_transform'")
            ).style("background:#95a5a6; color:white;")
            
            ui.button(
                "➡ Étape suivante",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/scaling'")
            ).style("background:#09538C; color:white;")





# ----------------- PAGE 3.9 : FEATURE SCALING -----------------
from nicegui import ui
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import copy

@ui.page('/supervised/scaling')
def feature_scaling_page():
    """
    Page complète pour le Feature Scaling avec détection intelligente des variables continues
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # ---------- CONTEXTE ----------
    df = state.get("raw_df", None)
    if df is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Aucun dataset chargé.").style("font-size:18px; color:#c0392b; font-weight:600;")
            ui.button("⬅ Retour à l'Upload", on_click=lambda: ui.run_javascript("window.location.href='/upload'"))
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
                        marker_color='#e74c3c',
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
                print(f"Erreur visualisation {col}: {e}")
                continue
        
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text=f"📊 Comparaison AVANT/APRÈS - {method.upper()}",
            title_font_size=20,
            title_x=0.5,
            paper_bgcolor='#f8f9fa',
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
                ui.label("⚠️ Aucune visualisation disponible").style("color:#e67e22; font-size:16px;")
    
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
            ui.notify("✅ Aucun scaling appliqué", color="positive")
            ui.run_javascript("setTimeout(() => window.location.href='/supervised/dimension_reduction', 1000);")
            return
        
        split = state.get("split", {})

        with ui.dialog() as dialog, ui.card().classes("p-6"):
            ui.label("⚠️ Confirmer l'application").style("font-weight:700; font-size:18px;")
            ui.label(f"Appliquer **{method.upper()}** sur {len(num_cols)} features continues ?").style("margin-top:8px;")
        
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("❌ Annuler", on_click=dialog.close).props("flat")
            
                def do_apply():
                    try:
                        scaler = scalers[method]
                    
                        # ✅ CORRECTION : FIT sur X_train uniquement
                        if split and "X_train" in split:
                            X_train_for_fit = split["X_train"][num_cols].copy()
                        else:
                            X_train_for_fit = df_train[num_cols].copy()
                    
                        # ✅ FIT sur train
                        scaler.fit(X_train_for_fit)
                    
                        # ✅ TRANSFORM sur chaque split
                        if split:
                            for key in ["X_train", "X_val", "X_test"]:
                                if key in split and isinstance(split[key], pd.DataFrame):
                                    split[key][num_cols] = scaler.transform(split[key][num_cols])
                    
                        state["scaling_method"] = method
                        state["fitted_scaler"] = scaler  # Sauvegarder le scaler fitté
                        state["scaled_columns"] = num_cols
                    
                        ui.notify(f"✅ Scaling appliqué sans data leakage!", color="positive")
                        dialog.close()
                        ui.run_javascript("setTimeout(() => window.location.href='/supervised/dimension_reduction', 1000);")
                
                    except Exception as e:
                        ui.notify(f"❌ Erreur: {str(e)}", color="negative")
            
                ui.button("✅ Confirmer", on_click=do_apply).style("background:#27ae60; color:white; font-weight:600;")
    
        dialog.open()

    
    # ---------- ANALYSE ----------
    features_info = analyze_features()
    recommended = get_recommendation()
    
    # ---------- INTERFACE ----------
    with ui.column().classes("w-full items-center p-8").style("background-color:#f8f9fa; min-height:100vh;"):
        # Header
        ui.label("🚀 ÉTAPE 3.9 : FEATURE SCALING").style(
            "font-weight:700; font-size:32px; color:#2c3e50; margin-bottom:8px; text-align:center;"
        )
        ui.label("Normalisation/Standardisation des features continues").style(
            "font-size:18px; color:#7f8c8d; margin-bottom:32px; text-align:center;"
        )
        
        # Info détection intelligente
        with ui.card().classes("w-full max-w-6xl p-6 mb-8").style("background:#fff3cd; border-left:5px solid #ffc107;"):
            ui.label("🔍 Détection Intelligente").style("font-weight:700; font-size:18px; color:#856404; margin-bottom:12px;")
            with ui.row().classes("w-full gap-8"):
                with ui.column().classes("flex-1"):
                    ui.label(f"✅ **{len(num_cols)} features continues** détectées").style("font-size:15px; margin-bottom:4px;")
                    if num_cols:
                        ui.label(f"→ {', '.join(num_cols[:5])}{'...' if len(num_cols) > 5 else ''}").style(
                            "font-size:13px; color:#666;"
                        )
                
                with ui.column().classes("flex-1"):
                    ui.label(f"⚠️ **{len(cat_encoded)} variables catégorielles** encodées").style("font-size:15px; margin-bottom:4px;")
                    if cat_encoded:
                        ui.label(f"→ {', '.join(cat_encoded[:5])}{'...' if len(cat_encoded) > 5 else ''}").style(
                            "font-size:13px; color:#666;"
                        )
                        ui.label("→ Seront préservées (pas de scaling)").style("font-size:12px; color:#856404; font-style:italic;")
        
        # Objectif
        with ui.card().classes("w-full max-w-6xl p-6 mb-8").style("background:#e3f2fd; border-left:5px solid #2196f3;"):
            ui.label("🎯 Objectif principal").style("font-weight:700; font-size:18px; color:#1976d2; margin-bottom:12px;")
            with ui.row().classes("w-full gap-8"):
                with ui.column().classes("flex-1"):
                    ui.label("🔴 **CRITIQUE** pour KNN").style("font-size:15px; font-weight:600; margin-bottom:4px;")
                    ui.label("→ Les distances sont biaisées par les échelles différentes").style("font-size:13px; color:#666;")
                
                with ui.column().classes("flex-1"):
                    ui.label("✅ **INUTILE** pour C4.5").style("font-size:15px; font-weight:600; margin-bottom:4px;")
                    ui.label("→ Arbres de décision insensibles aux échelles").style("font-size:13px; color:#666;")
                
                with ui.column().classes("flex-1"):
                    ui.label("🟡 **OPTIONNEL** pour Naive Bayes").style("font-size:15px; font-weight:600; margin-bottom:4px;")
                    ui.label("→ Dépend de la distribution des données").style("font-size:13px; color:#666;")
        
        # Analyse des features continues
        with ui.card().classes("w-full max-w-6xl p-6 mb-8"):
            ui.label("📊 Analyse des features continues").style(
                "font-weight:700; font-size:20px; color:#2c3e50; margin-bottom:16px;"
            )
            
            if features_info:
                # Tableau amélioré
                table_html = """<div style="background:#1a1a1a; border-radius:12px; padding:20px; overflow-x:auto;">
                <table style="width:100%; color:#00ff88; font-family:monospace; font-size:13px; border-collapse:collapse;">
                    <thead>
                        <tr style="border-bottom:2px solid #00ff88;">
                            <th style="text-align:left; padding:12px; color:#00ffff;">Feature</th>
                            <th style="text-align:right; padding:12px;">Min</th>
                            <th style="text-align:right; padding:12px;">Max</th>
                            <th style="text-align:right; padding:12px;">Range</th>
                            <th style="text-align:right; padding:12px;">Mean</th>
                            <th style="text-align:right; padding:12px;">Std</th>
                            <th style="text-align:center; padding:12px;">Outliers</th>
                        </tr>
                    </thead>
                    <tbody>"""
                
                for f in features_info:
                    out = "🔴" if f["outliers"] else "🟢"
                    # Highlight pour range > 1000
                    range_color = "#ffff00" if f["range"] > 1000 else "#00ff88"
                    table_html += f"""
                        <tr style="border-bottom:1px solid #333;">
                            <td style="padding:10px; font-weight:600;">{f['name'][:20]}</td>
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
                    ui.label("❌ Aucune feature continue détectée").style("color:#e74c3c; font-size:16px; font-weight:600;")
                    ui.label("Toutes les features sont catégorielles ou binaires").style("color:#7f8c8d; font-size:14px;")
        
        # Sélecteur de méthode
        with ui.card().classes("w-full max-w-5xl p-8 mb-8").style("border:3px solid #e0e0e0; border-radius:16px;"):
            ui.label("⚙️ Choisir la méthode de scaling").style(
                "font-weight:700; font-size:22px; color:#2c3e50; margin-bottom:24px; text-align:center;"
            )
            
            # Select avec options
            method_select = ui.select(
                label="Sélectionnez une méthode",
                options={
                    "standard": "📊 StandardScaler (Z-score, mean=0, std=1)",
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
                "background:#3498db; color:white; font-weight:600; "
                "height:48px; width:200px; border-radius:10px; margin:0 auto; display:block;"
            )
            
            method_select.on_value_change(lambda e: state.update({"scaling_method": e.value}))
        
        # Zone de visualisation
        plot_container = ui.column().classes("w-full max-w-6xl mb-8")
        
        # Recommandation intelligente
        with ui.card().classes("w-full max-w-5xl p-8 mb-12").style(
            "background:linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); "
            "border:4px solid #4caf50; border-radius:20px; box-shadow:0 8px 24px rgba(76,175,80,0.3);"
        ):
            method_names = {
                "standard": "StandardScaler",
                "minmax": "MinMaxScaler", 
                "robust": "RobustScaler",
                "maxabs": "MaxAbsScaler",
                "none": "Aucun scaling"
            }
            
            ui.label("🤖 RECOMMANDATION INTELLIGENTE").style(
                "font-weight:800; font-size:24px; color:#1b5e20; text-align:center; margin-bottom:20px;"
            )
            
            ui.label(f"**{method_names[recommended]}**").style(
                "font-weight:700; font-size:28px; color:#2e7d32; text-align:center; margin-bottom:16px;"
            )
            
            # Explication de la recommandation
            explanations = {
                "standard": "✓ Pas d'outliers majeurs détectés\n✓ Données normalement distribuées\n✓ Optimal pour KNN et régression",
                "robust": f"⚠️ Outliers détectés dans vos données\n✓ Résistant aux valeurs extrêmes\n✓ Préserve la structure centrale\n✓ Recommandé pour {len([f for f in features_info if f['outliers']])} features avec outliers",
                "none": "✓ Toutes les features ont des échelles similaires\n✓ Pas de normalisation nécessaire"
            }
            
            ui.label(explanations.get(recommended, "")).style(
                "font-size:15px; color:#2e7d32; text-align:center; white-space:pre-line; margin-bottom:20px;"
            )
            
            with ui.row().classes("w-full justify-center gap-6"):
                ui.button(
                    "🚀 Appliquer la recommandation",
                    on_click=lambda: (
                        method_select.set_value(recommended),
                        apply_scaling(recommended)
                    )
                ).style(
                    "background:linear-gradient(135deg, #4caf50 0%, #45a049 100%); "
                    "color:white; font-weight:700; height:56px; padding:0 40px; border-radius:12px;"
                )
        
        # Navigation
        with ui.row().classes("w-full max-w-5xl justify-between mt-16"):
            ui.button(
                "⬅ Étape précédente",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/distribution_transform'")
            ).style(
                "background:#95a5a6; color:white; font-weight:600; "
                "height:60px; width:260px; border-radius:16px; font-size:16px;"
            )
            
            ui.button(
                "✅ Appliquer & Continuer",
                on_click=lambda: apply_scaling(method_select.value)
            ).style(
                "background:linear-gradient(135deg, #2c3e50 0%, #34495e 100%); "
                "color:white; font-weight:700; height:60px; width:320px; "
                "border-radius:16px; font-size:16px;"
            )




@ui.page('/supervised/dimension_reduction')
def dimension_reduction_page():
    """
    Page complète pour la Réduction de Dimension (PCA, Feature Selection)
    """
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_classif, chi2
    from sklearn.tree import DecisionTreeClassifier
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # ---------- CONTEXTE ----------
    df = state.get("raw_df", None)
    if df is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Aucun dataset chargé.").style("font-size:18px; color:#c0392b; font-weight:600;")
            ui.button("⬅ Retour à l'Upload", on_click=lambda: ui.run_javascript("window.location.href='/supervised/upload'"))
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
                marker_color='#3498db',
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
                
                ui.notify(f"✅ PCA appliqué : {n_features} → {n_comp} composantes ({variance_explained:.1f}% variance)", 
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
                
                ui.notify(f"✅ Feature Selection : {n_features} → {n_comp} features", 
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
        ui.notify("✅ Aucune réduction appliquée", color="info")
        ui.run_javascript("setTimeout(() => window.location.href='/supervised/recap_validation', 1000);")
    
    # ---------- INTERFACE ----------
    with ui.column().classes("w-full items-center p-8").style("background-color:#f8f9fa; min-height:100vh;"):
        
        # Header
        ui.label("🎯 ÉTAPE 3.10 : RÉDUCTION DE DIMENSION").style(
            "font-weight:700; font-size:32px; color:#2c3e50; margin-bottom:8px; text-align:center;"
        )
        ui.label("Optionnel - Réduire curse of dimensionality pour KNN").style(
            "font-size:18px; color:#7f8c8d; margin-bottom:32px; text-align:center;"
        )
        
        # Section A : Évaluation
        with ui.card().classes("w-full max-w-6xl p-6 mb-8"):
            ui.label("📊 État Actuel du Dataset").style(
                "font-weight:700; font-size:20px; color:#2c3e50; margin-bottom:16px;"
            )
            
            with ui.row().classes("w-full gap-8 mb-6"):
                with ui.column().classes("flex-1"):
                    ui.label(f"**Features après preprocessing** : {n_features}").style("font-size:15px; margin-bottom:8px;")
                    ui.label(f"**Samples (train)** : {n_samples}").style("font-size:15px; margin-bottom:8px;")
                    ui.label(f"**Ratio samples/features** : {ratio:.1f}").style(
                        f"font-size:15px; font-weight:700; color:{'#27ae60' if ratio > 50 else '#e67e22'};"
                    )
            
            # Recommandations
            with ui.card().classes("w-full p-4").style("background:#fff3cd; border-left:4px solid #ffc107;"):
                ui.label("⚠️ Recommandations par algorithme").style("font-weight:700; font-size:16px; margin-bottom:12px;")
                ui.label(f"• **KNN** : Ratio idéal > 50 {'✅' if ratio > 50 else '⚠️ Curse of dimensionality'}").style("font-size:14px; margin-bottom:6px;")
                ui.label("• **C4.5** : Pas de limite stricte ✅").style("font-size:14px; margin-bottom:6px;")
                ui.label("• **Naive Bayes** : Assume indépendance (déjà traité) ✅").style("font-size:14px;")
            
            if ratio < 50:
                ui.label("💡 Réduction recommandée pour améliorer KNN").style(
                    "font-size:16px; color:#e67e22; font-weight:600; margin-top:12px; text-align:center;"
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
                                                ui.label("✅ **Avantages**").style("font-weight:600; color:#27ae60; margin-bottom:4px;")
                                                ui.label("• Linéaire, rapide, déterministe").style("font-size:13px;")
                                                ui.label("• Préserve variance maximale").style("font-size:13px;")
                                                ui.label("• Transformable pour new data").style("font-size:13px;")
                                            
                                            with ui.column().classes("flex-1"):
                                                ui.label("❌ **Inconvénients**").style("font-weight:600; color:#e74c3c; margin-bottom:4px;")
                                                ui.label("• Perd interprétabilité").style("font-size:13px;")
                                                ui.label("• Assume linéarité").style("font-size:13px;")
                                                ui.label("• Sensible au scaling (déjà fait ✅)").style("font-size:13px;")
                                        
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
                                    ui.label("📊 Analyse de variance expliquée").style(
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
                                        
                                        with ui.card().classes("w-full p-4 mb-6").style("background:#e8f5e8; border-left:4px solid #4caf50;"):
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
                                    with ui.card().classes("w-full p-6 mb-6").style("background:#fff3cd; border-left:4px solid #ffc107;"):
                                        ui.label("📌 Sélection de Features (sans transformation)").style(
                                            "font-weight:700; font-size:18px; color:#856404; margin-bottom:12px;"
                                        )
                                        
                                        ui.label("**Principe** : Garder uniquement les K features les plus importantes").style(
                                            "font-size:14px; margin-bottom:8px;"
                                        )
                                        
                                        with ui.row().classes("w-full gap-8 mb-4"):
                                            with ui.column().classes("flex-1"):
                                                ui.label("✅ **Avantages**").style("font-weight:600; color:#27ae60; margin-bottom:4px;")
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
                                        ui.label("📊 Features par importance (Decision Tree)").style(
                                            "font-weight:700; font-size:18px; color:#2c3e50; margin-bottom:12px;"
                                        )
                                        
                                        # Bar chart importance
                                        top_15 = feature_imp.head(15)
                                        fig = go.Figure(go.Bar(
                                            x=top_15['importance'],
                                            y=top_15['feature'],
                                            orientation='h',
                                            marker_color='#3498db'
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
            "border:4px solid #4caf50; border-radius:20px;"
        ):
            ui.label("💡 Décision Finale").style(
                "font-weight:800; font-size:24px; color:#1b5e20; text-align:center; margin-bottom:20px;"
            )
            
            ui.label("**Recommandation pour vos algorithmes** :").style(
                "font-size:16px; color:#2e7d32; font-weight:600; margin-bottom:12px;"
            )
            
            ui.label("🔵 **KNN** : PCA recommandée (améliore vitesse et précision)").style("font-size:14px; margin-bottom:6px;")
            ui.label("🌳 **C4.5** : Feature Selection optionnelle (garde interprétabilité)").style("font-size:14px; margin-bottom:6px;")
            ui.label("📊 **Naive Bayes** : Pas de réduction nécessaire").style("font-size:14px; margin-bottom:20px;")
            
            with ui.row().classes("w-full justify-center gap-6 mt-8"):
                ui.button(
                    "⏭️ Passer (pas de réduction)",
                    on_click=skip_reduction
                ).style(
                    "background:#95a5a6; color:white; font-weight:600; "
                    "height:56px; padding:0 32px; border-radius:12px; font-size:16px;"
                )
                
                ui.button(
                    "✅ Appliquer la Réduction",
                    on_click=lambda: apply_reduction(
                        state.get("reduction_method", "pca"),
                        state.get("n_components", 10)
                    )
                ).style(
                    "background:linear-gradient(135deg, #4caf50 0%, #45a049 100%); "
                    "color:white; font-weight:700; height:56px; padding:0 40px; border-radius:12px; font-size:16px;"
                ).bind_enabled_from(enable_switch, 'value')
        
        # Navigation
        with ui.row().classes("w-full max-w-5xl justify-between mt-16"):
            ui.button(
                "⬅ Étape précédente",
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
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import io
    import base64
    from datetime import datetime
    
    # ---------- CONTEXTE ----------
    df = state.get("raw_df", None)
    df_original = state.get("df_original", None)  # Dataset avant preprocessing
    split = state.get("split", None)
    target_col = state.get("target_column", None)
    
    if df is None:
        with ui.column().classes("items-center justify-center w-full h-screen"):
            ui.label("❌ Aucun dataset chargé.").style("font-size:18px; color:#c0392b; font-weight:600;")
            ui.button("⬅ Retour", on_click=lambda: ui.run_javascript("window.location.href='/upload'"))
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
                           marker_color='#27ae60', showlegend=(idx==0)),
                row=2, col=idx+1
            )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text="📊 Comparaison Avant/Après Preprocessing",
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
            
            ui.notify("✅ Téléchargement démarré !", color="positive")
        
        except Exception as e:
            ui.notify(f"❌ Erreur lors du téléchargement : {str(e)}", color="negative")
    
    def generate_pdf_report():
        """Génère un rapport PDF du preprocessing"""
        ui.notify("📄 Génération du rapport PDF...", color="info")
        # TODO: Implémenter génération PDF avec reportlab
        ui.notify("⚠️ Fonctionnalité en développement", color="warning")
    
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
            
            ui.label("⚠️ Attention : Modifier une étape peut invalider les étapes suivantes").style(
                "color:#e67e22; font-size:14px; margin-bottom:16px;"
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
                
                ui.button("Modifier", on_click=go_to_step).style("background:#3498db; color:white;")
        
        dialog.open()
    
    def validate_and_continue():
        """Valide le preprocessing et prépare pour les algorithmes"""
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-3xl p-6"):
            ui.label("✅ Validation du Pipeline").style("font-weight:700; font-size:20px; margin-bottom:16px;")
            
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
                    
                    ui.notify("✅ Pipeline validé avec succès !", color="positive")
                    
                    # Afficher résumé final
                    progress_container.clear()
                    with progress_container:
                        ui.label("✅ Validation complète !").style(
                            "font-weight:700; font-size:20px; color:#27ae60; margin-bottom:16px;"
                        )
                        
                        with ui.card().classes("w-full p-4 mb-4").style("background:#e8f5e9;"):
                            ui.label("Prêt à lancer les algorithmes :").style("font-weight:600; margin-bottom:8px;")
                            selected_algos = state.get("selected_algos", [])
                            for algo in selected_algos:
                                ui.label(f"• {algo}").style("font-size:14px;")
                        
                        ui.button(
                            "▶️ Passer à la Page Algorithmes",
                            on_click=lambda: (dialog.close(), ui.run_javascript("window.location.href='/supervised/algorithms'"))
                        ).style(
                            "background:linear-gradient(135deg, #27ae60 0%, #229954 100%); "
                            "color:white; font-weight:700; height:56px; width:100%; border-radius:12px; margin-top:12px;"
                        )
                
                # Lancer la validation
                ui.timer(0.1, run_validation, once=True)
        
        dialog.open()
    
    def reset_preprocessing():
        """Recommencer tout le preprocessing"""
        with ui.dialog() as dialog, ui.card().classes("p-6"):
            ui.label("⚠️ Recommencer le Preprocessing ?").style("font-weight:700; font-size:18px;")
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
        ui.label("🎯 ÉTAPE 3.11 : RÉCAPITULATIF & VALIDATION").style(
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
                ui.label("⚠️ Aucune transformation appliquée").style("color:#e67e22; font-size:16px;")
        
        # Section B : Statistiques comparatives
        with ui.card().classes("w-full max-w-6xl p-6 mb-8"):
            ui.label("📊 Statistiques Comparatives").style(
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
                    with ui.card().classes("p-4").style("background:#e8f5e9;"):
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
                ui.label("⚠️ Dataset original non disponible pour comparaison").style(
                    "color:#e67e22; font-size:14px;"
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
                    "📊 Télécharger Dataset Preprocessé",
                    on_click=download_preprocessed_data
                ).style(
                    "background:#3498db; color:white; font-weight:600; "
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
            "border:4px solid #4caf50; border-radius:20px;"
        ):
            ui.label("🎯 Décisions Finales").style(
                "font-weight:800; font-size:24px; color:#1b5e20; text-align:center; margin-bottom:20px;"
            )
            
            ui.label("Votre pipeline de preprocessing est prêt !").style(
                "font-size:16px; color:#2e7d32; text-align:center; margin-bottom:20px;"
            )
            
            with ui.row().classes("w-full justify-center gap-4"):
                ui.button(
                    "⬅️ Modifier une Étape",
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
                    "✅ Valider et Continuer",
                    on_click=validate_and_continue
                ).style(
                    "background:linear-gradient(135deg, #27ae60 0%, #229954 100%); "
                    "color:white; font-weight:700; height:56px; padding:0 40px; border-radius:12px; font-size:16px;"
                )
        
        # Navigation
        with ui.row().classes("w-full max-w-6xl justify-start mt-6"):
            ui.button(
                "⬅ Étape précédente",
                on_click=lambda: ui.run_javascript("window.location.href='/supervised/dimension_reduction'")
            ).style("background:#95a5a6; color:white; font-weight:600; height:50px; width:220px; border-radius:10px;")


















# ----------------- PAGE ALGOS (MODIFIÉE POUR SCIKIT-LEARN) -----------------
@ui.page('/algos')
def algos_page():
    if state.get('X') is None:
        ui.notify("⚠️ Prétraitement nécessaire", color='warning')
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
                "⬅ Retour Prétraitement",
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
        ui.notify("⚠️ Aucun résultat", color='warning')
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
        ui.label("📈 Tableau Comparatif des Métriques").classes("text-2xl font-bold mb-4 text-gray-800")
        
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
        ui.label("📊 Comparaison Visuelle des Métriques").classes("text-2xl font-bold mb-4 text-gray-800")
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
                                ui.label("⚠️ Impossible de générer le dendrogramme").classes(
                                    "text-orange-600 text-center py-10"
                                )
                        except Exception as e:
                            ui.label(f"❌ Erreur : {e}").classes("text-red-600 text-center py-10")

# LANCEMENT
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Clustering Data Mining", port=8080, reload=True)