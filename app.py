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

            X = df_active.drop(columns=[target_col])
            y = df_active[target_col]
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, train_size=tr, random_state=int(seed_input.value), stratify=stratify_col
            )
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
            for name, y_set in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
                counts = y_set.value_counts()
                total = len(y_set)
                ui.label(f"{name} set : {total} samples").style("font-weight:600; color:#01335A; margin-top:8px;")
                for cls, cnt in counts.items():
                    pct = round(cnt / total * 100, 1)
                    ui.label(f"  - {cls}: {cnt} ({pct}%)").style("font-family:monospace; margin-left:10px;")

            ui.notify("✅ Split effectué avec succès !", color="positive")

        ui.button("✅ Effectuer le split", on_click=do_split).style(
            "background:linear-gradient(135deg, #01335A, #09538C); color:white; font-weight:600; border-radius:8px; height:46px; width:250px; margin-top:20px;"
        )



















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