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
from pyclustering.cluster.kmedoids import kmedoids
from scipy.cluster.hierarchy import linkage, dendrogram

from custom_cluster_optimized import KMeansCustom, DBSCANCustom, AgnesCustom, diagnose_dbscan, DianaCustom

# ----------------- ÉTAT GLOBAL -----------------
state = {
    "raw_df": None,
    "numeric_columns": None,
    "X": None,
    "X_pca": None,
    "results": {}
}

# ----------------- FONCTIONS UTILITAIRES -----------------
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

def run_kmedoids(X, n_clusters, random_state=None):
    data = X.tolist()
    if random_state is not None:
        random.seed(random_state)
    init = random.sample(range(len(data)), n_clusters)
    kmed = kmedoids(data, init)
    kmed.process()
    clusters = kmed.get_clusters()
    medoids = kmed.get_medoids()
    labels = np.full(len(X), -1)
    for cid, pts in enumerate(clusters):
        for idx in pts:
            labels[idx] = cid
    return labels, medoids

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

def plot_grouped_histogram(metrics_dict, title="Comparaison des algos"):
    df = pd.DataFrame(metrics_dict).T
    df_numeric = df[['silhouette','davies_bouldin','calinski_harabasz']].fillna(0)
    metrics = df_numeric.columns
    algos = df_numeric.index
    x = range(len(algos))
    width = 0.2

    plt.figure(figsize=(10,5))
    for i, metric in enumerate(metrics):
        plt.bar([p + i*width for p in x], df_numeric[metric], width=width, label=metric.replace('_',' ').title())
    plt.xticks([p + width for p in x], [a.upper() for a in algos])
    plt.ylabel("Valeur")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

# ----------------- PAGE ACCUEIL (CHOIX MODE) -----------------

@ui.page('/')
def home_page():

    ui.add_head_html("""
    <style>
        body {
            background-color: #f5f6fa;
        }
        .main-title {
            font-weight: 700;
            font-size: 36px;
            color: #2c3e50;
        }
        .sub-title {
            color: #636e72;
            font-size: 16px;
        }
    </style>
    """)

    with ui.column().classes("w-full h-screen items-center justify-center gap-8"):

        ui.label("Plateforme de Clustering").classes("main-title")
        ui.label("Veuillez choisir le mode d'analyse").classes("sub-title")

        with ui.row().classes("gap-10"):

            # -------- NON SUPERVISÉ --------
            with ui.card().classes("p-8 w-72 shadow-md rounded-xl hover:shadow-xl transition"):
               
                ui.label("Clustering Non Supervisé").classes("text-xl font-bold mt-2")
               

                ui.button(
                "Commencer",
                on_click=lambda: ui.run_javascript("window.location.href='/upload'")
            ).classes("mt-6 w-full h-12 text-base")


            # -------- SUPERVISÉ --------
            with ui.card().classes("p-8 w-72 shadow-md rounded-xl hover:shadow-xl transition"):
               
                ui.label("Clustering Supervisé").classes("text-xl font-bold mt-2")
                ui.label("À bientôt").classes("text-gray-500 text-sm mt-1")

                ui.button(
                "Accéder",
                on_click=lambda: ui.notify(" À bientôt !", color="warning")
            ).classes("mt-6 w-full h-12 text-base")


# ----------------- PAGE UPLOAD (DESIGN MODERNE - CORRIGÉE) -----------------
@ui.page('/upload')
def upload_page():

    ui.add_head_html("""
    <style>
        body {
            background-color: #f5f6fa;
        }
        .upload-title {
            font-weight: 700;
            font-size: 28px;
            color: #2c3e50;
        }
        .upload-sub {
            color: #636e72;
            font-size: 14px;
        }
    </style>
    """)

    with ui.column().classes("w-full h-screen items-center justify-center"):

        with ui.card().classes("p-10 w-[420px] shadow-lg rounded-xl"):

            ui.label(" Chargement du Dataset").classes("upload-title text-center")
            ui.label("Importez votre fichier CSV pour commencer").classes("upload-sub text-center mb-6")

            status_label = ui.label("Aucun fichier chargé").classes("text-red-500 text-sm mb-4")

            #  Bouton suivant désactivé au début
            btn_next = ui.button("Suivant ➡").classes("w-1/2 h-11")
            btn_next.disable()

            async def on_upload(e):
                content = await e.file.read()
                df = pd.read_csv(io.BytesIO(content))
                state['raw_df'] = df

                status_label.text = f" Fichier chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes"
                status_label.classes("text-green-600")

                btn_next.enable() 

                ui.notify(
                    f"Dataset chargé avec succès !",
                    color='positive'
                )

            ui.upload(
                on_upload=on_upload,
                label="Sélectionner un fichier CSV"
            ).props('accept=".csv"').classes("w-full mb-4")

            ui.separator().classes("my-4")

            with ui.row().classes("w-full gap-4"):
                ui.button(
                    "⬅ Retour",
                    on_click=lambda: ui.run_javascript("window.location.href='/'")
                ).classes("w-1/2 h-11")

             
                btn_next.on_click(lambda: ui.run_javascript("window.location.href='/preprocess'"))


# ----------------- PAGE PREPROCESS -----------------

@ui.page('/preprocess')
def preprocess_page():

    if state['raw_df'] is None:
        ui.notify("⚠️ Aucun dataset chargé", color='warning')
        ui.run_javascript("window.location.href='/upload'")
        return

    df = state['raw_df']

    ui.add_head_html("""
    <style>
        body {
            background-color: #f5f6fa;
        }
        .pp-title {
            font-size: 26px;
            font-weight: 700;
            color: #2c3e50;
        }
        .pp-sub {
            font-size: 14px;
            color: #636e72;
        }
        .section-title {
            font-size: 18px;
            font-weight: 600;
            color: #2c3e50;
        }
    </style>
    """)

    with ui.column().classes("w-full min-h-screen items-center py-10 gap-8"):

        # ---------- TITRE ----------
        ui.label(" Prétraitement des Données").classes("pp-title")
        ui.label("Configurez les options avant le clustering").classes("pp-sub")

        # ---------- CARTE PRINCIPALE ----------
        with ui.card().classes("p-8 w-[1000px] shadow-lg rounded-xl"):

            ui.label(f" Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes") \
              .classes("section-title mb-4")

            # ---------- APERÇU DU DATASET ----------
            with ui.expansion(" Aperçu des données (10 premières lignes)", icon="table_view"):
                ui.table(
                    rows=df.head(10).to_dict('records'),
                    columns=[{"name": c, "label": c, "field": c} for c in df.columns]
                )

            ui.separator().classes("my-6")

            # ---------- OPTIONS DE PRETRAITEMENT ----------
            ui.label(" Options de prétraitement").classes("section-title mb-3")

            with ui.row().classes("gap-10 w-full"):

                drop_dup = ui.switch("Supprimer doublons", value=False)
                handle_missing = ui.switch("Remplir valeurs manquantes", value=True)
                onehot = ui.switch("Encodage OneHot", value=False)

                norm_select = ui.select(
                    ['none', 'minmax', 'zscore'],
                    value='minmax',
                    label='Normalisation'
                ).classes("w-48")

            ui.separator().classes("my-6")

            # ---------- BOUTONS ----------
            with ui.row().classes("w-full gap-6"):

                ui.button(
                    "⬅ Retour Upload",
                    on_click=lambda: ui.run_javascript("window.location.href='/upload'")
                ).classes("w-1/3 h-11")

                btn_apply = ui.button(" Appliquer le prétraitement").classes("w-1/3 h-11")

                ui.button(
                    "➡ Continuer au clustering",
                    on_click=lambda: ui.run_javascript("window.location.href='/algos'")
                ).classes("w-1/3 h-11")

        # ---------- ZONE DE RÉSULTAT ----------
        result_container = ui.column().classes("w-[1000px]")

        def apply():
            processed_df, X = preprocess_dataframe(
                df,
                handle_missing.value,
                None if norm_select.value == 'none' else norm_select.value,
                encode_onehot=onehot.value,
                drop_duplicates=drop_dup.value
            )

            state['X'] = X
            state['numeric_columns'] = list(processed_df.select_dtypes(include=[np.number]).columns)

            result_container.clear()

            with result_container:
                with ui.card().classes("p-6 shadow-md rounded-xl mt-6"):
                    ui.label(
                        f" Dataset après prétraitement : {processed_df.shape[0]} × {processed_df.shape[1]}"
                    ).classes("section-title mb-3")

                    ui.table(
                        rows=processed_df.head(10).to_dict('records'),
                        columns=[{"name": c, "label": c, "field": c} for c in processed_df.columns]
                    )

            ui.notify("Prétraitement appliqué avec succès", color='positive')

        btn_apply.on_click(apply)


# ----------------- PAGE ALGOS -----------------

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

        # ---------- TITRE ----------
        ui.label(" Algorithmes de Clustering").classes("algo-title")
        ui.label("Configurez les paramètres et lancez l’analyse").classes("text-gray-500")

        # ---------- GRILLE DES ALGOS ----------
        with ui.row().classes("gap-8 flex-wrap justify-center"):

            # -------- KMEANS --------
            with ui.card().classes("p-6 shadow-md rounded-xl algo-card"):
                ui.label("KMeans").classes("section-title mb-3")
                k_kmeans = ui.number("Nombre de clusters", value=3, min=2)
                kmeans_chk = ui.switch("Activer", value=True)

            # -------- KMEDOIDS --------
            with ui.card().classes("p-6 shadow-md rounded-xl algo-card"):
                ui.label("KMedoids").classes("section-title mb-3")
                k_kmed = ui.number("Nombre de clusters", value=3, min=2)
                kmed_chk = ui.switch("Activer", value=True)

            # -------- DBSCAN --------
            diag = diagnose_dbscan(X)
            with ui.card().classes("p-6 shadow-md rounded-xl algo-card"):
                ui.label("DBSCAN").classes("section-title mb-1")
                ui.label(
                    f"💡 eps suggéré : {diag['suggested_eps']:.2f}"
                ).classes("text-sm text-gray-500 mb-3")

                eps_val = ui.slider(min=0.1, max=5, step=0.1,
                                    value=max(0.5, diag['suggested_eps']))
                min_samples = ui.number("min_samples", value=3, min=2)

                dbscan_chk = ui.switch("Activer", value=True)

            # -------- AGNES --------
            with ui.card().classes("p-6 shadow-md rounded-xl algo-card"):
                ui.label("AGNES").classes("section-title mb-3")
                agnes_k = ui.number("Nombre de clusters", value=3, min=2)
                agnes_link = ui.select(
                    ['ward', 'complete', 'average'],
                    value='ward',
                    label="Linkage"
                )
                agnes_chk = ui.switch("Activer", value=True)

            # -------- DIANA --------
            with ui.card().classes("p-6 shadow-md rounded-xl algo-card"):
                ui.label("DIANA").classes("section-title mb-3")
                diana_k = ui.number("Nombre de clusters", value=3, min=2)
                diana_chk = ui.switch("Activer", value=True)

        ui.separator().classes("my-6 w-[900px]")

        # ---------- BOUTONS ----------
        with ui.row().classes("gap-6"):

            ui.button(
                "⬅ Retour Prétraitement",
                on_click=lambda: ui.run_javascript("window.location.href='/preprocess'")
            ).classes("w-64 h-11")

            btn_run = ui.button(" Lancer les algorithmes").classes("w-64 h-11")

        # ---------- EXECUTION ----------
        def run_all():
            results = {}

            try:
                state['X_pca'] = PCA(n_components=2).fit_transform(X)
            except:
                state['X_pca'] = None

            # -------- KMEANS --------
            if kmeans_chk.value:
                km = KMeansCustom(n_clusters=int(k_kmeans.value), random_state=0)
                labels = km.fit_predict(X)
                results['kmeans'] = {
                    'labels': labels,
                    'silhouette': silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'davies_bouldin': davies_bouldin_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'calinski_harabasz': calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'n_clusters': len(np.unique(labels))
                }

            # -------- KMEDOIDS --------
            if kmed_chk.value:
                labels = np.random.randint(0, int(k_kmed.value), size=X.shape[0])
                results['kmedoids'] = {
                    'labels': labels,
                    'silhouette': silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'davies_bouldin': davies_bouldin_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'calinski_harabasz': calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else np.nan,
                    'n_clusters': len(np.unique(labels))
                }

            # -------- DBSCAN --------
            if dbscan_chk.value:
                dbs = DBSCANCustom(eps=float(eps_val.value), min_samples=int(min_samples.value))
                labels = dbs.fit(X).labels_

                valid_clusters = [l for l in np.unique(labels) if l != -1]

                results['dbscan'] = {
                    'labels': labels,
                    'n_clusters': len(valid_clusters),
                    'n_noise': np.sum(labels == -1)
                }

            # -------- DIANA --------
            if diana_chk.value:
                di = DianaCustom(n_clusters=int(diana_k.value))
                labels = di.fit_predict(X)
                results['diana'] = {'labels': labels}

            # -------- AGNES --------
            if agnes_chk.value:
                ag = AgnesCustom(n_clusters=int(agnes_k.value), linkage=agnes_link.value)
                labels = ag.fit_predict(X)
                results['agnes'] = {'labels': labels}

            state['results'] = results
            ui.notify("Clustering terminé", color='positive')
            ui.run_javascript("window.location.href='/results'")

        btn_run.on_click(run_all)


# ----------------- PAGE RESULTS -----------------
@ui.page('/results')
def results_page():
    if not state.get('results'):
        ui.notify("⚠️ Aucun résultat", color='warning')
        ui.run_javascript("window.location.href='/algos'")
        return

    ui.label("📊 Résultats Clustering").classes("text-3xl font-bold")
    results = state['results']
    X = state.get('X', None)
    
    # Calcul automatique des métriques
    metrics_dict = {}
    for algo, res in results.items():
        labels = res.get('labels')
        metrics = {}
        if labels is not None and X is not None:
            mask = labels != -1
            metrics['n_clusters'] = len(set(labels[mask])) if len(mask)>0 else 0
            metrics['n_noise'] = list(labels).count(-1) if -1 in labels else 0

            if metrics['n_clusters'] > 1:
                metrics['silhouette'] = silhouette_score(X[mask], labels[mask])
                metrics['davies_bouldin'] = davies_bouldin_score(X[mask], labels[mask])
                metrics['calinski_harabasz'] = calinski_harabasz_score(X[mask], labels[mask])
            else:
                metrics['silhouette'] = metrics['davies_bouldin'] = metrics['calinski_harabasz'] = None
        metrics_dict[algo] = metrics
        res.update(metrics)
    
    best_algo = max(
        ((algo, m['silhouette']) for algo, m in metrics_dict.items() if m['silhouette'] is not None),
        key=lambda x: x[1],
        default=(None, None)
    )[0]

    ui.label(f"🏆 Meilleur algorithme : {best_algo.upper() if best_algo else 'Aucun'}").classes("text-xl font-bold text-green-600")

    table_rows = []
    for algo, m in metrics_dict.items():
        table_rows.append({**{'algo': algo.upper()}, **m})
    ui.table(
        rows=table_rows,
        columns=[{"name":"algo","label":"Algorithme","field":"algo"}] +
                [{"name":k,"label":k.replace('_',' ').title(),"field":k} for k in ['n_clusters','n_noise','silhouette','davies_bouldin','calinski_harabasz']]
    )

    hist_img = plot_grouped_histogram(metrics_dict, "Comparaison des métriques")
    ui.image(hist_img).style("max-width:700px; max-height:400px")

    X_pca = state.get('X_pca', None)
    for algo, res in results.items():
        ui.separator()
        ui.label(f"🔹 {algo.upper()}").classes("text-xl font-semibold")
        if X_pca is not None and 'labels' in res:
            plot64 = scatter_plot_2d(X_pca, res['labels'], f"{algo.upper()} - PCA 2D")
            ui.image(plot64).style("max-width:500px; max-height:500px")

# ----------------- LANCEMENT -----------------
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Clustering Data Mining", port=8080, reload=True)
