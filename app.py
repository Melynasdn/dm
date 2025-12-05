# app.py
from nicegui import ui
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import random

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from pyclustering.cluster.kmedoids import kmedoids

# -------- ÉTAT GLOBAL (simple) --------
state = {
    "raw_df": None,
    "numeric_columns": None,
    "X": None,
    "X_pca": None,
    "results": {},
}

# -------- UTILITAIRES PLOTS --------
def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

def scatter_pca_image(X_pca, labels, centers_pca=None, title="Scatter PCA"):
    fig, ax = plt.subplots(figsize=(6,4))
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        mask = labels == lab
        ax.scatter(X_pca[mask,0], X_pca[mask,1], label=str(lab), s=20, alpha=0.7)
    if centers_pca is not None:
        ax.scatter(centers_pca[:,0], centers_pca[:,1], marker='X', s=120, c='black', linewidths=1.5)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    return plot_to_base64(fig)

def histogram_image(labels, title="Distribution des clusters"):
    fig, ax = plt.subplots(figsize=(6,4))
    unique, counts = np.unique(labels, return_counts=True)
    ax.bar(unique.astype(str), counts, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Nombre de points")
    return plot_to_base64(fig)

def silhouette_image(score, title="Silhouette Score"):
    fig, ax = plt.subplots(figsize=(6,2))
    ax.axvline(score, linewidth=3)
    ax.set_xlim(-1,1)
    ax.set_title(f"{title} = {score:.4f}")
    return plot_to_base64(fig)

def dendrogram_image(X, method='ward', title="Dendrogramme"):
    from scipy.cluster.hierarchy import linkage, dendrogram
    Z = linkage(X, method=method)
    fig = plt.figure(figsize=(8,4))
    dendrogram(Z, color_threshold=None)
    plt.title(title)
    return plot_to_base64(fig)

# -------- PREPROCESSING --------
def preprocess_dataframe(df, normalization='minmax', handle_missing=True):
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.shape[1] == 0:
        raise ValueError("Aucune colonne numérique détectée dans le dataset.")
    if handle_missing:
        numeric = numeric.fillna(numeric.mean())
    X = numeric.values.astype(float)
    if normalization == 'minmax':
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    elif normalization == 'zscore':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    return numeric, X

# -------- K-Medoids --------
def run_kmedoids(X, n_clusters, random_state=None):
    data = X.tolist()
    if random_state is not None:
        random.seed(random_state)
        init = random.sample(range(len(data)), n_clusters)
    else:
        init = random.sample(range(len(data)), n_clusters)
    kmed = kmedoids(data, init)
    kmed.process()
    clusters = kmed.get_clusters()
    medoids = kmed.get_medoids()
    labels = np.zeros(len(X), dtype=int) - 1
    for cid, pts in enumerate(clusters):
        for idx in pts:
            labels[idx] = cid
    return labels, medoids

# -------- PAGES NiceGUI --------
@ui.page('/')
def index():
    ui.label("📥 Charger votre dataset CSV").classes("text-2xl font-bold")
    ui.label("Chargez un fichier CSV contenant vos données (les colonnes numériques seront utilisées).").classes("text-sm")

    async def on_upload(e):
        try:
            # Lire le contenu en bytes
            content = await e.file.read()
            df = pd.read_csv(io.BytesIO(content))
            state['raw_df'] = df
            ui.notify(f"Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes", color='positive')
            ui.button("Aller au prétraitement", on_click=lambda: ui.run_javascript("window.location.href='/preprocess'"))

        except Exception as ex:
            ui.notify(f"Erreur lecture CSV: {ex}", color='negative')

    ui.upload(on_upload=on_upload).props('accept=".csv"').classes('mt-4')


@ui.page('/preprocess')
def page_preprocess():
    if state['raw_df'] is None:
        ui.notify("Aucun dataset chargé — retour à l'accueil", color='warning')
        ui.run_javascript("window.location.href='/'")
        return

    ui.label("🧹 Prétraitement automatique").classes("text-2xl font-bold")
    df = state['raw_df']

    ui.label(f"Dimensions du dataset: {df.shape[0]} lignes × {df.shape[1]} colonnes").classes('mt-2')
    ui.label("Aperçu (5 premières lignes):").classes('mt-4')
    ui.table.from_pandas(df.head(5)).classes('w-full')

    with ui.row().classes('items-center gap-4 mt-4'):
        norm_select = ui.select(['minmax', 'zscore', 'none'], value='minmax', label='Normalisation').classes('w-48')
        missing_switch = ui.switch('Gérer valeurs manquantes (remplir par la moyenne)', value=True)

    def do_preprocess():
        try:
            norm = norm_select.value
            if norm == 'none':
                norm = None
            numeric_df, X = preprocess_dataframe(df, normalization=norm, handle_missing=missing_switch.value)
            state['numeric_columns'] = list(numeric_df.columns)
            state['X'] = X
            ui.notify(f"Prétraitement terminé — colonnes numériques: {len(state['numeric_columns'])}", color='positive')
            ui.run_javascript("window.location.href='/algos'")
        except Exception as ex:
            ui.notify(str(ex), color='negative')

    ui.button("Lancer prétraitement", on_click=do_preprocess).classes('mt-4 bg-blue-600 text-white')

    # --- Nouveau bouton pour aller directement au clustering si prétraitement déjà fait ---
    if state.get('X') is not None:
        ui.button("Aller au clustering", on_click=lambda: ui.run_javascript("window.location.href='/algos'"))\
          .classes('mt-2 bg-green-500 text-white')

    if state.get('numeric_columns') is not None:
        ui.separator().classes('my-4')
        ui.label("Colonnes numériques détectées:").classes('font-semibold')
        ui.label(", ".join(state['numeric_columns'])).classes('text-sm')


@ui.page('/algos')
def page_algos():
    if state.get('X') is None:
        ui.notify("Données non prétraitées — retour à preprocessing", color='warning')
        ui.run_javascript("window.location.href='/preprocess'")
        return

    ui.label("⚙️ Sélection des algorithmes & paramètres").classes("text-2xl font-bold")
    with ui.column().classes('w-full max-w-2xl'):
        kmeans_chk = ui.checkbox('K-Means', value=True)
        k_kmeans = ui.number('K (K-Means)', value=3, min=2, step=1)

        kmedoids_chk = ui.checkbox('K-Medoids', value=True)
        k_kmedoids = ui.number('K (K-Medoids)', value=3, min=2, step=1)

        dbscan_chk = ui.checkbox('DBSCAN', value=False)
        eps = ui.number('eps (DBSCAN)', value=0.5, step=0.1)
        min_samples = ui.number('min_samples (DBSCAN)', value=5, step=1, min=1)

        agnes_chk = ui.checkbox('AGNES (Agglomerative)', value=True)
        agnes_k = ui.number('K (AGNES)', value=3, min=2, step=1)
        agnes_link = ui.select(['ward', 'complete', 'average'], value='ward', label='linkage')

    def run_all():
        X = state['X']
        results = {}
        # PCA 2D for visualization
        try:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            state['X_pca'] = X_pca
        except Exception:
            state['X_pca'] = None

        # K-MEANS
        if kmeans_chk.value:
            try:
                k = int(k_kmeans.value)
                km = KMeans(n_clusters=k, random_state=0)
                labels = km.fit_predict(X)
                sil = silhouette_score(X, labels) if len(np.unique(labels))>1 else float('nan')
                db = davies_bouldin_score(X, labels) if len(np.unique(labels))>1 else float('nan')
                ch = calinski_harabasz_score(X, labels) if len(np.unique(labels))>1 else float('nan')
                results['kmeans'] = {
                    'labels': labels,
                    'centers': km.cluster_centers_,
                    'silhouette': sil,
                    'davies_bouldin': db,
                    'calinski_harabasz': ch,
                    'n_clusters': len(np.unique(labels))
                }
            except Exception as ex:
                ui.notify(f"K-Means erreur: {ex}", color='negative')

        # K-MEDOIDS
        if kmedoids_chk.value:
            try:
                k = int(k_kmedoids.value)
                labels, medoids = run_kmedoids(X, k, random_state=0)
                sil = silhouette_score(X, labels) if len(np.unique(labels))>1 else float('nan')
                db = davies_bouldin_score(X, labels) if len(np.unique(labels))>1 else float('nan')
                ch = calinski_harabasz_score(X, labels) if len(np.unique(labels))>1 else float('nan')
                results['kmedoids'] = {
                    'labels': labels,
                    'medoids': medoids,
                    'silhouette': sil,
                    'davies_bouldin': db,
                    'calinski_harabasz': ch,
                    'n_clusters': len(np.unique(labels))
                }
            except Exception as ex:
                ui.notify(f"K-Medoids erreur: {ex}", color='negative')

        # DBSCAN
        if dbscan_chk.value:
            try:
                dbs = DBSCAN(eps=float(eps.value), min_samples=int(min_samples.value))
                labels = dbs.fit_predict(X)
                valid_clusters = [lab for lab in np.unique(labels) if lab != -1]
                if len(valid_clusters) > 0 and len(np.unique(labels))>1:
                    sil = silhouette_score(X, labels)
                    db = davies_bouldin_score(X, labels)
                    ch = calinski_harabasz_score(X, labels)
                else:
                    sil = float('nan'); db = float('nan'); ch = float('nan')
                results['dbscan'] = {
                    'labels': labels,
                    'silhouette': sil,
                    'davies_bouldin': db,
                    'calinski_harabasz': ch,
                    'n_clusters': len(np.unique(labels)) - (1 if -1 in labels else 0)
                }
            except Exception as ex:
                ui.notify(f"DBSCAN erreur: {ex}", color='negative')

        # AGNES
        if agnes_chk.value:
            try:
                k = int(agnes_k.value)
                linkage = agnes_link.value
                ag = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                labels = ag.fit_predict(X)
                sil = silhouette_score(X, labels) if len(np.unique(labels))>1 else float('nan')
                db = davies_bouldin_score(X, labels) if len(np.unique(labels))>1 else float('nan')
                ch = calinski_harabasz_score(X, labels) if len(np.unique(labels))>1 else float('nan')
                results['agnes'] = {
                    'labels': labels,
                    'linkage': linkage,
                    'silhouette': sil,
                    'davies_bouldin': db,
                    'calinski_harabasz': ch,
                    'n_clusters': len(np.unique(labels))
                }
            except Exception as ex:
                ui.notify(f"AGNES erreur: {ex}", color='negative')

        state['results'] = results
        ui.notify("Clustering terminé", color='positive')
        ui.run_javascript("window.location.href='/results'")

    ui.button("Lancer tous les algorithmes sélectionnés", on_click=run_all).classes('mt-6 bg-green-600 text-white')



@ui.page('/results')
def page_results():
    if not state.get('results'):
        ui.notify("Aucun résultat — exécutez les algorithmes d'abord", color='warning')
        ui.run_javascript('/algos')
        return

    ui.label("📊 Résultats — onglets par algorithme").classes("text-2xl font-bold")
    results = state['results']
    X_pca = state.get('X_pca')

    for algo_name, res in results.items():
        ui.separator().classes('my-4')
        ui.label(f"🔷 {algo_name.upper()}").classes('text-xl font-semibold')

        with ui.tabs() as tabs:
            tab_summary = ui.tab("Résumé")
            tab_scatter = ui.tab("Scatter PCA 2D")
            tab_hist = ui.tab("Distribution")
            tab_sil = ui.tab("Silhouette")
            tab_dend = ui.tab("Dendrogramme")

        with ui.tab_panels(tabs, value=tab_summary):
            # Résumé
            with ui.tab_panel(tab_summary):
                ui.label(f"Nombre de clusters (excluant bruit) : {res.get('n_clusters', 'N/A')}")
                ui.label(f"Silhouette : {res.get('silhouette', float('nan'))}")
                ui.label(f"Davies-Bouldin : {res.get('davies_bouldin', float('nan'))}")
                ui.label(f"Calinski-Harabasz : {res.get('calinski_harabasz', float('nan'))}")
                if 'medoids' in res:
                    ui.label(f"Medoïdes indices : {res['medoids']}")
                if 'centers' in res:
                    ui.label(f"Centres disponibles (shape): {np.array(res['centers']).shape}")

            # Scatter PCA
            with ui.tab_panel(tab_scatter):
                if X_pca is None:
                    ui.label("PCA non disponible")
                else:
                    centers_pca = None
                    if 'centers' in res:
                        try:
                            centers_pca = PCA(n_components=2).fit_transform(res['centers'])
                        except Exception:
                            centers_pca = None
                    img = scatter_pca_image(X_pca, res['labels'], centers_pca=centers_pca, title=f"{algo_name.upper()} PCA")
                    ui.image(img).classes('mt-4')

            # Histogramme
            with ui.tab_panel(tab_hist):
                img = histogram_image(res['labels'], title=f"{algo_name.upper()} - Distribution des clusters")
                ui.image(img)

            # Silhouette
            with ui.tab_panel(tab_sil):
                s = res.get('silhouette', float('nan'))
                if np.isnan(s):
                    ui.label("Silhouette non calculable (moins de 2 clusters) ou non applicable")
                else:
                    img = silhouette_image(s, title=f"{algo_name.upper()} Silhouette")
                    ui.image(img)

            # Dendrogramme (AGNES)
            with ui.tab_panel(tab_dend):
                if algo_name == 'agnes':
                    try:
                        img = dendrogram_image(state['X'], method=res.get('linkage','ward'), title=f"AGNES - {res.get('linkage','ward')}")
                        ui.image(img)
                    except Exception as ex:
                        ui.label(f"Impossible de générer dendrogramme: {ex}")
                else:
                    ui.label("Dendrogramme disponible uniquement pour AGNES")


# -------- LANCEMENT --------
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Data Mining Clustering - Simple", port=8080, reload=True)
