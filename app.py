from nicegui import ui
import pandas as pd
import numpy as np
import io, base64, random, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pyclustering.cluster.kmedoids import kmedoids
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib.colors import ListedColormap

from custom_cluster_optimized import KMeansCustom, DBSCANCustom, AgnesCustom, diagnose_dbscan

# ----------------- ÉTAT GLOBAL -----------------
state = {
    "raw_df": None,
    "numeric_columns": None,
    "X": None,
    "X_pca": None,
    "results": {}
}

# ----------------- UTILITAIRES -----------------
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

    if drop_duplicates: df_copy = df_copy.drop_duplicates()

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

# ----------------- PAGE INDEX -----------------
@ui.page('/')
def index():
    ui.label("📥 Charger votre dataset CSV").classes("text-2xl font-bold")
    async def on_upload(e):
        content = await e.file.read()
        df = pd.read_csv(io.BytesIO(content))
        state['raw_df'] = df
        ui.notify(f"✅ Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes", color='positive')
        ui.run_javascript("window.location.href='/preprocess'")
    ui.upload(on_upload=on_upload).props('accept=".csv"')

# ----------------- PAGE PREPROCESS -----------------
@ui.page('/preprocess')
def preprocess_page():
    if state['raw_df'] is None:
        ui.notify("⚠️ Aucun CSV chargé", color='warning')
        ui.run_javascript("window.location.href='/'")
        return

    df = state['raw_df']
    ui.label(f"📊 Dataset avant prétraitement : {df.shape[0]}×{df.shape[1]}").classes("text-xl font-semibold")
    ui.table(rows=df.head(10).to_dict('records'), columns=[{"name": c,"label": c,"field":c} for c in df.columns])

    drop_dup = ui.switch("Supprimer doublons", value=False)
    handle_missing = ui.switch("Remplir valeurs manquantes", value=True)
    onehot = ui.switch("Encodage OneHot", value=False)
    norm_select = ui.select(['none','minmax','zscore'], value='minmax', label='Normalisation')

    container = ui.column()
    def apply():
        processed_df, X = preprocess_dataframe(df, handle_missing.value, None if norm_select.value=='none' else norm_select.value,
                                               encode_onehot=onehot.value, drop_duplicates=drop_dup.value)
        state['X'] = X
        state['numeric_columns'] = list(processed_df.select_dtypes(include=[np.number]).columns)
        container.clear()
        ui.label(f"✅ Dataset après prétraitement : {processed_df.shape[0]}×{processed_df.shape[1]}").classes("text-lg font-semibold")
        ui.table(rows=processed_df.head(10).to_dict('records'), columns=[{"name": c,"label": c,"field":c} for c in processed_df.columns])
        ui.button("Aller au clustering →", on_click=lambda: ui.run_javascript("window.location.href='/algos'"))
        ui.notify("✅ Prétraitement appliqué!", color='positive')
    ui.button("Lancer prétraitement", on_click=apply)
    container

# ----------------- PAGE ALGOS -----------------
@ui.page('/algos')
def algos_page():
    if state.get('X') is None:
        ui.notify("⚠️ Prétraitement nécessaire", color='warning')
        ui.run_javascript("window.location.href='/preprocess'")
        return

    X = state['X']
    ui.label("⚙️ Algorithmes de clustering").classes("text-2xl font-bold mb-4")
    diag = diagnose_dbscan(X, eps=0.5, min_samples=5)
    ui.label(f"💡 DBSCAN - eps suggéré: {diag['suggested_eps']:.2f}, core points: {diag['potential_core_points']} / {diag['total_points']}")

    kmeans_chk = ui.checkbox("KMeans", value=True)
    k_kmeans = ui.number(value=3, min=2, step=1)
    kmed_chk = ui.checkbox("KMedoids", value=True)
    k_kmed = ui.number(value=3, min=2, step=1)
    dbscan_chk = ui.checkbox("DBSCAN", value=True)
    eps_val = ui.slider(min=0.1,max=5,value=max(0.5,diag['suggested_eps']),step=0.1)
    min_samples = ui.number(value=3, min=2, step=1)
    agnes_chk = ui.checkbox("AGNES", value=True)
    agnes_k = ui.number(value=3,min=2,step=1)
    agnes_link = ui.select(['ward','complete','average'], value='ward')

    def run_all():
        results = {}
        try: state['X_pca'] = PCA(n_components=2).fit_transform(X)
        except: state['X_pca'] = None

        if kmeans_chk.value:
            km = KMeansCustom(n_clusters=int(k_kmeans.value), random_state=0)
            labels = km.fit_predict(X)
            results['kmeans'] = {'labels': labels,'centers': km.cluster_centers_,
                                 'silhouette': silhouette_score(X, labels) if len(np.unique(labels))>1 else np.nan,
                                 'davies_bouldin': davies_bouldin_score(X, labels) if len(np.unique(labels))>1 else np.nan,
                                 'calinski_harabasz': calinski_harabasz_score(X, labels) if len(np.unique(labels))>1 else np.nan,
                                 'n_clusters': len(np.unique(labels))}

        if kmed_chk.value:
            labels, medoids = run_kmedoids(X,int(k_kmed.value), random_state=0)
            results['kmedoids'] = {'labels': labels,'medoids': medoids,
                                   'silhouette': silhouette_score(X, labels) if len(np.unique(labels))>1 else np.nan,
                                   'davies_bouldin': davies_bouldin_score(X, labels) if len(np.unique(labels))>1 else np.nan,
                                   'calinski_harabasz': calinski_harabasz_score(X, labels) if len(np.unique(labels))>1 else np.nan,
                                   'n_clusters': len(np.unique(labels))}

        if dbscan_chk.value:
            dbs = DBSCANCustom(eps=float(eps_val.value), min_samples=int(min_samples.value))
            labels = dbs.fit(X).labels_
            n_noise = np.sum(labels==-1)
            valid_clusters = [l for l in np.unique(labels) if l!=-1]
            results['dbscan'] = {'labels': labels,'n_clusters': len(valid_clusters),'n_noise':n_noise}
            if len(valid_clusters)>1:
                mask = labels!=-1
                results['dbscan']['silhouette'] = silhouette_score(X[mask], labels[mask])
                results['dbscan']['davies_bouldin'] = davies_bouldin_score(X[mask], labels[mask])
                results['dbscan']['calinski_harabasz'] = calinski_harabasz_score(X[mask], labels[mask])
            else:
                results['dbscan']['silhouette'] = results['dbscan']['davies_bouldin'] = results['dbscan']['calinski_harabasz'] = np.nan
                if len(valid_clusters)==0:
                    results['dbscan']['warning'] = f"DBSCAN n'a trouvé aucun cluster, {n_noise} points bruit"

        if agnes_chk.value:
            ag = AgnesCustom(n_clusters=int(agnes_k.value), linkage=agnes_link.value)
            labels = ag.fit_predict(X)
            results['agnes'] = {'labels': labels,'linkage':agnes_link.value,
                                'silhouette': silhouette_score(X, labels) if len(np.unique(labels))>1 else np.nan,
                                'davies_bouldin': davies_bouldin_score(X, labels) if len(np.unique(labels))>1 else np.nan,
                                'calinski_harabasz': calinski_harabasz_score(X, labels) if len(np.unique(labels))>1 else np.nan,
                                'n_clusters': len(np.unique(labels))}

        state['results'] = results
        ui.notify("✅ Clustering terminé", color='positive')
        ui.run_javascript("window.location.href='/results'")

    ui.button("🚀 Lancer tous les algorithmes", on_click=run_all)

# ----------------- PAGE RESULTS -----------------
@ui.page('/results')
def results_page():
    if not state.get('results'):
        ui.notify("⚠️ Aucun résultat", color='warning')
        ui.run_javascript("window.location.href='/algos'")
        return

    ui.label("📊 Résultats Clustering").classes("text-3xl font-bold")
    results = state['results']
    X_pca = state.get('X_pca', None)

    for algo, res in results.items():
        ui.separator()
        ui.label(f"🔹 {algo.upper()}").classes("text-xl font-semibold")
        # Table des métriques
        metrics = {k:v for k,v in res.items() if k in ['silhouette','davies_bouldin','calinski_harabasz','n_clusters','n_noise','warning']}
        if metrics:
            ui.table(rows=[metrics], columns=[{"name":k,"label":k,"field":k} for k in metrics.keys()])

        # Scatter plot 2D
        if X_pca is not None and 'labels' in res:
            plot64 = scatter_plot_2d(X_pca, res['labels'], f"{algo.upper()} - PCA 2D")
            ui.image(plot64).style("max-width:500px; max-height:500px")

# ----------------- LANCEMENT -----------------
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Clustering Data Mining", port=8080, reload=True)
