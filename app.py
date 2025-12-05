# app_complete.py
from nicegui import ui
import pandas as pd
import numpy as np
import io, base64, random, matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from custom_cluster import KMeansCustom, DBSCANCustom, AgnesCustom
from custom_cluster_optimized import KMeansCustom, DBSCANCustom, AgnesCustom

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pyclustering.cluster.kmedoids import kmedoids
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_samples

# -------- ÉTAT GLOBAL --------
state = {
    "raw_df": None,
    "numeric_columns": None,
    "X": None,
    "X_pca": None,
    "results": {}
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
    for lab in np.unique(labels):
        mask = labels == lab
        ax.scatter(X_pca[mask,0], X_pca[mask,1], label=str(lab), s=20, alpha=0.7)
    if centers_pca is not None:
        ax.scatter(centers_pca[:,0], centers_pca[:,1], marker='X', s=120, c='black')
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
    Z = linkage(X, method=method)
    fig = plt.figure(figsize=(8,4))
    dendrogram(Z, color_threshold=None)
    plt.title(title)
    return plot_to_base64(fig)

# -------- PREPROCESSING --------
def preprocess_dataframe(df, normalization='minmax', handle_missing=True):
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.shape[1] == 0:
        raise ValueError("Aucune colonne numérique détectée")
    if handle_missing:
        numeric = numeric.fillna(numeric.mean())
    X = numeric.values.astype(float)
    if normalization == 'minmax':
        X = MinMaxScaler().fit_transform(X)
    elif normalization == 'zscore':
        X = StandardScaler().fit_transform(X)
    return numeric, X

# -------- K-MEDOIDS --------
def run_kmedoids(X, n_clusters, random_state=None):
    data = X.tolist()
    if random_state is not None:
        random.seed(random_state)
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

# -------- PAGES --------
@ui.page('/')
def index():
    ui.label("📥 Charger votre dataset CSV").classes("text-2xl font-bold")
    ui.label("Colonnes numériques utilisées pour le clustering").classes("text-sm")
    async def on_upload(e):
        content = await e.file.read()
        df = pd.read_csv(io.BytesIO(content))
        state['raw_df'] = df
        ui.notify(f"Dataset chargé : {df.shape}", color='positive')
        ui.button(
    "Aller au prétraitement",
    on_click=lambda: ui.run_javascript("window.location.href='/preprocess'")
)

    ui.upload(on_upload=on_upload).props('accept=".csv"').classes('mt-4')






import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

# -----------------------------
# Exemple d'état global
# -----------------------------
state = {
    'raw_df': None,  # à remplir lors du chargement CSV
    'X': None,
    'numeric_columns': []
}

# -----------------------------
# Fonction de prétraitement
# -----------------------------
def preprocess_dataframe(df, handle_missing=True, normalization=None, encode_onehot=False, drop_duplicates=False):
    df_copy = df.copy()
    
    # Remplacer '?' par NaN pour qu'ils soient reconnus comme valeurs manquantes
    df_copy = df_copy.replace('?', np.nan)

    # Supprimer doublons
    if drop_duplicates:
        df_copy = df_copy.drop_duplicates()

    # Remplir valeurs manquantes
    if handle_missing:
        for col in df_copy.columns:
            if df_copy[col].dtype in [np.float64, np.int64]:
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
            else:
                if len(df_copy[col].mode()) > 0:
                    df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
                else:
                    df_copy[col] = df_copy[col].fillna('Unknown')

    # Encodage OneHot
    if encode_onehot:
        cat_cols = df_copy.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            df_copy = pd.get_dummies(df_copy, columns=cat_cols)

    # Normalisation
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    if normalization == 'zscore':
        df_copy[numeric_cols] = StandardScaler().fit_transform(df_copy[numeric_cols])
    elif normalization == 'minmax':
        df_copy[numeric_cols] = MinMaxScaler().fit_transform(df_copy[numeric_cols])

    return df_copy, df_copy.values

# -----------------------------
# Page Preprocessing
# -----------------------------
@ui.page('/preprocess')
def preprocess_page():
    if state['raw_df'] is None:
        ui.notify("Aucun CSV chargé", color='warning')
        ui.run_javascript("window.location.href='/'")
        return

    df = state['raw_df']

    ui.label(f"📊 Dataset avant prétraitement : {df.shape[0]} lignes × {df.shape[1]} colonnes").classes("text-xl font-semibold")
    
    # CORRECTION: Convertir le DataFrame en liste de dictionnaires
    ui.table(
        rows=df.head(5).to_dict('records'),
        columns=[{"name": col, "label": col, "field": col} for col in df.columns]
    )

    # -----------------------------
    # Options de prétraitement
    # -----------------------------
    ui.label("⚙️ Choisissez les opérations de prétraitement").classes("text-lg font-semibold mt-4")
    drop_duplicates_switch = ui.switch("Supprimer doublons", value=False)
    missing_switch = ui.switch("Remplir valeurs manquantes", value=True)
    encode_onehot_switch = ui.switch("Encodage OneHot des catégories", value=False)
    norm_select = ui.select(['none', 'minmax', 'zscore'], value='minmax', label='Normalisation')

    # Conteneur pour afficher le dataset après prétraitement
    processed_table_container = ui.column().classes("mt-4")

    # -----------------------------
    # Fonction pour appliquer le prétraitement
    # -----------------------------
    def apply_preprocessing():
        processed_df, X = preprocess_dataframe(
            df,
            handle_missing=missing_switch.value,
            normalization=None if norm_select.value=='none' else norm_select.value,
            encode_onehot=encode_onehot_switch.value,
            drop_duplicates=drop_duplicates_switch.value
        )
        state['X'] = X
        state['numeric_columns'] = list(processed_df.select_dtypes(include=[np.number]).columns)

        # Affichage du dataset traité
        processed_table_container.clear()
        with processed_table_container:
            ui.label(f"✅ Dataset après prétraitement : {processed_df.shape[0]} lignes × {processed_df.shape[1]} colonnes").classes("text-xl font-semibold mt-2")
            
            # CORRECTION: Utiliser to_dict('records') ici aussi
            ui.table(
                rows=processed_df.head(5).to_dict('records'),
                columns=[{"name": col, "label": col, "field": col} for col in processed_df.columns]
            )
        
        ui.notify("✅ Prétraitement appliqué avec succès!", color='positive')

    # -----------------------------
    # Bouton lancer prétraitement
    # -----------------------------
    ui.button("Lancer prétraitement", on_click=apply_preprocessing).classes('mt-4 bg-blue-600 text-white px-4 py-2 rounded')

    # -----------------------------
    # Si prétraitement déjà fait
    # -----------------------------
    if state.get('X') is not None:
        ui.button("Aller au clustering", on_click=lambda: ui.run_javascript("window.location.href='/algos'")).classes('mt-2 bg-green-500 text-white px-4 py-2 rounded')






@ui.page('/algos')
def algos_page():
    if state.get('X') is None:
        ui.notify("Prétraitement nécessaire", color='warning')
        ui.run_javascript("window.location.href='/preprocess'")
        return

    X = state['X']
    ui.label("⚙️ Sélection des algorithmes de clustering").classes("text-2xl font-bold mb-4")

    # ------- LAYOUT DES CARTES -------
    with ui.row().classes('gap-4'):
        # K-MEANS
        with ui.card().classes('p-4 flex-1 shadow-lg'):
            ui.label("K-Means").classes("text-xl font-semibold mb-2")
            kmeans_chk = ui.checkbox('Activer', value=True).classes('mb-2')
            ui.label("Nombre de clusters (K)").classes('text-sm mb-1')
            k_kmeans = ui.number(value=3, min=2, step=1)

        # K-MEDOIDS
        with ui.card().classes('p-4 flex-1 shadow-lg'):
            ui.label("K-Medoids").classes("text-xl font-semibold mb-2")
            kmedoids_chk = ui.checkbox('Activer', value=True).classes('mb-2')
            ui.label("Nombre de clusters (K)").classes('text-sm mb-1')
            k_kmedoids = ui.number(value=3, min=2, step=1)

    with ui.row().classes('gap-4 mt-4'):
        # DBSCAN
        with ui.card().classes('p-4 flex-1 shadow-lg'):
            ui.label("DBSCAN").classes("text-xl font-semibold mb-2")
            dbscan_chk = ui.checkbox('Activer', value=False).classes('mb-2')
            ui.label("eps").classes('text-sm mb-1')
            eps = ui.slider(min=0.1, max=2.0, step=0.05, value=0.5)
            ui.label("min_samples").classes('text-sm mt-2 mb-1')
            min_samples = ui.number(value=5, min=1, step=1)

        # AGNES
        with ui.card().classes('p-4 flex-1 shadow-lg'):
            ui.label("AGNES").classes("text-xl font-semibold mb-2")
            agnes_chk = ui.checkbox('Activer', value=True).classes('mb-2')
            ui.label("Nombre de clusters (K)").classes('text-sm mb-1')
            agnes_k = ui.number(value=3, min=2, step=1)
            ui.label("Méthode linkage").classes('text-sm mb-1 mt-2')
            agnes_link = ui.select(['ward', 'complete', 'average'], value='ward')

    # ------- BOUTON LANCER -------
    def run_all():
        results = {}
        try:
            X_pca = PCA(n_components=2).fit_transform(X)
            state['X_pca'] = X_pca
        except:
            state['X_pca'] = None

        # --- K-MEANS ---
        if kmeans_chk.value:
            k = int(k_kmeans.value)
            km = KMeansCustom(n_clusters=k, random_state=0)
            labels = km.fit_predict(X)
            results['kmeans'] = {
                'labels': labels,
                'centers': km.cluster_centers_,
                'silhouette': silhouette_score(X, labels) if len(np.unique(labels))>1 else float('nan'),
                'davies_bouldin': davies_bouldin_score(X, labels) if len(np.unique(labels))>1 else float('nan'),
                'calinski_harabasz': calinski_harabasz_score(X, labels) if len(np.unique(labels))>1 else float('nan'),
                'n_clusters': len(np.unique(labels))
            }

        # --- K-MEDOIDS ---
        if kmedoids_chk.value:
            k = int(k_kmedoids.value)
            labels, medoids = run_kmedoids(X, k, random_state=0)
            results['kmedoids'] = {
                'labels': labels,
                'medoids': medoids,
                'silhouette': silhouette_score(X, labels) if len(np.unique(labels))>1 else float('nan'),
                'davies_bouldin': davies_bouldin_score(X, labels) if len(np.unique(labels))>1 else float('nan'),
                'calinski_harabasz': calinski_harabasz_score(X, labels) if len(np.unique(labels))>1 else float('nan'),
                'n_clusters': len(np.unique(labels))
            }

        # --- DBSCAN ---
        if dbscan_chk.value:
            dbs = DBSCANCustom(eps=float(eps.value), min_samples=int(min_samples.value))
            labels = dbs.fit_predict(X)
            valid_clusters = [l for l in np.unique(labels) if l != -1]
            results['dbscan'] = {
                'labels': labels,
                'silhouette': silhouette_score(X, labels) if len(valid_clusters)>1 else float('nan'),
                'davies_bouldin': davies_bouldin_score(X, labels) if len(valid_clusters)>1 else float('nan'),
                'calinski_harabasz': calinski_harabasz_score(X, labels) if len(valid_clusters)>1 else float('nan'),
                'n_clusters': len(valid_clusters)
            }

        # --- AGNES ---
        if agnes_chk.value:
            k = int(agnes_k.value)
            link = agnes_link.value
            ag = AgnesCustom(n_clusters=k, linkage=link)
            labels = ag.fit_predict(X)
            results['agnes'] = {
                'labels': labels,
                'linkage': link,
                'silhouette': silhouette_score(X, labels) if len(np.unique(labels))>1 else float('nan'),
                'davies_bouldin': davies_bouldin_score(X, labels) if len(np.unique(labels))>1 else float('nan'),
                'calinski_harabasz': calinski_harabasz_score(X, labels) if len(np.unique(labels))>1 else float('nan'),
                'n_clusters': len(np.unique(labels))
            }

        state['results'] = results
        ui.notify("Clustering terminé", color='positive')
        ui.run_javascript("window.location.href='/results'")

    ui.button("Lancer tous les algorithmes", on_click=run_all)\
        .classes('mt-6 bg-green-600 text-white text-lg px-6 py-3 rounded-lg shadow-lg')










@ui.page('/results')
def results_page():
    if not state.get('results'):
        ui.notify("Aucun résultat disponible — exécutez les algorithmes d'abord", color='warning')
        ui.run_javascript("window.location.href='/algos'")
        return

    ui.label("📊 Résultats").classes("text-2xl font-bold")
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
                ui.label(f"Nombre de clusters : {res.get('n_clusters','N/A')}")
                ui.label(f"Silhouette : {res.get('silhouette', 'N/A')}")
                ui.label(f"Davies-Bouldin : {res.get('davies_bouldin', 'N/A')}")
                ui.label(f"Calinski-Harabasz : {res.get('calinski_harabasz', 'N/A')}")
                if 'medoids' in res:
                    ui.label(f"Medoïdes : {res['medoids']}")
                if 'centers' in res:
                    ui.label(f"Centres (shape) : {np.array(res['centers']).shape}")

            # Scatter PCA
            with ui.tab_panel(tab_scatter):
                if X_pca is None:
                    ui.label("PCA non disponible")
                else:
                    centers_pca = None
                    if 'centers' in res:
                        try:
                            centers_pca = PCA(n_components=2).fit_transform(res['centers'])
                        except:
                            centers_pca = None
                    with ui.pyplot():
                        labels = res['labels']
                        unique_labels = np.unique(labels)
                        for lab in unique_labels:
                            mask = labels == lab
                            plt.scatter(X_pca[mask,0], X_pca[mask,1], label=str(lab), alpha=0.7)
                        if centers_pca is not None:
                            plt.scatter(centers_pca[:,0], centers_pca[:,1], marker='X', s=120, c='black', linewidths=1.5)
                        plt.title(f"{algo_name.upper()} PCA")
                        plt.xlabel("PC1")
                        plt.ylabel("PC2")
                        plt.legend(title="Cluster", bbox_to_anchor=(1.05,1), loc='upper left')

            # Histogramme
            with ui.tab_panel(tab_hist):
                with ui.pyplot():
                    labels = res['labels']
                    unique, counts = np.unique(labels, return_counts=True)
                    plt.bar(unique.astype(str), counts, alpha=0.8)
                    plt.title(f"{algo_name.upper()} - Distribution")
                    plt.xlabel("Cluster")
                    plt.ylabel("Nombre de points")

            # Silhouette
            with ui.tab_panel(tab_sil):
                s = res.get('silhouette', float('nan'))
                if np.isnan(s):
                    ui.label("Silhouette non calculable")
                else:
                    with ui.pyplot():
                        plt.axvline(s, linewidth=3, color='red')
                        plt.xlim(-1,1)
                        plt.title(f"{algo_name.upper()} Silhouette = {s:.4f}")

            # Dendrogramme
            with ui.tab_panel(tab_dend):
                if algo_name=='agnes':
                    try:
                        from scipy.cluster.hierarchy import linkage, dendrogram
                        Z = linkage(state['X'], method=res.get('linkage','ward'))
                        with ui.pyplot():
                            dendrogram(Z, color_threshold=None)
                            plt.title(f"AGNES - {res.get('linkage')}")
                    except:
                        ui.label("Dendrogramme impossible")
                else:
                    ui.label("Dendrogramme uniquement pour AGNES")











# -------- RUN --------
if __name__ in {"__main__","__mp_main__"}:
    ui.run(title="Clustering Data Mining - Full", port=8080, reload=True)
