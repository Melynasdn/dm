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



# ----------------- PAGE INDEX -----------------
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

                # Affichage de la valeur en temps réel
                eps_label = ui.label(f"Epsilon (eps) : {max(0.5, diag['suggested_eps']):.2f}")
                eps_val = ui.slider(
                    min=0.1, 
                    max=5, 
                    step=0.1,
                    value=max(0.5, diag['suggested_eps'])
                ).props('label-always')
    
                # Mise à jour du label quand le slider change
                eps_val.on_value_change(lambda e: eps_label.set_text(f"Epsilon (eps) : {e.value:.2f}"))

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






# ------------------ Helpers ------------------
import matplotlib.pyplot as plt
import io, base64
from scipy.cluster.hierarchy import dendrogram, linkage as sch_linkage
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import io
import base64


# ------------------ Helpers ------------------
import matplotlib.pyplot as plt
import io, base64
from scipy.cluster.hierarchy import dendrogram, linkage as sch_linkage
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import io
import base64


# ------------------ Helpers ------------------
import matplotlib.pyplot as plt
import io, base64
from scipy.cluster.hierarchy import dendrogram, linkage as sch_linkage
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import io
import base64


# ============== FONCTIONS DENDROGRAMME ==============

def fig_to_base64(fig):
    """Convertit une figure matplotlib en base64"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def diana_linkage(X):
    """Implémentation simplifiée de DIANA (DIvisive ANAlysis)"""
    n = X.shape[0]
    if n < 2:
        return np.array([])
    
    # Calcul de la matrice de distances
    dist_matrix = squareform(pdist(X, metric='euclidean'))
    
    # Structure pour stocker le linkage
    linkage_matrix = []
    cluster_id = n
    
    # Dictionnaire pour suivre les clusters actifs
    active_clusters = {i: [i] for i in range(n)}
    
    while len(active_clusters) > 1:
        # Trouver le cluster avec le diamètre maximum
        max_diameter = -1
        cluster_to_split = None
        
        for cid, members in active_clusters.items():
            if len(members) > 1:
                # Calculer le diamètre (distance max entre membres)
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
        
        # Trouver l'élément le plus éloigné du reste
        max_avg_dist = -1
        splinter = None
        
        for member in members:
            avg_dist = np.mean([dist_matrix[member, other] 
                               for other in members if other != member])
            if avg_dist > max_avg_dist:
                max_avg_dist = avg_dist
                splinter = member
        
        # Créer deux nouveaux clusters
        remaining = [m for m in members if m != splinter]
        
        # Ajouter à la matrice de linkage
        linkage_matrix.append([
            splinter if splinter < n else splinter,
            cluster_id if len(remaining) > 1 else remaining[0],
            max_avg_dist,
            len(members)
        ])
        
        # Mettre à jour les clusters actifs
        del active_clusters[cluster_to_split]
        active_clusters[splinter] = [splinter]
        if len(remaining) > 1:
            active_clusters[cluster_id] = remaining
            cluster_id += 1
        elif len(remaining) == 1:
            active_clusters[remaining[0]] = remaining
    
    return np.array(linkage_matrix) if linkage_matrix else np.array([])


def plot_grouped_histogram(metrics_dict, title):
    """Génère un histogramme groupé des métriques"""
    algos = list(metrics_dict.keys())
    n_algos = len(algos)
    
    # Extraction des métriques
    silhouettes = [metrics_dict[a].get("silhouette") for a in algos]
    davies = [metrics_dict[a].get("davies_bouldin") for a in algos]
    calinski = [metrics_dict[a].get("calinski_harabasz") for a in algos]
    
    # Remplacer les "N/A" par None pour le plotting
    silhouettes = [s if s != "N/A" else None for s in silhouettes]
    davies = [d if d != "N/A" else None for d in davies]
    calinski = [c if c != "N/A" else None for c in calinski]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(n_algos)
    width = 0.6
    
    colors = ['#3b82f6', '#8b5cf6', '#ec4899']
    
    # Silhouette
    valid_sil = [s if s is not None else 0 for s in silhouettes]
    axes[0].bar(x, valid_sil, width, color=colors[0], alpha=0.8)
    axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Silhouette ↑', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([a.upper() for a in algos], rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].set_ylim([-1, 1])
    
    # Davies-Bouldin
    valid_dav = [d if d is not None else 0 for d in davies]
    axes[1].bar(x, valid_dav, width, color=colors[1], alpha=0.8)
    axes[1].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Davies-Bouldin ↓', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([a.upper() for a in algos], rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Calinski-Harabasz
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
    """
    Génère un dendrogramme et retourne l'image en base64
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Données à clusteriser
    algo : str, default="agnes"
        Algorithme: "agnes" (agglomératif) ou "diana" (divisif)
    
    Returns:
    --------
    str : Image encodée en base64 avec le préfixe data:image/png;base64,
    """
    X = np.array(X)
    
    if X.shape[0] < 2:
        print("Erreur: Au moins 2 échantillons sont nécessaires")
        return None
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    try:
        if algo.lower() == "agnes":
            # AGNES: Agglomerative Nesting (bottom-up)
            X_scaled = StandardScaler().fit_transform(X)
            Z = linkage(X_scaled, method='ward')
            dendrogram(Z, ax=ax, color_threshold=0.7*max(Z[:,2]))
            ax.set_title("Dendrogramme - AGNES (Ward)", fontsize=16, fontweight='bold', pad=20)
            
        elif algo.lower() == "diana":
            # DIANA: DIvisive ANAlysis (top-down)
            X_scaled = StandardScaler().fit_transform(X)
            Z = diana_linkage(X_scaled)
            
            if Z.size > 0:
                dendrogram(Z, ax=ax, color_threshold=0.7*max(Z[:,2]))
                ax.set_title("Dendrogramme - DIANA (Divisif)", fontsize=16, fontweight='bold', pad=20)
            else:
                ax.text(0.5, 0.5, "Impossible de créer le dendrogramme DIANA", 
                       ha="center", va="center", fontsize=14)
                ax.set_title("Dendrogramme - DIANA", fontsize=16, fontweight='bold', pad=20)
        else:
            ax.text(0.5, 0.5, f"Algorithme '{algo}' non reconnu\nUtilisez 'agnes' ou 'diana'", 
                   ha="center", va="center", fontsize=14)
            ax.set_title("Erreur", fontsize=16, fontweight='bold', pad=20)
        
        ax.set_xlabel("Échantillons", fontsize=13, fontweight='bold')
        ax.set_ylabel("Distance", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        
        # Toujours retourner le base64 avec le préfixe data:image pour NiceGUI
        base64_str = fig_to_base64(fig)
        return f"data:image/png;base64,{base64_str}"
            
    except Exception as e:
        print(f"Erreur lors de la génération du dendrogramme: {e}")
        plt.close(fig)
        return None


# ============== PAGE RESULTS ==============

@ui.page('/results')
def results_page():

    if not state.get('results'):
        ui.notify("⚠️ Aucun résultat", color='warning')
        ui.run_javascript("window.location.href='/algos'")
        return

    

    results = state['results']
    X = state.get('X')
    X_pca = state.get('X_pca')

    # ----------------- Calcul des métriques -----------------
    metrics_dict = {}
    for algo, res in results.items():
        labels = res.get('labels')
        m = {}

        if labels is not None and X is not None:
            mask = labels != -1
            m["n_clusters"] = len(set(labels[mask])) if mask.sum() > 0 else 0
            m["n_noise"] = list(labels).count(-1) if -1 in labels else 0

            if m["n_clusters"] > 1:
                from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
                m["silhouette"] = round(float(silhouette_score(X[mask], labels[mask])), 3)
                m["davies_bouldin"] = round(float(davies_bouldin_score(X[mask], labels[mask])), 3)
                m["calinski_harabasz"] = round(float(calinski_harabasz_score(X[mask], labels[mask])), 2)
            else:
                m["silhouette"] = "N/A"
                m["davies_bouldin"] = "N/A"
                m["calinski_harabasz"] = "N/A"

        metrics_dict[algo] = m
        res.update(m)

    # ----------------- Tableau comparatif avec style amélioré -----------------
    with ui.card().classes("w-full shadow-2xl p-6 mb-8"):
        ui.label("📈 Tableau Comparatif des Métriques").classes(
            "text-2xl font-bold mb-4 text-gray-800"
        )
        
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
        ).classes("w-full").props("flat bordered").style(
            "font-size: 15px;"
        )

    # ----------------- Histogramme comparatif avec dimensions fixes -----------------
    with ui.card().classes("w-full shadow-2xl p-6 mb-10"):
        ui.label("📊 Comparaison Visuelle des Métriques").classes(
            "text-2xl font-bold mb-4 text-gray-800"
        )
        hist_img = plot_grouped_histogram(metrics_dict, "Comparaison des métriques")
        ui.image(hist_img).classes("mx-auto").style(
            "width: 100%; max-width: 1000px; height: auto; border-radius: 8px;"
        )

    # ----------------- Détail par algorithme -----------------
    for idx, (algo, res) in enumerate(results.items()):

        ui.separator().classes("my-8")
        
        # Couleur différente pour chaque algo
        colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981']
        color = colors[idx % len(colors)]
        
        with ui.element('div').classes('w-full p-5 rounded-lg mb-6').style(
            f'background: linear-gradient(135deg, {color}22 0%, {color}11 100%); border-left: 4px solid {color};'
        ):
            ui.label(f"{algo.upper()}").classes(
                "text-3xl font-bold"
            ).style(f'color: {color};')

        m = metrics_dict[algo]

        # ---------- Résumé dans une card élégante ----------
        with ui.card().classes("shadow-lg p-6 mb-6 w-full").style(
            "border-top: 3px solid " + color
        ):
            ui.label("📌 Résumé des Métriques").classes(
                "text-xl font-bold mb-4 text-gray-800"
            )
            
            with ui.row().classes("gap-4 w-full justify-center items-stretch"):
                # Clusters
                with ui.card().classes("p-4 bg-gray-50 text-center flex-1"):
                    ui.label("Clusters").classes("text-sm text-gray-600 font-semibold mb-2")
                    ui.label(str(m["n_clusters"])).classes("text-3xl font-bold").style(f'color: {color};')
                
                # Points de bruit
                with ui.card().classes("p-4 bg-gray-50 text-center flex-1"):
                    ui.label("Points de bruit").classes("text-sm text-gray-600 font-semibold mb-2")
                    ui.label(str(m["n_noise"])).classes("text-3xl font-bold text-orange-600")
                
                # Silhouette
                with ui.card().classes("p-4 bg-gray-50 text-center flex-1"):
                    ui.label("Silhouette ↑").classes("text-sm text-gray-600 font-semibold mb-2")
                    ui.label(str(m["silhouette"])).classes("text-2xl font-bold text-green-600")
                
                # Davies-Bouldin
                with ui.card().classes("p-4 bg-gray-50 text-center flex-1"):
                    ui.label("Davies-Bouldin ↓").classes("text-sm text-gray-600 font-semibold mb-2")
                    ui.label(str(m["davies_bouldin"])).classes("text-2xl font-bold text-blue-600")
                
                # Calinski-Harabasz
                with ui.card().classes("p-4 bg-gray-50 text-center flex-1"):
                    ui.label("Calinski-Harabasz ↑").classes("text-sm text-gray-600 font-semibold mb-2")
                    ui.label(str(m["calinski_harabasz"])).classes("text-2xl font-bold text-purple-600")

        # ---------- Visualisations côte à côte ----------
        with ui.grid(columns=1 if algo.lower() not in ["agnes", "diana"] else 2).classes("gap-8 mb-8 w-full"):
            
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

            # Dendrogramme (si applicable)
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
                        ui.label(f"❌ Erreur : {e}").classes(
                            "text-red-600 text-center py-10"
                        )



# ----------------- LANCEMENT -----------------
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Clustering Data Mining", port=8080, reload=True)