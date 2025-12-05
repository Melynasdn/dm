import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage as sch_linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ----------------- K-MEANS -----------------
class KMeansCustom:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
    
    def fit_predict(self, X):
        if self.random_state is not None: np.random.seed(self.random_state)
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        centers = X[indices].copy()
        for _ in range(self.max_iter):
            labels = np.argmin(cdist(X, centers), axis=1)
            new_centers = np.array([X[labels==k].mean(axis=0) if np.any(labels==k) else centers[k] for k in range(self.n_clusters)])
            if np.linalg.norm(new_centers - centers)<self.tol: break
            centers = new_centers
        self.cluster_centers_ = centers
        return labels

# ----------------- DBSCAN -----------------
class DBSCANCustom:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        self.X = StandardScaler().fit_transform(X)
        n_points = self.X.shape[0]
        labels = np.full(n_points, -1, dtype=int)
        cluster_id = 0
        neighbors = [self.region_query(i) for i in range(n_points)]
        for i in range(n_points):
            if labels[i]!=-1 or len(neighbors[i])<self.min_samples: continue
            self.expand_cluster(i, neighbors, labels, cluster_id)
            cluster_id += 1
        self.labels_ = labels
        return self

    def region_query(self, idx):
        distances = np.linalg.norm(self.X - self.X[idx], axis=1)
        return np.where(distances<=self.eps)[0].tolist()

    def expand_cluster(self, idx, neighbors, labels, cluster_id):
        labels[idx] = cluster_id
        seeds = neighbors[idx].copy()
        while seeds:
            current = seeds.pop()
            if labels[current]==-1: labels[current]=cluster_id
            if labels[current]!=-1: continue
            labels[current]=cluster_id
            cur_neighbors = self.region_query(current)
            if len(cur_neighbors)>=self.min_samples:
                seeds.extend([n for n in cur_neighbors if n not in seeds])

def diagnose_dbscan(X, eps=0.5, min_samples=5):
    neigh = NearestNeighbors(n_neighbors=min_samples).fit(StandardScaler().fit_transform(X))
    distances,_ = neigh.kneighbors(X)
    kth_dist = np.sort(distances[:,-1])
    suggested_eps = np.percentile(kth_dist,90)
    return {'suggested_eps': suggested_eps, 'total_points': X.shape[0], 'potential_core_points': np.sum(kth_dist<=suggested_eps), 'core_point_ratio': np.mean(kth_dist<=suggested_eps), 'avg_neighbors_at_eps': np.mean(np.sum(distances<=eps, axis=1))}

# ----------------- AGNES -----------------
class AgnesCustom:
    def __init__(self, n_clusters=3, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
    
    def fit_predict(self, X):
        Z = sch_linkage(X, method=self.linkage)
        labels = fcluster(Z, t=self.n_clusters, criterion='maxclust') - 1
        return labels