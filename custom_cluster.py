import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage as sch_linkage, fcluster

# ------------------ K-MEANS ------------------
class KMeansCustom:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit_predict(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        n_samples = X.shape[0]
        # Initialisation aléatoire des centres
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        centers = X[indices]
        for i in range(self.max_iter):
            # Assignation des clusters
            distances = cdist(X, centers, metric='euclidean')
            labels = np.argmin(distances, axis=1)
            # Calcul des nouveaux centres
            new_centers = np.array([X[labels==k].mean(axis=0) if np.any(labels==k) else centers[k] 
                                    for k in range(self.n_clusters)])
            # Convergence
            if np.linalg.norm(new_centers - centers) < self.tol:
                break
            centers = new_centers
        self.cluster_centers_ = centers
        return labels

# ------------------ DBSCAN ------------------
class DBSCANCustom:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, X):
        n_samples = X.shape[0]
        labels = -np.ones(n_samples, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        def region_query(idx):
            distances = np.linalg.norm(X - X[idx], axis=1)
            return np.where(distances <= self.eps)[0]

        def expand_cluster(idx, cluster_id):
            seeds = list(region_query(idx))
            if len(seeds) < self.min_samples:
                return False
            labels[seeds] = cluster_id
            while seeds:
                current = seeds.pop(0)
                neighbors = region_query(current)
                if len(neighbors) >= self.min_samples:
                    for n in neighbors:
                        if labels[n] == -1:
                            labels[n] = cluster_id
                            seeds.append(n)
            return True

        for i in range(n_samples):
            if not visited[i]:
                visited[i] = True
                if expand_cluster(i, cluster_id):
                    cluster_id += 1
        return labels

# ------------------ AGNES (Hierarchical Clustering) ------------------
class AgnesCustom:
    def __init__(self, n_clusters=3, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit_predict(self, X):
        Z = sch_linkage(X, method=self.linkage)
        labels = fcluster(Z, t=self.n_clusters, criterion='maxclust') - 1
        return labels
