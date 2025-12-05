import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage as sch_linkage, fcluster

# ------------------ K-MEANS vectorisé ------------------
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
            # Assignation des clusters (vectorisé)
            distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)
            # Mise à jour des centres
            new_centers = np.array([X[labels==k].mean(axis=0) if np.any(labels==k) else centers[k]
                                    for k in range(self.n_clusters)])
            # Convergence
            if np.linalg.norm(new_centers - centers) < self.tol:
                break
            centers = new_centers
        self.cluster_centers_ = centers
        return labels

# ------------------ DBSCAN vectorisé ------------------
class DBSCANCustom:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, X):
        n_samples = X.shape[0]
        labels = -np.ones(n_samples, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        # Pré-calcul de la matrice des distances (vectorisé)
        dist_matrix = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)

        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = np.where(dist_matrix[i] <= self.eps)[0]
            if len(neighbors) < self.min_samples:
                continue
            labels[neighbors] = cluster_id
            seeds = list(neighbors)
            while seeds:
                current = seeds.pop(0)
                if not visited[current]:
                    visited[current] = True
                    current_neighbors = np.where(dist_matrix[current] <= self.eps)[0]
                    if len(current_neighbors) >= self.min_samples:
                        for n in current_neighbors:
                            if labels[n] == -1:
                                labels[n] = cluster_id
                                seeds.append(n)
            cluster_id += 1
        return labels

# ------------------ AGNES ------------------
class AgnesCustom:
    def __init__(self, n_clusters=3, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit_predict(self, X):
        Z = sch_linkage(X, method=self.linkage)
        labels = fcluster(Z, t=self.n_clusters, criterion='maxclust') - 1
        return labels
