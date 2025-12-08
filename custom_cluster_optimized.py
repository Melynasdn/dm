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
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        centers = X[indices].copy()

        for _ in range(self.max_iter):
            labels = np.argmin(cdist(X, centers), axis=1)
            new_centers = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
                for k in range(self.n_clusters)
            ])

            if np.linalg.norm(new_centers - centers) < self.tol:
                break

            centers = new_centers

        self.cluster_centers_ = centers
        return labels

# ----------------- K-MEDOIDS (AJOUTÉ) -----------------
class KMedoidsCustom:
    def __init__(self, n_clusters=3, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit_predict(self, X):
        X = np.array(X)
        n_samples = X.shape[0]

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initial medoids = random points
        medoids = np.random.choice(n_samples, self.n_clusters, replace=False)

        distances = cdist(X, X)

        for _ in range(self.max_iter):

            # Assign each point to nearest medoid
            labels = np.argmin(distances[:, medoids], axis=1)

            new_medoids = medoids.copy()

            # Compute new medoids for each cluster
            for k in range(self.n_clusters):
                cluster_points = np.where(labels == k)[0]

                if len(cluster_points) == 0:
                    continue

                # Total distance of each point to all others in cluster
                intra_dists = distances[np.ix_(cluster_points, cluster_points)]

                best_point_idx = cluster_points[np.argmin(intra_dists.sum(axis=1))]
                new_medoids[k] = best_point_idx

            if np.array_equal(new_medoids, medoids):
                break

            medoids = new_medoids

        self.medoids_ = medoids
        return labels

# ----------------- DBSCAN (CORRIGÉ) -----------------
class DBSCANCustom:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        self.X = StandardScaler().fit_transform(X)
        n_points = self.X.shape[0]

        labels = np.full(n_points, -1)
        cluster_id = 0

        for i in range(n_points):
            if labels[i] != -1:
                continue

            neighbors = self.region_query(i)
            if len(neighbors) < self.min_samples:
                labels[i] = -1
                continue

            labels[i] = cluster_id
            seeds = neighbors.copy()
            if i in seeds:
                seeds.remove(i)

            while seeds:
                point = seeds.pop()

                if labels[point] == -1:
                    labels[point] = cluster_id

                if labels[point] != -1:
                    continue

                labels[point] = cluster_id
                point_neighbors = self.region_query(point)

                if len(point_neighbors) >= self.min_samples:
                    for n in point_neighbors:
                        if labels[n] == -1:
                            seeds.append(n)

            cluster_id += 1

        self.labels_ = labels
        return self

    def region_query(self, idx):
        distances = np.linalg.norm(self.X - self.X[idx], axis=1)
        return np.where(distances <= self.eps)[0].tolist()

def diagnose_dbscan(X, eps=0.5, min_samples=5):
    neigh = NearestNeighbors(n_neighbors=min_samples).fit(StandardScaler().fit_transform(X))
    distances, _ = neigh.kneighbors(X)
    kth_dist = np.sort(distances[:, -1])
    suggested_eps = np.percentile(kth_dist, 90)

    return {
        'suggested_eps': suggested_eps,
        'total_points': X.shape[0],
        'potential_core_points': np.sum(kth_dist <= suggested_eps),
        'core_point_ratio': np.mean(kth_dist <= suggested_eps),
        'avg_neighbors_at_eps': np.mean(np.sum(distances <= eps, axis=1))
    }

# ----------------- AGNES -----------------
class AgnesCustom:
    def __init__(self, n_clusters=3, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
    
    def fit_predict(self, X):
        Z = sch_linkage(X, method=self.linkage)
        labels = fcluster(Z, t=self.n_clusters, criterion='maxclust') - 1
        return labels
    
# ------------------ DIANA (CORRIGÉ) ------------------
class DianaCustom:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters

    def fit_predict(self, X):
        X = np.array(X)
        clusters = [list(range(X.shape[0]))]

        while len(clusters) < self.n_clusters:

            largest_cluster_idx = np.argmax([len(c) for c in clusters])
            cluster = clusters.pop(largest_cluster_idx)

            if len(cluster) <= 1:
                clusters.append(cluster)
                break

            sub_X = X[cluster]

            mean_dist = np.mean(cdist(sub_X, sub_X), axis=1)
            pivot_idx = np.argmax(mean_dist)

            group1 = [cluster[pivot_idx]]
            group2 = [cluster[i] for i in range(len(cluster)) if i != pivot_idx]

            clusters.append(group2)
            clusters.append(group1)

        labels = np.zeros(X.shape[0], dtype=int)
        for cid, cl in enumerate(clusters):
            for idx in cl:
                labels[idx] = cid

        return labels
# ----------------- ELBOW METHOD -----------------
def elbow_method(X, max_k=10, random_state=0):
    """Retourne l'inertie (= SSE) pour k = 1..max_k."""

    inertias = []
    
    for k in range(1, max_k + 1):
        model = KMeansCustom(n_clusters=k, random_state=random_state)
        labels = model.fit_predict(X)
        
        # Calcul de l'inertie : somme distances² aux centroïdes
        inertia = 0
        for i, center in enumerate(model.cluster_centers_):
            inertia += np.sum(np.linalg.norm(X[labels == i] - center, axis=1) ** 2)

        inertias.append(inertia)
    
    return inertias
