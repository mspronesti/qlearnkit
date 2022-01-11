from math import floor
import numpy as np


def random(X: np.ndarray,
           n_clusters: int,
           random_state: int = 42) -> np.ndarray:
    """
    Create random cluster centroids.

    Args:
        X:
            The dataset to be used for centroid initialization.
        n_clusters:
            The desired number of clusters for which centroids are required.
        random_state:
            Determines random number generation for centroid initialization.

    Returns:
        Collection of k centroids as a numpy ndarray.
    """
    np.random.seed(random_state)
    centroids = []
    m = np.shape(X)[0]

    for _ in range(n_clusters):
        r = np.random.randint(0, m - 1)
        centroids.append(X[r])

    return np.array(centroids)


def kmeans_plus_plus(X: np.ndarray,
                      k: int,
                      random_state: int = 42) -> np.ndarray:
    """
    Create cluster centroids using the k-means++ algorithm.

    Args:
        X:
            The dataset to be used for centroid initialization.
        k:
            The desired number of clusters for which centroids are required.
        random_state:
            Determines random number generation for centroid initialization.

    Returns:
        Collection of k centroids as a numpy ndarray.
    """
    np.random.seed(random_state)
    centroids = [X[0]]
    i = 0
    for _ in range(1, k):
        dist_sq = np.array([min([np.inner(c - x, c - x) for c in centroids]) for x in X])
        probs = dist_sq / dist_sq.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()
        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break

        centroids.append(X[i])

    return np.array(centroids)


def naive_sharding(X: np.ndarray,
                   k: int) -> np.ndarray:
    """
    Create cluster centroids using deterministic naive sharding algorithm.

    Args:
        X:
            The dataset to be used for centroid initialization.
        k:
            The desired number of clusters for which centroids are required.

    Returns:
        Collection of k centroids as a numpy ndarray.
    """
    n = np.shape(X)[1]
    m = np.shape(X)[0]
    centroids = np.zeros((k, n))

    composite = np.mat(np.sum(X, axis=1))
    ds = np.append(composite.T, X, axis=1)
    ds.sort(axis=0)

    step = floor(m / k)
    vfunc = np.vectorize(lambda sums, step_: sums/step_)

    for j in range(k):
        if j == k - 1:
            centroids[j:] = vfunc(np.sum(ds[j * step:, 1:], axis=0), step)
        else:
            centroids[j:] = vfunc(np.sum(ds[j * step:(j + 1) * step, 1:], axis=0), step)

    return centroids
