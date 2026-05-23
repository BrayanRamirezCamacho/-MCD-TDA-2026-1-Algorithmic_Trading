import numpy as np

from ripser import ripser
from gtda.time_series import SingleTakensEmbedding
from gtda.time_series import TakensEmbedding
from gtda.diagrams import PersistenceEntropy
from gtda.diagrams import BettiCurve


def takens_embedding(
        series: np.ndarray, 
        dim: int, 
        delay: int, 
        stride: int = 1, 
        n_jobs: int = -1
): 
    embedding = SingleTakensEmbedding(
        parameters_type='fixed', 
        dimension=dim, 
        time_delay=delay, 
        stride=stride, 
        n_jobs=n_jobs
    )

    point_cloud = embedding.fit_transform(series)

    return point_cloud


def batch_takens_embedding(
        windows: np.ndarray, 
        dim: int, 
        delay: int, 
        stride: int, 
): 
    embedding = TakensEmbedding(
        time_delay=delay, 
        dimension=dim, 
        stride=stride
    )

    return embedding.fit_transform(windows)


def compute_persistence_diagram(
        point_cloud: np.ndarray, 
        maxdim: int = 1  # H0, H1
):
    """
    Computa la homologia persistente de una nube de puntos usando
    la filtración de Vietoris-Rips.
    """

    result = ripser(point_cloud, maxdim=maxdim)   
    
    return result["dgms"]

def clean_diagram(D: np.ndarray):
    mask = (
        np.isfinite(D[:, 1]) &  
        ~np.isnan(D).any(axis=1)  
    )
    return D[mask]


def compute_persistence_vector(
        diagram: np.ndarray,
        sort: bool = True, 
        descending: bool = True
): 
    """Calcula el vector de persistencia de una diagrama birth-death."""

    births = diagram[:, 0]
    deaths = diagram[:, 1]

    persistence = deaths - births

    if sort: 
        persistence = np.sort(persistence)
        if descending: 
            persistence = persistence[::-1]

    return persistence


def to_gtda_format(diagrams, homology_dim):

    dims = np.full(
        diagrams.shape[:-1] + (1,),
        homology_dim
    )

    diagrams = np.concatenate(
        [diagrams, dims],
        axis=-1
    )

    return np.expand_dims(diagrams, axis=0)


def compute_persistence_entropy(
        diagram: np.ndarray,
        homology_dim: int = 0
): 
    """Calcula la persistence entropy de un diagrama birth-death."""

    import warnings
    warnings.filterwarnings("ignore")
   
    diagram = clean_diagram(diagram)

    diagram = to_gtda_format(diagram, homology_dim)

    pe = PersistenceEntropy()

    entropy = pe.fit_transform(diagram)

    return entropy


def betti_curve(
        diagram,
        homology_dim=0,
        n_bins=100
):
    """Calcula la Betti curve de un diagrama de persistencia."""

    import warnings
    warnings.filterwarnings("ignore")

    diagram = clean_diagram(diagram)

    diagram = to_gtda_format(diagram, homology_dim)

    bc = BettiCurve(
        n_bins=n_bins
    )

    beta = bc.fit_transform(diagram)

    t = bc.samplings_[homology_dim]

    return t, beta[0, 0]