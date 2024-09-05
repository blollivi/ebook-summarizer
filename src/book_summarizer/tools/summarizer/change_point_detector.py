import numpy as np
import ruptures as rpt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


class ChangePointDetector:
    def __init__(
        self,
        algorithm: str = "binseg",
        penalties: np.array = np.arange(2, 50, 0.25),
        denoise: bool = True,
        metric: str = "cosine",
    ):
        self.algorithm = algorithm
        self.penalties = penalties
        self.denoise = denoise
        self.metric = metric
        self.bkpts_matrix = None

    def fit(self, signal: np.array) -> None:
        """
        Find change points in the given signal using the specified penalty values.

        This method iterates through the penalty values, performing change point detection
        at each level and optionally denoising the signal between levels.

        Args:
            signal (np.array): The input signal.
        """
        bkpts = []
        centered_signal = signal

        for pen in tqdm(self.penalties, desc="Finding Change Points"):
            # cost = rpt.costs.CostCosine()
            if self.algorithm == "binseg":
                model = rpt.Binseg(model=self.metric, min_size=2, jump=1)
            elif self.algorithm == "window":
                model = rpt.Window(width=20, model=self.metric)
            elif self.algorithm == "pelt":
                model = rpt.KernelCPD(kernel=self.metric, min_size=2)
            else:
                raise ValueError("Invalid algorithm. Must be 'binseg' or 'pelt'.")

            model.fit(centered_signal)
            bkpts.append(model.predict(pen=pen))

            if self.denoise:
                centered_signal = compute_centered_signal(centered_signal, bkpts[-1])

        bkpt_matrix = np.zeros((len(self.penalties), len(signal)))
        for i, _bkps in enumerate(bkpts):
            for j in _bkps[:-1]:
                bkpt_matrix[i, j - 1] = 1

        self.bkpts_matrix = bkpt_matrix
        self.bkpts = bkpts

    def fit_predict(self, signal: np.array):
        """
        Perform change point detection on the given signal.

        Args:
            signal (np.array): The input signal.

        Returns:
            np.array: A matrix where each row corresponds to a penalty value and each column corresponds to a data point.
        A value of 1 indicates a change point.
        """
        self.fit(signal)
        return self.bkpts_matrix, self.bkpts


def medoid(arr: np.array):
    """
    Calculate the medoid of a set of vectors.

    The medoid is the point in the set with the smallest average distance to all other points.

    Args:
        arr (np.array): A 2D array of vectors.

    Returns:
        np.array: The medoid vector.
    """
    dist_matrix = cosine_similarity(arr)
    idx_medoid = np.argmin(np.sum(dist_matrix, axis=1))
    return arr[idx_medoid]


def compute_centered_signal(signal: np.array, bkpts: np.array):
    """
    Center a signal based on the medoid of each segment defined by the breakpoints.

    Args:
        signal (np.array): The input signal.
        bkpts (np.array): An array of breakpoint indices.

    Returns:
        np.array: The centered signal.
    """
    segments = get_segments_from_breakpoints(bkpts)
    centered_signal = np.zeros_like(signal)

    for seg in segments:
        _signal = signal[seg[0] : seg[1] + 1]
        center = medoid(_signal)
        centered_signal[seg[0] : seg[1] + 1] = center
    return centered_signal
