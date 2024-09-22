import numpy as np
import pandas as pd

from book_summarizer.tools.summarizer.change_point_detector import ChangePointDetector
from book_summarizer.tools.summarizer.summary_tree import SummaryTree


def detect_change_points(
    embeddings: pd.DataFrame, chpt_detection_params: dict
) -> np.ndarray:
    penalty_params = chpt_detection_params["penalty_params"]

    if penalty_params["scale"] == "log":
        sampler = np.geomspace
    elif penalty_params["scale"] == "linear":
        sampler = np.linspace
    else:
        raise ValueError("Invalid scale value. Must be 'log' or 'linear'.")
    penalties = sampler(
        penalty_params["start"], penalty_params["end"], penalty_params["steps"]
    )

    bkpts_matrix, bkpts = ChangePointDetector(
        penalties=penalties,
        denoise=chpt_detection_params["denoise"],
        algorithm=chpt_detection_params["algorithm"],
        metric=chpt_detection_params["metric"],
    ).fit_predict(embeddings.to_numpy())

    return bkpts_matrix, bkpts, penalties


def build_summary_tree(
    bkpts_matrix: np.ndarray, bkpts: np.ndarray, penalties: np.ndarray
) -> SummaryTree:

    summary_tree = SummaryTree(
        bkpts_matrix=bkpts_matrix, bkpts=bkpts, penalties=penalties
    ).fit()

    return summary_tree


def extract_tree_cut(summary_tree: SummaryTree, tree_cut_params: dict) -> SummaryTree:

    # Compute the number of nodes at each penalty level
    node_counts = []
    for pen in summary_tree.penalties:
        node_counts.append(len(summary_tree.get_penalty_level_cut(pen)))

    cumul_node_counts = np.cumsum(node_counts)
    # Find the penalty level that corresponds to the given cut threshold
    cut_thd = tree_cut_params["cut_thd"]
    cut_idx = np.where(cumul_node_counts >= cut_thd * cumul_node_counts[-1])[0][0]
    penalty_cut = summary_tree.penalties[cut_idx]

    tree_cut = summary_tree.get_penalty_level_cut(penalty_cut)

    return tree_cut
