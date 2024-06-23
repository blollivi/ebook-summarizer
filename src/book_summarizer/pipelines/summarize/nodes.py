import pandas as pd
import numpy as np

from book_summarizer.tools.summarizer.summary_tree import SummaryTree


def build_summary_tree(embeddings: pd.DataFrame, chpt_detection_params: dict) -> SummaryTree:
    penalty_params = chpt_detection_params["penalty_params"]
    penalties = np.arange(penalty_params["start"], penalty_params["end"], penalty_params["step"])
    denoise = chpt_detection_params["denoise"]
    summary_tree = SummaryTree(penalties=penalties, denoise=denoise).fit(embeddings.to_numpy())
    return summary_tree


