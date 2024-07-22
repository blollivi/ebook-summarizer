import pandas as pd
import numpy as np

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
from pathlib import Path

from book_summarizer.tools.summarizer.summary_tree import SummaryTree
from book_summarizer.tools.summarizer.llm_engine import LLMEngine
from book_summarizer.tools.summarizer.summarizer import Summarizer

project_root = Path(__file__).resolve().parents[4]
conf_path = str(Path(project_root) / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]


def build_summary_tree(
    embeddings: pd.DataFrame, chpt_detection_params: dict
) -> SummaryTree:
    penalty_params = chpt_detection_params["penalty_params"]
    algortihm = chpt_detection_params["algorithm"]
    if penalty_params["scale"] == "log":
        sampler = np.geomspace
    elif penalty_params["scale"] == "linear":
        sampler = np.linspace
    else:
        raise ValueError("Invalid scale value. Must be 'log' or 'linear'.")
    penalties = sampler(
        penalty_params["start"], penalty_params["end"], penalty_params["steps"]
    )
    denoise = chpt_detection_params["denoise"]
    summary_tree = SummaryTree(penalties=penalties, denoise=denoise).fit(
        embeddings.to_numpy()
    )
    return summary_tree


def summarize_all_tree_nodes(
    summary_tree: SummaryTree, chunks_df: pd.DataFrame
) -> dict:
    llm_engine = LLMEngine(google_api_key=credentials["google_api_key"])

    summarizer = Summarizer(summary_tree, llm_engine, chunks_df)

    summarizer.summarize()

    return summarizer.summarize_tree
