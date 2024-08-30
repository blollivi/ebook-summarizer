import pandas as pd
import numpy as np

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
from pathlib import Path

from book_summarizer.tools.summarizer.summary_tree import SummaryTree
from book_summarizer.tools.summarizer.llm_engine import LLMEngine
from book_summarizer.tools.summarizer.summarizer import Summarizer
from book_summarizer.tools.summarizer.prompts import build_summary_tree_prompt_template


project_root = Path(__file__).resolve().parents[4]
project_root = "."
conf_path = str(Path(project_root) / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]


def build_summary_tree(
    embeddings: pd.DataFrame, chpt_detection_params: dict
) -> SummaryTree:
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

    summary_tree = SummaryTree(
        penalties=penalties,
        denoise=chpt_detection_params["denoise"],
        algorithm=chpt_detection_params["algorithm"],
        metric=chpt_detection_params["metric"],
    ).fit(embeddings.to_numpy())
    return summary_tree


def extract_tree_cut(summary_tree: SummaryTree, tree_cut_params: dict) -> SummaryTree:

    # Compute the number of nodes at each penalty level
    node_counts = []
    for pen in summary_tree.penalties:
        node_counts.append(len(summary_tree.get_penalty_level_cut(pen)))

    cumul_node_counts = np.cumsum(node_counts)
    # Find the penalty level that corresponds to the 95th ot the last cumulative node count
    cut_thd = tree_cut_params["cut_thd"]
    cut_idx = np.where(cumul_node_counts >= cut_thd * cumul_node_counts[-1])[0][0]
    penalty_cut = summary_tree.penalties[cut_idx]

    tree_cut = summary_tree.get_penalty_level_cut(penalty_cut)

    return tree_cut


def summarize_tree(
    summary_tree: SummaryTree,
    head: str,
    chunks_df: pd.DataFrame,
    llm_engine_params: dict,
) -> dict:
    # Parse head
    head_tuple = tuple([int(i) for i in head.split("-")])
    print("Summarizing tree for head:", head_tuple)
    prompt_template = build_summary_tree_prompt_template()
    llm_engine = LLMEngine(
        prompt_template=prompt_template,
        google_api_key=credentials["google_api_key"],
        **llm_engine_params
    )

    summarizer = Summarizer(summary_tree, llm_engine, chunks_df)

    return {head: summarizer.summarize_tree(head_tuple)}
