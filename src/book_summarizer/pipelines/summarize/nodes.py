import pandas as pd
import numpy as np

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
from pathlib import Path
import json

from book_summarizer.tools.summarizer.change_point_detector import ChangePointDetector
from book_summarizer.tools.summarizer.summary_tree import SummaryTree
from book_summarizer.tools.summarizer.llm_engine import LLMEngine
from book_summarizer.tools.summarizer.summarizer import Summarizer
from book_summarizer.tools.summarizer.prompts import (
    build_summary_tree_prompt_template,
    build_aggregate_tree_prompt_template,
)


project_root = Path(__file__).resolve().parents[4]
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

    bkpts_matrix, bkpts = ChangePointDetector(
        penalties=penalties,
        denoise=chpt_detection_params["denoise"],
        algorithm=chpt_detection_params["algorithm"],
        metric=chpt_detection_params["metric"],
    ).fit_predict(embeddings.to_numpy())

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


def summarize_all(
    tree_cut: list,
    summary_tree: SummaryTree,
    chunks_df: pd.DataFrame,
    llm_engine_params: dict,
):
    prompt_template = build_summary_tree_prompt_template()
    llm_engine = LLMEngine(
        prompt_template=prompt_template,
        google_api_key=credentials["google_api_key"],
        **llm_engine_params,
    )

    summarizer = Summarizer(summary_tree, llm_engine, chunks_df)

    hierarchical_summary = {}
    for head in tree_cut:
        head_str = "-".join([str(i) for i in head])
        print(f"Summarizing tree for head: {head_str}")
        # Convert head to tuple
        head = tuple(head)
        output = summarizer.summarize_subtree(head)
        try:
            hierarchical_summary[head_str] = summarizer.check_output(output, head)
        except Exception as e:
            print(f"Output Check Error: {e}")
            hierarchical_summary[head_str] = output

    return hierarchical_summary


def write_global_summary(hierarchical_summary: dict, llm_engine_params: dict):
    prompt_template = build_aggregate_tree_prompt_template()
    llm_engine = LLMEngine(
        prompt_template=prompt_template,
        google_api_key=credentials["google_api_key"],
        **llm_engine_params,
    )

    all_sections = []
    for head_str, section_loader in hierarchical_summary.items():
        all_sections.extend(section_loader())

    return llm_engine.generate_response(
        dict(
            sections=json.dumps(all_sections),
            language="English",
        )
    )


def summarize_tree(
    head_str: str,
    summary_tree: SummaryTree,
    chunks_df: pd.DataFrame,
    llm_engine_params: dict,
):
    prompt_template = build_summary_tree_prompt_template()
    llm_engine = LLMEngine(
        prompt_template=prompt_template,
        google_api_key=credentials["google_api_key"],
        **llm_engine_params,
    )

    summarizer = Summarizer(summary_tree, llm_engine, chunks_df)

    hierarchical_summary = {}
    print(f"Summarizing tree for head: {head_str}")
    # Convert head_str to tuple
    head = tuple(map(int, head_str.split("-")))
    output = summarizer.summarize_subtree(head)
    try:
        hierarchical_summary[head_str] = summarizer.check_output(output, head)
    except Exception as e:
        print(f"Output Check Error: {e}")
        hierarchical_summary[head_str] = output

    return hierarchical_summary
