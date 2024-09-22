"""
This is a boilerplate pipeline 'summarize'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node


from .nodes import (
    summarize_tree,
    summarize_all,
    write_global_summary,
)


def create_pipelines(**kwargs) -> Pipeline:
    summarize_all_pipeline = pipeline(
        [
            node(
                func=summarize_all,
                inputs=["tree_cut", "summary_tree", "chunks", "params:llm_engine"],
                outputs="hierarchical_summary",
                name="summarize_all",
            ),
            node(
                func=write_global_summary,
                inputs=["hierarchical_summary", "params:llm_engine"],
                outputs="global_summary",
                name="write_global_summary",
            ),
        ]
    )

    summarize_tree_pipeline = pipeline(
        [
            node(
                func=summarize_tree,
                inputs=["params:head_str", "summary_tree", "chunks", "params:llm_engine"],
                outputs="hierarchical_summary",
                name="summarize_tree",
            ),
        ]
    )

    return summarize_all_pipeline, summarize_tree_pipeline
