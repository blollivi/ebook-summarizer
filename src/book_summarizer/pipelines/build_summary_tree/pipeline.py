"""
This is a boilerplate pipeline 'populate_database'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import (
    build_summary_tree,
    extract_tree_cut
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_summary_tree,
                inputs=["umap_projection", "params:change_point_detection"],
                outputs="summary_tree",
                name="build_summary_tree",
            ),
            node(
                func=extract_tree_cut,
                inputs=["summary_tree", "params:tree_cut"],
                outputs="tree_cut",
                name="extract_tree_cut",
            ),
        ]
    )