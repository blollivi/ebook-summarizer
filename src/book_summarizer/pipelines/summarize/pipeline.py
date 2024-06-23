"""
This is a boilerplate pipeline 'summarize'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import build_summary_tree


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_summary_tree,
                inputs=["pca_projection", "params:change_point_detection"],
                outputs="summary_tree",
                name="build_summary_tree",
            )
        ]
    )
