"""
This is a boilerplate pipeline 'compute_embedding'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import (
    make_api_call,
    compute_pca_projection,
    compute_change_points_graph
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=make_api_call,
            inputs=["chunks", "params:api_key", "params:model_name"],
            outputs="embeddings",
            name="make_api_call",
        ),
        node(
            func=compute_pca_projection,
            inputs=["embeddings", "params:pca_config"],
            outputs="pca_projection",
            name="compute_pca_projection",
        ),
        node(
            func=compute_change_points_graph,
            inputs=["pca_projection"],
            outputs="change_points_graph",
            name="compute_change_points_graph",
        )
    ])
