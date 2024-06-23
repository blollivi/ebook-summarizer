"""
This is a boilerplate pipeline 'compute_embedding'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import (
    make_api_call,
    compute_pca_projection,
    compute_umap_projection
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=make_api_call,
            inputs=["chunks", "params:voyageai_api_key", "params:embedding_model_name"],
            outputs="embeddings",
            name="make_api_call",
        ),
        node(
            func=compute_pca_projection,
            inputs=["embeddings", "params:pca"],
            outputs="pca_projection",
            name="compute_pca_projection",
        ),
        node(
            func=compute_umap_projection,
            inputs=["pca_projection", "params:umap"],
            outputs="umap_projection",
            name="compute_umap_projection",
        ),
    ])
