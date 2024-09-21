from kedro.pipeline import Pipeline, pipeline, node


from .nodes import build_payload, make_post_request


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_payload,
                inputs=["hierarchical_summary", "summary_tree", "global_summary"],
                outputs="payload",
                name="build_payload",
            ),
            node(
                func=make_post_request,
                inputs=["params:django_api_uri", "payload"],
                outputs=None,
                name="make_post_request",
            ),
        ]
    )
