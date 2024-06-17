"""
This is a boilerplate pipeline 'parse_ebook'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node


from .nodes import extract_chunks, infer_paragraph_levels, filter_chunks


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=infer_paragraph_levels,
                inputs="ebook_content",
                outputs="paragraph_levels",
                name="infer_paragraph_levels",
            ),
            node(
                func=extract_chunks,
                inputs=["ebook_content", "paragraph_levels"],
                outputs="raw_chunks",
                name="extract_chunks",
            ),
            node(
                func=filter_chunks,
                inputs="raw_chunks",
                outputs="chunks",
                name="filter_chunks",
            ),
        ]
    )
