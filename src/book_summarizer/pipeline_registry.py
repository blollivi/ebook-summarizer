"""Project pipelines."""

from typing import Dict

from kedro.pipeline import Pipeline

from book_summarizer.pipelines import parse_ebook, compute_embedding, summarize


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    parse_ebook_pipeline = parse_ebook.create_pipeline()
    compute_embedding_pipeline = compute_embedding.create_pipeline()
    summarize_pipeline = summarize.create_pipeline()

    complete_pipeline = parse_ebook_pipeline + compute_embedding_pipeline

    pipelines = {
        "__default__": complete_pipeline,
        "parse_ebook": parse_ebook_pipeline,
        "compute_embedding": compute_embedding_pipeline,
        "summarize": summarize_pipeline,
    }

    return pipelines
