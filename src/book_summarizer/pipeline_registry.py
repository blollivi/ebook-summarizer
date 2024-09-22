"""Project pipelines."""

from typing import Dict

from kedro.pipeline import Pipeline

from book_summarizer.pipelines import parse_ebook, compute_embedding, summarize, populate_database, build_summary_tree


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    parse_ebook_pipeline = parse_ebook.create_pipeline()
    compute_embedding_pipeline = compute_embedding.create_pipeline()
    build_summary_tree_pipeline = build_summary_tree.create_pipeline()
    summarize_all_pipeline, summarize_tree_pipeline = summarize.create_pipelines()
    populate_database_pipeline = populate_database.create_pipeline()

    complete_pipeline = parse_ebook_pipeline + compute_embedding_pipeline

    pipelines = {
        "__default__": complete_pipeline,
        "parse_ebook": parse_ebook_pipeline,
        "compute_embedding": compute_embedding_pipeline,
        "build_summary_tree": build_summary_tree_pipeline,
        "summarize_all": summarize_all_pipeline,
        "summarize_tree": summarize_tree_pipeline,
        "populate_database": populate_database_pipeline,
    }

    return pipelines
