import pandas as pd

from src.book_summarizer.tools.summarizer.summary_tree import SummaryTree
from src.book_summarizer.tools.summarizer.engine import Summarizer


def summarize_graph_nodes(chunks_df: pd.DataFrame, summary_tree: SummaryTree):
    summarizer = Summarizer(summary_tree, chunks_df)
    
    summarizer.compute_summary()
    return summarizer.progress
    
    