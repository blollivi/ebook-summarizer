from typing import Tuple
import networkx as nx
import pandas as pd
import numpy as np
import json

from book_summarizer.tools.summarizer.summary_tree import SummaryTree
from book_summarizer.tools.summarizer.llm_engine import LLMEngine


class Summarizer:
    """
    A class to summarize a book represented as a SummaryTree.

    This class uses an LLMEngine to generate summaries for each node in the SummaryTree,
    traversing the tree in a specific order to maintain context and coherence.
    """

    def __init__(
        self, summary_tree: SummaryTree, llm_engine: LLMEngine, chunks_df: pd.DataFrame
    ):
        """
        Initialize the Summarizer.

        Args:
            summary_tree (SummaryTree): The SummaryTree representing the book.
            llm_engine (LLMEngine): The LLMEngine to use for generating summaries.
            chunks_df (pd.DataFrame): Dataframe containing the chunks of text.
        """
        self.summary_tree = summary_tree
        self.llm_engine = llm_engine
        self.summary_path = summary_tree.summary_path
        self.outline_path = summary_tree.outline_path
        self.chunks_df = chunks_df
        self.progress = []  # List of nodes that have been summarized
        self.is_error = False

    def build_chunks_prompt(self, head: Tuple[int, int]):
        chunks = self.chunks_df.reset_index(drop=True)
        chunks = chunks.iloc[head[0] : head[1] + 1]["text"]
        return chunks.to_json()

    def summarize_subtree(self, head):
        """
        Summarize the subtree rooted at the given head node.

        Args:
            head: The head node of the subtree to summarize.

        Returns:
            dict: The generated summary for the head node, or None if an error occurred.
        """

        # Get context prompts
        chunks_prompt = self.build_chunks_prompt(head)
        sections = self.summary_tree.get_subtree_nodes_list(head)

        sections_with_summary = self.llm_engine.generate_response(
            dict(
                chunks=chunks_prompt,
                sections=json.dumps(sections),
                language="English",
            )
        )

        return sections_with_summary

    def summarize(self):
        """
        Summarize the entire summary tree.

        This method iterates through the heads of the summary tree, summarizing each subtree.
        """
        for head in self.summary_tree.heads:
            if self.is_error:
                break
            else:
                self.summarize_subtree(head)

    def check_output(self, output: dict, head: tuple) -> dict:
        """
        Check the output of the LLM engine for errors.

        Args:
            output (dict): The output of the LLM engine.

        """
        sections = self.summary_tree.get_subtree_nodes_list(head)
        output_sections = [(section["start"], section["end"]) for section in output]

        # Check if all sections are present in the output
        missing_sections = [
            section for section in sections if section not in output_sections
        ]

        print("Missing sections:", missing_sections)

        return [
            {
                "start": section["start"],
                "end": section["end"],
                "summary": section["summary"],
                "title": section["title"],
            }
            for section in output
        ]
