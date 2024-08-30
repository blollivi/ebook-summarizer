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

    def build_tree_outline(self, head, level, outline=""):
        """
        Recursively build the outline of the summary tree.

        Args:
            head: The current node being visited.
            level (int): The level of the current node in the tree.
            outline (str, optional): The outline string being built. Defaults to "".
        """
        child_nodes = list(self.summary_tree.graph.successors(head))

        for child in child_nodes:
            outline[child] = level
            self.build_tree_outline(child, level + 1, outline)

    def build_context_path(self, current_node):
        """
        Build the context path for a given node.

        The context path includes all nodes on the path from the root to the current node,
        up to the node with the same penalty level as the current node.

        Args:
            current_node: The node for which to build the context path.

        Returns:
            list: The list of nodes in the context path.
        """
        current_node_penalty = self.summary_tree.graph.nodes[current_node]["pen"]
        context_nodes = self.summary_tree.get_penalty_level_cut(current_node_penalty)
        current_node_idx = context_nodes.index(current_node)
        return context_nodes[:current_node_idx]

    def build_chunks_prompt(self, head: Tuple[int, int]):
        chunks =  self.chunks_df.reset_index(drop=True)
        chunks = chunks.iloc[head[0] : head[1] + 1]["text"]
        return chunks.to_json()

    def build_summary_tree_prompt(self, head: Tuple[int, int]):
        """
        Extract all the descendant tree of a given head and return it as a JSON string.
        """

        descendants = nx.descendants(self.summary_tree.graph, head)
        descendants.add(head)
        sub_graph = self.summary_tree.graph.subgraph(descendants)
        tree_dict = nx.readwrite.json_graph.tree_data(sub_graph, root=head)

        def _clean_tree(tree_dict):
            if "children" in tree_dict:
                tree_dict["children"] = [
                    _clean_tree(child) for child in tree_dict["children"]
                ]
            return {
                key: tree_dict[key]
                for key in ["start", "end", "children"]
                if key in tree_dict
            }

        tree_dict = _clean_tree(tree_dict)

        return json.dumps(tree_dict)

    def summarize_tree(self, head):
        """
        Summarize the subtree rooted at the given head node.

        Args:
            head: The head node of the subtree to summarize.

        Returns:
            dict: The generated summary for the head node, or None if an error occurred.
        """

        # Get context prompts
        chunks_prompt = self.build_chunks_prompt(head)
        summary_tree_prompt = self.build_summary_tree_prompt(head)

        return self.llm_engine.generate_response(
            dict(
                chunks=chunks_prompt,
                summary_tree=summary_tree_prompt,
                language="English",
            )
        )

    def summarize(self):
        """
        Summarize the entire summary tree.

        This method iterates through the heads of the summary tree, summarizing each subtree.
        """
        for head in self.summary_tree.heads:
            if self.is_error:
                break
            else:
                self.summarize_tree(head)
