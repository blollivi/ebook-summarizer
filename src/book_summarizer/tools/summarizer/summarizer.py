import networkx as nx
import pandas as pd

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

    def build_context_prompt(self, current_node):
        """
        Build the context prompt for a given node.

        The context prompt includes the summaries of all nodes in the context path,
        concatenated together.

        Args:
            current_node: The node for which to build the context prompt.

        Returns:
            str: The context prompt string.
        """
        context_path = self.build_context_path(current_node)

        return "\n".join(
            [
                self.summary_tree.graph.nodes[node_id]["summary"]
                for node_id in context_path
            ]
        )

    def summarize_node(self, node_id: int):
        """
        Generate a summary for a given node.

        This method retrieves the text chunks associated with the node, builds the context prompt,
        and uses the LLMEngine to generate a summary.

        Args:
            node_id (int): The ID of the node to summarize.

        Returns:
            dict: The generated summary, or None if an error occurred.
        """
        node = self.summary_tree.graph.nodes[node_id]

        child_nodes = list(self.summary_tree.graph.successors(node_id))
        if len(child_nodes) == 0:
            chunks = (
                self.chunks_df["text"].iloc[node["start"] : node["end"] + 1].to_list()
            )

        else:
            chunks = [
                self.summary_tree.graph.nodes[node_id]["summary"]
                for node_id in child_nodes
            ]

        context_path = self.build_context_path(node_id)
        context = self.build_context_prompt(node_id)
        print(
            f"Summarizing node {node_id} from {child_nodes} with context {context_path}"
        )
        chunk = "\n".join(chunks)
        try:
            return self.llm_engine.generate_response(
                dict(chunk=chunk, context=context, language="French")
            )
        except Exception as e:
            print(f"Error summarizing node {node_id}: {e}")
            return None

    def summarize_tree(self, head):
        """
        Summarize the subtree rooted at the given head node.

        This method iterates through the nodes in the subtree, generating a summary for each node
        that does not already have one.

        Args:
            head: The head node of the subtree to summarize.

        Returns:
            dict: The generated summary for the head node, or None if an error occurred.
        """
        i = 0
        head_idx = self.summary_path.index(head) + 1

        for i in range(head_idx):
            node = self.summary_path[i]
            if "summary" in self.summary_tree.graph.nodes[node]:
                continue
            node_summary = self.summarize_node(node)
            if node_summary is None:
                # Stop the loop
                self.is_error = False
                break
            nx.set_node_attributes(
                self.summary_tree.graph,
                {
                    node: {
                        "summary": node_summary["summary"],
                        "title": node_summary["title"],
                    }
                },
            )

        return node_summary

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