import networkx as nx
import pandas as pd


from book_summarizer.tools.summarizer.summary_tree import SummaryTree
from book_summarizer.tools.summarizer.llm_engine import LLMEngine


class Summarizer:
    def __init__(
        self, summary_tree: SummaryTree, llm_engine: LLMEngine, chunks_df: pd.DataFrame
    ):
        self.summary_tree = summary_tree
        self.llm_engine = llm_engine
        self.summary_path = summary_tree.get_summary_path()
        self.outline_path = summary_tree.get_outline_path()
        self.chunks_df = chunks_df
        self.progress = []  # List of nodes that have been summarized
        self.is_error = False

    def build_tree_outline(self, head, level, outline=""):
        # Navigate the tree to build the outline
        child_nodes = list(self.summary_tree.graph.successors(head))

        for child in child_nodes:
            outline[child] = level
            self.build_tree_outline(child, level + 1, outline)

    def build_context_path(self, current_node):
        outline_path = list(self.outline_path)
        current_node_idx = outline_path.index(current_node)
        outline_path = outline_path[:current_node_idx]

        book_coverage = 0
        context_path = []
        for node_id in outline_path:
            if "short_summary" in self.summary_tree.graph.nodes[node_id]:
                if node_id[1] > book_coverage:
                    context_path.append(node_id)
                    book_coverage = node_id[1]
        return context_path

    def build_context_prompt(self, current_node):
        context_path = self.build_context_path(current_node)

        return "\n".join(
            [
                self.summary_tree.graph.nodes[node_id]["long_summary"]
                for node_id in context_path
            ]
        )

    def summarize_node(self, node_id: int):
        node = self.summary_tree.graph.nodes[node_id]

        child_nodes = list(self.summary_tree.graph.successors(node_id))
        if len(child_nodes) == 0:
            chunks = (
                self.chunks_df["text"].iloc[node["start"] : node["end"] + 1].to_list()
            )

        else:
            chunks = [
                self.summary_tree.graph.nodes[node_id]["long_summary"]
                for node_id in child_nodes
            ]

        context_path = self.build_context_path(node_id)
        context = self.build_context_prompt(node_id)
        print(
            f"Summarizing node {node_id} from {child_nodes} with context {context_path}"
        )

        try:
            return self.llm_engine.generate_response(dict(chunk=chunks, context=context, language="French"))
        except Exception as e:
            print(f"Error summarizing node {node_id}: {e}")
            return None

    def summarize_tree(self, head):
        i = 0
        head_idx = self.summary_path.index(head) + 1

        for i in range(head_idx):
            node = self.summary_path[i]
            if "long_summary" in self.summary_tree.graph.nodes[node]:
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
                        "long_summary": node_summary["long"],
                        "short_summary": node_summary["short"],
                        "title": node_summary["title"],
                    }
                },
            )

        return node_summary

    def summarize(self):
        for head in self.summary_tree.graph.heads:
            if self.is_error:
                break
            else:
                self.summarize_tree(head)
