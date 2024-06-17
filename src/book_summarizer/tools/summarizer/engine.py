from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import RetryOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from src.book_summarizer.tools.summarizer.summary_tree import SummaryTree
import networkx as nx
import pandas as pd


from .prompts import build_prompt_template

GOOGLE_API_KEY = "AIzaSyBBUP5cLkckeHhariMLznIwnUYMv1jc0vM"

chat = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=GOOGLE_API_KEY,
    generation_config={"response_mime_type": "application/json"},
    temperature=0.5,
)

parser = JsonOutputParser()

retry_parser = RetryOutputParser.from_llm(
    parser=parser,
    llm=ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=GOOGLE_API_KEY,
        generation_config={"response_mime_type": "application/json"},
        temperature=0.0,
    ),
)


def invoke_llm(chunks, context=""):

    chunk = "_".join(chunks)

    prompt_template = build_prompt_template()
    chain = prompt_template | chat
    output = chain.invoke(dict(chunk=chunk, language="French", outline=context))
    try:
        return parser.parse(output.content)
    except Exception as e:
        return retry_parser.parse_with_prompt(output.content, prompt_template)


class Summarizer:
    def __init__(self, summary_tree: SummaryTree, chunks_df: pd.DataFrame):
        self.summary_tree = summary_tree
        self.chunks_df = chunks_df
        self.progress = []  # List of nodes that have been summarized
        self.is_error = False

    def build_tree_outline(self, head, level, outline=""):
        # Navigate the tree to build the outline
        child_nodes = list(self.summary_tree.predecessors(head))

        for child in child_nodes:
            outline[child] = level
            self.build_tree_outline(child, level + 1, outline)

    def build_context_prompt(self, current_node):
        current_tree_idx = self.summary_tree.connected_trees.index(
            self.summary_tree.nodes[current_node]["tree"]
        )

        # Select previous nodes in the summary path
        history = self.summary_tree.summary_path[
            : self.summary_tree.summary_path.index(current_node)
        ]
        pass

    def summarize_node(self, node_id: int):
        node = self.summary_tree.nodes[node_id]

        child_nodes = list(self.summary_tree.predecessors(node_id))
        if len(child_nodes) == 0:
            chunks = (
                self.chunks_df["text"].iloc[node["start"] : node["end"] + 1].to_list()
            )

        else:
            chunks = [
                self.summary_tree.nodes[node_id]["long_summary"]
                for node_id in child_nodes
            ]

        context = self.build_context_prompt(node_id)
        print(context + "\n")

        try:
            return invoke_llm(chunks, context)
        except Exception as e:
            print(f"Error summarizing node {node_id}: {e}")
            return None

    def summarize_tree(self, head):
        i = 0
        head_idx = self.summary_tree.heads.index(head)
        tree_nodes = self.summary_tree.connected_trees[head_idx]
        # Order nodes in the tree by their position in the summary path
        tree_nodes = sorted(
            tree_nodes, key=lambda x: self.summary_tree.summary_path.index(x)
        )

        for node in tree_nodes:
            print(f"Summarizing node {node}")
            node_summary = self.summarize_node(node)
            if node_summary is None:
                # Stop the loop
                self.is_error = False
                break
            nx.set_node_attributes(
                self.summary_tree,
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
        for head in self.summary_tree.heads:
            if self.is_error:
                break
            else:
                self.summarize_tree(head)
