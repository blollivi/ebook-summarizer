from langchain_core.prompts import PromptTemplate
import networkx as nx
from pathlib import Path
import json


SYSTEM_PROMPT = """<TASK DESCRIPTION>
    You are a skilled text analyzer and summarizer. Your task is to summarize sections of a given text based on a hierarchical structure.

    Input:
    1. Text chunks in JSON format: <chunk_id>: <chunk_text>
    2. Tree structure representing the hierarchical outline of the text in JSON format.
        Each node represents a section and follows this schema:
        "start" (int): <The first chunk of the section>,
        "end" (int): <The last chunk of the section>,
        "children" (list, optional): <List of child nodes (sub-sections)>.


    Task Instructions:
    For each node of the tree, apply the followiong process:
        1. Identify the content of the section based on start and end chunk ids.
        2. Write a 2-3 sentence summary of the section.
        3. Write a title.
    Use the author's perspective, avoid third-person references.
    Minimize redundancy between parent and child summaries.
    Write in {language}.
    Use only JSON-compatible characters.

    Output:
    A summary tree with the same nodes as the input, with the addition of the "summary" and "title" keys:
        "start": <start_chunk_id>,
        "end": <end_chunk_id>,
        "summary": "<2-3 sentence summary>",
        "title": "<section title>",
        "children": [<sub-sections>]

    **IMPORTANT**
    Make sure that all nodes present in the input tree are present in the output tree

    </TASK DESCRIPTION>


    <EXAMPLES>
    Use the following examples as guidance for your task.
    {examples_prompt}
    </EXAMPLES>

    <CHUNKS>
    {chunks}
    </CHUNKS>

    <TREE STRUCTURE>
    {tree_structure}
    </TREE STRUCTURE>

    Output:
"""


def _extract_tree_structure(summary_tree: str) -> str:
    """remove the title and summary keys from all nodes of the tree"""
    tree_dict = json.loads(summary_tree)

    # remove the title and summary keys from all nodes
    # Also remove the children key if it is empty
    def remove_keys(node):
        if "title" in node:
            del node["title"]
        if "summary" in node:
            del node["summary"]
        if "children" in node:
            if not node["children"]:
                del node["children"]
            else:
                for child in node["children"]:
                    remove_keys(child)

    remove_keys(tree_dict)

    return json.dumps(tree_dict)


def _build_examples_prompt(
    examples_path: str = "data/examples",
) -> str:
    # list all folders in the examples directory
    examples = Path(examples_path).iterdir()

    prompt = ""

    for i, example in enumerate(examples):
        if example.is_dir():
            with open(example / "chunks.json", "r") as file:
                chunks = file.read()

            with open(example / "summary_tree.json", "r") as file:
                summary_tree = file.read()

            tree_structure = _extract_tree_structure(summary_tree)

        prompt += f"""
        <EXAMPLE {i}>
        Input:
            <CHUNKS>
            {chunks}
            <CHUNKS>
            <TREE STRUCTURE>
            {tree_structure}
            <\TREE STRUCTURE>
        Output:
        {summary_tree}

        """

    return prompt


def build_prompt_template() -> PromptTemplate:

    examples_prompt = _build_examples_prompt()

    return PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["chunks", "tree"],
        partial_variables={"examples_prompt": examples_prompt},
    )
