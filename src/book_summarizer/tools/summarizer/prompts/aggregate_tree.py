from langchain_core.prompts import PromptTemplate
from typing import List
from pathlib import Path
import json


SYSTEM_PROMPT = """<TASK DESCRIPTION>
    **Context:**
    You are a skilled text analyzer and summarizer. Your task is to write a global summary given
    the summmaries of all sections of a given text.
    The sections are defined by the first and last chunk of text that belong to that section.
    Sections are nested in a tree like structure: Higher-level sections contain lower-level subsections.

    **Input**:
    The flatten list of sections.
    JSON schema: List({{
        "start": <start_chunk_id>,
        "end": <end_chunk_id>,
        "summary": <section_summary>,
        "title": <section_title>
    }}).

    **Task Instructions**:
    Write a global summary of the book using the summaries of all sections. Take into account the hierarchy of the sections.
    The summary should be 6-8 sentences long.
    Write in {language}.
    Use only JSON-compatible characters.

    **Output**:
    A JSON object containing the summary:
    {{
        "summary": <section_summary>
    }}

    </TASK DESCRIPTION>


    <EXAMPLES>
    Use the following examples as guidance for your task.
    {examples_prompt}
    </EXAMPLES>

    <SECTIONS>
    {sections}
    </SECTIONS>

    Output:
"""


def _extract_section_lists(summary_tree: str, is_summary: bool = False) -> List[str]:
    """Breath-first search to extract the sections of the tree"""

    tree_dict = json.loads(summary_tree)

    sections = []

    def bfs(node):
        if "children" in node:
            for child in node["children"]:
                bfs(child)
        section = {"start": node["start"], "end": node["end"]}
        if is_summary:
            section["title"] = node["title"]
            section["summary"] = node["summary"]
        sections.append(section)

    bfs(tree_dict)

    return json.dumps(sections)


def _build_examples_prompt(
    examples_path: str = "data/examples",
) -> str:
    # list all folders in the examples directory
    examples = Path(examples_path).iterdir()

    prompt = ""

    for i, example in enumerate(examples):
        if example.is_dir():
            with open(example / "summary_tree.json", "r") as file:
                summary_tree = file.read()

            with open(example / "global_summary.json", "r") as file:
                summary = file.read()

            sections = _extract_section_lists(
                summary_tree, is_summary=True
            )

        prompt += f"""
        <EXAMPLE {i}>
        Input:
            <SECTIONS>
            {sections}
            <\SECTIONS>
        Output:
        {summary}

        """

    return prompt


def build_prompt_template() -> PromptTemplate:

    examples_prompt = _build_examples_prompt()

    return PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["sections"],
        partial_variables={"examples_prompt": examples_prompt},
    )
