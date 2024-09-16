from langchain_core.prompts import PromptTemplate
from typing import List
from pathlib import Path
import json


SYSTEM_PROMPT = """<TASK DESCRIPTION>
    **Context:**
    You are a skilled text analyzer and summarizer. Your task is to summarize all sections of a given text.
    The sections are defined by the first and last chunk of text that belong to that section.
    Sections are nested in a tree like structure: Higher-level sections contain lower-level subsections.

    **Input**:
    1. A text, represented as a set of subsquent chunks. And formatted as a JSON object with given schema:
            {{<chunk_id>: <chunk_text>}}
    2. The flatten list of sections.
    JSON schema: List({{
        "start": <start_chunk_id>,
        "end": <end_chunk_id>,
    }}).

    **Task Instructions**:
    For each section in the list, Identify the content of the section based on start and end chunk ids and, apply the followiong process:
        - Write a 2-3 sentence summary of the section
        - Write a title
    Use the author's perspective, avoid third-person references.
    Minimize redundancy between parent and child summaries.
    Write in {language}.
    Use only JSON-compatible characters.

    **Output**:
    A JSON list containing the same elements of the input list, with the addition of the "title" and "summary" keys:
    {{
        "start": <start_chunk_id>,
        "end": <end_chunk_id>,
        "title": <section_title>,
        "summary": <section_summary>
    }}


    **IMPORTANT**
    Make sure that all sections present in the input list are present in the output.
    Each summary must only reflect the information in the corresponding section.

    </TASK DESCRIPTION>


    <EXAMPLES>
    Use the following examples as guidance for your task.
    {examples_prompt}
    </EXAMPLES>

    <CHUNKS>
    {chunks}
    </CHUNKS>

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
            with open(example / "chunks.json", "r") as file:
                chunks = file.read()

            with open(example / "summary_tree.json", "r") as file:
                summary_tree = file.read()

            sections_json = _extract_section_lists(summary_tree, is_summary=False)
            sections_with_summary_json = _extract_section_lists(
                summary_tree, is_summary=True
            )

        prompt += f"""
        <EXAMPLE {i}>
        Input:
            <CHUNKS>
            {chunks}
            <CHUNKS>
            <SECTIONS>
            {sections_json}
            <\SECTIONS>
        Output:
        {sections_with_summary_json}

        """

    return prompt


def build_prompt_template() -> PromptTemplate:

    examples_prompt = _build_examples_prompt()

    return PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["chunks", "sections"],
        partial_variables={"examples_prompt": examples_prompt},
    )
