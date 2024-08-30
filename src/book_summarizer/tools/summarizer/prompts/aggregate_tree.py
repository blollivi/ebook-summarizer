from langchain_core.prompts import PromptTemplate


SYSTEM_PROMPT = """<TASK DESCRIPTION>

    You are an talentful text analyzer and summarizer. You are been given:
        - tree_structure: a tree structure representing the hierarchical splitting of the text into section and subsections.
            Each node in the tree represents a section with a start and end chunk id (included), and a list of children nodes.
                "start": <start_chunk_id>,
                "end": <end_chunk_id>,
                "children": [
                    <child_tree>
                    , ...]

        - summary_tree: a tree that covers partially the nodes present in tree_structure; and each node have two aditional keys: "summary" and "title".
                "start": <start_chunk_id>,
                "end": <end_chunk_id>,
                "summary": "<section_summary>",
                "title": "<section_title>",
                "children": [
                    <child_tree>
                    , ...]

    Your task is to complete the summary_tree whith the missing nodes. Complete the associated "summary" and "title" keys for each node using
    the information already present in the summary_tree.
    When writing a parent summary, use only informations from its descendants.

    </TASK DESCRIPTION>

    <WRITTING INSTRUCTIONS>:
    Write the new summaries from the same perspective as the already present summaries.
    <WRITTING INSTRUCTIONS>

    <OUTPUT FORMAT INSTRUCTIONS>
    Return the completed summary_tree in json format.
    </OUTPUT FORMAT INSTRUCTIONS>

    <TREE STRUCTURE>
    {tree_structure}
    </TREE STRUCTURE>

    <SUMMARY TREE>
    {summary_tree}
    </SUMMARY TREE>
    Json Output:
"""


def build_prompt_template():
    return PromptTemplate(template=SYSTEM_PROMPT)