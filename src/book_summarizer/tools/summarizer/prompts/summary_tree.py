from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


SYSTEM_PROMPT = """<TASK DESCRIPTION>
    You are an talentful text analyzer and summarizer. You are been given:
    - an input text, formated with JSON as a set of subsequent chunks
            <chunk_id>: <chunk_text>
    - a tree structure representing the hierarchical splitting of the text into section and subsections.
        Each node in the tree represents a section with a start and end chunk id (included), and a list of children nodes if any.
            "start": <start_chunk_id>,
            "end": <end_chunk_id>,
            "children": <List of sub section nodes>

    Your task is to write a summary of each section node of the tree.

    SECTION NODE SUMMARIZATION STEPS:
    Follow the given steps to summarize a section node:
    1. Identify the chunks that belong to the section node given its start and end chunk id.
    2. Read the chunks and write a 2-3 sentences long summary. It must accurately synthetise the content of the section node.
    3. Write a title for the section node

    </TASK DESCRIPTION>

    <WRITTING INSTRUCTIONS>:
    Write the summaries from the same perspective the author.
    Do not use a third person perspective such as "the author says ..." or "the text states ...", "this section is about ...".
    Do not add any personal opinion or interpretation.
    Child and parent summaries must be the less redundant as possible.
    Write in {language}.
    </WRITTING INSTRUCTIONS>


    <OUTPUT FORMAT INSTRUCTIONS>
    The ouput, called summary_tree, must have the same structure as the input tree,
    but with the two new keys "summary" and "title" added to all nodes.

    Each node must have the following structure:
            "start": <start_chunk_id>,
            "end": <end_chunk_id>,
            "summary": "<section_summary>",
            "title": "<section_title>",
            "children": <List of sub section nodes>

    Very important: Use only JSON compatible characters.

    Reminder: The tree must have the exact same structure as the input tree.

    </OUTPUT FORMAT INSTRUCTIONS>

    <EXAMPLE>
    Use the following example as guidance for your task.

    Input:
        <EXAMPLE CHUNKS>
        {example_chunks}
        </EXAMPLE CHUNKS>

        <EXAMPLE TREE>
        {example_tree}
        </EXAMPLE TREE>

    Json Output:
    {example_json_output}
    </EXAMPLE>

    <CHUNKS>
    {chunks}
    </CHUNKS>

    <SUMMARY TREE>
    {summary_tree}
    </SUMMARY TREE>

    Json Output:
"""


def build_prompt_template() -> PromptTemplate:

    with open("data/examples/example_essay.json", "r") as file:
        example_essay = file.read()

    with open("data/examples/example_essay_tree.json", "r") as file:
        example_essay_tree = file.read()

    with open("data/examples/example_essay_tree_output.json", "r") as file:
        example_essay_tree_output = file.read()

    return PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["chunks", "tree"],
        partial_variables={
            "example_chunks": example_essay,
            "example_tree": example_essay_tree,
            "example_json_output": example_essay_tree_output,
        },
    )
