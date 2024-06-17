from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


SYSTEM_PROMPT = """You are an talentful writing assistant that follows instructions carefully."""


BASE_SUMMARY_PROMPT = """Given the input text, write three shorter new verions of it:
- long: a well written, coherent and detailled synthesis of the main concepts.
- short: a concise version, mentioning only the main idea.
- title: a title that captures the essence of the text."""

SUMMARY_PROMT = """You are summarization engine, that help me summarize an entire book by applying a recursive procedure to
its subsequent chunks.
The summary of the ith chunk is written given the following informations:
- the text of the ith chunk
- the outline of the book until the i-1th chunk, serving as context to make the summary coherent with the already produced summaries.

You must write three versions of the summary:
- long: a well written, coherent and detailled synthesis of the main concepts.
- short: a concise version, mentioning only the main idea.
- title: a title that captures the essence of the text.

The long summary is the final chunk summary, while the short summary and the title are used to build the outine used as
context for the next chunk summary.

If no outline is provided, this means that the current chunk is the first one of the book, and the context is empty.
"""


FORMAT_PROMPT = """MANDATORY WRITTING INSTRUCTIONS:
Write from the same perspective as the original author: Do no use forumulations such as
"the author says that" or "the text explains that".
Write in {language}.
Return a JSON object with the following keys:
- long: str
- short: str
- title: str"""

INPUT_PROMPT = """Here is the chunk to summarize:
{chunk}
Here is the outline of the book until now:
{outline}
"""


CONTEXT_PROMPT = """In order to make the synthesis coherent with the previous text, use the given context."""

CONTEXT_INPUT_PROMPT = """Here is the context:"""


def build_prompt_template() -> PromptTemplate:

    # system_prompt_template = PromptTemplate.from_template(SYSTEM_PROMPT)

    summary_prompt_template = PromptTemplate.from_template(SUMMARY_PROMT)
    format_prompt_template = PromptTemplate.from_template(FORMAT_PROMPT)
    input_prompt_template = PromptTemplate.from_template(INPUT_PROMPT)


    return summary_prompt_template + format_prompt_template + input_prompt_template
