from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


SYSTEM_PROMPT = """You are an talentful writing assistant. Your task is help me summarize a book, chunk by chunk,
with strict respect of the given instructions."""


SUMMARY_PROMT = """<summarization-instructions>
Summarize the given chunk oh the book, given:
- the text of the current chunk
- the summary the book until now that you should respect to ensure coherent and smooth transitions between the chunks.

Write the summary in three versions:
- long: A detailled summary, focusing on the high level ideas and concepts.
- short: a concise version, mentioning only the main idea.
- title: a title that captures the essence of the text.

If no context is provided, this means that the current chunk is the first one of the book, and the context is empty.
</summarization-instructions>
"""

FORMAT_PROMPT = """<format-instructions>
MANDATORY WRITTING INSTRUCTIONS:
Write from the same perspective as the original author: Do not use forumulations such as
"the author says that", "the text explains that", "this chapter is about", etc.
Write in {language}.
Return a JSON object with the following keys:
- long: str
- short: str
- title: str
</format-instructions>"""

INPUT_PROMPT = """Summarize this chunk:
<chunk>
{chunk}
</chunk>

With the following context:
<context>
{context}
</context>"""


def build_prompt_template() -> PromptTemplate:

    system_prompt_template = PromptTemplate.from_template(SYSTEM_PROMPT)
    summary_prompt_template = PromptTemplate.from_template(SUMMARY_PROMT)
    format_prompt_template = PromptTemplate.from_template(FORMAT_PROMPT)
    input_prompt_template = PromptTemplate.from_template(INPUT_PROMPT)


    return system_prompt_template + summary_prompt_template +  input_prompt_template + format_prompt_template
