from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


SYSTEM_PROMPT = """You are an talentful writing assistant. Your task is to help me summarize an entire essay, chunk by chunk,
with strict respect of the following instructions. Your summarization skills include:
- a strong abstraction capacity: you can grasp the high level ideas and concepts and rephrase them in an elegant style
- Analytical skills: the summary reflect your overall understanding of the book, how it is structured and the relationships
  between the different parts

<SUMMARIZATION-INSTRUCTIONS>
You will receive the next chunk of text from the book to be summarized, and the summary of the book until now. Write the summary of the
new chunk as if it was the natural continuation of the previous summary. Be concise, with a focus on the high levels concepts and the
main arguments.

<MANDATORY FORMAT INSTRUCTIONS>:
A reader must not be able to distinguish between the original text and the summary: the summary must be written in the same style and
tone as the original text.
Write in {language}.
Format the output as a JSON object with the following keys:
- summary: the summary of the new chunk
- title: a title that captures the essence of the new chunk
"""


INPUT_PROMPT = """New chunk:
<chunk>
{chunk}
</chunk>

Previous summary:
<summary>
{context}
</summary>"""


def build_prompt_template(with_context: bool = True) -> PromptTemplate:

    if with_context:
        return ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", INPUT_PROMPT),
            ]
        )
    else:
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Summarize the given text with strict respect of the given instructions.
            Return a JSON object with the following keys:
            - summary: summary that synthesizes the concepts and ideas, three hundred words maximum
            - abstract: a very concise version of the text
            - title: a title that captures the essence of the text

            Write from the same perspective as the original author.
            Write in {language}.
            """,
                ),
                ("human", "<text>{chunk}</text>"),
            ]
        )
