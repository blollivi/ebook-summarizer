from typing import Dict, Any

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser



def build_llm_chain(
    system_prompt: str, few_shot_examples: Dict, llm_config: Dict[str, Any]
):
    """
    Builds an LLM chain for a given set of examples and prompt.

    Parameters
    ----------
    system_prompt : str
        The system prompt to be used for the LLM chain.
    few_shot_examples : dict
        A dictionary containing few-shot examples.

    Returns
    -------
    object
        The constructed LLM chain.
    """
    llm = AzureChatOpenAI(
        model=llm_config["model"],
        temperature=llm_config["temperature"],
        max_tokens=llm_config["max_tokens"],
    ).bind(response_format=llm_config["response_format"])

    few_shot_examples_template = FewShotChatMessagePromptTemplate(
        input_variables=["input"],
        examples=few_shot_examples,
        example_prompt=ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{output}")]
        ),
    )
    final_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), few_shot_examples_template, ("human", "{input}")]
    )
    parser = JsonOutputParser()
    chain = final_prompt | llm | parser
    return chain