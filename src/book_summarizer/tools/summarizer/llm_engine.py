import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import RetryOutputParser

from langchain_google_genai._enums import (
    HarmBlockThreshold,
    HarmCategory,
)

safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}


class LLMEngine:
    """
    A class to manage interactions with a language model, including rate limiting and response parsing.

    This class provides an interface to generate responses using a language model while
    managing token and call limits, and parsing the output into a structured format.
    """

    def __init__(
        self,
        prompt_template: str,
        google_api_key: str,
        google_model_name: str,
        temperature: float,
        token_limit_per_minute: int,
        call_limit_per_minute: int,
    ):
        """
        Initialize the LLMEngine.

        Args:
            google_api_key (str): The API key for Google's generative AI service.
            token_limit_per_minute (float, optional): Maximum number of tokens allowed per minute. Defaults to 1e6.
            call_limit_per_minute (int, optional): Maximum number of API calls allowed per minute. Defaults to 15.
        """
        self.token_limit_per_minute = token_limit_per_minute
        self.call_limit_per_minute = call_limit_per_minute
        self.token_count = 0
        self.call_count = 0
        self.last_reset_time = time.time()
        self.prompt_template = prompt_template
        self.chat = ChatGoogleGenerativeAI(
            model=google_model_name,
            google_api_key=google_api_key,
            generation_config={"response_mime_type": "application/json"},
            temperature=temperature,
            safety_settings=safety_settings,
        )
        self.chain = self.prompt_template | self.chat
        self.parser = JsonOutputParser()
        self.retry_parser = RetryOutputParser.from_llm(
            parser=self.parser,
            llm=ChatGoogleGenerativeAI(
                model=google_model_name,
                google_api_key=google_api_key,
                generation_config={"response_mime_type": "application/json"},
                temperature=0,
                safety_settings=safety_settings,
            )
        )

    def _reset_counters(self):
        """Reset the token and call counters, and update the last reset time."""
        self.token_count = 0
        self.call_count = 0
        self.last_reset_time = time.time()

    def _check_limits(self):
        """
        Check if the token or call limits have been reached and wait if necessary.

        This method resets the counters if a minute has passed since the last reset.
        If either limit is reached, it waits for the appropriate amount of time before proceeding.
        """
        current_time = time.time()
        if current_time - self.last_reset_time >= 60:
            self._reset_counters()

        if self.token_count >= self.token_limit_per_minute:
            time_to_wait = 60 - (current_time - self.last_reset_time)
            if time_to_wait > 0:
                print(f"TKPM limit reached. Waiting {time_to_wait}s.")
                time.sleep(time_to_wait)
            self._reset_counters()

        if self.call_count >= self.call_limit_per_minute:
            time_to_wait = 60 - (current_time - self.last_reset_time)
            if time_to_wait > 0:
                print(f"RPM limit reached. Waiting {time_to_wait}s.")
                time.sleep(time_to_wait)
            self._reset_counters()

    def generate_response(self, chain_args: dict, parse_output: bool = True):
        """
        Generate a response using the language model and optionally parse the output.

        Args:
            chain_args (dict): Arguments to be passed to the language model chain.
            parse_output (bool, optional): Whether to parse the output as JSON. Defaults to True.

        Returns:
            If parse_output is True, returns the parsed JSON output.
            If parse_output is False, returns the raw content of the response.

        Raises:
            Exception: If parsing fails and cannot be recovered using the retry parser.
        """
        self._check_limits()
        print(chain_args.keys())
        response = self.chain.invoke(chain_args)
        used_tokens = (
            response.usage_metadata["input_tokens"]
            + response.usage_metadata["output_tokens"]
        )
        self.token_count += used_tokens
        self.call_count += 1
        print(f"Used {used_tokens} tokens. Total: {self.token_count}")

        if parse_output:
            try:
                return self.parser.parse(response.content)
            except Exception as e:
                print(e)
                prompt_value = self.prompt_template.format_prompt(**chain_args)
                return self.retry_parser.parse_with_prompt(
                    response.content, prompt_value
                )
        else:
            return response.content
