import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import RetryOutputParser

from .prompts import build_prompt_template


class LLMEngine:
    def __init__(self, token_limit_per_minute=1e6, call_limit_per_minute=15):
        self.token_limit_per_minute = token_limit_per_minute
        self.call_limit_per_minute = call_limit_per_minute
        self.token_count = 0
        self.call_count = 0
        self.last_reset_time = time.time()

        GOOGLE_API_KEY = "AIzaSyBBUP5cLkckeHhariMLznIwnUYMv1jc0vM"

        self.prompt = build_prompt_template()

        self.chat = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            generation_config={"response_mime_type": "application/json"},
            temperature=0.3,
        )

        self.chain = self.prompt | self.chat

        self.parser = JsonOutputParser()

        self.retry_parser = RetryOutputParser.from_llm(
            parser=self.parser,
            llm=ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=GOOGLE_API_KEY,
                generation_config={"response_mime_type": "application/json"},
                temperature=0.1,
            ),
        )

    def _reset_counters(self):
        self.token_count = 0
        self.call_count = 0
        self.last_reset_time = time.time()

    def _check_limits(self):
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

    def generate_response(self, llm_args: dict):
        self._check_limits()
        response = self.chain.invoke(**llm_args)
        used_tokens = (
            response.usage_metadata["input_tokens"]
            + response.usage_metadata["output_tokens"]
        )
        self.token_count += used_tokens
        self.call_count += 1

        try:
            return self.parser.parse(response.content)
        except Exception as e:
            return self.retry_parser.parse_with_prompt(response.content, self.prompt)
