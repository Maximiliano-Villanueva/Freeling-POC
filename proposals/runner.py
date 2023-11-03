# Standard imports
import json
import logging
from math import ceil
from typing import Any, List, Dict
import tiktoken

# Third-party imports
import openai

# Constants
MAX_TOKENS = 4096  # GPT-3.5 Turbo token limit


class ProposalInferenceRunner:
    """Class to manage the inference for proposals using the OpenAI API."""

    def __init__(self, model_engine: str = "gpt-3.5-turbo", language: str = "es"):
        """Initialize the ProposalInferenceRunner."""
        self._model_engine = model_engine
        self._language = language

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        enc = tiktoken.encoding_for_model(self._model_engine)
        encoded = enc.encode(text)
        return len(encoded)
    
    def _split_in_paragraphs(self, text: str) -> List[str]:
        """Splits the text into smaller chunks."""
        paragraphs = text.split('.\n')
        return paragraphs        

    def _make_request(self, messages: List[Dict[str, str]]) -> Dict[Any, Any]:
        """Internal method to make API requests."""
        try:
            response = openai.ChatCompletion.create(
                model=self._model_engine,
                messages=messages
            )
            logging.debug("API request successful.")
            return response
        except Exception as e:
            logging.error(f"API request failed: {e}")
            return None

    def run(self, text: str) -> List[str]:
        """Run the inference process on the given text."""
        proposals = []
        text_chunks = self._split_in_paragraphs(text)

        for idx, chunk in enumerate(text_chunks):
            prompt = f"""
                Given the text in {self._language}: "{chunk}", please identify proposals based on the information in the text.
                Keep in mind the complexity of the language, the complexity of the text, and any other relevant context.
                The proposals should be simple enough to be parsed into triplets later.
                Respond with an array in JSON format containing all the proposals, each proposals should end with a dot.
            """

            messages = [
                {"role": "user", "content": prompt}
            ]

            response = self._make_request(messages)
            if response:
                logging.info(f"Received response from API for request nº {idx}.")
                try:
                    response_content = response['choices'][0]['message']['content']
                    proposal_chunk = json.loads(response_content)
                    for propo in proposal_chunk:
                        if isinstance(propo, dict):
                            for k,v in propo.items():
                                proposals.append(str(v))
                        else:
                            proposals.append(propo)
                except Exception as e:
                    logging.error(f"Failed to process the API response: {e}")

        logging.info(f"Total proposals retrieved: {len(proposals)}")
        return proposals


if __name__ == "__main__":
    runner = ProposalInferenceRunner()
    large_text = "Your very large text goes here..."
    proposals = runner.run(large_text)
    print(proposals)
