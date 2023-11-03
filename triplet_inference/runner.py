"""
Triplet Inference Runner for OpenAI API

This script provides a mechanism to infer relations in a text and to generate
new, logical relations based on existing ones. The script uses OpenAI's GPT-3 API 
to perform the inference. It is designed with retry logic to handle API request 
failures, providing up to 3 retries with a 10-second wait time between each retry.

example usage:
    runner = TripletInferenceRunner()
    runner.run([], [], "example text")

"""


# Standard imports
import json
import logging
from math import ceil

from dataclasses import dataclass
from typing import Any, List, Dict
from utils.decorators import retry

# Third party imports
import openai

# Internal imports
from common.entities import GraphNode, GraphRelation, Triplet
from text_processing.module import RelationType

LANGUAGE_ACRONYMOUS_MAP = {
    "en": {
        "acronym": "en",
        "full_name": "English",
        "locale": "es-US"
    },
    "es": {
        "acronym": "es",
        "full_name": "Spanish",
        "locale": "es-ES"
    },

}

MAX_TOKENS = 4096  # GPT-3.5 Turbo token limit

@dataclass
class TripletInferenceRunnerConfig:
    """Configuration for the TripletInferenceRunner class."""
    model_engine: str = "gpt-3.5-turbo"
    language: str = "es"


class TripletInferenceRunner:
    """Class to manage the inference for triplets using the OpenAI API."""
    def __init__(self, config=TripletInferenceRunnerConfig()):
        """Initialize the TripletInferenceRunner."""
        self._config = config

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = config

    def _truncate_list_for_token_limit(self, relations_list: List[Triplet], prompt: str) -> List[Triplet]:
        """Truncates the list of relations to ensure the prompt doesn't exceed the token limit."""
        while len(prompt) > MAX_TOKENS:
            logging.warning("Prompt exceeds maximum token limit. Truncating relations list.")
            relations_list.pop()  # Remove one relation from the list
            prompt = prompt.format(relations=str(relations_list))  # Update the prompt
        return relations_list
    
    @retry(retries=3, wait_time=10)
    def _make_request(self, messages: List[Dict[str, str]], model_engine: str) -> Dict[Any, Any]:
        """Internal method to make API requests."""
        response = openai.ChatCompletion.create(
            model=model_engine,
            messages=messages
        )
        logging.debug("API request successful.")
        return response

    @retry(retries=3, wait_time=2)
    def run(self, relations: List[Triplet],
            incomplete_relations: List[Triplet],
            text: str) -> List[Triplet]:
        """Run the inference process on the given text and relations."""
        model_engine = self._config.model_engine
        language = LANGUAGE_ACRONYMOUS_MAP.get(self._config.language)

        base_tokens = 3000 + len(text)  # Tokens for static part of prompt and other constant parameters
        average_triplet_tokens = 20  # Generous estimate of tokens for a single triplet

        # Calculate available tokens and CHUNK_SIZE
        available_tokens = MAX_TOKENS - base_tokens
        CHUNK_SIZE = available_tokens // average_triplet_tokens

        # Attempt to retrieve the triplet type otherwise set a default value
        triplet_type = relations[0].relation.properties.get("triplet_type", "SVO") if len(relations) > 0 else "SVO"

        # Split the relations and incomplete_relations into smaller chunks
        num_chunks = ceil(len(relations) / CHUNK_SIZE)
        
        num_chunks_incomplete = ceil(len(incomplete_relations) / CHUNK_SIZE)
        incomplete_chunks = [incomplete_relations[i * CHUNK_SIZE: (i + 1) * CHUNK_SIZE] for i in range(num_chunks_incomplete)]

        relation_type_description = RelationType[triplet_type.upper()].value

        for incomplete_chunk in incomplete_chunks:
            prompt = f"""
                Given the {language.get("full_name")}({language.get("locale")}) text "{text}", and the triplets of type {relation_type_description}: {str(incomplete_chunk)} are incomplete. Please identify the missing element to complete the triplet based on the information in the text.
                Keep in mind the complexity of the language, the complexity of the text, the person involved, and any other relevant context.
                Respond only an array in json in {language.get("full_name")} format with all the triplets fixed. Each entry of the array must contain the entries "entity1", "relation", "entity2".

            """

            messages = [
                {"role": "user", "content": prompt},
            ]


            response = self._make_request(messages, model_engine)
            if response:
                logging.info(f"Fixed relations: {response}")

                response = response['choices'][0]['message']['content']
                response = json.loads(response)

                logging.info(f"Fixed relations: {response}")
                for entry in response:
                    logging.info(f"{entry}")
                    new_triplet = Triplet(
                        node1=GraphNode(
                            name=entry["entity1"],
                            properties={"language": language.get("acronym")}
                        ),
                        relation=GraphRelation(
                            name=entry["relation"],
                            properties={"language": language.get("acronym"), "triplet_type": triplet_type}
                        ),
                        node2=GraphNode(
                            name=entry["entity2"],
                            properties={"language": language.get("acronym")}
                        )
                    )
                    relations.append(new_triplet)
                else:
                    # Force try
                    raise Exception("Force retry")
            
        relations_chunks = [relations[i * CHUNK_SIZE: (i + 1) * CHUNK_SIZE] for i in range(num_chunks)]
        
        for relations_chunk in relations_chunks:
            prompt = f"""
                Given the {language.get("full_name")}({language.get("locale")}) text "{text}", and the triplets {str(relations_chunk)}. Please generate new triplets that could logically exist within this context.
                Respond only an array in json format in {language.get("full_name")} with all the new triplets.
            """

            prompt = f"""
                a partir del texto {text}.
                He conseguido las tripletas de tipo {relation_type_description} {str(relations)}.
                Infiere nuevas tripletas que puedan existir lógicamente dentro del contexto y las tripletas proporcionadas.
                De las tripletas proporcionadas elimina o arregla aquellas que no tengan sentido.
                El resultado debe ser un array en formato json en idioma {language.get("full_name")}, cada entrada del array debe contener las emtradas "entity1", "relation", "entity2".
            """

            messages = [
                {"role": "user", "content": prompt},
            ]

            response = self._make_request(messages, model_engine)
            if response:
                logging.info(f"Inferred relations: {response}")
                response = response['choices'][0]['message']['content']
                response = json.loads(response)
                print(response)

                logging.info(f"Infeered relations: {response}")
                for entry in response:
                    logging.info(entry)
                    new_triplet = Triplet(
                        node1=GraphNode(
                            name=entry["entity1"],
                            properties={"language": language.get("acronym")}
                        ),
                        relation=GraphRelation(
                            name=entry["relation"],
                            properties={"language": language.get("acronym"), "triplet_type": triplet_type}
                        ),
                        node2=GraphNode(
                            name=entry["entity2"],
                            properties={"language": language.get("acronym")}
                        )
                    )
                    relations.append(new_triplet)
            else:
                # Force try
                raise Exception("Force retry")
        
        # Split the relations and incomplete_relations into smaller chunks
        # num_chunks = ceil(len(relations) / CHUNK_SIZE)
        # all_relations = []
        # relations_chunks = [relations[i * CHUNK_SIZE: (i + 1) * CHUNK_SIZE] for i in range(num_chunks)]
        # for relations_chunk in relations_chunks:
        #     prompt = f"""
        #         Given the {language.get("full_name")}({language.get("locale")}) text "{text}", and the triplets {str(relations_chunk)}. Please remove or replace those triplets that are incorrect that could logically exist within this context.
        #         Keep in mind the complexity of the language, the complexity of the text, the person involved, and any other relevant context.
        #         Respond only an array in json in {language.get("full_name")} format with all the triplets fixed. Each entry of the array must contain the entries "entity1", "relation", "entity2".
        #     """
        #     prompt = f"""
        #         Given the {language.get("full_name")}({language.get("locale")}) text "{text}", and the triplets {str(relations_chunk)},
        #         please remove or replace those triplets that are incorrect that could not logically exist within this context.
        #         Keep in mind the complexity of the language, the complexity of the text, the person involved, and any other relevant context.
        #         Return only an array in JSON format in {language.get("full_name")} with all the valid or fixed triplets.
        #         Each entry of the array must contain the entries "entity1", "relation", "entity2". Do not include eliminated triplets.
        #     """

        #     messages = [
        #         {"role": "user", "content": prompt},
        #     ]

        #     response = self._make_request(messages, model_engine)
        #     if response:
        #         logging.info(f"All relations relations: {response}")
        #         response = response['choices'][0]['message']['content']
        #         response = json.loads(response)
        #         print(response)

        #         logging.info(f"All relations relations: {response}")
        #         for entry in response:
        #             logging.info(entry)
        #             new_triplet = Triplet(
        #                 node1=GraphNode(
        #                     name=entry["entity1"],
        #                     properties={"language": language.get("acronym")}
        #                 ),
        #                 relation=GraphRelation(
        #                     name=entry["relation"],
        #                     properties={"language": language.get("acronym"), "triplet_type": triplet_type}
        #                 ),
        #                 node2=GraphNode(
        #                     name=entry["entity2"],
        #                     properties={"language": language.get("acronym")}
        #                 )
        #             )
        #             all_relations.append(new_triplet)

        # return all_relations
        return relations