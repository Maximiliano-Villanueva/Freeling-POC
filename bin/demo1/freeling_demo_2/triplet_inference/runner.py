# Standard immports
import json
import logging

from dataclasses import dataclass

# Third party imports
import openai

@dataclass
class TripletInferenceRunnerConfig:
    model_engine: str = "gpt-3.5-turbo"


class TripletInferenceRunner:
    def __init__(self, config=TripletInferenceRunnerConfig()):
        self.config = config

    def run(self, relations, incomplete_relations, text: str):

        model_engine = self.config.model_engine

        prompt = f"""
            Given the spanish(ES-es) text "{text}", and the triplets {str(incomplete_relations)} are incomplete. Please identify the missing element to complete the triplet based on the information in the text.
            Keep in mind the complexity of the language, the complexity of the text, the person involved, and any other relevant context.
            Respond only an array in json in spanish format with all the triplets fixed. Each entry of the array must contain the entries "entity1", "relation", "entity2".

        """

        messages = [
            {"role": "user", "content": prompt},
        ]

        response = openai.ChatCompletion.create(
            model=model_engine,
            messages=messages
        )

        response = response['choices'][0]['message']['content']
        response = json.loads(response)

        logging.info(f"Fixed relations: {response}")
        for entry in response:
            logging.info(f"{entry}")
            relations.append((
                entry["entity1"],
                entry["relation"],
                entry["entity2"])
            )


        prompt = f"""
            Given the spanish(ES-es) text "{text}", and the triplets {str(relations)}. please generate new triplets that could logically exist within this context. Consider the text's complexity, the person involved, and other relevant contextual factors.
            Respond only an array in json format in spanish with all the new triplets. Each entry of the array must contain the entries "entity1", "relation", "entity2". Do not respond anything else than the json.
        """

        prompt = f"""
            a partir del texto {text}.
            He conseguido las tripletas {str(relations)}.
            Infiere nuevas tripletas que puedan existir lógicamente dentro del contexto y las tripletas proporcionadas.
            El resultado debe ser un array en formato json, cada entrada del array debe contener las emtradas "entity1", "relation", "entity2".
        """

        messages = [
            {"role": "user", "content": prompt},
        ]

        response = openai.ChatCompletion.create(
            model=model_engine,
            messages=messages
        )

        response = response['choices'][0]['message']['content']
        response = json.loads(response)
        print(response)

        logging.info(f"Infeered relations: {response}")
        for entry in response:
            logging.info(entry)
            relations.append((
                entry["entity1"],
                entry["relation"],
                entry["entity2"])
            )
        
        return relations