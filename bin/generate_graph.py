
""""
This script executes the pipeline for kwnoledge extraction from text files, from reading files to inserting knowledge in the database.
    - Read data from files.
    - Extract relations.
    - Complete and infeer new relations.
    - Insert relations in database.
"""
# Standard imports
import logging
import os
from typing import List, Optional, Set

# Internal imports
from common.entities import Triplet
from data_extractor.module import PdfDataExtractor
from data_extractor.runners import Runner, RunConfig
from proposals.runner import ProposalInferenceRunner
from text_processing.module import TestAnalyzer

from text_processing.runner import TextProcessingRunnerConfig, TextProcessingRunner
from neo4j_operations.runner import Neo4JOperationsRunnerConfig, Neo4JOperationsRunner
from triplet_inference.runner import TripletInferenceRunnerConfig, TripletInferenceRunner

# Initialize logging
logging.basicConfig(level=logging.INFO)


class MainRunnerConfig:
    def __init__(self):
        self.text_processing_config = TextProcessingRunnerConfig()
        self.text_processing_config.analyzer = TestAnalyzer()
        self.neo4j_operations_config = Neo4JOperationsRunnerConfig()
        self.triplet_inference_config = TripletInferenceRunnerConfig()


def get_unique_relation_types(complete_relations: List[Triplet],
                              incomplete_relations: Optional[List[Triplet]] = [] ) -> Set[str]:
    """
    Get all unique relation names from a list of complete and incomplete Triplet instances.
    
    Parameters:
        complete_relations (List[Triplet]): List of complete Triplet instances
        incomplete_relations (List[Triplet], optional): List of incomplete Triplet instances
    
    Returns:
        Set[str]: A set containing all unique relation names
    """
    # Initialize an empty set to store unique relation names
    unique_relation_names = set()

    # Combine complete and incomplete relations into one list for processing
    all_relations = complete_relations + incomplete_relations

    # Validate the input lists
    if not all_relations:
        logging.warning("Both complete and incomplete relation lists are empty.")
        return unique_relation_names

    # Iterate over each Triplet instance in the list
    for triplet in all_relations:
        try:
            # Get the relation type using the get_relation_type method
            relation_type = triplet.get_relation_type()

            # Add relation type to the set (ignores duplicate entries)
            if relation_type:
                unique_relation_names.add(relation_type)
                
        except Exception as e:
            logging.error(f"An error occurred while processing a triplet: {e}")

    if not unique_relation_names:
        logging.info("No unique relation names were found.")
    else:
        logging.info(f"Found unique relation names: {unique_relation_names}")

    return unique_relation_names


def text_pipeline(text: str, source: str = "", infeer: bool= True):
    """
    Execute the pipiline for a single string
    """
    config = MainRunnerConfig()
    text_processing_runner = TextProcessingRunner(config.text_processing_config)

    # Detect language
    sample_text = text.split("\n")[0]
    sample_text = sample_text if len(sample_text) > 1 else text
    language = text_processing_runner.detect_language(sample_text=sample_text)
    config.text_processing_config.lang = language

    # create proposals
    proposals_runner = ProposalInferenceRunner(language=language)
    proposals = proposals_runner.run(text=text)
    proposals = '\n'.join(proposals).lower()

    # Set the language for the text processor
    text_processing_runner.config = config.text_processing_config
    config.triplet_inference_config.language = language
    # Set the source of the information for neo4j
    config.neo4j_operations_config.source = source
    # Initialize the neo4j and triplet infeerer objects
    neo4j_operations_runner = Neo4JOperationsRunner(config.neo4j_operations_config)
    tirplet_inferer_runner = TripletInferenceRunner(config.triplet_inference_config)

    # Pipeline - Extract relations.
    relations_extracted = text_processing_runner.run_on_document(document=proposals)

    # Pipeline - Complete and infeer new relations.
    # Pipeline - Insert relations in database.
    for relation_i in relations_extracted:
        complete_relations = relation_i["complete_relations"]
        incomplete_relations = relation_i["incomplete_relations"]
        paragraph = relation_i["paragraph"]

        # Get all unique relation names
        all_relations = complete_relations + incomplete_relations
        unique_relation_names = get_unique_relation_types(all_relations)

        # Run the inference process for each unique relation type
        for relation_name in unique_relation_names:
            filtered_complete_relations = [r for r in complete_relations if r.get_relation_type() == relation_name]
            filtered_incomplete_relations = [r for r in incomplete_relations if r.get_relation_type() == relation_name]
            
            logging.info(f"Running inference for relation type: {relation_name}")

            # Run the inference
            if infeer:
                relations = tirplet_inferer_runner.run(relations=filtered_complete_relations,
                                                       incomplete_relations=filtered_incomplete_relations,
                                                       text=paragraph)
            
            # Run neo4j operations
            neo4j_operations_runner.run(relations)

def load_data():
    """
    Load data from the specified folder.
    Default folder : data folder.
    """
    config = RunConfig()
    config.data_directory = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset')
    runner = Runner(config=config, extractor=PdfDataExtractor())
    data = runner.run()

    return data

def reset_db():
    config = MainRunnerConfig()
    neo4j_operations_runner = Neo4JOperationsRunner(config.neo4j_operations_config)
    neo4j_operations_runner.clear_db()


if __name__ == "__main__":
    reset_db()

    data = load_data()
    for key in list(data.keys()):
        # text_pipeline(text='\n\n'.join(data.get(key)[0:2]), source=key)
        text_pipeline(text='\n\n'.join(data.get(key)[0:2]), source=key)
    
