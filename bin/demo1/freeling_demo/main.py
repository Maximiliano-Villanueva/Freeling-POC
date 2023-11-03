from text_processing.runner import TextProcessingRunnerConfig, TextProcessingRunner
from neo4j_operations.runner import Neo4JOperationsRunnerConfig, Neo4JOperationsRunner
from triplet_inference.runner import TripletInferenceRunnerConfig, TripletInferenceRunner

class MainRunnerConfig:
    def __init__(self):
        self.text_processing_config = TextProcessingRunnerConfig()
        self.neo4j_operations_config = Neo4JOperationsRunnerConfig()
        self.triplet_inference_config = TripletInferenceRunnerConfig()

def main(text: str):
    config = MainRunnerConfig()
    text_processing_runner = TextProcessingRunner(config.text_processing_config)
    neo4j_operations_runner = Neo4JOperationsRunner(config.neo4j_operations_config)
    tirplet_inferer_runner = TripletInferenceRunner(config.triplet_inference_config)
    complete_relations, incomplete_relations = text_processing_runner.run(text=text)
    relations = tirplet_inferer_runner.run(relations=complete_relations, incomplete_relations=incomplete_relations, text=text)
    # neo4j_operations_runner.run(relations)


if __name__ == "__main__":
    main(text ="Sobre la mesa, ves una manzana, un sombrero, una llave y un paraguas. Coges la manzana y el paraguas.")
