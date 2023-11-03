"""
This module contains the Neo4jDB class for interacting with a Neo4j database.
It inherits from the abstract class GraphDB.
"""

# Standard imports
import logging

from abc import ABC, abstractmethod
from typing import List

# Third party imports
from py2neo import Graph, Node, Relationship

# Internal imports
from common.entities import Triplet
from text_processing.module import FreelingAnalyzer


class GraphDB(ABC):
    """
    Abstract class for GraphDB interactions.
    """

    @abstractmethod
    def clear_db(self):
        """
        Abstract method to clear the database.
        """
        pass

    @abstractmethod
    def insert_relations(self, relations, source: str):
        """
        Abstract method to insert relations into the database.

        Parameters:
            relations: The relations to insert.
        """
        pass


class Neo4jDB(GraphDB):
    """
    Neo4jDB class implements the GraphDB abstract class for Neo4j database interactions.
    """

    def __init__(self, uri, user, password):
        """
        Initialize the Neo4jDB with URI, user, and password.

        Parameters:
            uri (str): The URI for the Neo4j database.
            user (str): The user name for the Neo4j database.
            password (str): The password for the Neo4j database.
        """
        self.uri = uri
        self.user = user
        self.password = password
        logging.info("Neo4jDB initialized.")

    def clear_db(self):
        """
        Clear the Neo4j database.
        """
        logging.info("Clearing Neo4j database.")
        
        graph = Graph(self.uri, auth=(self.user, self.password))
        delete_query = "MATCH (n) DETACH DELETE n"
        graph.run(delete_query)

        logging.info("End clearing Neo4j database.")
        
    def insert_relations(self, relations: List[Triplet], source: str):
        """
        Insert relations into the Neo4j database.

        Parameters:
            relations: The relations to insert.
        """
        relations = FreelingAnalyzer.remove_duplicates(relations=relations)
        logging.info("Inserting relations into the Neo4j database.")
        logging.debug(f"Inserting relations: {relations}")
            
        graph = Graph(uri=self.uri, auth=(self.user, self.password))
        for triplet in relations:
            logging.info(f"Inserting relation of type {str(triplet.get_relation_type())}: {str(triplet)}")
            verb = str(triplet.node1) if triplet.node1 and triplet.node1.name is not None else 'None'
            relation = str(triplet.relation)
            word = str(triplet.node2) if triplet.node2 and triplet.node2.name is not None else 'None'
            
            # Ignore triplets without relation
            if relation is None:
                continue
            verb_node = Node("Verb", name=verb, source=source)
            word_node = Node("Word", name=word, source=source)
            graph.merge(verb_node, "Verb", "name")
            graph.merge(word_node, "Word", "name")
            rel = Relationship(verb_node, relation.upper(), word_node)
            rel["source"] = source
            graph.merge(rel)

        logging.info("End Inserting relations into the Neo4j database.")
