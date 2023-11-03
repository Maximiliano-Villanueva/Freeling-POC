"""
This module contains the Neo4jDB class for interacting with a Neo4j database.
It inherits from the abstract class GraphDB.
"""

# Standard imports
import logging

from abc import ABC, abstractmethod

# Third party imports
from py2neo import Graph, Node, Relationship


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
        
    def insert_relations(self, relations, source: str):
        """
        Insert relations into the Neo4j database.

        Parameters:
            relations: The relations to insert.
        """
        logging.info("Inserting relations into the Neo4j database.")
        logging.debug(f"Inserting relations: {relations}")
            
        graph = Graph(uri=self.uri, auth=(self.user, self.password))
        for verb, relation, word in relations:
            verb_node = Node("Verb", name=verb, source=source)
            word_node = Node("Word", name=word, source=source)
            graph.merge(verb_node, "Verb", "name")
            graph.merge(word_node, "Word", "name")
            rel = Relationship(verb_node, relation.upper(), word_node)
            rel["source"] = source
            graph.merge(rel)

        logging.info("End Inserting relations into the Neo4j database.")
