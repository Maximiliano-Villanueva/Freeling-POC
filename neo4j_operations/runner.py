# Standard imports
from dataclasses import dataclass
from typing import List

# Internal imports
from common.entities import Triplet
from .module import Neo4jDB


@dataclass
class Neo4JOperationsRunnerConfig:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "recipes"
    source: str = "my.pdf"
    

class Neo4JOperationsRunner:
    def __init__(self, config=Neo4JOperationsRunnerConfig()):
        self.config = config

        self.db = Neo4jDB(
            uri=self.config.uri,
            user=self.config.user,
            password=self.config.password
        )
    
    def clear_db(self):
        self.db.clear_db()

    def run(self, relations: List[Triplet]):
        self.db.insert_relations(
            relations=relations,
            source=self.config.source,
        )
