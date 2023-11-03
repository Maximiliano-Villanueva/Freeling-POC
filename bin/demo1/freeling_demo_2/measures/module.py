"""
This module provides utility to measure the quantity and quality of the extracted knowledge
"""
# Third party imports
from py2neo import Graph
from neo4j import GraphDatabase


class Measures:
    def __init__(self, uri, user, password):
        """
        Initialize the Measures class with a connection to the Neo4j database.
        """
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self.user = user
        self.password = password
        self.uri = uri

class AmountMeasures(Measures):
    """
    This class provides utility to measure the amount of the extracted knowledge
    """

    def __init__(self, uri : str, user : str, password : str):
         Measures.__init__(self, uri=uri, user=user, password=password)

    def measure_amount(self):
        """
        Measure the quantity of the extracted knowledge
        Returns: number of nodes and number of relations in the graphs
        """
        graph = Graph(self.uri, auth=(self.user, self.password))

        # Count all nodes
        count_all_nodes = graph.run("MATCH (n) RETURN COUNT(n)").data()[0]['COUNT(n)']

        # Count all relationships
        count_all_rels = graph.run("MATCH ()-[r]->() RETURN COUNT(r)").data()[0]['COUNT(r)']

        return count_all_nodes, count_all_rels


class StructuralMeasures(Measures):

    def __init__(self, uri : str, user : str, password : str):
         Measures.__init__(self, uri=uri, user=user, password=password)

    def close(self):
        """
        Close the database connection.
        """
        self._driver.close()

    def execute_query(self, query):
        """
        Execute a single query and return the result.
        """
        with self._driver.session() as session:
            result = session.run(query)
            return [record for record in result]
    
    def node_degree_distribution(self):
        """
        Check the distribution of node degrees to understand the connectivity of nodes.
        """
        query = "MATCH (n) RETURN id(n), size((n)--()) as degree"
        return self.execute_query(query)

    def clustering_coefficient(self):
        """
        Measures how nodes tend to cluster together.
        """
        query = """MATCH (n)--(m)
                   WITH n, collect(id(m)) AS nbrs
                   WITH n, nbrs, size(nbrs) as degree
                   MATCH (n)--(m)
                   WHERE id(m) IN nbrs
                   WITH n, degree, count(m) as interconnections
                   RETURN id(n), 2.0 * interconnections / (degree * (degree - 1)) AS clustering_coeff"""
        return self.execute_query(query)

    def diameter_and_average_path_length(self):
        """
        Diameter is the longest shortest path in the graph.
        Average path length is the average of all shortest paths.
        """
        query_diameter = "CALL gds.alpha.diameter.stream({nodeProjection: '*', relationshipProjection: '*'}) YIELD diameter"
        diameter = self.execute_query(query_diameter)
        # Note: Calculating average path length might require iterating through all pairs of nodes, which can be computationally expensive.
        return diameter, "Average path length computation can be expensive and is not implemented here."

    def connected_components(self):
        """
        Analyze the number and size of connected components in the graph.
        """
        query = "CALL gds.wcc.stream({nodeProjection: '*', relationshipProjection: '*'}) YIELD componentId, nodeId"
        return self.execute_query(query)


class SemanticEvaluation(Measures):
   
    def __init__(self, uri, user, password):
        """
        Initialize the Measures class with a connection to the Neo4j database.
        """
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self.user = user
        self.password = password
        self.uri = uri
        Measures.__init__(self, uri=uri, password=password, user=user)

    def entity_resolution(self):
        """
        Ensure that entities with the same meaning are represented as a single node, to minimize redundancy.
        Note: This is a placeholder. The actual implementation would require specific logic to identify duplicate entities.
        """
        query = "MATCH (n) RETURN n LIMIT 10"
        return self.execute_query(query)

    def type_consistency(self):
        """
        Ensure that relationships between nodes accurately reflect real-world interactions or hierarchical structures.
        Note: This is a placeholder. Actual checks would require domain-specific knowledge.
        """
        query = "MATCH ()-[r]->() RETURN type(r), count(*)"
        return self.execute_query(query)

    def attribute_accuracy(self):
        """
        Check the attributes associated with nodes and edges for accuracy and completeness.
        Note: This is a placeholder. Actual checks would require domain-specific knowledge.
        """
        query = "MATCH (n) RETURN keys(n) LIMIT 10"
        return self.execute_query(query)

# Initialization and usage example
if __name__ == "__main__":
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password"

    measures = Measures(uri, user, password)
    
    print("Node Degree Distribution:")
    print(measures.node_degree_distribution())
    
    print("Clustering Coefficient:")
    print(measures.clustering_coefficient())

    print("Diameter and Average Path Length:")
    print(measures.diameter_and_average_path_length())

    print("Connected Components:")
    print(measures.connected_components())

    measures.close()
