# Stnadard imports
import logging

from typing import Dict, Union, Optional
from abc import ABC, abstractmethod


class GraphItemWithProperties(ABC):
    """
    This abstract class represents the common features of a graph item with properties.
    """
    def __init__(self, name: Union[str, None], properties: Optional[Dict[str, str]] = None):
        self._properties: Dict[str, str] = properties if properties is not None else []
        self.name = name

    @property
    def properties(self) -> Dict[str, str]:
        """Get the properties of the GraphItem"""
        return self._properties
    
    def update_properties(self, new_properties: Dict[str, str]):
        self._properties.update(new_properties)

    @properties.setter
    def properties(self, value: Union[Dict[str, str], str]):
        """Set the properties of the GraphItem"""
        if not all(isinstance(item, str) for item in value):
            logging.error("All elements in properties must be of type str.")
        else:
            self._properties = value

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash((self.name, frozenset(self.properties.items()) if self.properties else None))

    def __eq__(self, other):
        if not isinstance(other, GraphItemWithProperties):
            return False
        equality_check = self.name == other.name and self.properties == other.properties
        return equality_check

    def __str__(self) -> str:
        return self.name


class GraphNode(GraphItemWithProperties):
    def __init__(self, name: Union[str, None], properties: Optional[Dict[str, str]] = None):
        super().__init__(name=name, properties=properties)


class GraphRelation(GraphItemWithProperties):
    def __init__(self, name: Union[str, None], properties: Optional[Dict[str, str]] = None):
        super().__init__(name=name, properties=properties)


class Triplet:
    """
    Provide functionality for triplets.
    """
    def __init__(self, node1: Union[GraphNode, None],
                 relation: Union[GraphRelation, None],
                 node2: Union[GraphNode, None]) -> None:
        self._node1 = node1
        self._relation = relation
        self._node2 = node2

    @property
    def node1(self) -> Union[GraphNode, None]:
        return self._node1

    @node1.setter
    def node1(self, value: Union[GraphNode, None]) -> None:
        if value is not None and not isinstance(value, GraphNode):
            logging.error("node1 must be an instance of GraphNode or None.")
        else:
            self._node1 = value

    @property
    def relation(self) -> Union[GraphRelation, None]:
        return self._relation

    @relation.setter
    def relation(self, value: Union[GraphRelation, None]) -> None:
        if value is not None and not isinstance(value, GraphRelation):
            logging.error("relation must be an instance of GraphRelation or None.")
        else:
            self._relation = value

    @property
    def node2(self) -> Union[GraphNode, None]:
        return self._node2

    @node2.setter
    def node2(self, value: Union[GraphNode, None]) -> None:
        if value is not None and not isinstance(value, GraphNode):
            logging.error("node2 must be an instance of GraphNode or None.")
        else:
            self._node2 = value

    def get_relation_type(self):
        """
        Return the type of relation
        """
        return self._relation.properties["triplet_type"]

    def __eq__(self, other) -> bool:
        if not isinstance(other, Triplet):
            return False

        return self.node1 == other.node1 and \
            self.relation == other.relation and \
            self.node2 == other.node2

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash((self.node1, self.relation, self.node2))

    def __str__(self) -> str:

        node1_str = str(self.node1) if self.node1 and self.node1.name else 'None'
        relation_str = str(self.relation) if self.relation and self.relation.name else 'None'
        node2_str = str(self.node2) if self.node2 and self.node2.name else 'None'
        
        return f"('{node1_str}', '{relation_str}', '{node2_str}')"

