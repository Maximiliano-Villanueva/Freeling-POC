"""
Module for text analysis using an abstract TextAnalyzer class and
a concrete implementation using pyfreeling.

This module provides functionalities to:
- Configure morphological analyzer options
- Extract object-action-location triplets
- Get all relations from a list of sentences
"""

from abc import ABC, abstractmethod
import logging
import pyfreeling


class TextAnalyzer(ABC):
    """Abstract base class for text analysis."""

    @abstractmethod
    def extract_object_action_location_triplets(self, node, current_state=None, triplets=None):
        """
        Abstract method to recursively extract object-action-location triplets from text.

        Args:
            node: Node object to start extraction.
            current_state (dict, optional): State dictionary to hold verb, location, preposition.
            triplets (list, optional): List to store the extracted triplets.

        Returns:
            list: List of triplets.
        """
        pass

    @abstractmethod
    def get_all_relations(self, ls):
        """
        Abstract method to extract relations from a list of text elements.

        Args:
            ls: List of text elements to analyze.

        Returns:
            list: List of all relations.
        """
        pass


class FreelingAnalyzer(TextAnalyzer):
    """Freeling Analyzer class for text analysis, implementing TextAnalyzer."""

    @staticmethod
    def my_maco_options(lang: str, data: str):
        """
        Configure morphological analyzer options for pyfreeling.

        Args:
            lang (str): Language configuration.
            data (str): Path to data folder.

        Returns:
            pyfreeling.analyzer_config: Analyzer configuration options.
        """
        logging.info("Configuring morphological analyzer options.")

        opt = pyfreeling.analyzer_config()

        # define creation options for morphological analyzer modules
        opt.config_opt.Lang = lang
        opt.config_opt.MACO_PunctuationFile = data + "/common/punct.dat"
        opt.config_opt.MACO_DictionaryFile = f"{data}/{lang}" + "/dicc.src"
        opt.config_opt.MACO_AffixFile = f"{data}/{lang}" + "/afixos.dat"
        opt.config_opt.MACO_CompoundFile = f"{data}/{lang}" + "/compounds.dat"
        opt.config_opt.MACO_LocutionsFile = f"{data}/{lang}" + "/locucions.dat"
        opt.config_opt.MACO_NPDataFile = f"{data}/{lang}" + "/np.dat"
        opt.config_opt.MACO_QuantitiesFile = f"{data}/{lang}" + "/quantities.dat"
        opt.config_opt.MACO_ProbabilityFile = f"{data}/{lang}" + "/probabilitats.dat"
        opt.config_opt.TAGGER_HMMFile = f"{data}/{lang}" + "/tagger.dat"

        # choose which modules among those available will be used by default
        opt.invoke_opt.MACO_AffixAnalysis = True
        opt.invoke_opt.MACO_CompoundAnalysis = True
        opt.invoke_opt.MACO_MultiwordsDetection = True
        opt.invoke_opt.MACO_NumbersDetection = True
        opt.invoke_opt.MACO_PunctuationDetection = True
        opt.invoke_opt.MACO_DatesDetection = True
        opt.invoke_opt.MACO_QuantitiesDetection = True
        opt.invoke_opt.MACO_DictionarySearch = True
        opt.invoke_opt.MACO_ProbabilityAssignment = True
        opt.invoke_opt.MACO_NERecognition = True
        opt.invoke_opt.MACO_RetokContractions = True

        return opt

    @staticmethod
    def extract_object_action_location_triplets(node, current_state=None, triplets=None, object_memory=None, location_memory=None, flush_memory=False):
        if triplets is None:
            triplets = []
        if current_state is None:
            current_state = {'verb': None, 'location': None, 'preposition': None}
        if object_memory is None:
            object_memory = []
        if location_memory is None:
            location_memory = {}
        
        word = node.get_word().get_lemma()
        label = node.get_label()
        tag = node.get_word().get_tag()

        if label == 'top':
            flush_memory = True

        if tag[0] == 'V':
            if flush_memory:
                object_memory.clear()
                flush_memory = False
            current_state['verb'] = word

        if label == 'comp':
            current_state['location'] = word
            for obj in object_memory:
                location_memory[obj] = word

        if label == 'adjt':
            current_state['preposition'] = word

        if label == 'coor':
            object_memory.append(word)
            location = location_memory.get(word, current_state['location'])
            triplet = (word, current_state['verb'], f"{current_state['preposition']} {location}" if current_state['preposition'] else location)
            triplets.append(triplet)

        if label == 'dobj' and current_state['verb']:
            flush_memory = True

        for ch in range(0, node.num_children()):
            child = node.nth_child(ch)
            FreelingAnalyzer.extract_object_action_location_triplets(child, current_state, triplets, object_memory, location_memory, flush_memory)

        return triplets




    @staticmethod
    def get_all_relations(ls):
        """
        Extract relations from a list of sentences.

        Args:
            ls: List of sentences to analyze.

        Returns:
            tuple: Two lists, one with complete relations and another with incomplete ones.
        """
        logging.info("Getting all relations from list of sentences.")

        all_relations = []

        for s in ls:
            dp = s.get_dep_tree()
            relations = FreelingAnalyzer.extract_object_action_location_triplets(dp.begin())
            all_relations.extend(relations)

        complete_relations = [(obj, action, location) for obj, action, location in all_relations if location]
        incomplete_relations = [(obj, action, location) for obj, action, location in all_relations if not location]

        logging.info(f"Extracted complete relations: {complete_relations}")
        logging.info(f"Extracted incomplete relations: {incomplete_relations}")

        return complete_relations, incomplete_relations
