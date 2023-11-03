"""
Module for text analysis using an abstract TextAnalyzer class and
a concrete implementation using pyfreeling.

This module provides functionalities to:
- Configure morphological analyzer options
- Extract object-action-location triplets
- Get all relations from a list of sentences
"""

# Standard imports
import logging
import inspect

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Set, Tuple

# Third party imports
import pyfreeling

# Internal imports
from common.entities import GraphNode, GraphRelation, Triplet


class RelationType(Enum):
    SVO = "(Subject-Verb-object)"
    EAV = "(Entity-Attribute-Value): Used mainly to describe attributes or properties of an entity"
    EAT = "(Entity-Action-Time): Describes an action taken by an entity at a specific time."
    EPT = "(Entity-Part-Tag): Describes a part or a component of an entity and tags it."
    ERT = "(Entity-Relation-Type): Describes the type or nature of a relation between two entities."
    EET = "(Entity-Event-Time): Describes an event that an entity is involved in at a specific time."
    ECV = "(Entity-Cause-Value): Explains why an entity has a certain value or state."
    EPV = "(Entity-Property-Value): Similar to EAV but more focused on intrinsic properties."
    EST = "(Entity-Source-Time): Identifies the source of information and the time it was acquired."
    EWT = "(Entity-Work-Time): Describes the work done by an entity during a specific time frame."
    ECO = "(Entity-Category-Object): Classifies an entity into a certain category and links it to an object within that category."
    ELOC = "(Entity-Location-Object): Describes an entity's location relative to another object."
    EIV = "(Entity-IsA-Value): Defines what type or class an entity belongs to."
    EPS = "(Entity-Part-Status): Describes the status or condition of a part of an entity."
    EOO = "(Entity-Owns-Object): Describes ownership relationships between entities and objects."
    ECL = "(Entity-Coordinates-Location): Provides geographical coordinates for an entity's location."
    ETP = "(Entity-Topic-Polarity): Links an entity to a topic and provides sentiment polarity."
    EMM = "(Entity-Material-Measurement): Describes what material an entity is made of and its measurements."
    EAF = "(Entity-Affiliation-Factor): Describes an entity's affiliation with certain factors like organizations or causes."
    EPP = "(Entity-Purpose-Process): Describes the intended purpose of an entity and the process to achieve it."


class TextAnalyzer(ABC):
    """Abstract base class for text analysis."""

    @abstractmethod
    def extract_svo(self, node, current_state=None, triplets=None):
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
    def get_all_relations(self, ls, language: str):
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
        opt.config_opt.TOK_TokenizerFile = f"{data}/{lang}" + "/tokenizer.dat"
        opt.config_opt.SPLIT_SplitterFile = f"{data}/{lang}" + "/splitter.dat"
        opt.config_opt.MACO_Decimal = "."
        opt.config_opt.MACO_Thousand = ","
        opt.config_opt.MACO_LocutionsFile = f"{data}/{lang}" + "/locucions.dat"
        opt.config_opt.MACO_QuantitiesFile = f"{data}/{lang}" + "/quantities.dat"
        opt.config_opt.MACO_AffixFile = f"{data}/{lang}" + "/afixos.dat"
        opt.config_opt.MACO_ProbabilityFile = f"{data}/{lang}" + "/probabilitats.dat"
        opt.config_opt.MACO_DictionaryFile = f"{data}/{lang}" + "/dicc.src"
        opt.config_opt.MACO_NPDataFile = f"{data}/{lang}" + "/np.dat"
        opt.config_opt.MACO_PunctuationFile = data + "/common/punct.dat"
        opt.config_opt.MACO_ProbabilityThreshold = 0.001
        
        
        opt.config_opt.NEC_NECFile = f"{data}/{lang}" + "/nerc/nec/nec-ab-poor1.dat"
        opt.config_opt.SENSE_ConfigFile = f"{data}/{lang}" + "/senses.dat"
        opt.config_opt.UKB_ConfigFile = f"{data}/{lang}" + "/ukb.dat"
        opt.config_opt.TAGGER_HMMFile = f"{data}/{lang}" + "/tagger.dat"
        opt.config_opt.PARSER_GrammarFile = f"{data}/{lang}" + "/chunker/grammar-chunk.dat"
        opt.config_opt.DEP_TxalaFile = f"{data}/{lang}" + "/dep_txala/dependences.dat"
        opt.config_opt.DEP_TreelerFile = f"{data}/{lang}" + "/treeler/dependences.dat"

        opt.config_opt.MACO_CompoundFile = f"{data}/{lang}" + "/compounds.dat"
        opt.config_opt.COREF_CorefFile = f"{data}/{lang}" + "/coref/relaxcor_constit/relaxcor.dat"
        
        
        opt.invoke_opt.TAGGER_ForceSelect = pyfreeling.RETOK
        opt.invoke_opt.SENSE_WSD_which = pyfreeling.UKB
        opt.invoke_opt.TAGGER_which = pyfreeling.HMM
        opt.invoke_opt.DEP_which = pyfreeling.TXALA
        opt.invoke_opt.SRL_which = pyfreeling.TREELER
        opt.invoke_opt.InputLevel = pyfreeling.TEXT
        opt.invoke_opt.OutputLevel = pyfreeling.SEMGRAPH

        # choose which modules among those available will be used by default
        opt.invoke_opt.MACO_UserMap=False
        opt.invoke_opt.MACO_AffixAnalysis = True
        opt.invoke_opt.MACO_MultiwordsDetection = True
        opt.invoke_opt.MACO_NumbersDetection = True
        opt.invoke_opt.MACO_PunctuationDetection = True
        opt.invoke_opt.MACO_DatesDetection = True
        opt.invoke_opt.MACO_QuantitiesDetection = True
        opt.invoke_opt.MACO_DictionarySearch = True
        opt.invoke_opt.MACO_ProbabilityAssignment = True
        opt.invoke_opt.MACO_CompoundAnalysis = False
        opt.invoke_opt.MACO_NERecognition = True
        opt.invoke_opt.MACO_RetokContractions = False
        
        opt.invoke_opt.NEC_NEClassification = True
        opt.invoke_opt.PHON_Phonetics = False

        return opt

    @staticmethod
    def extract_svo(node, current_state=None, triplets=None, object_memory=None, location_memory=None, flush_memory=False):
        """
        Subject verb object
        """
        logging.debug("Start extracting svo")

        # Initialization (existing code)
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

        logging.debug(f"Processing node with word: {word}, label: {label}, tag: {tag}")

        if label == 'top':
            flush_memory = True

        if tag[0] == 'V':
            current_state['verb'] = word

        if label == 'comp':
            current_state['location'] = word

        if label == 'adjt':
            current_state['preposition'] = word

        # Added 'dobj' and 'subj' to collect objects
        if label in ('coor', 'dobj', 'subj'):
            object_memory.append(word)
            location = current_state.get('location', None)
            triplet = (word, current_state['verb'], f"{current_state['preposition']} {location}" if current_state['preposition'] else location)
            triplets.append(triplet)

        if label == 'dobj' and current_state['verb']:
            flush_memory = True

        for ch in range(0, node.num_children()):
            child = node.nth_child(ch)
            FreelingAnalyzer.extract_svo(child, current_state, triplets, object_memory, location_memory, flush_memory)

        logging.debug("End extracting svo")
        return triplets

    @staticmethod
    def extract_eat(node, current_state=None, triplets=None, entity_memory=None, time_memory=None, language='en'):
        """
        EAT (Entity-Action-Time): Describes an action taken by an entity at a specific time.
        """
        logging.debug("Start extracting eat")

        if triplets is None:
            triplets = []
        if current_state is None:
            current_state = {'verb': None, 'time': None}
        if entity_memory is None:
            entity_memory = []
        if time_memory is None:
            time_memory = []

        word = node.get_word().get_lemma()
        label = node.get_label()
        tag = node.get_word().get_tag()

        verb_tags = {'en': ['VBD'], 'es': ['VMIP3S0']}
        time_tags = {'en': ['Z'], 'es': ['Z']}
        entity_labels = {'en': ['ncsubj'], 'es': ['subj']}
        entity_tags = {'en': [], 'es': ['NP00000']}

        if tag in verb_tags.get(language, []):
            current_state['verb'] = word

        if tag in time_tags.get(language, []):
            current_state['time'] = word

        if label in entity_labels.get(language, []) or tag in entity_tags.get(language, []):
            entity_memory.append(word)
            time = current_state.get('time', None)
            triplet = (word, current_state['verb'], time)
            triplets.append(triplet)

        for ch in range(0, node.num_children()):
            child = node.nth_child(ch)
            FreelingAnalyzer.extract_eat(child, current_state, triplets, entity_memory, time_memory, language)
        
        logging.debug("Start extracting eat")
        return triplets

    @staticmethod
    def extract_ept(node, current_state=None, triplets=None, entity_memory=None, part_memory=None, language='en'):
        """
        EPT (Entity-Part-Tag): Describes a part or a component of an entity and tags it.
        """
        # Initialization
        if triplets is None:
            triplets = []
        if current_state is None:
            current_state = {'entity': None, 'part': None, 'tag': None}
        if entity_memory is None:
            entity_memory = []
        if part_memory is None:
            part_memory = []

        word = node.get_word().get_lemma()
        label = node.get_label()
        tag = node.get_word().get_tag()

        entity_labels = {'en': ['top'], 'es': ['top']}
        part_labels = {'en': ['mod'], 'es': ['mod']}
        tag_labels = {'en': ['mod', 'modnomatch'], 'es': ['mod']}

        if label in entity_labels.get(language, []):
            current_state['entity'] = word

        if label in part_labels.get(language, []):
            current_state['part'] = word

        if label in tag_labels.get(language, []):
            current_state['tag'] = word

        if current_state['entity'] and current_state['part']:
            triplet = (current_state['entity'], current_state['part'], current_state['tag'])
            triplets.append(triplet)

        for ch in range(0, node.num_children()):
            child = node.nth_child(ch)
            FreelingAnalyzer.extract_ept(child, current_state, triplets, entity_memory, part_memory, language)

        return triplets

    @staticmethod
    def extract_eet(node, triplets=None, current_state=None, language='en'):
        if triplets is None:
            triplets = []
        
        if current_state is None:
            current_state = {'entity': None, 'event': None, 'time': None}

        word = node.get_word().get_lemma()
        label = node.get_label()
        tag = node.get_word().get_tag()

        logging.debug(f"Processing node with word: {word}, label: {label}, tag: {tag}")

        if label == 'subj':
            current_state['entity'] = word
        elif label == 'top':
            current_state['event'] = word
        elif label == 'dobj':
            current_state['event'] = f"{current_state['event']} ({word})"
        elif label == 'adjt' or label == 'comp':
            if current_state['time'] is not None:
                current_state['time'] = f"{current_state['time']} {word}"
            else:
                current_state['time'] = word

        logging.debug(f"Updated state: {current_state}")

        for ch in range(node.num_children()):
            child = node.nth_child(ch)
            FreelingAnalyzer.extract_eet(child, triplets, current_state, language)

        if all(val is not None for val in current_state.values()):
            logging.debug(f"Appending to triplets: {current_state}")
            triplets.append(((current_state['entity'], current_state['event'], current_state['time'])))            

            # Reset current_state
            for key in current_state.keys():
                current_state[key] = None

        return triplets

    @staticmethod
    def extract_est(node, triplets=None, current_state=None, language='en'):
        if triplets is None:
            triplets = []
        
        if current_state is None:
            current_state = {'entity': None, 'source': None, 'time': None}

        word = node.get_word().get_lemma()
        label = node.get_label()
        tag = node.get_word().get_tag()

        logging.debug(f"Processing node with word: {word}, label: {label}, tag: {tag}")

        if label in ['ncsubj', 'subj']:
            current_state['entity'] = word
        elif label in ['ncmod', 'ccomp']:
            current_state['source'] = word
        elif label in ['dobj', 'ncmod', 'adjt', 'comp']:
            current_state['time'] = word

        logging.debug(f"Updated state: {current_state}")

        for ch in range(node.num_children()):
            child = node.nth_child(ch)
            FreelingAnalyzer.extract_est(child, triplets, current_state, language)

        if len([element for element in current_state.values() if element is None]) < 2:
            logging.debug(f"Appending to triplets: {current_state}")
            triplets.append((current_state['entity'], current_state['source'], current_state['time']))

            # Reset current_state
            for key in current_state.keys():
                current_state[key] = None

        return triplets

    @staticmethod
    def extract_epp(node, triplets=None, current_state=None, language='en'):
        if triplets is None:
            triplets = []

        if current_state is None:
            current_state = {'entity': None, 'purpose': None, 'process': None}

        word = node.get_word().get_lemma()
        label = node.get_label()
        tag = node.get_word().get_tag()

        logging.debug(f"Processing node with word: {word}, label: {label}, tag: {tag}")

        if label in ['subj', 'ncsubj', 'top']:
            current_state['entity'] = word
        elif label in ['top', 'mod']:
            current_state['process'] = word
        elif label in ['dobj', 'ncmod', 'adjt', 'comp']:
            if current_state['purpose'] is not None:
                current_state['purpose'] = f"{current_state['purpose']} {word}"
            else:
                current_state['purpose'] = word

        logging.debug(f"Updated state: {current_state}")

        for ch in range(node.num_children()):
            child = node.nth_child(ch)
            FreelingAnalyzer.extract_epp(child, triplets, current_state, language)

        if all(val is not None for val in current_state.values()):
            logging.debug(f"Appending to triplets: {current_state}")
            triplets.append((current_state["entity"], current_state["purpose"], current_state["process"]))
            

            # Reset current_state
            for key in current_state.keys():
                current_state[key] = None

        return triplets

    @staticmethod
    def _extract_eaf_es(node, triplets=None, current_state=None):
        if triplets is None:
            triplets = []

        if current_state is None:
            current_state = {'entity': None, 'affiliation': None, 'factor': None}

        word = node.get_word().get_lemma()
        label = node.get_label()
        tag = node.get_word().get_tag()

        logging.debug(f"Processing node with word: {word}, label: {label}, tag: {tag}")

        if label == 'subj':
            current_state['entity'] = word
        elif label == 'top' or label == 'dobj':
            current_state['affiliation'] = word
        elif label == 'ncmod' or label == 'adjt':
            if current_state['factor'] is not None:
                current_state['factor'] = f"{current_state['factor']} {word}"
            else:
                current_state['factor'] = word

        logging.debug(f"Updated state: {current_state}")

        for ch in range(node.num_children()):
            child = node.nth_child(ch)
            FreelingAnalyzer._extract_eaf_es(child, triplets, current_state)

        if all(val is not None for val in current_state.values()):
            logging.debug(f"Appending to triplets: {current_state}")
            triplets.append((current_state['entity'], current_state['affiliation'], current_state['factor']))

            # Reset current_state
            for key in current_state.keys():
                current_state[key] = None

        return triplets

    @staticmethod
    def _extract_eaf_en(node, triplets=None, current_state=None, language='en'):
        if triplets is None:
            triplets = []

        if current_state is None:
            current_state = {'entity': None, 'affiliation': None, 'factor': None}

        word = node.get_word().get_lemma()
        label = node.get_label()
        tag = node.get_word().get_tag()

        logging.debug(f"Processing node with word: {word}, label: {label}, tag: {tag}")

        if language == 'en':
            subj_label = 'ncsubj'
            obj_label = 'dobj'
            mod_label = 'ncmod'
        else:
            subj_label = 'subj'
            obj_label = 'dobj'
            mod_label = 'ncmod'

        if label == subj_label:
            current_state['entity'] = word
        elif label == 'top' or label == obj_label:
            current_state['affiliation'] = word
        elif label == mod_label:
            if current_state['factor'] is not None:
                current_state['factor'] = f"{current_state['factor']} {word}"
            else:
                current_state['factor'] = word

        logging.debug(f"Updated state: {current_state}")

        for ch in range(node.num_children()):
            child = node.nth_child(ch)
            FreelingAnalyzer._extract_eaf_en(child, triplets, current_state, language)

        if all(val is not None for val in current_state.values()):
            logging.debug(f"Appending to triplets: {current_state}")
            triplets.append((current_state['entity'], current_state['affiliation'], current_state['factor']))

            # Reset current_state
            for key in current_state.keys():
                current_state[key] = None

        return triplets

    @staticmethod
    def extract_eaf(node, triplets=None, current_state=None, language='en'):
        if language.lower() == "es":
            return FreelingAnalyzer._extract_eaf_es(node=node, triplets=triplets, current_state=current_state)
        else:
            return FreelingAnalyzer._extract_eaf_en(node=node, triplets=triplets, current_state=current_state, language=language)

    @staticmethod
    def extract_emm(node, triplets=None, current_state=None, language='en'):
        if triplets is None:
            triplets = []
        
        if current_state is None:
            current_state = {'entity': None, 'material': None, 'measurement': None}
        
        word = node.get_word().get_lemma()
        label = node.get_label()
        tag = node.get_word().get_tag()
        
        logging.debug(f"Processing node with word: {word}, label: {label}, tag: {tag}")
        
        # For English
        if language == 'en':
            if label == 'subj' or label == 'ncsubj':
                current_state['entity'] = word
            elif label == 'dobj' or label == 'mod':
                current_state['material'] = word
            elif label == 'modnomatch' or label == 'spec':
                current_state['measurement'] = word

        # For Spanish
        elif language == 'es':
            if label == 'subj':
                current_state['entity'] = word
            elif label == 'attr' or label == 'pobj':
                current_state['material'] = word
            elif label == 'dobj' or label == 'comp':
                current_state['measurement'] = word
        
        logging.debug(f"Updated state: {current_state}")

        for ch in range(node.num_children()):
            child = node.nth_child(ch)
            FreelingAnalyzer.extract_emm(child, triplets, current_state, language)
        
        if all(val is not None for val in current_state.values()):
            logging.debug(f"Appending to triplets: {current_state}")
            triplets.append((current_state["entity"], current_state["measurement"], current_state["material"]))

            # Reset current_state
            for key in current_state.keys():
                current_state[key] = None

        return triplets

    @staticmethod
    def extract_eoo(node, triplets=None, current_state=None, language='en'):
        if triplets is None:
            triplets = []
        
        if current_state is None:
            current_state = {'entity': None, 'object': None}

        logging.debug(f"Initializing EOO extraction for language: {language}")
        
        if language == 'en':
            current_state['relation'] = "owns"
            FreelingAnalyzer._extract_eoo_en(node, triplets, current_state)
        elif language == 'es':
            current_state['relation'] = "Posee"
            FreelingAnalyzer._extract_eoo_es(node, triplets, current_state)
        else:
            logging.error(f"Unsupported language: {language}")
            return None

        return triplets

    @staticmethod
    def _extract_eloc_en(node, triplets, current_state):
        word = node.get_word().get_lemma()
        label = node.get_label()
        tag = node.get_word().get_tag()

        logging.debug(f"Start extracting ELOC: Processing node with word: {word}, label: {label}, tag: {tag}")

        if label == 'ncsubj' and tag in ["NN", "NNS", "NNP", "NNPS"]:
            current_state['entity'] = word
        elif label in ['prep', 'case'] and tag == "IN":
            current_state['location'] = word
        elif (label == 'dobj' or label == 'pobj') and tag in ["NN", "NNS"]:
            current_state['object'] = word

        # logging.debug(f"End extracting ELOC: Updated English state: {current_state}")

        # Only traverse child nodes if the current node had a relevant label
        if label in ['ncsubj', 'prep', 'case', 'dobj', 'pobj']:
            for ch in range(node.num_children()):
                child = node.nth_child(ch)
                FreelingAnalyzer._extract_eloc_en(child, triplets, current_state)

        if all(val is not None for val in current_state.values()):
            logging.debug(f"Appending to English triplets for ELOC: {current_state}")
            triplets.append((current_state["entity"], current_state["location"], current_state["object"]))
            
            # Reset current_state
            for key in current_state.keys():
                current_state[key] = None

    @staticmethod
    def _extract_eoo_en(node, triplets, current_state):
        word = node.get_word().get_lemma()
        label = node.get_label()
        tag = node.get_word().get_tag()
        
        logging.debug(f"Start extracting EOO: Processing node with word: {word}, label: {label}, tag: {tag}")

        if label == 'subj' or label == 'ncsubj':
            current_state['entity'] = word
        elif label == 'dobj':
            if current_state['object'] is None:
                current_state['object'] = word
            else:
                current_state['object'] = f"{current_state['object']} {word}"
        
        logging.debug(f"End extracting EOO: Updated English state: {current_state}")

        for ch in range(node.num_children()):
            child = node.nth_child(ch)
            FreelingAnalyzer._extract_eoo_en(child, triplets, current_state)
            
        if all(val is not None for val in current_state.values()):
            logging.debug(f"Appending to English triplets: {current_state}")
            state = (current_state["entity"], current_state["relation"], current_state["object"])
            triplets.append(state)

            # Reset current_state
            for key in current_state.keys():
                current_state[key] = None

    @staticmethod
    def _extract_eoo_es(node, triplets, current_state):
        word = node.get_word().get_lemma()
        label = node.get_label()
        tag = node.get_word().get_tag()

        logging.debug(f"Processing Spanish node with word: {word}, label: {label}, tag: {tag}")
        
        if label == 'subj':
            current_state['entity'] = word
        elif label == 'dobj' or label == 'pobj':
            current_state['object'] = word

        logging.debug(f"Updated Spanish state: {current_state}")

        for ch in range(node.num_children()):
            child = node.nth_child(ch)
            FreelingAnalyzer._extract_eoo_es(child, triplets, current_state)
            
        if all(val is not None for val in current_state.values()):
            logging.debug(f"Appending to Spanish triplets: {current_state}")
            state = (current_state["entity"], current_state["relation"], current_state["object"])
            triplets.append(state)

            # Reset current_state
            for key in current_state.keys():
                if key == "relation":
                    continue

    @staticmethod
    def extract_eps(node, triplets=None, current_state=None, language='en'):
        if triplets is None:
            triplets = []
        
        if current_state is None:
            current_state = {'entity': None, 'part': None, 'status': None}
        
        logging.debug(f"Initializing EPS extraction for language: {language}")
        
        if language == 'en':
            FreelingAnalyzer._extract_eps_en(node, triplets, current_state)
        elif language == 'es':
            FreelingAnalyzer._extract_eps_es(node, triplets, current_state)
        else:
            logging.error(f"Unsupported language: {language}")
            return None

        return triplets

    @staticmethod
    def _extract_eps_en(node, triplets, current_state):
        word = node.get_word().get_lemma()
        label = node.get_label()
        tag = node.get_word().get_tag()
        
        logging.debug(f"Start extracting EPS: Processing node with word: {word}, label: {label}, tag: {tag}")

        if label == 'ncsubj':
            current_state['part'] = word
        elif label == 'ncmod':
            current_state['entity'] = word
        elif label == 'top' and tag == 'VBN':  # VBN is the past participle, which often indicates status
            current_state['status'] = word

        logging.debug(f"End extracting EPS: Updated English state: {current_state}")

        for ch in range(node.num_children()):
            child = node.nth_child(ch)
            FreelingAnalyzer._extract_eps_en(child, triplets, current_state)
            
        if all(val is not None for val in current_state.values()):
            logging.debug(f"Appending to English triplets for EPS: {current_state}")
            triplets.append((current_state["entity"], current_state["status"], current_state["part"]))

            # Reset current_state
            for key in current_state.keys():
                current_state[key] = None

    @staticmethod
    def _extract_eps_es(node, triplets, current_state):
        word = node.get_word().get_lemma()
        label = node.get_label()
        tag = node.get_word().get_tag()
        
        logging.debug(f"Start extracting EPS: Processing node with word: {word}, label: {label}, tag: {tag}")

        if label == 'subj':
            current_state['part'] = word
        elif label == 'comp':
            current_state['entity'] = word
        elif label == 'attr':
            current_state['status'] = word

        logging.debug(f"End extracting EPS: Updated Spanish state: {current_state}")

        for ch in range(node.num_children()):
            child = node.nth_child(ch)
            FreelingAnalyzer._extract_eps_es(child, triplets, current_state)
            
        if all(val is not None for val in current_state.values()):
            logging.debug(f"Appending to Spanish triplets for EPS: {current_state}")
            triplets.append((current_state["entity"], current_state["status"], current_state["part"]))

            # Reset current_state
            for key in current_state.keys():
                current_state[key] = None

    @staticmethod
    def extract_eiv(node, triplets=None, current_state=None, language='en'):
        if triplets is None:
            triplets = []
        if current_state is None:
            current_state = {'entity': None, 'isa': None, 'value': None}
        
        if language == 'en':
            FreelingAnalyzer._extract_eiv_en(node, triplets, current_state)
        elif language == 'es':
            FreelingAnalyzer._extract_eiv_es(node, triplets, current_state)
        
        return triplets

    @staticmethod
    def _extract_eiv_en(node, triplets, current_state):
        word = node.get_word().get_lemma()
        label = node.get_label()
        tag = node.get_word().get_tag()

        logging.debug(f"Start extracting EIV: Processing node with word: {word}, label: {label}, tag: {tag}")

        if label == 'ncsubj':
            current_state['entity'] = word
        elif word.lower() == 'be':
            current_state['isa'] = 'is_a'
        elif label == 'dobj':
            current_state['value'] = word

        logging.debug(f"End extracting EIV: Updated English state: {current_state}")

        for ch in range(node.num_children()):
            child = node.nth_child(ch)
            FreelingAnalyzer._extract_eiv_en(child, triplets, current_state)

        if all(val is not None for val in current_state.values()):
            logging.debug(f"Appending to English triplets for EIV: {current_state}")
            triplets.append((current_state["entity"], current_state["isa"], current_state["value"]))

            # Reset current_state
            for key in current_state.keys():
                current_state[key] = None


    @staticmethod
    def _extract_eiv_es(node, triplets, current_state):
        word = node.get_word().get_lemma()
        label = node.get_label()
        tag = node.get_word().get_tag()

        logging.debug(f"Start extracting EIV: Processing node with word: {word}, label: {label}, tag: {tag}")

        if label == 'subj':
            current_state['entity'] = word
        elif word.lower() == 'ser':
            current_state['isa'] = 'es_un'
        elif label == 'attr':
            current_state['value'] = word

        logging.debug(f"End extracting EIV: Updated Spanish state: {current_state}")

        for ch in range(node.num_children()):
            child = node.nth_child(ch)
            FreelingAnalyzer._extract_eiv_es(child, triplets, current_state)

        if all(val is not None for val in current_state.values()):
            logging.debug(f"Appending to Spanish triplets for EIV: {current_state}")
            triplets.append((current_state["entity"], current_state["isa"], current_state["value"]))

            # Reset current_state
            for key in current_state.keys():
                current_state[key] = None

    @staticmethod
    def extract_eloc(node, triplets=None, current_state=None, language='en'):
        if triplets is None:
            triplets = []

        if current_state is None:
            current_state = {'entity': None, 'location': None, 'object': None}

        if language == 'en':
            FreelingAnalyzer._extract_eloc_en(node, triplets, current_state)
        elif language == 'es':
            FreelingAnalyzer._extract_eloc_es(node, triplets, current_state)

        return triplets

    @staticmethod
    def _extract_eloc_en(node, triplets, current_state):
        word = node.get_word().get_lemma()
        label = node.get_label()
        tag = node.get_word().get_tag()

        logging.debug(f"Start extracting ELOC: Processing node with word: {word}, label: {label}, tag: {tag}")

        if label == 'ncsubj' and tag in ["NN", "NNS", "NNP", "NNPS"]:
            current_state['entity'] = word
        elif label in ['prep', 'case', 'ncmod'] and tag == "IN":
            current_state['location'] = word
        elif (label == 'dobj' or label == 'pobj') and tag in ["NN", "NNS"]:
            current_state['object'] = word

        logging.debug(f"End extracting ELOC: Updated English state: {current_state}")

        for ch in range(node.num_children()):
            child = node.nth_child(ch)
            FreelingAnalyzer._extract_eloc_en(child, triplets, current_state)

        if all(val is not None for val in current_state.values()):
            logging.debug(f"Appending to English triplets for ELOC: {current_state}")
            triplets.append((current_state["entity"], current_state["location"], current_state["object"]))

            # Reset current_state
            for key in current_state.keys():
                current_state[key] = None

    @staticmethod
    def _extract_eloc_es(node, triplets, current_state):
        word = node.get_word().get_lemma()
        label = node.get_label()
        tag = node.get_word().get_tag()

        logging.debug(f"Start extracting ELOC: Processing node with word: {word}, label: {label}, tag: {tag}")

        if label == 'subj':
            current_state['entity'] = word
        elif label in ['pobj', 'adjt']:
            current_state['location'] = word
        elif label == 'comp':
            current_state['object'] = word

        logging.debug(f"End extracting ELOC: Updated Spanish state: {current_state}")

        for ch in range(node.num_children()):
            child = node.nth_child(ch)
            FreelingAnalyzer._extract_eloc_es(child, triplets, current_state)

        if all(val is not None for val in current_state.values()):
            logging.debug(f"Appending to Spanish triplets for ELOC: {current_state}")
            triplets.append((current_state["entity"], current_state["location"], current_state["object"]))

            # Reset current_state
            for key in current_state.keys():
                current_state[key] = None

    @staticmethod
    def extract_eco(node, triplets=None, current_state=None, language='en'):
        if triplets is None:
            triplets = []

        if current_state is None:
            current_state = {'entity': None, 'category': None, 'object': None}

        if language == 'en':
            FreelingAnalyzer._extract_eco_en(node, triplets, current_state)
        elif language == 'es':
            FreelingAnalyzer._extract_eco_es(node, triplets, current_state)

        return triplets

    @staticmethod
    def _extract_eco_en(node, triplets, current_state):
        word = node.get_word().get_lemma()
        label = node.get_label()
        tag = node.get_word().get_tag()

        logging.debug(f"Start extracting ECO: Processing node with word: {word}, label: {label}, tag: {tag}")

        if label == 'top' and tag in ["NN", "NNS", "NNP", "NNPS"]:
            current_state['entity'] = word
        elif label == 'ncmod' and tag == "IN":
            current_state['category'] = word
        elif label == 'dobj' and tag in ["NN", "NNS"]:
            current_state['object'] = word

        logging.debug(f"End extracting ECO: Updated English state: {current_state}")

        for ch in range(node.num_children()):
            child = node.nth_child(ch)
            FreelingAnalyzer._extract_eco_en(child, triplets, current_state)

        if all(val is not None for val in current_state.values()):
            logging.debug(f"Appending to English triplets for ECO: {current_state}")
            triplets.append((current_state["entity"], current_state["category"], current_state["object"]))

            for key in current_state.keys():
                current_state[key] = None

    @staticmethod
    def _extract_eco_es(node, triplets, current_state):
        word = node.get_word().get_lemma()
        label = node.get_label()
        tag = node.get_word().get_tag()

        logging.debug(f"Start extracting ECO: Processing Spanish node with word: {word}, label: {label}, tag: {tag}")

        if label == 'subj':
            current_state['entity'] = word
        elif label in ['pobj', 'adjt']:
            current_state['category'] = word
        elif label == 'comp':
            current_state['object'] = word

        logging.debug(f"End extracting ECO: Updated Spanish state: {current_state}")

        for ch in range(node.num_children()):
            child = node.nth_child(ch)
            FreelingAnalyzer._extract_eco_es(child, triplets, current_state)

        if all(val is not None for val in current_state.values()):
            logging.info(f"Appending to Spanish triplets for ECO: {current_state}")
            triplets.append((current_state["entity"], current_state["category"], current_state["object"]))

            # Reset current_state
            for key in current_state.keys():
                current_state[key] = None

    @staticmethod
    def extract_ewt(node, triplets=None, current_state=None, language='en'):
        if triplets is None:
            triplets = []

        if current_state is None:
            current_state = {'entity': None, 'work': None, 'time': None}

        if language == 'en':
            FreelingAnalyzer._extract_ewt_en(node, triplets, current_state)
        elif language == 'es':
            FreelingAnalyzer._extract_ewt_es(node, triplets, current_state)

        return triplets

    @staticmethod
    def _extract_ewt_en(node, triplets, current_state):
        word = node.get_word().get_lemma()
        label = node.get_label()
        tag = node.get_word().get_tag()

        logging.debug(f"Start extracting EWT: Processing node with word: {word}, label: {label}, tag: {tag}")

        if label == 'ncsubj' and tag in ["NN", "NNS", "NNP", "NNPS"]:
            current_state['entity'] = word
        elif label == 'top' and tag in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
            current_state['work'] = word
        elif label == 'dobj' and tag in ["NN", "NNS"]:
            current_state['time'] = word

        logging.debug(f"End extracting EWT: Updated English state: {current_state}")

        for ch in range(node.num_children()):
            child = node.nth_child(ch)
            FreelingAnalyzer._extract_ewt_en(child, triplets, current_state)

        if all(val is not None for val in current_state.values()):
            logging.debug(f"Appending to English triplets for EWT: {current_state}")
            triplets.append((current_state["entity"], current_state["time"], current_state["work"]))
            for key in current_state.keys():
                current_state[key] = None

    @staticmethod
    def _extract_ewt_es(node, triplets, current_state):
        word = node.get_word().get_lemma()
        label = node.get_label()
        tag = node.get_word().get_tag()

        logging.debug(f"Start extracting EWT: Processing node with word: {word}, label: {label}, tag: {tag}")

        if label == 'subj' and tag.startswith("NC"):
            current_state['entity'] = word
        elif label == 'top' and tag.startswith("VM"):
            current_state['work'] = word
        elif label == 'adjt' and tag.startswith("RG"):
            current_state['time'] = word

        logging.debug(f"End extracting EWT: Updated Spanish state: {current_state}")

        for ch in range(node.num_children()):
            child = node.nth_child(ch)
            FreelingAnalyzer._extract_ewt_es(child, triplets, current_state)

        if all(val is not None for val in current_state.values()):
            logging.debug(f"Appending to Spanish triplets for EWT: {current_state}")
            triplets.append((current_state["entity"], current_state["time"], current_state["work"]))
            for key in current_state.keys():
                current_state[key] = None

    @staticmethod
    def remove_duplicates(relations: List[Triplet]):
        # Convert to sets to remove duplicates
        relations_set: Set[Triplet] = set(relations)

        # Convert back to lists
        relations = list(relations_set)
        return relations

    @staticmethod
    def get_all_relations(ls, language: str) -> Tuple[List[Triplet], List[Triplet]]:
        logging.info("Getting all relations from list of sentences.")
        logging.info(f"language {language}")
        all_relations = []

        for s in ls:
            dp = s.get_dep_tree()
            
            # Dynamically find all functions starting with "extract_" and call them
            for func_name in dir(FreelingAnalyzer):
                if func_name.startswith("extract_"):
                    func = getattr(FreelingAnalyzer, func_name)
                    params = inspect.signature(func).parameters.keys()
                    
                    # Preparing arguments based on function signature
                    args_dict = {}
                    if "node" in params:
                        args_dict["node"] = dp.begin()
                    if "language" in params:
                        args_dict["language"] = language

                    relations = func(**args_dict)
                    logging.debug("-------------------------------------------------------")
                    logging.debug(f"relations extacted from function {func_name}: {relations}")
                    logging.debug("-------------------------------------------------------")
                    # Hacer if para ver si relation tiene 3 o 4 elementos
                    for relation in relations:
                        # Ensure the 4th element is unpacked correctly if it exists
                        n1, rel, n2, *extra = relation
                        if len(extra) > 0:
                            extra_properties = extra[0]
                            if not isinstance(extra_properties, dict):
                                logging.debug(f"{str(n1)}, {str(rel)}, {str(n2)}, {str(extra)}")
                                raise TypeError(f"Expected a dictionary as 4th element in function {func_name}, got {extra_properties} when extra is: {extra} and of type {type(extra_properties)} instead: {extra_properties}, {str(n1)}, {str(rel)}, {str(n2)}, {str(extra)}, {relation} and relations: {relations}")
                        else:
                            extra_properties = {}


                        properties = {"language": language, "triplet_type": func_name.replace("extract_", ""), **extra_properties}
                        all_relations.append(
                            Triplet(
                                node1=GraphNode(name=n1, properties={"language": language}),
                                relation=GraphRelation(name=rel, properties=properties),
                                node2=GraphNode(name=n2, properties={"language": language})
                            )
                        )

        complete_relations = [triplet for triplet in all_relations if triplet.relation.name]
        incomplete_relations = [triplet for triplet in all_relations if not triplet.relation.name]
        complete_relations = FreelingAnalyzer.remove_duplicates(complete_relations)
        incomplete_relations = FreelingAnalyzer.remove_duplicates(incomplete_relations)

        logging.debug("-------------------------------------------------------")
        logging.debug(f"Extracted complete relations: {[str(r) for r in complete_relations]}")
        logging.debug(f"Extracted incomplete relations: {[str(r) for r in incomplete_relations]}")
        logging.debug("-------------------------------------------------------")
        logging.info("End Getting all relations from list of sentences.")

        return complete_relations, incomplete_relations


class TestAnalyzer(FreelingAnalyzer):

    @staticmethod
    def get_all_relations(ls, language: str) -> Tuple[List[Triplet], List[Triplet]]:
        logging.info("Getting all relations from list of sentences.")
        logging.info(f"language {language}")
        all_relations = []

        for s in ls:
            dp = s.get_dep_tree()
            # Get the parse tree of the sentence
            parse_tree = s.get_parse_tree()
            all_relations.extend(TestAnalyzer.extract_svo(s, language))

        complete_relations = [triplet for triplet in all_relations if triplet.relation.name]
        incomplete_relations = [triplet for triplet in all_relations if not triplet.relation.name]
        complete_relations = FreelingAnalyzer.remove_duplicates(complete_relations)
        incomplete_relations = FreelingAnalyzer.remove_duplicates(incomplete_relations)

        logging.debug("-------------------------------------------------------")
        logging.debug(f"Extracted complete relations: {[str(r) for r in complete_relations]}")
        logging.debug(f"Extracted incomplete relations: {[str(r) for r in incomplete_relations]}")
        logging.debug("-------------------------------------------------------")
        logging.info("End Getting all relations from list of sentences.")

        return complete_relations, incomplete_relations

    @staticmethod
    def collect_terminal_nodes(node):
        """
        Recursively iterates over the children of a given node and 
        collects terminal nodes as GraphNode objects in a list.
        """
        terminal_nodes = []
        nch = node.num_children()
        
        if nch == 0:  # Base case: Terminal node
            return TestAnalyzer.extract_terminal_node(node)

        # Recursive case: Non-terminal node
        for i in range(nch):
            child = node.nth_child(i)
            if child.get_label().lower() == "cc" or child.get_label().lower() == "fc" or child.get_label().lower() == "fp" or child.get_label().lower().startswith("di0") or child.get_label().lower().startswith("da0"):
                continue
            terminal_nodes.extend(TestAnalyzer.collect_terminal_nodes(child))
        
        return terminal_nodes
    
    @staticmethod
    def extract_terminal_node(node):
        w = node.get_info().get_word()
        logging.debug('Terminal node encountered.')
        return [GraphNode(
            name=w.get_lemma(),
            properties={
                "form": w.get_form(),
                "lemma" : w.get_lemma(),
                "tag": w.get_tag(),
                "node_label": node.get_info().get_label(),
                "parent_label": node.get_parent().get_info().get_label(),
                "node_id": node.get_node_id()
            }
        )]
    
    @staticmethod
    def extract_svo(s, language):
        """ Extract SVO triplets. """
        parse_tree = s.get_parse_tree()
        potential_triplets = TestAnalyzer.SvoTriplets.extract_potential_triplets(parse_tree)

        dp = s.get_dep_tree()
        subjects = TestAnalyzer.SvoTriplets._find_subject(s)
        
        triplets = []
        for triplet in potential_triplets:
            if not isinstance(triplet, list):
                #logging.info(f"Skipping triplet because no relation was found for text {[w.get_form() for w in s.get_words()]}")
                #logging.info(f"Skipping triplet: {str(triplet)}")
                pass
            else:
                properties = {"language": language, "triplet_type": "svo"}
                triplet[1].update_properties(properties)
                triplets.append(Triplet(triplet[0], triplet[1], triplet[2]))
        
        # potential_triplets = TestAnalyzer.SvoTriplets._remove_incorrect_subject_triplets(potential_triplets, subjects)
        # potential_triplets = TestAnalyzer.SvoTriplets.associate_triplets_based_on_tag(potential_triplets, subject=subjects)
        return triplets
    class SvoTriplets():
        """
        This class implements triplet extraction by parsing different parts of the  freeling analyzis
        """
        @staticmethod
        def extract_potential_triplets(ptree):
            triplet_combinations = []
            TestAnalyzer.SvoTriplets._extract_triplets_helper(
                ptree.begin(),
                triplet_combinations
            )
            triplet_combinations = [triplet for triplet in triplet_combinations if isinstance(triplet,list)]
            return triplet_combinations

        @staticmethod
        def _remove_incorrect_subject_triplets(potential_triplets, subject):
            """ Remove triplets which subject is incorrect. """
            triplets = []
            for p in potential_triplets:
                if p[0].name.lower() == subject.lower():
                    triplets.append(p)
            return triplets

        @staticmethod
        def find_nodes_with_string(node, target_string):
            """
            Recursively search for nodes containing the target_string.
            """

            found_nodes = []
            if node.get_word.get_lemma() == target_string:
                found_nodes.append(node)

            for i in range(node.num_children()):
                child = node.nth_child(i)
                TestAnalyzer.find_nodes_with_string(child, target_string)

            return found_nodes

        @staticmethod
        def associate_triplets_based_on_tag(potential_triplets, dp_tree):
            """
            Iterate over each element in potential_triplets and find the corresponding nodes in dp_tree.
            """
            logging.info("Starting to remove incorrect subject triplets.")

            final_triplets = []
            for idx, triplet in enumerate(potential_triplets):
                node_string = triplet.node2.name
                
                if triplet.node2.properties["tag"].lower().startswith("a"):
                    # find nodes with certain string
                    found_nodes = TestAnalyzer.find_nodes_with_string(node=dp_tree.begin(),
                                                        target_string=node_string
                                                        )
                    node = found_nodes[0]
                    parents = []
                    node_afected = None
                    # find parents that start with grup
                    while not node.is_root() and node_afected is None:
                        if node.get_label().startswith('grup'):
                            # parents.append(node)
                            terminal_nodes = TestAnalyzer.collect_terminal_nodes(node)
                            # TODO                          
                        
                        node = node.get_parent() 

                    if not found_nodes:
                        logging.warning(f"No nodes found for element: {node_string}")
                    else:
                        logging.info(f"Nodes found for element {element}: {found_nodes}")
                else:
                    final_triplets.append(triplet)

            logging.info("Finished removing incorrect subject triplets.")

        @staticmethod
        def coreference_parsing(doc, potential_triplets):
            """
            TODO
            """
            sem_graph = doc.get_semantic_graph()
            entities = doc.get_semantic_graph().get_entities()

            node = entities.begin()
            while(node != entities.end()):
                node.incr()

        @staticmethod
        def _extract_triplets_helper(node, triplet_combinations):
            info = node.get_info()
            label = info.get_label().lower()
            nch = node.num_children()

            if nch == 0:
                return TestAnalyzer.extract_terminal_node(node)

            elif label == "grup-verb":
                return TestAnalyzer.SvoTriplets.handle_grup_verb(
                    node,
                    triplet_combinations
                )

            elif label == "verb" or label.startswith("v"):
                return TestAnalyzer.SvoTriplets.handle_verb_case(
                    node
                )

            else:
                return TestAnalyzer.SvoTriplets.handle_generic_node(
                    node,
                    triplet_combinations
                )

        @staticmethod
        def handle_grup_verb(node, triplet_combinations):
            logging.info('Handling grup-verb.')
            
            potential_triplet = [[], None, []]
            nch = node.num_children()

            # Extracting potential triplets within 'grup-verb'
            for i in range(nch):
                child = node.nth_child(i)
                child_label = child.get_info().get_label().lower()

                if child_label == 'verb':
                    verb_child_word = child.nth_child(0).get_info().get_word()
                    potential_triplet[1] = GraphRelation(
                        name=verb_child_word.get_lemma(),
                        properties={
                            "form": verb_child_word.get_form(),
                            "tag": verb_child_word.get_tag(),
                            "node_label": child.nth_child(0).get_info().get_label(),
                            "node_id": child.nth_child(0).get_node_id()
                        }
                    )
                elif potential_triplet[1] is None:
                    potential_triplet[0].extend(TestAnalyzer.collect_terminal_nodes(child))
                else:
                    potential_triplet[2].extend(TestAnalyzer.collect_terminal_nodes(child))

            unfolded_candidates_before_verb = potential_triplet[0]
            unfolded_candidates_after_verb = potential_triplet[2]

            for unfolded_before in unfolded_candidates_before_verb:
                if len(unfolded_candidates_after_verb)>0:
                    for unfolded_after in unfolded_candidates_after_verb:
                        triplet_combinations.append([unfolded_before, potential_triplet[1], unfolded_after])
                else:
                    triplet_combinations.append([unfolded_before, potential_triplet[1], None])
            if len(unfolded_candidates_before_verb) < 1:
                for unfolded_after in unfolded_candidates_after_verb:
                    triplet_combinations.append([None, potential_triplet[1], unfolded_after])

            return triplet_combinations

        @staticmethod
        def handle_verb_case(node):
            logging.info('Handling verb case.')
            verb_relation_object = GraphRelation(
                name=node.nth_child(0).get_word().get_lemma(),
                properties={
                    "form": node.nth_child(0).get_word().get_form(),
                    "tag": node.nth_child(0).get_word().get_tag(),
                    "node_label": node.nth_child(0).get_info().get_label(),
                    "node_id": node.nth_child(0).get_node_id()
                }
            )
            potential_triplet = [[], verb_relation_object, []]
            parent = node.get_parent()
            candidate_pos = 0
            nch = parent.num_children()

            for i in range(nch):
                child = parent.nth_child(i)
                if child.get_node_id() == node.get_node_id():
                    candidate_pos = 2
                    continue
                
                potential_triplet[candidate_pos].extend(
                    TestAnalyzer.collect_terminal_nodes(child)
                )

            unfolded_candidates_before_verb = potential_triplet[0]
            unfolded_candidates_after_verb = potential_triplet[2]
            triplet_combinations = []

            for unfolded_before in unfolded_candidates_before_verb:
                if len(unfolded_candidates_after_verb)>0:
                    for unfolded_after in unfolded_candidates_after_verb:
                        triplet_combinations.append(
                            [
                                unfolded_before,
                                potential_triplet[1],
                                unfolded_after
                            ]
                        )
                else:
                    triplet_combinations.append([unfolded_before, potential_triplet[1], None])
            if len(unfolded_candidates_before_verb) < 1:
                for unfolded_after in unfolded_candidates_after_verb:
                    triplet_combinations.append([None, potential_triplet[1], unfolded_after])

            return triplet_combinations

        @staticmethod
        def handle_generic_node(node, triplet_combinations):
            logging.debug('Handling generic node.')
            nch = node.num_children()
            children_nodes = []

            for i in range(nch):
                child = node.nth_child(i)
                if child.get_label().lower() == "cc" or child.get_label().lower() == "fc" or child.get_label().lower() == "fp" or child.get_label().lower().startswith("di0") or child.get_label().lower().startswith("d0"):
                    continue
                recursive_child = TestAnalyzer.SvoTriplets._extract_triplets_helper(child, [])
                children_nodes.extend(recursive_child)
            
            triplet_combinations.extend(children_nodes)

            return triplet_combinations

        @staticmethod
        def _find_subject(s) -> List[str]:
            """ Find the subject of the sentence."""
            subject = []
            dep_tree = s.get_dep_tree()
            # Iterate dependency tree
            node = dep_tree.begin()

            while(node != dep_tree.end()):
                label = node.get_label()
                if 'suj' in label.lower() or 'subj' in label.lower():
                    subject = node.get_word().get_lemma()
                node.incr()
            return subject

    @staticmethod
    def _get_node_relations(dp_tree, node_id):
        """ Get all words related to the given node. """

    @staticmethod
    def _extract_svo_parse_dp_tree(dp_tree, triplet_combinations):
        
        node = dp_tree.begin()
        while node != dp_tree.end() :
            # if node.get_word().get_tag()[0]=='V' :
            #     for ch in range(0,node.num_children()) :
            #         child = node.nth_child(ch)
            #         if child.get_label()=="SBJ" :
            #            (lsubj,ssubj) = extract_lemma_and_sense(child.get_word())
            #         elif child.get_label()=="OBJ" :
            #            (ldobj,sdobj) = extract_lemma_and_sense(child.get_word())
            print(f"{node.get_word().get_lemma()}, {node.get_label().lower()}")
            if node.get_label().lower() =="sbj":
                found = True
            node.incr()
        # for idx, possible_triplet in triplet_combinations:
        #     pass
        # TestAnalyzer._extract_triplets_helper_dp_tree(
        #     dp_tree.begin(),
        #     triplet_combinations
        # )
        # return triplet_combinations

    @staticmethod
    def _extract_triplets_helper_dp_tree(node, triplet_combinations):
        info = node.get_info()
        label = info.get_label().lower()
        nch = node.num_children()

        if nch == 0:
            return TestAnalyzer.extract_terminal_node(node)

        elif label == "grup-verb":
            return TestAnalyzer.handle_grup_verb(node, triplet_combinations)

        elif label == "verb" or label.startswith("v"):
            return TestAnalyzer.handle_verb_case(node, triplet_combinations)

        else:
            return TestAnalyzer.handle_generic_node(node, triplet_combinations)
