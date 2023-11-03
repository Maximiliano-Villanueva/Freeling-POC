"""
This module contains classes and methods for text processing.
It uses pyfreeling library to perform a set of NLP tasks.
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
from collections import deque

# Third-party imports
import pyfreeling  # pylint: disable=E0401

# Local imports
from common.entities import Triplet
from .module import TextAnalyzer, FreelingAnalyzer, TestAnalyzer


@dataclass
class FREELING_CONF_ES:
    """
    This class contains all necessary paths for spanish processing
    """
    
@dataclass
class TextProcessingRunnerConfig:
    """
    Configuration Data Class for TextProcessingRunner
    """
    lang: str = "es"
    data_path: str = "/usr/local/share/freeling/"
    ipath: str = "/usr/local/"
    locale: str = "default"
    analyzer: TextAnalyzer = FreelingAnalyzer()
        

class TextProcessingRunner:
    """
    A class to encapsulate the text processing logic.
    """
    def __init__(self, config=TextProcessingRunnerConfig()):
        """
        Initialize the TextProcessingRunner with a given configuration.

        Parameters:
            config (TextProcessingRunnerConfig): The configuration settings.
        """
        self._config = config
        logging.info("TextProcessingRunner initialized.")

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = config  # Storing square of the value

    def detect_language(self, sample_text: str):
        """
        Detect language from a sample text
        """
        logging.info("Start language detection.")
        # set locale to an UTF8 compatible locale 
        pyfreeling.util.init_locale(self._config.locale)
        # get installation path
        ipath = self._config.ipath
        # path to language data 
        cpath = os.path.join(ipath, "share", "freeling", "common")
        # creates a language identifier with the default config file
        ident_path = os.path.join(cpath, "lang_ident", "ident.dat")
        di = pyfreeling.lang_ident(ident_path)
        best_l = di.identify_language(sample_text)
        logging.info(f"Language detected {best_l}")
        logging.info("End language detection.")

        return best_l

    def run_on_document(self, document: str) -> List[Dict[str, Union[List[Triplet], str]]]:
        """
        Perform text analysis on the input document.

        Parameters:
            document (str): The document to analyze.

        Returns:
            relations: A dictionary containing lists of triplets and the paragraph of processed relations from the input document.
        """
        logging.info("Starting document processing.")

        # Split the document into paragraphs
        paragraphs = document.split("\n\n")  # Assuming paragraphs are separated by two newlines
        all_relations = []

        for paragraph in paragraphs:
            # Analyze each paragraph
            complete_relations, incomplete_relations = self.run(paragraph)
            self.run(paragraph)
            if len(complete_relations) == 0 and len(incomplete_relations) == 0:
                continue
            temp = {"complete_relations": complete_relations,
                    "incomplete_relations": incomplete_relations,
                    "paragraph": paragraph}
            all_relations.append(temp)

        logging.info("End Document processing")
        return all_relations
    
    def run(self, text: str) -> Tuple[List[Triplet], List[Triplet]]:
        """
        Perform text analysis on the input text.

        Parameters:
            text (str): The text to analyze.

        Returns:
            relations: Processed relations from the input text.
        """
        logging.info("Starting text processing.")

        # set locale to an UTF8 compatible locale 
        pyfreeling.util.init_locale(self._config.locale)

        lang = self._config.lang

        # get installation path
        ipath = self._config.ipath

        # path to language data 
        lpath = os.path.join(ipath, "share", "freeling", lang)
        data_path = os.path.join(ipath, "share", "freeling")

        # create analyzers
        tk = pyfreeling.tokenizer(os.path.join(lpath, "tokenizer.dat"))
        sp = pyfreeling.splitter(os.path.join(lpath, "splitter.dat"))
        neclass = pyfreeling.nec(lpath + "/nerc/nec/nec-ab-poor1.dat")

        config_analyzer = self._config.analyzer.my_maco_options(lang, data_path)
        # create the analyzer with the required set of maco_options  
        morfo = pyfreeling.maco(config_analyzer)
                        
        # create tagger
        tagger = pyfreeling.hmm_tagger(config_analyzer)

        # create sense annotator
        sen = pyfreeling.senses(os.path.join(lpath, "senses.dat"))
        # create sense disambiguator
        wsd = pyfreeling.ukb(os.path.join(lpath, "ukb.dat"))
        # create dependency parser
        parser_chart = pyfreeling.chart_parser(
            os.path.join(
                lpath,
                "chunker",
                "grammar-chunk.dat"
                )
            )
        parser = pyfreeling.dep_treeler(
            os.path.join(
                lpath,
                "treeler",
                "dependences.dat"
                )
            )
        srl_parser = pyfreeling.srl_treeler(
            os.path.join(
                lpath,
                "treeler",
                "srl.dat"
                )
            )

        coref_analysis = pyfreeling.relaxcor(
            os.path.join(
                lpath,
                "coref",
                "relaxcor_constit",
                "relaxcor.dat"
                )
        )
        
        dep = pyfreeling.dep_txala(
            os.path.join(
                lpath,
                "dep_txala",
                "dependences.dat"
            ),
            parser_chart.get_start_symbol()
        )

        nerc_module = pyfreeling.crf_nerc(
            os.path.join(
                lpath,
                "nerc",
                "nerc",
                "nerc.dat"
            )
        )

        # tokenize input line into a list of words
        lw = tk.tokenize(text)
        
        # split list of words in sentences, return list of sentences
        ls = sp.split(lw)
        # perform morphosyntactic analysis and disambiguation
        ls = morfo.analyze_sentence_list(ls)
        ls = tagger.analyze_sentence_list(ls)
        ls = sen.analyze_sentence_list(ls)
        ls = neclass.analyze_sentence_list(ls)
        ls = nerc_module.analyze_sentence_list(ls)
        ls = wsd.analyze_sentence_list(ls)
        
        # ls = dep.analyze_sentence_list(ls)
        ls = parser_chart.analyze_sentence_list(ls)
        ls = dep.analyze_sentence_list(ls)
        ls = parser.analyze_sentence_list(ls)        
        ls = srl_parser.analyze_sentence_list(ls)

        # self.run_old(text=text)

        complete_relations, incomplete_relations = self._config.analyzer.get_all_relations(ls=ls, language=self._config.lang)

        logging.info("Text processing completed.")
        return complete_relations, incomplete_relations



    def run_old(self, text: str) -> Tuple[List[Triplet], List[Triplet]]:
            """
            Perform text analysis on the input text.

            Parameters:
                text (str): The text to analyze.

            Returns:
                relations: Processed relations from the input text.
            """
            logging.info("Starting text processing.")

            # set locale to an UTF8 compatible locale 
            pyfreeling.util.init_locale(self._config.locale)

            lang = self._config.lang

            # get installation path
            ipath = self._config.ipath

            # path to language data 
            lpath = os.path.join(ipath, "share", "freeling", lang)
            data_path = os.path.join(ipath, "share", "freeling")

            config_analyzer = self._config.analyzer.my_maco_options(lang, data_path)
            analyzer = pyfreeling.analyzer(config_analyzer)

            doc = pyfreeling.document()
            analyzer.analyze(text, doc)

            output = pyfreeling.output_freeling()
            output.output_senses(False)
            output.output_dep_tree(False)
            output.output_corefs(False)
            output.output_semgraph(True)

            # print(output.PrintResults(doc))

            conll = pyfreeling.output_conll()
            print(conll.PrintResults(doc))
