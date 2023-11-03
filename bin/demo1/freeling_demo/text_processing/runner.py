"""
This module contains classes and methods for text processing.
It uses pyfreeling library to perform a set of NLP tasks.
"""

import os
import logging
from dataclasses import dataclass

# Third-party imports
import pyfreeling  # pylint: disable=E0401

# Local imports
from .module import TextAnalyzer, FreelingAnalyzer


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
        self.config = config
        logging.info("TextProcessingRunner initialized.")

    def run(self, text: str):
        """
        Perform text analysis on the input text.

        Parameters:
            text (str): The text to analyze.

        Returns:
            relations: Processed relations from the input text.
        """
        logging.info("Starting text processing.")

        # set locale to an UTF8 compatible locale 
        pyfreeling.util.init_locale(self.config.locale)

        lang = self.config.lang

        # get installation path
        ipath = self.config.ipath

        # path to language data 
        lpath = os.path.join(ipath, "share", "freeling", lang)
        data_path = os.path.join(ipath, "share", "freeling")

        # create analyzers
        tk = pyfreeling.tokenizer(os.path.join(lpath, "tokenizer.dat"))
        sp = pyfreeling.splitter(os.path.join(lpath, "splitter.dat"))

        config_analyzer = self.config.analyzer.my_maco_options(lang, data_path)
        # create the analyzer with the required set of maco_options  
        morfo = pyfreeling.maco(config_analyzer)
                        
        # create tagger
        tagger = pyfreeling.hmm_tagger(config_analyzer)

        # create sense annotator
        sen = pyfreeling.senses(os.path.join(lpath, "senses.dat"))
        # create sense disambiguator
        wsd = pyfreeling.ukb(os.path.join(lpath, "ukb.dat"))
        # create dependency parser
        parser = pyfreeling.chart_parser(
            os.path.join(
                lpath,
                "chunker",
                "grammar-chunk.dat"
                )
            )
        dep = pyfreeling.dep_txala(
            os.path.join(
                lpath,
                "dep_txala",
                "dependences.dat"
            ),
            parser.get_start_symbol()
        )

        # tokenize input line into a list of words
        lw = tk.tokenize(text)
        print(lw)
        # split list of words in sentences, return list of sentences
        ls = sp.split(lw)

        # perform morphosyntactic analysis and disambiguation
        ls = morfo.analyze_sentence_list(ls)
        ls = tagger.analyze_sentence_list(ls)
        # annotate and disambiguate senses     
        ls = sen.analyze_sentence_list(ls)
        ls = wsd.analyze_sentence_list(ls)
        # parse sentences
        ls = parser.analyze_sentence_list(ls)
        ls = dep.analyze_sentence_list(ls)

        relations = self.config.analyzer.get_all_relations(ls)

        logging.info("Text processing completed.")
        return relations
