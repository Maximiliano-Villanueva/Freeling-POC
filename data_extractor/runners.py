"""
This module contains the Runner class used to extract data from different sources
and prepare them for further processing.

A specific configuration for the run can be set using the RunConfig dataclass.

The data extract function can be provided at runtime, allowing for flexibility 
in the data pipeline.

Example usage:

    config = RunConfig(data_directory='/path/to/your/data/directory')
    runner = Runner(config=config, data_extracor=ZipDataLoader(
        filepath=config.data_directory
    ))
    data = runner.run()
"""

# Standard imports
import logging

from dataclasses import dataclass
from typing import Dict, Optional

# Internal imports
from data_extractor.module import DataExtractor


@dataclass
class RunConfig:
    """ 
    Configuration for the run. 

    Attributes
    ----------
    data_directory: str
        The directory from where to extract the data.
    """
    data_directory: str = ''



class Runner:
    """
    A class that represents a runner to process data
    with specific configuration.

    Attributes
    ----------
    config: RunConfig
        The configuration to use for the run.
    data_extractor: DataExtractor
        The data extracter to extract data.

    Methods
    -------
    run() -> Any:
        Runs the script with the specified configuration.
    """
    def __init__(self,
                 config: Optional[RunConfig] = RunConfig(),
                 extractor: Optional[DataExtractor] = None) -> None:
        self.config: RunConfig = config
        self.extractor: DataExtractor = extractor

    def run(self) -> Dict[str, str]:
        """ Run the script with specific configuration. """
        logging.info(f"Start extracting data from {self.config.data_directory}")      

        file_paths = DataExtractor.get_file_paths(self.config.data_directory)

        data = {}
        for file_path in file_paths:
            logging.debug(f"Loading data from path {file_path}")
            data[str(file_path)] = self.extractor.extract(file_path)

        logging.info(f"End extracting data from {self.config.data_directory}")
        
        return data

