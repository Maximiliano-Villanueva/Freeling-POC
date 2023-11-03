"""
This script contains classes and a factory for data extraction and processing
from various sources.
"""
# Standard imports
import abc
import os
import pickle
import zipfile

from enum import Enum
from typing import Callable, Dict, List, Optional, Union

# Third party libraries
import pandas as pd

# Internal imports
from common.data_transformers import DataTransformer, DataTransformerFactory  # noqa


class DataExtractor(abc.ABC):
    """Abstract class for data Extractors."""

    CHUNKSIZE: int = 10000

    @staticmethod
    def get_file_paths(directory: str) -> List[str]:
        """ Iterate over a directory recursively and return a list of paths. """  # noqa
        file_paths = []
        for root, dirs, files in os.walk(directory):  # Pylint: disable=W0612
            for file in files:
                file_paths.append(os.path.join(root, file))
        return file_paths

    @abc.abstractmethod
    def load(self, source):
        """ Load data from source."""
        raise NotImplementedError()


class ZipDataExtractor(DataExtractor):
    """DataExtractor for ZIP files."""

    def load(self, source: str) -> Union[List[str], None]:
        """Load data from zip file."""
        if source.endswith(".zip"):
            tmp_folder = "./tmp"
            extracted_files = []
            with zipfile.ZipFile(source, 'r') as zip_ref:
                for file in zip_ref.namelist():
                    # if not os.path.exists(os.path.join(tmp_folder, file)):
                    extracted_files.append(zip_ref.extract(file, tmp_folder))

            return extracted_files
        return None


class PickleDataExtractor(DataExtractor):
    """DataExtractor for Pickle files."""
    def load(self, source: str):
        """Unpickles the file and returns it. """
        with open(source, 'rb') as opened_file:
            return pickle.load(opened_file)


class CSVDataExtractor(DataExtractor):
    """DataExtractor for CSV files."""
    def load(self, source: str) -> pd.DataFrame:
        """Reads each CSV file into a pandas DataFrame and returns it. """
        return pd.read_csv(source)


class TextDataExtractor(DataExtractor):
    """DataExtractor for CSV files."""
    def load(self, source: str) -> str:
        """Reads each CSV file into a pandas DataFrame and returns it. """
        with open(source, 'r') as opened_file:
            return opened_file.read()


class DataExtractorFactory:
    """Factory for creating DataExtractor instances based on the data source. """  # noqa
    @staticmethod
    def create(source: str):
        """ Create and return a dataextractor based on the extion of the input file. """  # noqa
        if source.endswith('.zip'):
            return ZipDataExtractor()
        elif source.endswith('.pickle'):
            return PickleDataExtractor()
        elif source.endswith('.csv'):
            return CSVDataExtractor()
        elif source.endswith('.html') or source.endswith('.txt'):
            return TextDataExtractor()
        

        raise ValueError("The provided source does not have a valid file extension.")  # noqa


class OutputFormat(Enum):
    """ This class defines the possible outputs for the Extractor. """
    DICT = "DICT"
    LIST = "LIST"
    
    @staticmethod
    def formatFactory(format: str) -> Callable:
        """ format the data. """
        format_functions = {
            OutputFormat.DICT: OutputFormat.format_as_dict,
            OutputFormat.LIST: OutputFormat.format_as_list,
        }
        
        if format in format_functions:
            return format_functions[format]
        else:
            raise ValueError(f"Invalid format type: {format}. Valid formats are {list(format_functions.keys())}")  # noqa
        
    @staticmethod
    def format_as_dict(sources: List[str], data: List[str]) -> Dict[str, str]:
        """ Format data as dictionary. """
        result_dict = {}
        for source, data in zip(sources, data):
            if isinstance(data, dict):
                # The data from {source} is already in dictionary format.
                result_dict.update(data)
            else:
                result_dict[source] = data
        return result_dict
    
    @staticmethod
    def format_as_list(sources: List[str], data: List[str]) -> List[str]:
        """ Format data as list. """
        return data


class Extractor:
    """Class responsible for loading and processing data from multiple sources."""  # noqa

    def load_processed_data(self,
                            sources: List[str],
                            output_format: Optional[OutputFormat] = OutputFormat.DICT,  # noqa
                              data_extractor: DataExtractor = None  # noqa
                            ) -> List[str]:
        """Loads and processes data from all sources, and returns a list of the processed data."""  # noqa
        processed_data = []
        forced_data_extractor = data_extractor is not None
        
        for source in sources:
            if not forced_data_extractor:
                data_extractor: DataExtractor = DataExtractorFactory.create(source=source)  # noqa
            data = data_extractor.load(source)

            # Handle compressed files
            if isinstance(data, list) and all(os.path.exists(str(path)) for path in data):  # noqa
                processed_data.append(self.load_processed_data(sources=data))
            else:
                transformer: DataTransformer = DataTransformerFactory.create(source=source)  # noqa
                transformed_data = transformer.transform(data)

                processed_data.append(transformed_data)

        # Get output formatter
        formatter = OutputFormat.formatFactory(
            format=output_format
        )
        # Format output before returning
        return formatter(sources=sources, data=processed_data)
