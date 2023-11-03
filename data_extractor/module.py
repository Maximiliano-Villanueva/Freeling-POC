# Standard imports
import abc
import os
import pickle

from typing import Dict, List
from zipfile import ZipFile

# Third party imports
import pandas as pd
import pdfplumber


class DataExtractor(abc.ABC):
    """Abstract base class for data extractors."""

    CHUNKSIZE: int = 10000

    @abc.abstractmethod
    def extract(self):
        """Load data."""
        raise NotImplementedError

    @staticmethod
    def get_file_paths(directory: str) -> List[str]:
        """Iterate over a directory recursively and return a list of paths."""
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_paths.append(os.path.join(root, file))
        return file_paths


class ZipDataExtractor(DataExtractor):
    """A data extractor for zip files."""

    @staticmethod
    def extract_file(file):
        """Load data from a single file."""
        ext = os.path.splitext(file)[1]
        if ext == '.csv':
            return pd.read_csv(file, chunksize=DataExtractor.CHUNKSIZE)
        elif ext == '.pickle':
            with open(file, 'rb') as pickle_file:
                return pickle.load(pickle_file)
        else:
            return None

    def extract(self, filepath: str) -> Dict[str, str]:
        """Load data from zip file."""
        if filepath.endswith(".zip"):
            tmp_folder = "./tmp"
            extracted_files = []
            with ZipFile(filepath, 'r') as zip_ref:
                for file in zip_ref.namelist():
                    extracted_files.append(zip_ref.extract(file, tmp_folder))

            data = {}
            for file in extracted_files:
                data[file] = self.extract_file(file)

            return data


class PdfDataExtractor(DataExtractor):
    """A data extractor for PDF files."""

    def extract(self, filepath: str) -> List[str]:
        """Load data from PDF file."""
        if filepath.endswith('.pdf'):
            with pdfplumber.open(filepath) as pdf:
                text_data = list()
                for page in pdf.pages:
                    text_data.append(page.extract_text())
            return text_data
