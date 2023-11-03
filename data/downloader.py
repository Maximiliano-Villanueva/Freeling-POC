"""
A class to download files from different sources.

Attributes
----------
None

Methods
-------
download_zip(url, path)
    Downloads a zip file from the given URL and extracts it to the given path.
download_csv(csv_path, url_column, dest_folder)
    Loads a CSV file from the given path and downloads each document in the URL column to the given destination folder.
"""

# Standard imports
import os
import zipfile

# Third party libraries
from bs4 import BeautifulSoup
import pandas as pd
import requests


class Downloader:
    """
    A class to download files from different sources.

    Attributes
    ----------
    None

    Methods
    -------
    download_zip(url, path)
        Downloads a zip file from the given URL and extracts it to
        the given path.
    download_csv(csv_path, url_column, dest_folder)
        Loads a CSV file from the given path and downloads each document in
        the URL column to the given destination folder.
    """

    @staticmethod
    def download_zip(url, path):
        """
        Downloads a zip file from the given URL and extracts it
         to the given path.

        Parameters
        ----------
        url : str
            The URL of the zip file to download.
        path : str
            The path where the zip file should be extracted.

        Returns
        -------
        None
        """
        # Create the directory if it does not exist
        os.makedirs(path, exist_ok=True)

        # Download the zip file
        response = requests.get(url)
        with open(os.path.join(path, 'dataset.zip'), 'wb') as f:
            f.write(response.content)

        # Extract the zip file
        with zipfile.ZipFile(os.path.join(path, 'dataset.zip'), 'r') as zip_ref:  # noqa
            zip_ref.extractall(path)

    @staticmethod
    def download_csv(csv_path, url_column, dest_folder):
        """
        Loads a CSV file from the given path and downloads each document in
        the URL column to the given destination folder.

        Parameters
        ----------
        csv_path : str
            The path of the CSV file to load.
        url_column : str
            The name of the column in the CSV file that contains the
            URLs to download.
        dest_folder : str
            The path where the downloaded documents should be saved.

        Returns
        -------
        None
        """
        # Create the directories if they do not exist
        os.makedirs(dest_folder, exist_ok=True)

        # Load the CSV file
        df = pd.read_csv(csv_path)
        
        # Download each document in the URL column
        for article_id, url in zip(df['article_id'], df[url_column]):
            try:
                filename = f"{article_id} - {url.replace('/','_')}"
                filepath = os.path.join(dest_folder, filename)
                if not os.path.exists(filepath):
                    response = requests.get(url)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text = soup.get_text()
                    text = text.replace('\n', ' ')
                    with open(filepath, 'w') as f:
                        f.write(text)
            except:
                pass


downloader = Downloader()

# Download the zip file
# downloader.download_zip('https://files.osf.io/v1/resources/xqwa4/providers/osfstorage/62a7566ea8bdf51af26fb9ba/?zip=', './data/dataset1')  # noqa

# Download the documents from the CSV file
downloader.download_csv('./tmp/sentence_df_articleId_to_url.csv', 'url', './tmp/documents')  # noqa