"""
Module to provide custom logging functionality.

This module includes a `Logger` class which simplifies the process of logging
in both console and file with specific formatting.
"""

import logging


class Logger:
    """
    Logger class for application debug, info and error messages.
    Usage example:

        logger = Logger('my_logger', 'log.txt').get_logger()

        def some_function() -> None:
            try:
                # Some computation here
                pass
            except Exception as error:
                logger.error("Error occurred: %s", str(error))

        # Function call
        some_function()

    """

    def __init__(self, logger_name: str, log_file: str) -> None:
        """Initialize Logger class.

        Args:
            logger_name (str): Name of the logger.
            log_file (str): File name where logs will be stored.
        """

        self.logger: logging.Logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        # Create a file handler
        file_handler: logging.FileHandler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Create a console handler
        console_handler: logging.StreamHandler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Create a formatter and add it to the handlers
        formatter: logging.Formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self) -> logging.Logger:
        """Getter for logger object.

        Returns:
            logging.Logger: logger object.
        """

        return self.logger
