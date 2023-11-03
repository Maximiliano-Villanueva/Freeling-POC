"""
This Python file provides data transformation utilities for multiple
file formats using appropriate DataTransformer classes and a factory
to create instances based on the data source.
"""
class DataTransformer:
    """Abstract class for data DataTransformers."""
    def transform(self, data: str) -> str:
        """ Transform input. """
        raise NotImplementedError()


class EmptyTransformer(DataTransformer):
    def transform(self, data: str) -> str:
        """ Transform input. """
        return data


class DataTransformerFactory:
    """Factory for creating DataTransformer instances
    based on the data source."""
    @staticmethod
    def create(source) -> DataTransformer:
        """ Genereate and return a DataTransformer based on file extension. """
        # Decide which DataTransformer to use based on the file extension.
        if True:  # if source.endswith('.pickle'):
            return EmptyTransformer()
        else:
            raise ValueError("The provided source does not have a valid file extension.")  # noqa
