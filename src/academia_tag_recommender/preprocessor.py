"""This module handles text preprocessing."""

import re


class BasicPreprocessor:
    """This preprocessor changes all characters to lower case and removes html tags and punctuaion for a given text."""
    _html_re = re.compile(r'<[^>]+>')
    _punctuation_re = re.compile(r'[^\w\s]')

    def __call__(self, text):
        """Changes all characters to lower case and removes html tags and punctuaion for the given text.

        Args:
            text: The unprocessed :class:`str`.

        Returns:
            The processed :class:`str`.
        """
        return self._punctuation_re.sub(' ', self._html_re.sub(' ', text.lower()))
