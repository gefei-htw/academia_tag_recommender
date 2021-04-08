class Document:
    """This is a class representation of a document.

    Attributes:
        title: The documents title as string.
        body: The documents body as string.
        text: A combination of the documents title and body as string.
        tags: A list of the documents tags as list of strings.
    """

    def __init__(self, title, body, tags):
        """Constructor method"""
        self.title = title
        self.body = body
        self.text = title + ' ' + body
        self.tags = _format_tags(tags)

    def __repr__(self):
        """Representation method"""
        return '[title: {}, body: {}, text: {}, tag: {}]'.format(self.title, self.body, self.text, self.tags)

    def __str__(self):
        """String method"""
        return '[title: {}, body: {}, text: {}, tag: {}]'.format(self.title, self.body, self.text, self.tags)


def _format_tags(tags):
    """Reformats a string of tags.

    Extracts a list of strings from a string in diamond notation.

    Args:
        tags: Tags in diamond notation as one string.

    Returns: 
        A list of tags as list of strings.
        Example:

        ['publications', 'online-publications]
    """
    tags_ = tags.replace('<', '').split('>')
    return tags_[0:(len(tags_)-1)]
