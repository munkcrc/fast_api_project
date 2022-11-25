class Doc(object):
    """
    The documentation of a function/output
    """
    def __init__(self, doc_string):
        self.doc_string = doc_string

    def __str__(self):
        return self.doc_string
    
    def __repr__(self):
        return f"Doc({self.doc_string})"
