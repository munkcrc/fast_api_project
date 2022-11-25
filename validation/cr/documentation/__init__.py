from .doc import Doc


def get_doc(func):
    if hasattr(func, "_cr_doc"):
        return func._cr_doc
    return None


# Documentation
# TODO: Can we instead just resolve documentation by using the 
#       automation recording_uid to identify the function 
#       and pull doc -> as opposed to this where we attach everywhere 
def doc(documentation, output_docs=None):
    def documentation_wrapper(func):
        def doc_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if output_docs:
                for output in result.outputs:
                    if output in output_docs:
                        result[output]._cr_doc = Doc(output_docs[output])
            result._cr_doc = Doc(documentation)    
            return result

        doc_wrapper._wrapped_func = func
        doc_wrapper._cr_doc = Doc(documentation)
        return doc_wrapper

    return documentation_wrapper