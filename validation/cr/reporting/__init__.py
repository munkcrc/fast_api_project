from .report import Report, Chapter, Section
from .writers import LatexWriter


def ref(output):
    if output.has_source:
        return f"!<REF;{output._recording_uid}.{output._output_key}>!"
    return f"!<REF;UNRECORDED>!"

def val(output):
    if output.has_source:
        return f"!<RES;{output._recording_uid}.{output._output_key};{output.output_type.name}>!"
    return f"!<RES;UNRECORDED>!"