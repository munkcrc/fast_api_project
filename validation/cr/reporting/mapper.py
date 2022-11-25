import re
from cr.testing.output import OutputType

class ContentMapper(object):

    def __init__(self, runner, context):
        self.regex = re.compile("\!\<(.*?)\>\!")
        self.runner = runner
        self.context = context
        self.tag_maps = {
            "CTX": self.map_context,
            "RES": self.map_result
        }

    def map(self, content):
        return re.sub(self.regex, lambda x: self.map_content_piece(x.groups()[0]), content)

    def unknown_tag_type(tag_type, tag):
        raise Exception(f"Unable to write tag with tag_type: {tag_type}")

    def map_content_piece(self, tag):
        tag_type, tag = tag.split(";", 1)
        if tag_type in self.tag_maps:
            return self.tag_maps[tag_type](tag)
        return self.unknown_tag_type(tag_type, tag)

    def map_context(self, context_tag):
        if context_tag in self.context:
            return self.context[context_tag]
        else:
            # TODO: Raise warning
            return f"<<{context_tag}>>"
        
    def map_result(self, tag):
        full_source, rules = tag.split(";", 1)
        test_uid, output_name = full_source.split(".")

        # Recompute the output/result
        result = self.runner.run(test_uid)
        output = result[output_name]

        return self._map_output(output, test_uid, output_name, tag)

    def _map_output(self, output, test_uid, output_name, tag):
        if output.output_type == OutputType.FIGURE:
            return "!FIGURE:{output_name}!"
        else:
            return str(output.value)