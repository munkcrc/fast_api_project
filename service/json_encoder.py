import json

import numpy as np
import pandas
from cr.testing.result import Result
from cr.testing.output import Output, OutputType
from plotly.graph_objects import Figure
from cr.documentation import Doc


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Result):
            return vars(obj)
        elif isinstance(obj, Output):
            obj_dict = {'value': obj.value,
                        'formatted_value': obj.formatted_value,
                        'output_type': obj.output_type,
                        'has_source': obj.has_source
                        }
            obj_dict.update(vars(obj))
            return obj_dict
        elif isinstance(obj, OutputType):
            return {'name': obj.name}
        elif isinstance(obj, Figure):
            return obj.to_dict()
        elif isinstance(obj, Doc):
            return obj.doc_string
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, float) and np.isnan(obj):
            return str(obj)
        elif isinstance(obj, pandas.DataFrame):
            return obj.to_dict(orient='index')
        return json.JSONEncoder.default(self, obj)

