from cr.data.dataset import DataSet
from cr.data.ingestion import from_excel
from cr.data.segmentation import ByGroup
from cr.testing.metric.representativeness import psi_numerical
from cr.testing.metric.performance.discriminatory_power import gini
import numpy as np
import pandas as pd

from cr.automation import record, get_session_tape

with record():
    dataset = from_excel(r"C:\Users\Rasmu\OneDrive\Mansen ApS\Projects\Sparnord\OLD2\lgl_gen_mst.xlsx")
    segments_mdl_segment = dataset.segment("modelsegment", ByGroup())
    segments_afsluttet = dataset.segment("AFSLUTTET", ByGroup())
    segments = segments_mdl_segment.composite_with(segments_afsluttet)
    EAD_GINI = gini(dataset["EAD"], dataset["LGL_INKAS"])
    psi_result = psi_numerical(segments[0]["LGL_INKAS"], segments[1]["LGL_INKAS"])

from cr.reporting import Report, Chapter, Section, LatexWriter, val, ref
from cr.reporting.writers.latex.template import CR
from cr.plotting.plotly import Theme
theme = Theme().set_as_default()

report = Report("Validation Report", [
    Chapter("Data Quality", [
        Section(
            "Distribution", f"""
Between Privat and Erhverv the PSI result is {val(psi_result["value"])} for LTV. This gives it the color {val(psi_result["color"])} as {val(psi_result["reasoning"])}
 """
        )
    ]),
    Chapter("Performance", [
        Section(
            "GINI - in sample", f"""
The overall performance is poor, with a gini of {val(EAD_GINI["value"])}.
As can be seen from the cap curve {ref(EAD_GINI["cap_curve"])}:
{val(EAD_GINI["cap_curve"])}
        """)
    ])
])

report.context['SUMMARY'] = "This is the report summary"

with open("tape.yaml", 'w') as f:
    get_session_tape().to_yaml(f)

with open("report.yaml", 'w') as f:
    report.to_yaml(f)