from cr.data.dataset import DataSet
from cr.testing.metric.representativeness import psi
from cr.testing.controller import Tester
import numpy as np
import pandas as pd

def generate_data(obs, id):
    df = pd.DataFrame(np.random.binomial(1, 0.73, [obs,1]), columns=["cr"])
    df['lgl'] = np.random.uniform(0, 1.5, [obs,1])
    df["modpartstype"] = np.random.choice(["privat", "erhverv"], [obs,1])
    df["ltv"] = np.random.standard_normal([obs,1])
    df["pbs aftaler"] = np.random.randint(0, 10, [obs,1])
    df["default dato"] = np.datetime64('2016-01-01') + np.random.randint(0, 365*5, [obs,1])
    return DataSet(id, df,
        target_cols=["cr", "lgl"], 
        factor_cols=["pbs aftaler", "ltv"],
        segmentor_cols=['modpartstype', "default dato"]
        )

# Generate som random dataset
training_sample = generate_data(1000, "training_sample")
testing_sample = generate_data(400, "testing_sample")

### Running a test
result = psi(training_sample.get_factor("ltv"), testing_sample.get_factor("ltv"))
if result.passed:
    print("PSI test passed")

"""

### Core In Python -> Run test
tester = Tester()

with tester.between_samples(training_sample, testing_sample):
    tester.factor_test(PSITest, "ltv")
    tester.calibration_test(CAPTest, "prediction", "result")

with tester.between_samples(training_sample, testing_sample):
    with tester.on_segment({"modpartstype":"privat"}):
        tester.factor_test(PSITest, "ltv")

    with tester.between_segments("modpartstype"):
        tester.factor_test(PSITest, "ltv")

with tester.in_sample(training_sample).by_segments("default dato", ordinal=True, strategy="all_vs_all"):
    tester.factor_test(PSITest, "ltv")

with tester.by_segments(by="modpartstype", include_all=False):
    tester.factor_test(PSITest, "ltv")
"""