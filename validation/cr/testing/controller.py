from contextlib import contextmanager

class PipeStep(object):
    def __init__(self, compare):
        self.compare=compare

class SampleStep(PipeStep):
    def __init__(self, compare, samples):
        super().__init__(compare)
        if isinstance(samples, list):
            self.samples = samples
        else: # Potentially check that it is a dataset
            self.samples = [samples]

    def pipe(self, input=None):
        for sample in self.samples:
            yield sample
        return self.samples

class SegmentSelectStep(PipeStep):
    def __init__(self, key, value):
        super().__init__(False)
        self.key = key
        self.value = value

    def pipe(self, input):
        for dataset in input:
            yield dataset.get_segment(self.key, self.value)

class SegmentSplitStep(PipeStep):
    def __init__(self, compare, by, ordinal, nominal_strategy):
        super().__init__(compare)
        self.by = by
        self.ordinal = ordinal
        self.nominal_strategy = nominal_strategy

    def pipe(self, input):
        for dataset in input:
            for segment in dataset.get_segments(self.by, self.ordinal, self.nominal_strategy):
                yield segment

class Tester(object):

    def __init__(self):
        self._results = {}
        self._pipe = []

    @contextmanager
    def to_pipe(self, step):
        try:
            self._decompositions.append(step)
        finally:
            # Remove the last decomposition stack
            self._decompositions.pop()

    @property
    def _datasets(self):
        if not isinstance(self._pipe[0], SampleStep):
            raise ValueError("The Tester has no set sample")

        datasets = None
        for steps in self._pipe:
            datasets = steps.pipe(datasets)

        return datasets

    @property
    def describe_testing(self):
        for dataset in self._datasets:
            print(dataset.id)

    # Sampling logic
    def in_sample(self, sample):
        return self.to_pipe(SampleStep(False, sample))

    def between_samples(self, samples):
        return self.to_pipe(SampleStep(True, samples))

    def by_samples(self, samples):
        return self.to_pipe(SampleStep(False, samples))

    # Segmentation logic
    def in_segment(self, key, value):
        return self.to_pipe(SegmentSelectStep(key, value))

    def between_segments(self, key, ordinal=False, ordinal_strategy=None):
        return self.to_pipe(SegmentSplitStep(compare=True, by=key, ordinal=ordinal, nominal=ordinal_strategy))

    def by_segments(self, key, ordinal=False, ordinal_strategy=None):
        return self.to_pipe(SegmentSplitStep(compare=False, by=key, ordinal=ordinal, nominal=ordinal_strategy))

    # Testing logic
    def factor_test(self, test, factor_name):
        pass
        # DATASET
        # UDVÆLG DATA
        # KØR TEST