from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer


def sdv_copula_generator(data, num_rows):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)

    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)

    return synthesizer.sample(num_rows=num_rows)
