import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer


def sdv_ctgan_generator(data, num_rows):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)

    synthesizer = CTGANSynthesizer(metadata)
    synthesizer.fit(data)

    return synthesizer.sample(num_rows=num_rows)


df = pd.read_csv('../heart_uci.csv')

print(sdv_ctgan_generator(df, 500))
