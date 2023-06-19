from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer


def sdv_ctgan_generator(data, num_rows, metadata_file=None):

    if metadata_file is None:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)
        # metadata.save_to_json('maternal_metadata.json')
    else:
        metadata = SingleTableMetadata.load_from_json(filepath=metadata_file)
        # metadata.load_from_dict(metadata_file)

    synthesizer = CTGANSynthesizer(metadata)
    synthesizer.fit(data)

    return synthesizer.sample(num_rows=num_rows)
