import os.path

from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.lib.utils import display_bayesian_network


def data_synthesizer_random_generator(data_in_csv, num_rows, description_file='description.json'):
    describer = DataDescriber(category_threshold=10)
    describer.describe_dataset_in_random_mode(data_in_csv)
    if os.path.isfile(description_file):
        os.remove(description_file)
    describer.save_dataset_description_to_file(description_file)

    generator = DataGenerator()
    generator.generate_dataset_in_random_mode(num_rows, description_file)
    return generator.synthetic_dataset


def data_synthesizer_bayes_generator(data_in_csv, num_rows, description_file='description.json'):
    print("Please, use formula: if __name__ == '__main__':")

    describer = DataDescriber(category_threshold=10)
    describer.describe_dataset_in_correlated_attribute_mode(dataset_file=data_in_csv, k=3)

    if os.path.isfile(description_file):
        os.remove(description_file)
    describer.save_dataset_description_to_file(description_file)

    display_bayesian_network(describer.bayesian_network)

    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(num_rows, description_file)
    return generator.synthetic_dataset
