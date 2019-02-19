import os
import pickle
import random
from pathlib import Path
from typing import List, Dict, Union, Callable

from flair.data import TaggedCorpus
from tqdm import tqdm

from classyflair.classes import ClassySentence


def _convert_to_sentences(parsed_dataset: List[Dict],
                          tokenizer: Callable[[str], List[Dict]]) -> List[ClassySentence]:
    """
    Takes a list of dict with keys 'labels' and 'text'
    and returns a list of Sentences
    """
    return [
        ClassySentence(text=datapoint['text'], tokenizer=tokenizer, labels=datapoint['labels'])
        for datapoint in tqdm(parsed_dataset)
    ]


def save_corpus(corpus: TaggedCorpus,
                dataset_dir: Union[str, Path]):
    """
    Pickle a Tagged Corpus
    """
    directory = os.path.join('datasets', dataset_dir)
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    save_path = os.path.join(directory, 'dataset.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(corpus, file)

    print()
    print(f'Saved TaggedCorpus to {save_path}')
    print()


def load_corpus(dataset_dir: Union[str, Path]):
    """
    Unpickle a Tagged Corpus
    """
    load_file_path = os.path.join('datasets', dataset_dir, 'dataset.pkl')
    with open(load_file_path, 'rb') as file:
        corpus = pickle.load(file)

    print()
    print(f'Loaded TaggedCorpus from {load_file_path}')
    print()

    return corpus


def create_corpus(
        dataset: List[Dict],
        tokenizer: Callable[[str], List[Dict]],
        train_percent: float = 0.7,
        dev_percent: float = 0.1,
        seed: int = 1234) -> TaggedCorpus:
    """
    Adapted from:
    https://github.com/zalandoresearch/custom-flair-classifier/blob/
    4b3562eed534e4e305a36d2f22d12961bc6c25d5/custom-flair-classifier/data_fetcher.py#L328
    """
    sentences = _convert_to_sentences(dataset, tokenizer)

    random.seed(seed)
    random.shuffle(sentences)

    length = len(sentences)
    train_index = int(train_percent * length)
    dev_index = int(dev_percent * length)

    sentences_train = sentences[:train_index]
    sentences_dev = sentences[train_index:train_index + dev_index]
    sentences_test = sentences[train_index + dev_index:]

    print()
    print(f'{length} total labeled Sentences.')
    print(f'{train_index} Sentences for training at {100 * train_percent}% of total.')
    print(f'{dev_index} Sentences for development at {100 * dev_percent}% of total.')
    print(f'{length - train_index - dev_index} Sentences for '
          f'testing at {round(100 * (1 - dev_percent - train_percent), 1)}% of total.')
    print()

    return TaggedCorpus(sentences_train, sentences_dev, sentences_test)
