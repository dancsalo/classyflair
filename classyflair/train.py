import os
from typing import Union, List, Dict, Callable, Any

from flair.data import TaggedCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, BertEmbeddings, Embeddings

from classyflair.classes import ClassyTrainer, ClassyClassifier

EMBEDS = {
    'word': WordEmbeddings,
    'flair': FlairEmbeddings,
    'bert': BertEmbeddings,
}


def _loop_embeddings(func: Callable[[str, Union[str, List]], Any],
                     embeddings: Dict):
    """Create list of embeddings from user input"""
    embedding = []
    for _type, embeds in embeddings.items():
        if isinstance(embeds, (list, tuple)):
            value = [func(_type, embed) for embed in embeds]
        elif isinstance(embeds, str):
            value = [func(_type, embeds)]
        else:
            raise Exception('Invalid embedding selector.')
        embedding.extend(value)
    return embedding


def get_embedding(embeddings: Dict) -> List[Embeddings]:
    """Get list of Flair embeddings"""
    return _loop_embeddings(lambda t, e: EMBEDS[t](e), embeddings)


def get_embedding_info(embeddings: Dict, embedding: List[Embeddings]) -> Dict:
    """Returns a dict of information on the embeddings"""
    embed_labels = _loop_embeddings(lambda t, e: f'{t}_{e}', embeddings)
    embed_size = sum([mbd.embedding_length for mbd in embedding])
    return {
        'labels': embed_labels,
        'size': embed_size
    }


def create_trainer(corpus: TaggedCorpus,
                   model: ClassyClassifier,
                   word_embeddings: Union[str, List],
                   flair_embeddings: Union[str, List],
                   bert_embeddings: Union[str, List],
                   multi_label: bool = False,
                   ):
    embeddings = {
        'word': word_embeddings,
        'flair': flair_embeddings,
        'bert': bert_embeddings
    }
    embedding = get_embedding(embeddings)
    embedding_info = get_embedding_info(embeddings, embedding)
    label_dict = corpus.make_label_dictionary()
    classifier = model(embedding, label_dict, multi_label)
    return ClassyTrainer(classifier, corpus)


def get_train_path(model_name: str, dataset_dir: str):
    """Generate path for ClassyTrainer to save models"""
    base_path = os.path.join('datasets', dataset_dir, 'models')

    if os.path.isdir(base_path):
        model_dirs = filter(lambda d: model_name in d.path and os.path.isdir(d.path),
                            os.scandir(base_path))
        numbers = set(int(folder.path.split('/')[-1].split('_')[1]) for folder in model_dirs)
        new_number = max(numbers) + 1 if numbers else 0
    else:
        new_number = 0

    model_dir = f'{model_name}_{str(new_number).zfill(3)}'
    train_path = os.path.join('datasets', dataset_dir, 'models', model_dir)
    os.makedirs(train_path)
    return train_path
