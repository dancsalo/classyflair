import logging
from pathlib import Path
from typing import Union, List, Dict, Callable

import spacy
import torch
from flair.data import Sentence, Dictionary, Label, Token, TaggedCorpus
from flair.embeddings import Embeddings, DocumentLSTMEmbeddings, CharacterEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric, log_line, clear_embeddings

nlp = spacy.load('en_core_web_sm')
LOG = logging.getLogger('flair')


class ClassyParser:

    def __init__(self, load_file_dir):
        self.load_file_dir = load_file_dir

    @classmethod
    def __str__(cls):
        return cls.__name__

    def parse(self):
        raise NotImplementedError

    @staticmethod
    def tokenizer(text: str):
        """Custom Tokenizer the uses spacy"""
        return [{'text': token.text} for token in nlp(text)]


class ClassyCorpus(TaggedCorpus):

    def __init__(self, sentences_train: List[Sentence],
                 sentences_dev: List[Sentence],
                 sentences_test: List[Sentence],
                 multi_label: bool):
        self.multi_label = multi_label
        super(ClassyCorpus, self).__init__(sentences_train, sentences_dev, sentences_test)


class ClassyClassifier(TextClassifier):
    """
    Adapted from:
    https://github.com/zalandoresearch/flair/blob/
    4b3562eed534e4e305a36d2f22d12961bc6c25d5/flair/models/text_classification_model.py#L19
    """

    def __init__(self,
                 embeddings: List[Embeddings],
                 label_dictionary: Dictionary,
                 multi_label: bool = False):
        # Code to trick TextClassifier + Flair and make them Happy :)
        document_embeddings = DocumentLSTMEmbeddings([CharacterEmbeddings()],
                                                     hidden_size=512,
                                                     reproject_words=True,
                                                     reproject_words_dimension=256,
                                                     )
        super(ClassyClassifier, self).__init__(document_embeddings,
                                               label_dictionary,
                                               multi_label)

        # Start other init code
        self.embeddings = embeddings
        self.embedder = self.init_embedder()
        self.embedding_length = self.embedder.embedding_length
        self.init_layers()

    def get_scores(self, sentence):
        """
        Predicts the class labels for the given sentences. The labels are directly added to the sentences.
        :param sentences: list of sentences
        :param mini_batch_size: mini batch size to use
        :return: the list of sentences containing the labels
        """
        with torch.no_grad():
            batch = [sentence]
            scores = self.forward(batch)
            clear_embeddings(batch)
            return scores

    def save(self, model_file: Union[str, Path]):
        """
        Saves the current model to the provided file.
        :param model_file: the model file
        """
        model_state = {
            'state_dict': self.state_dict(),
            'embeddings': self.embeddings,
            'label_dictionary': self.label_dictionary,
            'multi_label': self.multi_label,
        }
        torch.save(model_state, str(model_file), pickle_protocol=4)

    def save_checkpoint(self, model_file: Union[str, Path],
                        optimizer_state: dict,
                        scheduler_state: dict,
                        epoch: int,
                        loss: float):
        """
        Saves the current model to the provided file.
        :param model_file: the model file
        """
        model_state = {
            'state_dict': self.state_dict(),
            'embeddings': self.embeddings,
            'label_dictionary': self.label_dictionary,
            'multi_label': self.multi_label,
            'optimizer_state_dict': optimizer_state,
            'scheduler_state_dict': scheduler_state,
            'epoch': epoch,
            'loss': loss
        }
        torch.save(model_state, str(model_file), pickle_protocol=4)

    @classmethod
    def load_from_file(cls, model_file: Union[str, Path]):
        """
        Loads the model from the given file.
        :param model_file: the model file
        :return: the loaded text classifier model
        """
        state = cls._load_state(model_file)

        model = cls(
            embeddings=state['embeddings'],
            label_dictionary=state['label_dictionary'],
            multi_label=state['multi_label']
        )
        model.load_state_dict(state['state_dict'])
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        return model

    def init_embedder(self) -> Embeddings:
        """Initializes the sentence embedding object"""
        raise NotImplementedError

    def init_layers(self):
        """Initializes any layers to be used in forward()"""

    def forward(self, sentences: Union[List[Sentence], Sentence]) -> torch.Tensor:
        """Defines one forward pass of the neural net"""
        raise NotImplementedError


class ClassyTrainer(ModelTrainer):

    def __init__(self, *args, **kwargs):
        super(ClassyTrainer, self).__init__(*args, **kwargs)

    def final_test(self,
                   base_path: Path,
                   embeddings_in_memory: bool,
                   evaluation_metric: EvaluationMetric,
                   eval_mini_batch_size: int):

        log_line(LOG)
        LOG.info('Testing using best model ...')

        self.model.eval()

        if (base_path / 'best-model.pt').exists():
            self.model = self.model.load_from_file(base_path / 'best-model.pt')

        test_metric, test_loss = self.evaluate(self.model,
                                               self.corpus.test,
                                               eval_mini_batch_size=eval_mini_batch_size,
                                               embeddings_in_memory=embeddings_in_memory)

        LOG.info(f'MICRO_AVG: acc {test_metric.micro_avg_accuracy()} -'
                 f'f1-score {test_metric.micro_avg_f_score()}')
        LOG.info(f'MACRO_AVG: acc {test_metric.macro_avg_accuracy()} -'
                 f'f1-score {test_metric.macro_avg_f_score()}')
        for class_name in test_metric.get_classes():
            LOG.info(f'{class_name:<10}'
                     f'tp: {test_metric.get_tp(class_name)} -'
                     f'fp: {test_metric.get_fp(class_name)} - '
                     f'fn: {test_metric.get_fn(class_name)} -'
                     f'tn: {test_metric.get_tn(class_name)} -'
                     f'precision: {test_metric.precision(class_name):.4f} -'
                     f'recall: {test_metric.recall(class_name):.4f} - '
                     f'accuracy: {test_metric.accuracy(class_name):.4f} -'
                     f'f1-score: {test_metric.f_score(class_name):.4f}')
        log_line(LOG)

        # get and return the final test score of best model
        if evaluation_metric == EvaluationMetric.MACRO_ACCURACY:
            final_score = test_metric.macro_avg_accuracy()
        elif evaluation_metric == EvaluationMetric.MICRO_ACCURACY:
            final_score = test_metric.micro_avg_accuracy()
        elif evaluation_metric == EvaluationMetric.MACRO_F1_SCORE:
            final_score = test_metric.macro_avg_f_score()
        else:
            final_score = test_metric.micro_avg_f_score()

        return final_score


class ClassySentence(Sentence):
    """
    A Sentence is a list of Tokens and is used to represent a sentence or text fragment.
    """

    def __init__(self, text: str,
                 tokenizer: Callable,
                 labels: Union[List[Label], List[str]] = None):

        self.labels: List[Label] = []
        if labels is not None:
            self.add_labels(labels)

        self._embeddings: Dict = {}

        self.tokens: List[Token] = []

        # determine offsets for whitespace_after field
        index = text.index
        running_offset = 0
        last_word_offset = -1
        last_token = None
        for word in [token for token in tokenizer(text)]:
            try:
                word_offset = index(word['text'], running_offset)
                start_position = word_offset
            except:
                word_offset = last_word_offset + 1
                start_position = running_offset + 1 if running_offset > 0 else running_offset

            token = Token(word['text'], start_position=start_position)
            self.add_token(token)

            if word_offset - 1 == last_word_offset and last_token is not None:
                last_token.whitespace_after = False

            word_len = len(word['text'])
            running_offset = word_offset + word_len
            last_word_offset = running_offset - 1
            last_token = token
