from typing import Union, List

import torch
import torch.nn as nn
from flair.data import Sentence
from flair.embeddings import DocumentLSTMEmbeddings, StackedEmbeddings, Embeddings

from classyflair.classes import ClassyClassifier


class DocLSTM(ClassyClassifier):

    def __init__(self, *args, **kwargs):
        super(DocLSTM, self).__init__(*args, **kwargs)

    def init_embedder(self) -> Embeddings:
        """Initializes the sentence embedding object"""
        embedder = DocumentLSTMEmbeddings(self.embeddings,
                                          hidden_size=512,
                                          reproject_words=True,
                                          reproject_words_dimension=256,
                                          )
        return embedder

    def init_layers(self):
        """Initializes the layers to be used in forward()"""
        self.decoder = nn.Linear(self.embedding_length, len(self.label_dictionary))

    def forward(self, sentences: Union[List[Sentence], Sentence]) -> torch.Tensor:
        """Defines one forward pass of the neural net"""
        self.embedder.embed(sentences)

        text_embedding_list = [sentence.get_embedding().unsqueeze(0) for sentence in sentences]
        text_embedding_tensor = torch.cat(text_embedding_list, 0)

        label_scores = self.decoder(text_embedding_tensor).squeeze()
        return label_scores


class OneLayerMLP(ClassyClassifier):

    def __init__(self, *args, **kwargs):
        super(OneLayerMLP, self).__init__(*args, **kwargs)

    def init_embedder(self) -> Embeddings:
        """Initializes the sentence embedding object"""
        return StackedEmbeddings(self.embeddings)

    def init_layers(self):
        """Initializes the layers to be used in forward()"""
        self.decoder = nn.Linear(self.embedding_length, len(self.label_dictionary))

    def forward(self, sentences: Union[List[Sentence], Sentence]) -> torch.Tensor:
        """Defines one forward pass of the neural net"""
        self.embedder.embed(sentences)

        # Average
        text_embedding_tensors = torch.stack(
            [torch.stack([word.embedding for word in sentence]).mean(dim=0)
             for sentence in sentences])

        label_scores = self.decoder(text_embedding_tensors).squeeze()
        return label_scores


class TwoLayerMLP(ClassyClassifier):

    def __init__(self, *args, **kwargs):
        super(TwoLayerMLP, self).__init__(*args, **kwargs)

    def init_embedder(self) -> Embeddings:
        """Initializes the sentence embedding object"""
        return StackedEmbeddings(self.embeddings)

    def init_layers(self):
        """Initializes the layers to be used in forward()"""
        hidden = 128
        self.decoder = torch.nn.Sequential(
                torch.nn.Linear(self.embedding_length, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, len(self.label_dictionary)),
            )

    def forward(self, sentences: Union[List[Sentence], Sentence]) -> torch.Tensor:
        """Defines one forward pass of the neural net"""
        self.embedder.embed(sentences)

        # Average
        text_embedding_tensors = torch.stack(
            [torch.stack([word.embedding for word in sentence]).mean(dim=0)
             for sentence in sentences])

        label_scores = self.decoder(text_embedding_tensors).squeeze()
        return label_scores


class TwoLayerMLPdp(ClassyClassifier):

    def __init__(self, *args, **kwargs):
        super(TwoLayerMLPdp, self).__init__(*args, **kwargs)

    def init_embedder(self) -> Embeddings:
        """Initializes the sentence embedding object"""
        return StackedEmbeddings(self.embeddings)

    def init_layers(self):
        """Initializes the layers to be used in forward()"""
        hidden = 128
        self.decoder = torch.nn.Sequential(
                torch.nn.Linear(self.embedding_length, hidden),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(hidden, len(self.label_dictionary))
            )

    def forward(self, sentences: Union[List[Sentence], Sentence]) -> torch.Tensor:
        """Defines one forward pass of the neural net"""
        self.embedder.embed(sentences)

        # Average
        text_embedding_tensors = torch.stack(
            [torch.stack([word.embedding for word in sentence]).mean(dim=0)
             for sentence in sentences])

        label_scores = self.decoder(text_embedding_tensors).squeeze()
        return label_scores


class TwoLayerMLPbn(ClassyClassifier):

    def __init__(self, *args, **kwargs):
        super(TwoLayerMLPbn, self).__init__(*args, **kwargs)

    def init_embedder(self) -> Embeddings:
        """Initializes the sentence embedding object"""
        return StackedEmbeddings(self.embeddings)

    def init_layers(self):
        """Initializes the layers to be used in forward()"""
        hidden = 128
        self.decoder = torch.nn.Sequential(
                torch.nn.Linear(self.embedding_length, hidden),
                torch.nn.BatchNorm1d(hidden, affine=False),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, len(self.label_dictionary)),
            )

    def forward(self, sentences: Union[List[Sentence], Sentence]) -> torch.Tensor:
        """Defines one forward pass of the neural net"""
        self.embedder.embed(sentences)

        # Average
        text_embedding_tensors = torch.stack(
            [torch.stack([word.embedding for word in sentence]).mean(dim=0)
             for sentence in sentences])

        label_scores = self.decoder(text_embedding_tensors).squeeze()
        return label_scores


class TwoLayerMLPbndp(ClassyClassifier):

    def __init__(self, *args, **kwargs):
        super(TwoLayerMLPbndp, self).__init__(*args, **kwargs)

    def init_embedder(self) -> Embeddings:
        """Initializes the sentence embedding object"""
        return StackedEmbeddings(self.embeddings)

    def init_layers(self):
        """Initializes the layers to be used in forward()"""
        hidden = 128
        self.decoder = torch.nn.Sequential(
                torch.nn.Linear(self.embedding_length, hidden),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hidden),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(hidden, len(self.label_dictionary)),
            )

    def forward(self, sentences: Union[List[Sentence], Sentence]) -> torch.Tensor:
        """Defines one forward pass of the neural net"""
        self.embedder.embed(sentences)

        # Average
        text_embedding_tensors = torch.stack(
            [torch.stack([word.embedding for word in sentence]).mean(dim=0)
             for sentence in sentences])

        label_scores = self.decoder(text_embedding_tensors).squeeze()
        return label_scores

# Add additional models below #
