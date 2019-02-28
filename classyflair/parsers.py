import os

import spacy

from classyflair.classes import ClassyParser

nlp = spacy.load('en_core_web_sm')


class SciArticles(ClassyParser):

    def __init__(self, *args, **kwargs):
        super(SciArticles, self).__init__(*args, **kwargs)

    @staticmethod
    def _process_line(_line):
        """Handles inconsistencies in line spliting"""
        if '\t' in _line:
            values = _line.split('\t')
        elif '--' in _line:
            values = _line.split('--', 1)
        else:
            values = _line.split(' ', 1)
        return (v.strip() for v in values[:2])

    def _parse_file(self, file_path):
        """Generator individual datapoints from one file"""
        with open(file_path) as f:
            for line in filter(lambda l: l[0] != '#', f.readlines()):
                label, text = self._process_line(line)
                yield {'labels': [label], 'text': text}

    def parse(self):
        """
        Generators a dict with keys 'labels' and 'text'
        """
        article_file_paths = [
            article.path
            for article in os.scandir(self.load_file_dir)
            if article.path[-4:] == '.txt'
        ]
        return [
            datapoint
            for file_path in article_file_paths
            for datapoint in self._parse_file(file_path)
        ]

    @staticmethod
    def tokenizer(text: str):
        """Custom Tokenizer the uses spacy"""
        return [{'text': token.text} for token in nlp(text)]
