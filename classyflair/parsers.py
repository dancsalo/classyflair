import os

from classyflair.classes import ClassyParser


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
        Outputs a list of dict with keys 'labels' and 'text'
        """
        article_file_paths = [
            article.path
            for article in os.scandir(self.load_file_dir)
            if article.path[-6:] == '_1.txt'
        ]
        return [
            datapoint
            for file_path in article_file_paths
            for datapoint in self._parse_file(file_path)
        ]


class Sst1(ClassyParser):

    def __init__(self, *args, **kwargs):
        super(Sst1, self).__init__(*args, **kwargs)

    def load_file(self, split: str):
        data = []
        path = os.path.join(self.load_file_dir, f'stsa.binary.{split}')
        with open(path) as f:
            for row in f.readlines():
                label, sentence = row.split(' ', 1)
                data.append({'labels': [label], 'text': sentence.strip()})
        return data

    def parse(self):
        """
        Outputs a list of dict with keys 'labels' and 'text'
        """
        train = self.load_file('train')
        dev = self.load_file('dev')
        test = self.load_file('test')
        return {'train': train, 'dev': dev, 'test': test}
