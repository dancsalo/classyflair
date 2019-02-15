import argparse
import os

from flair.visual.training_curves import Plotter

import models
from classyflair import create_trainer, get_train_path, load_corpus, save_corpus, create_corpus


def _process_line(_line):
    """Handles inconsistencies in line spliting"""
    if '\t' in _line:
        values = _line.split('\t')
    elif '--' in _line:
        values = _line.split('--', 1)
    else:
        values = _line.split(' ', 1)
    return (v.strip() for v in values[:2])


def _parse_file(file_path):
    """Generator individual datapoints from one file"""
    with open(file_path) as f:
        for line in filter(lambda l: l[0] != '#', f.readlines()):
            label, text = _process_line(line)
            yield {'labels': [label], 'text': text}


def parse(load_file_dir):
    """
    Outputs list of dict with keys 'labels' and 'text'
    """
    article_file_paths = [
        article.path
        for article in os.scandir(load_file_dir)
        if article.path[-4:] == '.txt'
    ]
    return [
        datapoint
        for file_path in article_file_paths
        for datapoint in _parse_file(file_path)
    ]


def main():
    # Parse Args
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', type=str, default='arxiv_dataset')
    parser.add_argument('-d', type=str, default='sci_articles')
    parser.add_argument('-a', type=str, default='OneLayerMLP')
    parser.add_argument('-m', type=str, default='preprocess')
    parser.add_argument('-b', type=str, nargs='*', default=[])
    parser.add_argument('-f', type=str, nargs='*', default=[])
    parser.add_argument('-w', type=str, nargs='*', default=['glove'])
    parser.add_argument('-c', dest='char', action='store_true')
    parser.add_argument('-nc', dest='char', action='store_false')
    parser.add_argument('-e', type=int, default=20)
    parser.add_argument('-r', type=float, default=0.075)
    parser.set_defaults(char=False)
    args = vars(parser.parse_args())

    # Preprocess or Train
    if args['m'] == 'preprocess':
        dataset = parse(args['l'])
        corpus = create_corpus(dataset)
        if args['d']:
            save_corpus(corpus, args['d'])
    elif args['m'] == 'train':
        corpus = load_corpus(args['d'])
        model = getattr(models, args['a'])
        trainer = create_trainer(corpus,
                                 model,
                                 bert_embeddings=args['b'],
                                 flair_embeddings=args['f'],
                                 word_embeddings=args['w'])
        train_path = get_train_path(args['a'], args['d'])
        trainer.train(train_path,
                      learning_rate=args['r'],
                      mini_batch_size=32,
                      anneal_factor=0.5,
                      patience=5,
                      max_epochs=args['e'])
        plotter = Plotter()
        plotter.plot_training_curves(os.path.join(train_path, 'loss.tsv'))


if __name__ == '__main__':
    main()
