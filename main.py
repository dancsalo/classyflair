import argparse
import os

from flair.visual.training_curves import Plotter

from classyflair import create_trainer, get_train_path, load_corpus, save_corpus, create_corpus, parsers, models


def main():
    # Parse Args
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', type=str, default='arxiv_dataset')
    parser.add_argument('-p', type=str, default='SciArticles')
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
        parser = getattr(parsers, args['p'])
        _parser = parser(load_file_dir=args['l'])
        dataset = _parser.parse()
        corpus = create_corpus(dataset, tokenizer=parser.tokenizer)
        save_corpus(corpus, dataset_dir=str(_parser))
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
                      max_epochs=args['e'],
                      weight_decay=0.0001)
        plotter = Plotter()
        plotter.plot_training_curves(os.path.join(train_path, 'loss.tsv'))


if __name__ == '__main__':
    main()
