import datasets

from small_text import balanced_sampling
from active_learning_lab.utils.experiment import set_random_seed


def main():
    dbp_dataset = datasets.load_dataset('dbpedia_14')

    set_random_seed(42)
    indices = balanced_sampling(dbp_dataset['train']['label'], 140_000)

    labels = dbp_dataset['train']['label']
    titles = dbp_dataset['train']['title']
    content = dbp_dataset['train']['content']

    dbp_dataset['train'] = datasets.Dataset.from_dict({
        'label': [labels[i] for i in indices],
        'title': [titles[i] for i in indices],
        'content': [content[i] for i in indices]
    })

    dbp_dataset['train'].to_parquet('./data/dbp-140k/train.parquet')
    dbp_dataset['test'].to_parquet('./data/dbp-140k/test.parquet')


if __name__ == '__main__':
    main()
