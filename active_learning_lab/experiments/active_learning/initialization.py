from small_text.data import Dataset, TextDataset


def get_initial_indices(train_set: Dataset, train_text: TextDataset, initialization_strategy: str,
                        initialization_strategy_kwargs: dict, num_samples: int):

    _unused = train_text, initialization_strategy_kwargs

    if initialization_strategy == 'random':
        from small_text.initialization import random_initialization
        x_ind_init = random_initialization(train_set, n_samples=num_samples)
    elif initialization_strategy == 'srandom':
        from small_text.initialization import random_initialization_stratified
        y_train = train_set.y
        x_ind_init = random_initialization_stratified(y_train, n_samples=num_samples)
    elif initialization_strategy == 'balanced':
        from small_text.initialization import random_initialization_balanced
        y_train = train_set.y
        x_ind_init = random_initialization_balanced(y_train, n_samples=num_samples)
    else:
        raise ValueError('Invalid initialization strategy: ' + initialization_strategy)

    return x_ind_init
