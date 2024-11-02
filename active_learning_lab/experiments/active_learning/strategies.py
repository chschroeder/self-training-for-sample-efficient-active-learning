from small_text import GreedyCoreset
from small_text.query_strategies import (RandomSampling,
                                         ContrastiveActiveLearning)
from small_text.query_strategies.strategies import BreakingTies


def query_strategy_from_str(query_strategy_name, _query_strategy_kwargs, _num_classes):

    if query_strategy_name == 'lc-bt':
        return BreakingTies()
    elif query_strategy_name == 'cal':
        return ContrastiveActiveLearning()
    elif query_strategy_name == 'gc':
        return GreedyCoreset()
    elif query_strategy_name == 'random':
        return RandomSampling()
    else:
        raise ValueError('Unknown query strategy string: ' + str)
