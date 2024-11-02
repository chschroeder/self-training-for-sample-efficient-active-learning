from collections import OrderedDict


def get_stopping_criteria_from_str(sc_strings, num_classes) -> OrderedDict:
    criteria = OrderedDict()
    for sc_string in sc_strings:
        criteria[sc_string] = _get_stopping_criterion_from_str(sc_string, num_classes)

    if len(criteria) == 0:
        return None

    return criteria


def _get_stopping_criterion_from_str(sc_string, num_classes):
    if sc_string == 'kappa':
        from small_text.stopping_criteria.kappa import KappaAverage
        return KappaAverage(num_classes)
    else:
        raise ValueError('Unknown stopping criterion string: ' + str)
