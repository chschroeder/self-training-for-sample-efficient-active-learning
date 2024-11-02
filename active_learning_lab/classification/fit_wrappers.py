def _clf_fit_wrapper(fit_method, early_stopping):
    def fit_wrap(*args, **kwargs):
        if 'early_stopping' in kwargs:
            raise ValueError
        kwargs['early_stopping'] = early_stopping

        # temp fix
        import numpy as np
        early_stopping._history = np.empty((0,), dtype=early_stopping._dtype)

        return fit_method(*args, **kwargs)

    return fit_wrap


def wrap_clf(clf, early_stopping=None):
    fit_method = getattr(clf, 'fit')
    wrapped_fit_method = _clf_fit_wrapper(fit_method, early_stopping)
    setattr(clf, 'fit', wrapped_fit_method)
    return clf
