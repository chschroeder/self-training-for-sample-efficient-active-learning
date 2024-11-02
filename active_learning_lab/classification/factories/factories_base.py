import types

from small_text import (
    AbstractClassifierFactory,
    SetFitModelArguments
)

from active_learning_lab.experiments.active_learning.self_training.classification.factories.setfit_extended import SetFitClassificationExtendedFactory
from active_learning_lab.classification.transformers import (
    HuggingfaceTransformersClassificationFactory
)


class FactoryWrappingApplier(AbstractClassifierFactory):

    def __init__(self, base_factory, wrappers):
        self.base_factory = base_factory
        self.wrappers = wrappers

    def new(self):
        new_obj = self.base_factory.new()
        for wrapper in self.wrappers:
            new_obj = wrapper(new_obj)
        return new_obj


class SetfitProgressbarSuppressor(AbstractClassifierFactory):

    def __init__(self, base_factory):
        self.base_factory = base_factory

    def new(self):

        def fit(self, train_set, validation_set=None, setfit_train_kwargs=dict(), **kwargs):
            setfit_train_kwargs['show_progress_bar'] = False
            return self.fit_old(train_set, validation_set, setfit_train_kwargs=setfit_train_kwargs, **kwargs)

        new_obj = self.base_factory.new()
        new_obj.fit_old = new_obj.fit
        new_obj.fit = types.MethodType(fit, new_obj)
        return new_obj


def get_factory(classifier_name, num_classes, classifier_kwargs={}):

    if classifier_name == 'transformer':
        return HuggingfaceTransformersClassificationFactory(classifier_name, num_classes, kwargs=classifier_kwargs)
    elif classifier_name == 'setfit-ext':
        import copy
        classifier_kwargs_new = copy.deepcopy(classifier_kwargs)
        for key in ['transformer_model', 'early_stopping']:
            if key in classifier_kwargs_new:
                del classifier_kwargs_new[key]
        setfit_model_args = SetFitModelArguments(classifier_kwargs['transformer_model'])
        return SetfitProgressbarSuppressor(
            SetFitClassificationExtendedFactory(setfit_model_args, num_classes, classification_kwargs=classifier_kwargs_new)
        )

    raise ValueError('No factory found')
