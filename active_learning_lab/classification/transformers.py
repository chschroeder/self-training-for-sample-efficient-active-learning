import copy
import warnings
import types
import torch

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LambdaLR

from small_text.classifiers.factories import AbstractClassifierFactory
from small_text.integrations.transformers.classifiers.classification import FineTuningArguments, TransformerModelArguments, TransformerBasedClassification


class HuggingfaceTransformersClassificationFactory(AbstractClassifierFactory):

    def __init__(self, classifier_name, num_classes, kwargs={}):
        self.classifier_name = classifier_name
        self.num_classes = num_classes
        self.kwargs = kwargs

    def new(self):

        kwargs_new = copy.deepcopy(self.kwargs)
        for key in ['transformer_model', 'transformer_config', 'transformer_tokenizer']:
            if key in kwargs_new:
                del kwargs_new[key]

        fine_tuning_args = None

        transformer_model = TransformerModelArguments(
            self.kwargs['transformer_model'],
            self.kwargs.get('transformer_tokenizer', None),
            self.kwargs.get('transformer_config', None)
        )

        classifier_cls = self.kwargs.get('classifier_cls', TransformerBasedClassification)
        if 'classifier_cls' in kwargs_new:
            del kwargs_new['classifier_cls']

        clf = classifier_cls(transformer_model,
                             num_classes=self.num_classes,
                             fine_tuning_arguments=fine_tuning_args,
                             **kwargs_new)

        clf._initialize_optimizer_and_scheduler = types.MethodType(_initialize_optimizer_and_scheduler, clf)
        clf.initialize_transformer = types.MethodType(_initialize_transformer, clf)
        clf._perform_model_selection = types.MethodType(_perform_model_selection, clf)

        return clf


# https://discuss.pytorch.org/t/load-state-dict-causes-memory-leak/36189/13
# https://github.com/pytorch/pytorch/issues/7415
def _perform_model_selection(self, optimizer, model_selection):
    model_selection_result = model_selection.select()
    if model_selection_result is not None:
        self.model.load_state_dict(torch.load(model_selection_result.model_path, map_location='cpu'))
        optimizer_path = model_selection_result.model_path.with_suffix('.pt.optimizer')
        optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu'))


from transformers import logging as transformers_logging
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def _initialize_transformer(self, cache_dir):

    self.config = AutoConfig.from_pretrained(
        self.transformer_model.config,
        num_labels=self.num_classes,
        cache_dir=cache_dir,
    )
    self.tokenizer = AutoTokenizer.from_pretrained(
        self.transformer_model.tokenizer,
        cache_dir=cache_dir,
    )

    # Suppress "Some weights of the model checkpoint at [model name] were not [...]"-warnings
    previous_verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    self.model = AutoModelForSequenceClassification.from_pretrained(
        self.transformer_model.model,
        from_tf=False,
        config=self.config,
        cache_dir=cache_dir,
    )

    try:
        import torch
        from torch import _dynamo
        #torch._dynamo.config.suppress_errors = True
        # self.model.bert = torch.compile(self.model.bert)
        # https://github.com/pytorch/pytorch/commit/e071d72f3c9ba7e58ddb4cfcf0f4563e0e522bcf
        # https://discuss.pytorch.org/t/torch-dynamo-exc-unsupported-tensor-backward/169246/2
        #self.model.bert = _dynamo.optimize('aot_eager')(self.model.bert)
    except:
        warnings.warn('torch dynamo not found: could not compile model')

    transformers_logging.set_verbosity(previous_verbosity)


def _initialize_optimizer_and_scheduler(self, optimizer, scheduler, num_epochs,
                                        sub_train, base_lr):

    steps = (len(sub_train) // self.mini_batch_size) \
            + int(len(sub_train) % self.mini_batch_size != 0)

    if optimizer is None:
        params, optimizer = self._default_optimizer(base_lr) \
            if optimizer is None else optimizer

    if scheduler == 'linear':
        try:
            from transformers import get_linear_schedule_with_warmup
            total_steps = steps * self.num_epochs
            warmup_steps = min(0.1 * total_steps, 100)

            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=warmup_steps,
                                                        num_training_steps=total_steps - warmup_steps)
        except ImportError:
            raise ValueError('Linear scheduler is only available when the transformers '
                             'integration is installed ')

    elif scheduler is None:
        # constant learning rate
        scheduler = LambdaLR(optimizer, lambda _: 1)
    elif not isinstance(scheduler, _LRScheduler):
        raise ValueError(f'Invalid scheduler: {scheduler}')

    return optimizer, scheduler
