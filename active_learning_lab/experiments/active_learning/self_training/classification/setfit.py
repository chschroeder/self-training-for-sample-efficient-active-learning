# Based on:
# https://github.com/webis-de/small-text/blob/f3456d34df56267cadbcbc1dff5f7d165be06b2f/small_text/integrations/transformers/classifiers/setfit.py

import types
import numpy as np

from small_text.base import check_optional_dependency
from small_text.classifiers.classification import Classifier
from small_text.exceptions import UnsupportedOperationException

from small_text.utils.classification import (
    empty_result,
    _multi_label_list_to_multi_hot,
    prediction_result
)
from small_text.utils.labels import csr_to_list
from small_text.utils.logging import VERBOSITY_MORE_VERBOSE

try:
    import torch

    from datasets import Dataset
    from active_learning_lab.thirdparty.setfit.trainer import SetFitTrainerExtended
    from active_learning_lab.thirdparty.setfit.modeling import SetFitModel

    from small_text.integrations.pytorch.utils.misc import _compile_if_possible, enable_dropout
    from small_text.integrations.transformers.utils.classification import (
        _get_arguments_for_from_pretrained_model
    )
    from small_text.integrations.transformers.classifiers.setfit import SetFitClassificationEmbeddingMixin
    from small_text.integrations.transformers.utils.setfit import (
        _check_model_kwargs,
        _check_trainer_kwargs,
        _check_train_kwargs,
        _truncate_texts
    )
except ImportError as e:
    print(e)


class SetFitClassificationExtended(SetFitClassificationEmbeddingMixin, Classifier):
    """A classifier that operates through Sentence Transformer Finetuning (SetFit, [TRE+22]_).

    This class is a wrapper which encapsulates the
    `Hugging Face SetFit implementation <https://github.com/huggingface/setfit>_` .

    .. note ::
       This strategy requires the optional dependency `setfit`.

    .. versionadded:: 1.2.0
    """
    def __init__(self, setfit_model_args, num_classes, multi_label=False, max_seq_len=512,
                 use_differentiable_head=False, mini_batch_size=32, model_kwargs=dict(),
                 trainer_kwargs=dict(), amp_args=None, device=None,
                 verbosity=VERBOSITY_MORE_VERBOSE):
        """
        sentence_transformer_model : SetFitModelArguments
            Settings for the sentence transformer model to be used.
        num_classes : int
            Number of classes.
        multi_label : bool, default=False
            If `False`, the classes are mutually exclusive, i.e. the prediction step results in
            exactly one predicted label per instance.
        use_differentiable_head : bool
            Uses a differentiable head instead of a logistic regression for the classification head.
            Corresponds to the keyword argument with the same name in
            `SetFitModel.from_pretrained()`.
        model_kwargs : dict
            Keyword arguments used for the SetFit model. The keyword `use_differentiable_head` is
            excluded and managed by this class. The other keywords are directly passed to
            `SetFitModel.from_pretrained()`.

            .. seealso:: `SetFit: src/setfit/modeling.py
                         <https://github.com/huggingface/setfit/blob/main/src/setfit/modeling.py>`_

        trainer_kwargs : dict
            Keyword arguments used for the SetFit model. The keyword `batch_size` is excluded and
            is instead controlled by the keyword `mini_batch_size` of this class. The other
            keywords are directly passed to `SetFitTrainer.__init__()`.

            .. seealso:: `SetFit: src/setfit/trainer.py
                         <https://github.com/huggingface/setfit/blob/main/src/setfit/trainer.py>`_
        amp_args : AMPArguments, default=None
            Configures the use of Automatic Mixed Precision (AMP).

            .. seealso:: :py:class:`~small_text.integrations.pytorch.classifiers.base.AMPArguments`
            .. versionadded:: 2.0.0

        device : str or torch.device, default=None
            Torch device on which the computation will be performed.
        verbosity : int, default=VERBOSITY_MORE_VERBOSE
            Controls the verbosity of logging messages. Lower values result in less log messages.
            Set this to `VERBOSITY_QUIET` or `0` for the minimum amount of logging.
        compile_model : bool, default=False
            Compiles the model (using `torch.compile`) if `True` and PyTorch version is greater than or equal 2.0.0.

            .. versionadded:: 2.0.0
        """
        check_optional_dependency('setfit')

        self.setfit_model_args = setfit_model_args
        self.num_classes = num_classes
        self.multi_label = multi_label

        self.model_kwargs = _check_model_kwargs(model_kwargs)
        self.trainer_kwargs = _check_trainer_kwargs(trainer_kwargs)

        self.model = None

        self.max_seq_len = max_seq_len
        self.use_differentiable_head = use_differentiable_head
        self.mini_batch_size = mini_batch_size

        self.amp_args = amp_args
        self.device = device

        self.verbosity = verbosity

    # <change>
    def fit(self, train_set, validation_set=None, weights=None, setfit_train_kwargs=dict()):
        """Trains the model using the given train set.

        Parameters
        ----------
        train_set : TextDataset
            A dataset used for training the model.
        validation_set : TextDataset or None, default None
            A dataset used for validation during training.
        weights : np.ndarray[np.float32] or None, default=None
            Sample weights or None.
        setfit_train_kwargs : dict
            Additional keyword arguments that are passed to `SetFitTrainer.train()`

        Returns
        -------
        self : SetFitClassification
            Returns the current classifier with a fitted model.
        """
        setfit_train_kwargs = _check_train_kwargs(setfit_train_kwargs)
        if self.model is None:
            self.model = self.initialize()

        if validation_set is None:
            train_set = _truncate_texts(self.model, self.max_seq_len, train_set)[0]
        else:
            train_set, validation_set = _truncate_texts(self.model, self.max_seq_len, train_set, validation_set)

        x_valid = validation_set.x if validation_set is not None else None
        y_valid = validation_set.y if validation_set is not None else None

        if self.multi_label:
            y_valid = _multi_label_list_to_multi_hot(csr_to_list(y_valid), self.num_classes) \
                if y_valid is not None else None
            y_train = _multi_label_list_to_multi_hot(csr_to_list(train_set.y), self.num_classes)
        else:
            y_valid = y_valid.tolist() if isinstance(y_valid, np.ndarray) else y_valid
            y_train = train_set.y

        sub_train, sub_valid = self._get_train_and_valid_sets(train_set.x,
                                                             y_train,
                                                             x_valid,
                                                             y_valid)

        return self._fit_main(sub_train, sub_valid, weights, setfit_train_kwargs)

    def _fit_main(self, sub_train, sub_valid, sub_train_weights, setfit_train_kwargs):
        if self.use_differentiable_head:
            raise NotImplementedError
        else:
            self.model.model_body.to(self.device)
            return self._fit(sub_train, sub_valid, sub_train_weights, setfit_train_kwargs)

    def _get_train_and_valid_sets(self, x_train, y_train, x_valid, y_valid):
        sub_train = Dataset.from_dict({'text': x_train, 'label': y_train})
        if x_valid is not None:
            sub_valid = Dataset.from_dict({'text': x_valid, 'label': y_valid})
        else:
            if self.use_differentiable_head:
                raise NotImplementedError
            else:
                sub_valid = None

        return sub_train, sub_valid

    def _fit(self, sub_train, sub_valid, sub_train_weights, setfit_train_kwargs):
        trainer = SetFitTrainerExtended(
            self.model,
            sub_train,
            eval_dataset=sub_valid,
            batch_size=self.mini_batch_size,
            train_weights=sub_train_weights,
            **self.trainer_kwargs
        )
        if not 'show_progress_bar' in setfit_train_kwargs:
            setfit_train_kwargs['show_progress_bar'] = self.verbosity >= VERBOSITY_MORE_VERBOSE
        trainer.train(max_length=self.max_seq_len,
                      **setfit_train_kwargs)
        return self
    # </change>

    def initialize(self):
        from_pretrained_options = _get_arguments_for_from_pretrained_model(
            self.setfit_model_args.model_loading_strategy
        )
        model_kwargs = self.model_kwargs.copy()
        if self.multi_label and 'multi_target_strategy' not in model_kwargs:
            model_kwargs['multi_target_strategy'] = 'one-vs-rest'

        model = SetFitModel.from_pretrained(
            self.setfit_model_args.sentence_transformer_model,
            use_differentiable_head=self.use_differentiable_head,
            force_download=from_pretrained_options.force_download,
            local_files_only=from_pretrained_options.local_files_only,
            **model_kwargs
        )
        model.model_body = _compile_if_possible(model.model_body, compile_model=self.setfit_model_args.compile_model)
        return model

    def validate(self, _validation_set):
        if self.use_differentiable_head:
            raise NotImplementedError()
        else:
            raise UnsupportedOperationException(
                'validate() is not available when use_differentiable_head is set to False'
            )

    def predict(self, dataset, return_proba=False):
        """Predicts the labels for the given dataset.

        Parameters
        ----------
        dataset : TextDataset
            A dataset on whose instances predictions are made.
        return_proba : bool, default=False
            If True, additionally returns the confidence distribution over all classes.

        Returns
        -------
        predictions : np.ndarray[np.int32] or csr_matrix[np.int32]
            List of predictions if the classifier was fitted on single-label data,
            otherwise a sparse matrix of predictions.
        probas : np.ndarray[np.float32], optional
            List of probabilities (or confidence estimates) if `return_proba` is True.
        """
        if len(dataset) == 0:
            return empty_result(self.multi_label, self.num_classes, return_prediction=True,
                                return_proba=return_proba)

        proba = self.predict_proba(dataset)
        predictions = prediction_result(proba, self.multi_label, self.num_classes)

        if return_proba:
            return predictions, proba

        return predictions

    def predict_proba(self, dataset, dropout_sampling=1):
        """Predicts the label distributions.

        Parameters
        ----------
        dataset : TextDataset
            A dataset whose labels will be predicted.
        dropout_sampling : int, default=1
            If `dropout_sampling > 1` then all dropout modules will be enabled during prediction and
            multiple rounds of predictions will be sampled for each instance.

        Returns
        -------
        scores : np.ndarray
            Confidence score distribution over all classes of shape (num_samples, num_classes).
            If `dropout_sampling > 1` then the shape is (num_samples, dropout_sampling, num_classes).

        .. warning::
           This function is not thread-safe if `dropout_sampling > 1`, since the underlying model gets
           temporarily modified.
        """
        if len(dataset) == 0:
            return empty_result(self.multi_label, self.num_classes, return_prediction=False,
                                return_proba=True)
        dataset = _truncate_texts(self.model, self.max_seq_len, dataset)[0]

        if self.use_differentiable_head:
            raise NotImplementedError()

        with torch.no_grad():
            if dropout_sampling <= 1:
                return self._predict_proba(dataset)
            else:
                return self._predict_proba_dropout_sampling(dataset, dropout_samples=dropout_sampling)

    def _predict_proba(self, dataset):
        proba = np.empty((0, self.num_classes), dtype=float)

        num_batches = int(np.ceil(len(dataset) / self.mini_batch_size))
        for batch in np.array_split(dataset.x, num_batches, axis=0):
            proba_tmp = np.zeros((batch.shape[0], self.num_classes), dtype=float)
            proba_tmp[:, self.model.model_head.classes_] = self.model.predict_proba(batch)
            proba = np.append(proba, proba_tmp, axis=0)

        return proba

    def _predict_proba_dropout_sampling(self, dataset, dropout_samples=2):
        # this whole method be done much more efficiently but this solution works without modifying setfit's code

        self.model.model_body.train()
        model_body_eval = self.model.model_body.eval
        self.model.model_body.eval = types.MethodType(lambda x: x, self.model.model_body)

        proba = np.empty((0, dropout_samples, self.num_classes), dtype=float)
        proba[:, :, :] = np.inf

        with enable_dropout(self.model.model_body):
            num_batches = int(np.ceil(len(dataset) / self.mini_batch_size))
            for batch in np.array_split(dataset.x, num_batches, axis=0):
                samples = np.empty((dropout_samples, len(batch), self.num_classes), dtype=float)
                for i in range(dropout_samples):
                    proba_tmp = np.zeros((batch.shape[0], self.num_classes), dtype=float)
                    proba_tmp[:, self.model.model_head.classes_] = self.model.predict_proba(batch)
                    samples[i] = proba_tmp

                samples = np.swapaxes(samples, 0, 1)
                proba = np.append(proba, samples, axis=0)

        self.model.model_body.eval = model_body_eval

        return proba

    def __del__(self):
        try:
            attrs = ['model']
            for attr in attrs:
                delattr(self, attr)
        except Exception:
            pass
