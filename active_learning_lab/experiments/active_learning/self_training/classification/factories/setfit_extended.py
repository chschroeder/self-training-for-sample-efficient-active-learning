from small_text import AbstractClassifierFactory

from active_learning_lab.experiments.active_learning.self_training.classification.setfit import SetFitClassificationExtended


class SetFitClassificationExtendedFactory(AbstractClassifierFactory):
    """
    .. versionadded:: 1.2.0
    """

    def __init__(self, setfit_model_args, num_classes, classification_kwargs={}):
        """
        Parameters
        ----------
        setfit_model_args : SetFitModelArguments
            Name of the sentence transformer model.
        num_classes : int
            Number of classes.
        classification_kwargs : dict
            Keyword arguments which will be passed to `SetFitClassification`.
        """
        self.setfit_model_args = setfit_model_args
        self.num_classes = num_classes
        self.classification_kwargs = classification_kwargs

    def new(self):
        """Creates a new SetFitClassification instance.

        Returns
        -------
        classifier : SetFitClassification
            A new instance of SetFitClassification which is initialized with the given keyword args `kwargs`.
        """
        return SetFitClassificationExtended(self.setfit_model_args,
                                            self.num_classes,
                                            **self.classification_kwargs)
