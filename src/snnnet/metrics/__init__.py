"""This module implements utility classes for measuring model training and outputs."""

import torch


class BinaryClassificationSummary:
    """This class implements a summary system for keeping track of a binary
    classification summary. It stores the number of (true / false) (positives / negatives),
    and computes the accuracy, precision and recall from those.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all the counters in this summary."""
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def record_results(self, true_labels, predicted_labels):
        """Records the given true and predicted labels.

        Parameters
        ----------
        true_labels : torch.Tensor
            A binary tensor indicating the true labels.
        predicted_labels : torch.Tensor
            A binary tensor representing the predicted labels.
        """
        self.true_positives += torch.logical_and(true_labels, predicted_labels).sum()
        self.false_positives += torch.logical_and(torch.logical_not(true_labels), predicted_labels).sum()
        self.true_negatives += torch.logical_not(torch.logical_or(true_labels, predicted_labels)).sum()
        self.false_negatives += torch.logical_and(true_labels, torch.logical_not(predicted_labels)).sum()

    @property
    def count(self):
        return (self.true_positives + self.true_negatives + self.false_positives + self.false_negatives)

    @property
    def accuracy(self):
        return (self.true_positives + self.true_negatives).true_divide(self.count)

    @property
    def precision(self):
        return self.true_positives.true_divide(self.true_positives + self.false_positives)

    @property
    def recall(self):
        return self.true_positives.true_divide(self.true_positives + self.false_negatives)

    @property
    def f1(self):
        p = self.precision
        r = self.recall
        p_and_r = p + r
        return torch.where(p_and_r > 0, 2 * p * r / p_and_r, 0.0)


class ClassificationSummary:
    """ Simple class to keep track of summaries of a classification problem. """
    def __init__(self, num_classes=2, device=None):
        """ Initializes a new summary class with the given number of outcomes.

        Parameters
        ----------
        num_classes : int
            The number of possible classes of the classification problem.
        """
        self.recorded = torch.zeros(num_classes * num_classes, dtype=torch.int32, device=device)
        self.num_classes = num_classes

    @property
    def prediction_matrix(self):
        return self.recorded.view((self.num_classes, self.num_classes))

    def record_results(self, labels, predictions):
        """ Records statistics for a batch of predictions.

        Parameters
        ----------
        labels : torch.Tensor
            An array of true labels in integer format. Each label must correspond to an
            integer in 0 to `num_classes` - 1 inclusive.
        predictions : torch.Tensor
            an array of predicted labels. Must follow the same format as `labels`.
        """
        indices = torch.add(labels.int(), self.num_classes, predictions.int()).long()
        self.recorded = self.recorded.scatter_add_(
            0, indices, torch.ones_like(indices, dtype=torch.int32))

    def reset(self):
        """ Resets statistics recorded in this accumulator. """
        self.recorded = torch.zeros_like(self.recorded)

    @property
    def accuracy(self):
        """ Compute the accuracy of the recorded problem. """
        num_correct = self.prediction_matrix.diag().sum()
        num_total = self.recorded.sum()

        return num_correct.float() / num_total.float()

    @property
    def confusion_matrix(self):
        return self.prediction_matrix.float() / self.prediction_matrix.sum().float()

    @property
    def cohen_kappa(self):
        pm = self.prediction_matrix.float()
        N = self.recorded.sum().float()

        p_observed = pm.diag().sum() / N
        p_expected = torch.dot(pm.sum(dim=0), pm.sum(dim=1)) / (N * N)

        return 1 - (1 - p_observed) / (1 - p_expected)

    @property
    def marginal_labels(self):
        return self.prediction_matrix.sum(dim=0).float() / self.recorded.sum().float()

    @property
    def marginal_predicted(self):
        return self.prediction_matrix.sum(dim=1).float() / self.recorded.sum().float()

    @property
    def count(self):
        return self.recorded.sum()
